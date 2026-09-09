"""Forecast entry point: validate, predict, normalise, record.

Covers all four RFC forecasting modes: A (univariate), B (multiple independent
series), C (multivariate / multi-target) and D (covariate-informed, past-only
and known-future). Phase 2 lifted the multi-target and covariate refusals that
Phase 1 carried; the validation layer already implemented the rules for both.

**Numeric covariates only.** See :func:`chronos2_pipeline.validation
._coerce_covariates` for why a categorical covariate is refused rather than
handed to an encoder whose route depends on the number of targets.

Five upstream behaviours are asserted on every call rather than trusted:

* ``predict_df`` returns the columns the rename map expects;
* it returns one row per (series, target, step), in that order — so a forecast
  cannot be exported under the wrong series or target label;
* the returned point and quantile values are finite;
* when ``0.5`` is requested, ``predictions`` is *exactly* the ``"0.5"`` column;
* every exported ``q<level>`` column names a level the loaded model was actually
  trained on, so no substituted quantile can leave under a borrowed label.

The second matters only once there is more than one target, and it is the
multi-target half of RFC C-1: upstream lays the frame out as ``target_name =
np.tile(np.repeat(target, prediction_length), n_inputs)``
(``chronos/chronos2/pipeline.py`` L955 in 2.3.1) and the values arrive in that
same order from a ravelled array. Nothing in the frame ties a row's number back
to the series it came from, so a change to either order would relabel forecasts
silently rather than fail.

The fourth is the median oracle (RFC C-2). Upstream 2.3.1 computes it that way
at ``chronos/chronos2/pipeline.py`` L816-818 — ``# NOTE: the median is returned
as the mean here`` — while its own docstrings call the value a mean. If a future
release makes it an actual mean, this assertion fails loudly instead of
relabelling a different statistic as ``prediction``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import ForecastConfig
from .errors import UpstreamContractError
from .model import LoadedModel
from .provenance import build_provenance
from .validation import DEFAULT_LIMITS, ResourceLimits, validate_forecast_request

__all__ = [
    "ForecastResult",
    "NORMALIZED_ID_COLUMN",
    "NORMALIZED_TIMESTAMP_COLUMN",
    "NORMALIZED_TARGET_COLUMN",
    "NORMALIZED_POINT_COLUMN",
    "RAW_POINT_COLUMN",
    "build_rename_map",
    "normalized_columns",
    "quantile_column_name",
    "forecast",
]

NORMALIZED_ID_COLUMN = "series_id"
NORMALIZED_TIMESTAMP_COLUMN = "timestamp"
NORMALIZED_TARGET_COLUMN = "target_name"
NORMALIZED_POINT_COLUMN = "prediction"
RAW_POINT_COLUMN = "predictions"


def quantile_column_name(level: float) -> str:
    """``0.1 -> "q0.1"``. Upstream names the raw column ``str(level)``."""
    return f"q{level}"


def build_rename_map(config: ForecastConfig) -> dict[str, str]:
    """The deterministic upstream-raw to DIMER-normalised column map.

    ``target_name`` is deliberately absent: it is carried through unchanged, and
    a rename entry would imply it is optional.
    """
    mapping = {
        config.id_column: NORMALIZED_ID_COLUMN,
        config.timestamp_column: NORMALIZED_TIMESTAMP_COLUMN,
        RAW_POINT_COLUMN: NORMALIZED_POINT_COLUMN,
    }
    for level in config.quantile_levels:
        mapping[str(level)] = quantile_column_name(level)
    return mapping


def normalized_columns(config: ForecastConfig) -> list[str]:
    """Exact column order of the normalised forecast frame."""
    return [
        NORMALIZED_ID_COLUMN,
        NORMALIZED_TIMESTAMP_COLUMN,
        NORMALIZED_TARGET_COLUMN,
        NORMALIZED_POINT_COLUMN,
        *(quantile_column_name(q) for q in config.quantile_levels),
    ]


@dataclass(frozen=True)
class ForecastResult:
    """A normalised forecast, the raw upstream frame, and full provenance."""

    forecast: pd.DataFrame
    raw: pd.DataFrame
    provenance: dict[str, Any]

    @property
    def model(self) -> dict[str, Any]:
        return self.provenance["model"]

    @property
    def inference(self) -> dict[str, Any]:
        return self.provenance["inference"]


def _assert_raw_contract(raw: pd.DataFrame, config: ForecastConfig) -> None:
    expected = [
        config.id_column,
        config.timestamp_column,
        NORMALIZED_TARGET_COLUMN,
        RAW_POINT_COLUMN,
        *(str(q) for q in config.quantile_levels),
    ]
    missing = [c for c in expected if c not in raw.columns]
    if missing:
        raise UpstreamContractError(
            f"predict_df returned a frame missing expected column(s) {missing}. "
            f"Present: {list(raw.columns)}. The pinned upstream contract recorded in "
            f"docs/rfc/0001-chronos-2.md no longer holds."
        )


def _assert_row_layout(
    raw: pd.DataFrame, config: ForecastConfig, series_ids: list[Any], horizon: int
) -> None:
    """One row per (series, target, step), in that order.

    Upstream builds the frame by ravelling a ``[n_tasks, n_variates, horizon]``
    array against row labels it generates separately — ``target_name`` from
    ``np.tile(np.repeat(target, prediction_length), n_inputs)`` and the id/
    timestamp block from ``future.iloc[np.repeat(item_rows, n_variates)]``
    (``chronos/chronos2/pipeline.py`` L951-957 in 2.3.1). The labels and the
    values are therefore aligned by construction and by nothing else: if a
    future release reorders either, every forecast still arrives, each one
    attached to the wrong series or the wrong target, and no column in the frame
    would contradict it.

    Series order is upstream's ``PreparedInput`` order, "the order in which
    item_ids first appear in df" (``preprocess.py`` L164). The validated history
    is sorted by id, so first appearance is that sorted order — the same list
    :func:`validate_forecast_request` returns.
    """
    targets = config.target_names
    expected_rows = len(series_ids) * len(targets) * horizon
    if len(raw) != expected_rows:
        raise UpstreamContractError(
            f"predict_df returned {len(raw)} row(s); the pinned contract is one row per "
            f"(series, target, step) = {len(series_ids)} x {len(targets)} x {horizon} = "
            f"{expected_rows}. The recorded upstream layout no longer holds, and rows "
            f"cannot be attributed to a series and target by position."
        )

    expected_targets = np.tile(np.repeat(targets, horizon), len(series_ids))
    observed_targets = raw[NORMALIZED_TARGET_COLUMN].to_numpy()
    if not np.array_equal(observed_targets, expected_targets):
        first = int(np.argmax(observed_targets != expected_targets))
        raise UpstreamContractError(
            f"predict_df returned target_name in an unexpected order: row {first} is "
            f"{observed_targets[first]!r}, the pinned layout puts "
            f"{expected_targets[first]!r} there. Values are aligned to these labels by "
            f"position alone, so a reordering would relabel forecasts silently."
        )

    expected_ids = np.repeat(np.asarray(series_ids, dtype=object), len(targets) * horizon)
    observed_ids = raw[config.id_column].to_numpy(dtype=object)
    if not np.array_equal(observed_ids, expected_ids):
        first = int(np.argmax(observed_ids != expected_ids))
        raise UpstreamContractError(
            f"predict_df returned series ids in an unexpected order: row {first} is "
            f"{observed_ids[first]!r}, the pinned layout puts {expected_ids[first]!r} "
            f"there. Values are aligned to these labels by position alone, so a "
            f"reordering would attribute forecasts to the wrong series."
        )


def _assert_finite(raw: pd.DataFrame, config: ForecastConfig) -> None:
    """A non-finite forecast is a distinct failure, and must be named as one.

    Checked before the median oracle: ``NaN == NaN`` is ``False``, so a
    NaN-valued output used to surface as "the median contract broke", sending a
    maintainer to the wrong upstream line for a condition that has nothing to do
    with the median (review R-9).
    """
    columns = [RAW_POINT_COLUMN, *(str(q) for q in config.quantile_levels)]
    offenders = {}
    for column in columns:
        values = pd.to_numeric(raw[column], errors="coerce").to_numpy(dtype=float)
        n_bad = int((~np.isfinite(values)).sum())
        if n_bad:
            offenders[column] = n_bad
    if offenders:
        raise UpstreamContractError(
            f"predict_df returned non-finite values (NaN/inf) in column(s) {offenders}. "
            f"A forecast frame containing NaN is not a forecast; v1 refuses it rather "
            f"than exporting it."
        )


def _assert_median_oracle(raw: pd.DataFrame, config: ForecastConfig) -> None:
    if 0.5 not in [float(q) for q in config.quantile_levels]:
        return
    point = raw[RAW_POINT_COLUMN].to_numpy()
    median = raw["0.5"].to_numpy()
    if point.shape != median.shape or not np.array_equal(point, median, equal_nan=True):
        n_diff = int((point != median).sum()) if point.shape == median.shape else -1
        raise UpstreamContractError(
            f"Upstream `predictions` is no longer exactly the 0.5 quantile "
            f"({n_diff} differing row(s)). This pipeline labels the point forecast "
            f"`prediction` on the recorded basis that it is the median; that basis "
            f"has changed and the label would now be wrong."
        )


def _assert_quantile_labels(
    config: ForecastConfig, trained_quantiles: tuple[float, ...] | list[float]
) -> None:
    """No ``q<level>`` column may be exported for a level the model cannot produce.

    Validation already hard-fails an out-of-grid request, so this can only fire
    if that gate is ever weakened or bypassed. It is here because the failure it
    guards against is silent: upstream substitutes the nearest trained level and
    returns it under the requested name, so a mislabelled column looks exactly
    like a correct one (RFC C-5, review R-2).
    """
    grid = {float(q) for q in trained_quantiles}
    mislabelled = [float(q) for q in config.quantile_levels if float(q) not in grid]
    if mislabelled:
        raise UpstreamContractError(
            f"Refusing to export quantile column(s) "
            f"{[quantile_column_name(q) for q in mislabelled]}: level(s) {mislabelled} "
            f"are not members of the trained grid {sorted(grid)}, so upstream can only "
            f"have substituted a different level under that name."
        )


def forecast(
    history_df: pd.DataFrame,
    config: ForecastConfig,
    model: LoadedModel,
    future_df: pd.DataFrame | None = None,
    *,
    limits: ResourceLimits = DEFAULT_LIMITS,
    allow_unroll: bool = False,
    target_policy: str = "strict",
    measure_latency: bool = False,
) -> ForecastResult:
    """Forecast one or more targets across one or more series.

    Parameters
    ----------
    history_df
        Long-format history: id column, timestamp column, one column per target.
        Every remaining column is a covariate — past-only unless it also appears
        in ``future_df``. Covariates must be numeric.
    config
        User parameters. ``config.target`` names one column or several.
    model
        The result of :func:`chronos2_pipeline.model.load_pinned_model`.
    future_df
        Known-future covariate values: the id and timestamp columns, plus at
        least one covariate that also exists in ``history_df``, with exactly
        ``prediction_length`` rows per series starting at the forecast origin.
        Target columns here are refused as leakage.
    allow_unroll
        Permit ``prediction_length`` beyond the model's native horizon, which
        upstream satisfies by autoregressive unrolling. Off by default; when on,
        ``autoregressive_unrolled`` is recorded in provenance.
    measure_latency
        Run one discarded warm-up ``predict_df`` before the scored call. Doubles
        the cost; off by default, in which case ``latency_seconds`` is a cold
        measurement and ``warm_up_performed`` is ``False``.

    Raises
    ------
    ValidationError
        Any of RFC rules 1-21.
    UpstreamContractError
        ``predict_df`` no longer matches the recorded pinned contract.
    """
    validated = validate_forecast_request(
        history_df,
        config,
        trained_quantiles=model.trained_quantiles,
        model_context_length=model.model_context_length,
        model_prediction_length=model.model_prediction_length,
        future_df=future_df,
        limits=limits,
        target_policy=target_policy,
        allow_unroll=allow_unroll,
    )

    predict_kwargs: dict[str, Any] = {
        "future_df": validated.future,
        "id_column": config.id_column,
        "timestamp_column": config.timestamp_column,
        #: Always a list, even for one target. Upstream wraps a bare string in a
        #: list anyway (``pipeline.py`` L900-901), so passing one is the same
        #: request with the shape stated rather than inferred.
        "target": list(config.target_names),
        "prediction_length": config.prediction_length,
        "quantile_levels": [float(q) for q in config.quantile_levels],
        "batch_size": config.batch_size,
        "context_length": config.context_length,
        "cross_learning": config.cross_learning,
        "validate_inputs": True,
        #: Only an alias validation affirmatively confirmed against the observed
        #: interval, never the raw request: upstream uses ``freq`` as-is to lay
        #: out the horizon and does not check it against the data (review R-1).
        "freq": validated.confirmed_frequency,
    }

    if measure_latency:
        model.pipeline.predict_df(validated.history, **predict_kwargs)

    started = time.perf_counter()
    raw = model.pipeline.predict_df(validated.history, **predict_kwargs)
    latency_seconds = time.perf_counter() - started

    _assert_raw_contract(raw, config)
    _assert_row_layout(raw, config, validated.series_ids, config.prediction_length)
    _assert_finite(raw, config)
    _assert_median_oracle(raw, config)
    _assert_quantile_labels(config, model.trained_quantiles)

    rename_map = build_rename_map(config)
    normalized = raw.rename(columns=rename_map)[normalized_columns(config)].reset_index(drop=True)

    model_ctx = model.model_context_length
    requested_ctx = config.context_length
    #: The context bound actually in force. ``min(requested, model)`` alone reported a
    #: ceiling rather than an effective value: a 100-observation series with
    #: ``context_length=None`` exported the model's full context length, a number larger
    #: than the history that existed. Upstream can only consume what a series holds, so
    #: the longest series caps it too (RFC C-6). Series shorter than this contributed
    #: less, which is why both observed lengths are exported alongside it.
    effective_ctx = min(requested_ctx or model_ctx, model_ctx, validated.max_series_length)
    effective_horizon = config.prediction_length

    provenance = build_provenance(
        model.identity,
        device=model.device,
        dtype=model.dtype,
        n_ids=validated.n_ids,
        n_targets=validated.n_targets,
        n_covariates=validated.n_covariates,
        past_covariate_names=list(validated.past_covariate_names),
        known_future_covariate_names=list(validated.known_future_covariate_names),
        requested_context_length=requested_ctx,
        effective_context_length=effective_ctx,
        longest_series_length=validated.max_series_length,
        shortest_series_length=validated.min_series_length,
        requested_prediction_length=config.prediction_length,
        effective_prediction_length=effective_horizon,
        autoregressive_unrolled=effective_horizon > model.model_prediction_length,
        requested_quantiles=validated.requested_quantiles,
        effective_quantiles=validated.requested_quantiles,
        batch_size=config.batch_size,
        cross_learning=config.cross_learning,
        latency_seconds=latency_seconds,
        warm_up_performed=measure_latency,
        frequency=config.frequency if config.frequency is not None else "",
        observed_frequency=str(validated.frequency),
    )

    return ForecastResult(forecast=normalized, raw=raw, provenance=provenance)
