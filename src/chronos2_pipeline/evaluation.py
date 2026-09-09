"""Chronological, leakage-safe evaluation and naive baselines.

This module deliberately operates on already-normalised forecast frames and on
held-out truth. It never trains or refits the model. The only split helper is
chronological: the last ``horizon`` timestamps of each series are removed from
history before inference, so future targets cannot leak into model context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import ForecastConfig
from .errors import ValidationError
from .inference import (
    NORMALIZED_ID_COLUMN,
    NORMALIZED_POINT_COLUMN,
    NORMALIZED_TARGET_COLUMN,
    NORMALIZED_TIMESTAMP_COLUMN,
    quantile_column_name,
)

__all__ = [
    "EvaluationResult",
    "HoldoutSplit",
    "chronological_holdout",
    "evaluate_forecast",
    "seasonal_naive_baseline",
    "last_value_baseline",
]


@dataclass(frozen=True)
class HoldoutSplit:
    """History and future truth produced by a chronological tail split."""

    history: pd.DataFrame
    truth: pd.DataFrame
    horizon: int


@dataclass(frozen=True)
class EvaluationResult:
    """Aligned row-level evidence plus per-series and aggregate metrics."""

    aligned: pd.DataFrame
    per_series: pd.DataFrame
    aggregate: dict[str, Any]
    quantiles: pd.DataFrame


def _fail(code: str, message: str, **details: Any) -> None:
    raise ValidationError(code, message, details)


def _prepared(frame: pd.DataFrame, config: ForecastConfig, table: str) -> pd.DataFrame:
    required = [config.id_column, config.timestamp_column, *config.target_names]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        _fail(
            "EVALUATION_MISSING_COLUMNS",
            f"{table} is missing required column(s) {missing}.",
            table=table,
            missing=missing,
        )
    out = frame.copy()
    out[config.timestamp_column] = pd.to_datetime(out[config.timestamp_column], errors="coerce")
    if out[config.timestamp_column].isna().any():
        _fail(
            "EVALUATION_BAD_TIMESTAMP",
            f"{table} contains timestamp values that cannot be parsed.",
            table=table,
        )
    if out[config.id_column].isna().any():
        _fail("EVALUATION_NULL_ID", f"{table} contains null series ids.", table=table)
    if out.duplicated([config.id_column, config.timestamp_column]).any():
        _fail(
            "EVALUATION_DUPLICATE_TIMESTAMP",
            f"{table} contains duplicate (id, timestamp) rows.",
            table=table,
        )
    for target in config.target_names:
        values = pd.to_numeric(out[target], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
            _fail(
                "EVALUATION_BAD_TARGET",
                f"{table} target {target!r} contains missing or non-finite values.",
                table=table,
                target=target,
            )
        out[target] = values.astype(float)
    return out.sort_values(
        [config.id_column, config.timestamp_column], kind="mergesort"
    ).reset_index(drop=True)


def chronological_holdout(
    frame: pd.DataFrame,
    config: ForecastConfig,
    *,
    horizon: int | None = None,
    minimum_context: int = 3,
) -> HoldoutSplit:
    """Remove the last ``horizon`` timestamps of every series as future truth.

    No random split helper exists in the public API. Every target value in the
    returned ``truth`` frame is absent from ``history``.
    """

    data = _prepared(frame, config, "evaluation source")
    holdout = config.prediction_length if horizon is None else horizon
    if isinstance(holdout, bool) or not isinstance(holdout, int) or holdout < 1:
        _fail(
            "EVALUATION_INVALID_HORIZON",
            f"horizon must be a positive integer, got {holdout!r}.",
            horizon=holdout,
        )
    if (
        isinstance(minimum_context, bool)
        or not isinstance(minimum_context, int)
        or minimum_context < 1
    ):
        _fail(
            "EVALUATION_INVALID_MINIMUM_CONTEXT",
            f"minimum_context must be a positive integer, got {minimum_context!r}.",
            minimum_context=minimum_context,
        )

    history_parts: list[pd.DataFrame] = []
    truth_parts: list[pd.DataFrame] = []
    for series_id, block in data.groupby(config.id_column, sort=False):
        if len(block) < holdout + minimum_context:
            _fail(
                "EVALUATION_SERIES_TOO_SHORT",
                f"series {series_id!r} has {len(block)} row(s); need at least "
                f"{holdout + minimum_context} for a {holdout}-step holdout with "
                f"{minimum_context} context rows.",
                series_id=series_id,
                n_rows=len(block),
                horizon=holdout,
                minimum_context=minimum_context,
            )
        history_parts.append(block.iloc[:-holdout].copy())
        truth_parts.append(block.iloc[-holdout:].copy())

    history = pd.concat(history_parts, ignore_index=True)
    truth = pd.concat(truth_parts, ignore_index=True)

    for series_id in history[config.id_column].drop_duplicates().tolist():
        h = history[history[config.id_column] == series_id]
        t = truth[truth[config.id_column] == series_id]
        if not h[config.timestamp_column].max() < t[config.timestamp_column].min():
            _fail(
                "EVALUATION_HOLDOUT_OVERLAP",
                f"chronological split for series {series_id!r} overlaps the held-out future.",
                series_id=series_id,
            )

    return HoldoutSplit(history=history, truth=truth, horizon=holdout)


def _truth_long(truth_df: pd.DataFrame, config: ForecastConfig) -> pd.DataFrame:
    truth = _prepared(truth_df, config, "truth")
    long = truth.melt(
        id_vars=[config.id_column, config.timestamp_column],
        value_vars=config.target_names,
        var_name=NORMALIZED_TARGET_COLUMN,
        value_name="truth",
    ).rename(
        columns={
            config.id_column: NORMALIZED_ID_COLUMN,
            config.timestamp_column: NORMALIZED_TIMESTAMP_COLUMN,
        }
    )
    return long[
        [
            NORMALIZED_ID_COLUMN,
            NORMALIZED_TIMESTAMP_COLUMN,
            NORMALIZED_TARGET_COLUMN,
            "truth",
        ]
    ].sort_values(
        [NORMALIZED_ID_COLUMN, NORMALIZED_TARGET_COLUMN, NORMALIZED_TIMESTAMP_COLUMN],
        kind="mergesort",
    ).reset_index(drop=True)


def _normalised_forecast(forecast_df: pd.DataFrame, config: ForecastConfig) -> pd.DataFrame:
    required = [
        NORMALIZED_ID_COLUMN,
        NORMALIZED_TIMESTAMP_COLUMN,
        NORMALIZED_TARGET_COLUMN,
        NORMALIZED_POINT_COLUMN,
    ]
    missing = [column for column in required if column not in forecast_df.columns]
    if missing:
        _fail(
            "EVALUATION_FORECAST_SCHEMA",
            f"forecast is missing required normalised column(s) {missing}.",
            missing=missing,
        )
    out = forecast_df.copy()
    out[NORMALIZED_TIMESTAMP_COLUMN] = pd.to_datetime(
        out[NORMALIZED_TIMESTAMP_COLUMN], errors="coerce"
    )
    if out[NORMALIZED_TIMESTAMP_COLUMN].isna().any():
        _fail("EVALUATION_FORECAST_SCHEMA", "forecast contains unparseable timestamps.")
    if out.duplicated(
        [NORMALIZED_ID_COLUMN, NORMALIZED_TIMESTAMP_COLUMN, NORMALIZED_TARGET_COLUMN]
    ).any():
        _fail(
            "EVALUATION_FORECAST_SCHEMA",
            "forecast contains duplicate (series_id, timestamp, target_name) rows.",
        )
    point = pd.to_numeric(out[NORMALIZED_POINT_COLUMN], errors="coerce")
    if point.isna().any() or not np.isfinite(point.to_numpy(dtype=float)).all():
        _fail("EVALUATION_FORECAST_NONFINITE", "forecast point predictions must be finite.")
    out[NORMALIZED_POINT_COLUMN] = point.astype(float)

    for level in config.quantile_levels:
        column = quantile_column_name(level)
        if column in out.columns:
            values = pd.to_numeric(out[column], errors="coerce")
            if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
                _fail(
                    "EVALUATION_FORECAST_NONFINITE",
                    f"forecast quantile column {column!r} must be finite.",
                    column=column,
                )
            out[column] = values.astype(float)
    return out


def evaluate_forecast(
    forecast_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    config: ForecastConfig,
) -> EvaluationResult:
    """Score a DIMER-normalised forecast against held-out truth.

    Point metrics are MAE/RMSE. Quantile metrics use pinball loss. If at least
    two requested quantile columns are present, empirical coverage is measured
    between the lowest and highest requested levels.
    """

    forecast = _normalised_forecast(forecast_df, config)
    truth = _truth_long(truth_df, config)
    keys = [NORMALIZED_ID_COLUMN, NORMALIZED_TIMESTAMP_COLUMN, NORMALIZED_TARGET_COLUMN]
    merged = truth.merge(forecast, on=keys, how="outer", indicator=True, validate="one_to_one")
    if not (merged["_merge"] == "both").all():
        missing_forecast = merged.loc[merged["_merge"] == "left_only", keys].to_dict("records")
        missing_truth = merged.loc[merged["_merge"] == "right_only", keys].to_dict("records")
        _fail(
            "EVALUATION_COVERAGE_MISMATCH",
            "forecast and truth do not cover exactly the same (series, timestamp, target) rows.",
            missing_forecast=missing_forecast[:10],
            missing_truth=missing_truth[:10],
        )
    merged = merged.drop(columns="_merge")
    merged["error"] = merged[NORMALIZED_POINT_COLUMN] - merged["truth"]
    merged["absolute_error"] = merged["error"].abs()
    merged["squared_error"] = merged["error"] ** 2

    per_series = (
        merged.groupby([NORMALIZED_ID_COLUMN, NORMALIZED_TARGET_COLUMN], sort=True)
        .agg(
            n=("truth", "size"),
            mae=("absolute_error", "mean"),
            mse=("squared_error", "mean"),
        )
        .reset_index()
    )
    per_series["rmse"] = np.sqrt(per_series.pop("mse"))
    per_series = per_series[
        [NORMALIZED_ID_COLUMN, NORMALIZED_TARGET_COLUMN, "n", "mae", "rmse"]
    ]

    aggregate: dict[str, Any] = {
        "n": int(len(merged)),
        "mae": float(merged["absolute_error"].mean()),
        "rmse": float(np.sqrt(merged["squared_error"].mean())),
    }

    quantile_rows: list[dict[str, Any]] = []
    present_levels: list[float] = []
    for level in config.quantile_levels:
        column = quantile_column_name(level)
        if column not in merged.columns:
            continue
        present_levels.append(float(level))
        residual = merged["truth"] - merged[column]
        loss = np.where(
            residual >= 0.0,
            float(level) * residual,
            (float(level) - 1.0) * residual,
        )
        quantile_rows.append(
            {
                "quantile": float(level),
                "n": int(len(merged)),
                "pinball_loss": float(np.mean(loss)),
            }
        )
    quantiles = pd.DataFrame(quantile_rows, columns=["quantile", "n", "pinball_loss"])

    if len(present_levels) >= 2:
        lower = min(present_levels)
        upper = max(present_levels)
        lower_col = quantile_column_name(lower)
        upper_col = quantile_column_name(upper)
        covered = (merged["truth"] >= merged[lower_col]) & (merged["truth"] <= merged[upper_col])
        aggregate.update(
            {
                "interval_lower_quantile": lower,
                "interval_upper_quantile": upper,
                "interval_coverage": float(covered.mean()),
            }
        )

    return EvaluationResult(
        aligned=merged.sort_values(keys, kind="mergesort").reset_index(drop=True),
        per_series=per_series,
        aggregate=aggregate,
        quantiles=quantiles,
    )


def last_value_baseline(
    history_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    config: ForecastConfig,
) -> pd.DataFrame:
    """Repeat each series/target's final observed history value over the truth window."""

    history = _prepared(history_df, config, "history")
    truth = _truth_long(truth_df, config)
    rows: list[pd.DataFrame] = []
    for target in config.target_names:
        last = (
            history.sort_values([config.id_column, config.timestamp_column], kind="mergesort")
            .groupby(config.id_column, sort=False)[target]
            .last()
        )
        block = truth[truth[NORMALIZED_TARGET_COLUMN] == target].copy()
        block[NORMALIZED_POINT_COLUMN] = block[NORMALIZED_ID_COLUMN].map(last)
        if block[NORMALIZED_POINT_COLUMN].isna().any():
            _fail(
                "EVALUATION_BASELINE_MISSING_HISTORY",
                f"last-value baseline has no history for at least one series for "
                f"target {target!r}.",
                target=target,
            )
        rows.append(
            block[
                [
                    NORMALIZED_ID_COLUMN,
                    NORMALIZED_TIMESTAMP_COLUMN,
                    NORMALIZED_TARGET_COLUMN,
                    NORMALIZED_POINT_COLUMN,
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True).sort_values(
        [NORMALIZED_ID_COLUMN, NORMALIZED_TARGET_COLUMN, NORMALIZED_TIMESTAMP_COLUMN],
        kind="mergesort",
    ).reset_index(drop=True)


def seasonal_naive_baseline(
    history_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    config: ForecastConfig,
    *,
    season_length: int,
) -> pd.DataFrame:
    """Repeat the final explicit season over the truth window.

    ``season_length`` is intentionally required. The pipeline does not infer a
    seasonal period from timestamps and thereby smuggle an unreviewed heuristic
    into the benchmark.
    """

    if isinstance(season_length, bool) or not isinstance(season_length, int) or season_length < 1:
        _fail(
            "EVALUATION_INVALID_SEASON_LENGTH",
            f"season_length must be a positive integer, got {season_length!r}.",
            season_length=season_length,
        )
    history = _prepared(history_df, config, "history")
    truth = _truth_long(truth_df, config)
    rows: list[dict[str, Any]] = []
    for series_id, truth_series in truth.groupby(NORMALIZED_ID_COLUMN, sort=False):
        h = history[history[config.id_column] == series_id].sort_values(
            config.timestamp_column, kind="mergesort"
        )
        if len(h) < season_length:
            _fail(
                "EVALUATION_SEASON_TOO_LONG",
                f"series {series_id!r} has {len(h)} history row(s), fewer than "
                f"season_length={season_length}.",
                series_id=series_id,
                n_history=len(h),
                season_length=season_length,
            )
        for target in config.target_names:
            season = h[target].tail(season_length).to_numpy(dtype=float)
            target_truth = truth_series[
                truth_series[NORMALIZED_TARGET_COLUMN] == target
            ].sort_values(NORMALIZED_TIMESTAMP_COLUMN, kind="mergesort")
            for step, (_, truth_row) in enumerate(target_truth.iterrows()):
                rows.append(
                    {
                        NORMALIZED_ID_COLUMN: series_id,
                        NORMALIZED_TIMESTAMP_COLUMN: truth_row[NORMALIZED_TIMESTAMP_COLUMN],
                        NORMALIZED_TARGET_COLUMN: target,
                        NORMALIZED_POINT_COLUMN: float(season[step % season_length]),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        [NORMALIZED_ID_COLUMN, NORMALIZED_TARGET_COLUMN, NORMALIZED_TIMESTAMP_COLUMN],
        kind="mergesort",
    ).reset_index(drop=True)
