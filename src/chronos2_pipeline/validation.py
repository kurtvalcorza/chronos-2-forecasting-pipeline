"""DIMER-side validation — RFC rules 1-21.

Every check here runs *before* ``predict_df`` and is independent of the pinned
upstream validator. Two reasons the duplication is deliberate:

* Upstream's ``freq=`` argument bypasses frequency inference entirely. Its own
  docstring says so (``chronos/chronos2/pipeline.py`` L881-885 in 2.3.1: "the
  provided ``freq`` is used as-is and is not checked against the data, even when
  ``validate_inputs=True``"). Supplying ``frequency`` must not be a way to smuggle
  a gappy series past a regularity check, so rules 7-10 are ours (RFC C-3).
* Upstream raises bare ``ValueError`` with prose messages. DIMER needs stable
  machine-readable codes for its own error surface.

Frequency is established by explicit diff equality — ``diff(timestamps)`` must
have exactly one distinct value per series — not by ``pd.infer_freq``, which
tolerates patterns this contract rejects and needs three points before it will
say anything at all.

**Phase 1 supports fixed-width frequencies only.** A frequency is fixed-width
when one period is always the same ``Timedelta``: "15min", "h", "D", "7D". Data
spaced monthly, quarterly, yearly or business-daily has no constant period, so
diff equality can never hold for it; it is rejected with
``CALENDAR_FREQUENCY_UNSUPPORTED`` rather than mislabelled "irregular".
Widening the rule to calendar offsets is a Phase-2 design decision; see
``MODEL_CARD.md`` and ``README.md``, which state the limitation. Weekly data is
in scope — a week is a constant seven days — even though the pandas *alias*
``W`` is anchored and therefore not fixed-width; declare it as ``"7D"``.

The same rule governs a *declared* ``frequency``: an alias that is not
fixed-width cannot be compared against the observed ``Timedelta``, and upstream
would use it as-is to build the horizon index, so it is rejected
(``FREQUENCY_NOT_FIXED_WIDTH``) rather than forwarded unchecked. Only a declared
alias that has been affirmatively confirmed equal to the observed interval is
passed to ``predict_df`` (see :attr:`ValidationResult.confirmed_frequency`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pandas.api.types as ptypes

from .config import ForecastConfig
from .errors import ValidationError

__all__ = [
    "ResourceLimits",
    "ValidationResult",
    "DEFAULT_LIMITS",
    "MIN_OBSERVATIONS",
    "validate_forecast_request",
    "quantiles_in_grid",
    "nearest_grid_level",
]

#: RFC rule 10. Upstream's own frequency inference needs three points
#: (``chronos/df_utils.py`` L30), and a two-point series has no repeated
#: interval to validate, so regularity is unfalsifiable below three.
MIN_OBSERVATIONS = 3

#: How close a rejected level has to be to a grid member before the error
#: message names that member as the probable intent. Purely cosmetic: it never
#: widens what is accepted (see :func:`quantiles_in_grid`).
_QUANTILE_HINT_TOL = 1e-6


@dataclass(frozen=True)
class ResourceLimits:
    """DIMER-side resource guards (RFC rule 20).

    Deliberately conservative: these bound what a single request may ask of a
    shared serving process, and are separate from the model's own limits.
    """

    max_ids: int = 1000
    max_targets: int = 64
    max_covariates: int = 64
    max_rows: int = 5_000_000
    max_context_length: int = 8192
    #: Deliberately **above** the model's native horizon (1024) so that the
    #: unroll gate below, not this guard, is what answers a request beyond the
    #: model's capacity. With the two equal, ``allow_unroll`` was unreachable
    #: under the default limits and ``autoregressive_unrolled`` could never be
    #: ``True`` on the shipped configuration (review R-4).
    max_prediction_length: int = 4096
    min_observations: int = MIN_OBSERVATIONS


DEFAULT_LIMITS = ResourceLimits()


@dataclass(frozen=True)
class ValidationResult:
    """The normalised request, plus everything derived while proving it valid."""

    history: pd.DataFrame
    future: pd.DataFrame | None
    frequency: pd.Timedelta
    series_ids: list[Any]
    target_names: list[str]
    covariate_names: list[str]
    n_rows: int
    max_series_length: int
    min_series_length: int
    requested_quantiles: list[float] = field(default_factory=list)
    #: Covariates whose future values the caller supplied, i.e. the covariate
    #: columns present in the future table. Upstream decides this the same way:
    #: ``[c for c in covariate_columns if c in future_df.columns]``
    #: (``chronos/chronos2/preprocess.py`` L195).
    known_future_covariate_names: list[str] = field(default_factory=list)
    #: Covariates the model may read up to the forecast origin and no further.
    #: Every covariate is one or the other, so these two partition
    #: :attr:`covariate_names` (RFC Mode D asks that they be distinguished).
    past_covariate_names: list[str] = field(default_factory=list)
    #: The caller's declared ``frequency`` alias, and **only** when it was
    #: affirmatively confirmed equal to ``frequency``. ``None`` whenever the
    #: caller declared nothing. Nothing else may be forwarded to ``predict_df``,
    #: whose own docstring says ``freq`` "is used as-is and is not checked
    #: against the data, even when ``validate_inputs=True``" (review R-1).
    confirmed_frequency: str | None = None

    @property
    def n_ids(self) -> int:
        return len(self.series_ids)

    @property
    def n_targets(self) -> int:
        return len(self.target_names)

    @property
    def n_covariates(self) -> int:
        return len(self.covariate_names)

    @property
    def n_known_future_covariates(self) -> int:
        return len(self.known_future_covariate_names)

    @property
    def n_past_covariates(self) -> int:
        return len(self.past_covariate_names)


def quantiles_in_grid(
    requested: list[float], grid: tuple[float, ...] | list[float]
) -> list[float]:
    """Return the requested levels that are *not* in the trained grid.

    Membership is **exact float identity**, matching the gate upstream itself
    uses (``set(quantile_levels).issubset(training_quantile_levels)`` at
    ``chronos/chronos2/pipeline.py`` L797 in 2.3.1). A tolerance here would
    accept levels that upstream then routes to ``interpolate_quantiles``
    silently — e.g. ``0.1 * 7 == 0.7000000000000001`` is not ``0.7`` — which is
    exactly the silent substitution RFC C-5 exists to prevent (review R-7).
    """
    grid_set = {float(g) for g in grid}
    return [float(q) for q in requested if float(q) not in grid_set]


def nearest_grid_level(level: float, grid: tuple[float, ...] | list[float]) -> float | None:
    """The grid member a rejected level was probably meant to be, if any is close."""
    if len(grid) == 0:
        return None
    grid_array = np.asarray(grid, dtype=float)
    nearest = float(grid_array[int(np.argmin(np.abs(grid_array - float(level))))])
    if abs(nearest - float(level)) <= _QUANTILE_HINT_TOL:
        return nearest
    return None


def _fail(code: str, message: str, **details: Any) -> None:
    raise ValidationError(code, message, details)


# --------------------------------------------------------------------------
# Rules 1-6: shape, parsing, duplicates, ordering
# --------------------------------------------------------------------------


def _require_columns(df: pd.DataFrame, required: list[str], code: str, table: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        _fail(
            code,
            f"{table} table is missing required column(s): {missing}.",
            table=table,
            missing=missing,
            present=list(df.columns),
        )


def _parse_timestamps(df: pd.DataFrame, column: str, table: str) -> pd.Series:
    try:
        parsed = pd.to_datetime(df[column], errors="raise")
    except (ValueError, TypeError, pd.errors.ParserError) as exc:
        _fail(
            "TIMESTAMP_UNPARSEABLE",
            f"{table} column {column!r} could not be parsed as timestamps: {exc}",
            table=table,
            column=column,
        )
    if isinstance(parsed, pd.DataFrame):  # duplicate column names
        _fail(
            "TIMESTAMP_UNPARSEABLE",
            f"{table} column {column!r} is duplicated; a single timestamp column is required.",
            table=table,
            column=column,
        )
    if parsed.isna().any():
        n_bad = int(parsed.isna().sum())
        _fail(
            "TIMESTAMP_UNPARSEABLE",
            f"{table} column {column!r} has {n_bad} unparseable/null timestamp(s).",
            table=table,
            column=column,
            n_null=n_bad,
        )
    return parsed


def _coerce_targets(df: pd.DataFrame, targets: list[str], policy: str) -> pd.DataFrame:
    """RFC rule 4 and rule 19: explicit numeric and missing-value policies.

    ``policy="strict"`` (the only v1 policy) rejects any value that is not
    numeric-coercible, and rejects nulls outright — v1 does not impute
    (RFC "explicitly out of scope": silent interpolation).
    """
    if policy != "strict":
        _fail(
            "TARGET_POLICY_UNSUPPORTED",
            f"target coercion policy {policy!r} is not supported in v1; only 'strict' is.",
            policy=policy,
        )
    out = df.copy()
    for col in targets:
        original = out[col]
        coerced = pd.to_numeric(original, errors="coerce")
        newly_bad = coerced.isna() & original.notna()
        if newly_bad.any():
            offenders = original[newly_bad].unique()[:5].tolist()
            _fail(
                "TARGET_NOT_NUMERIC",
                f"target column {col!r} contains {int(newly_bad.sum())} value(s) that are "
                f"not numeric-coercible under the 'strict' policy; e.g. {offenders}.",
                column=col,
                n_offending=int(newly_bad.sum()),
                examples=[str(v) for v in offenders],
            )
        if coerced.isna().any():
            _fail(
                "TARGET_MISSING_VALUES",
                f"target column {col!r} contains {int(coerced.isna().sum())} missing value(s). "
                f"v1 rejects gaps and nulls rather than interpolating them.",
                column=col,
                n_missing=int(coerced.isna().sum()),
            )
        if not np.isfinite(coerced.to_numpy(dtype=float)).all():
            _fail(
                "TARGET_NOT_FINITE",
                f"target column {col!r} contains non-finite values (inf/-inf).",
                column=col,
            )
        out[col] = coerced.astype(float)
    return out


def _coerce_covariates(
    df: pd.DataFrame, covariates: list[str], policy: str, table: str
) -> pd.DataFrame:
    """Rules 4 and 19 applied to covariate columns (Phase 2).

    **Phase 2 admits numeric covariates only.** Upstream would accept a string
    or categorical covariate, but it encodes one by a route that depends on how
    many targets the request has: ``target_encode = use_target_encoding and
    n_targets == 1`` (``chronos/chronos2/preprocess.py`` L415), so the same
    holiday-name column is target-encoded in a single-target request and
    ordinal-encoded in a two-target one. A covariate whose meaning changes with
    an unrelated field of the request is not a contract this phase can export
    honestly, so non-numeric covariates are refused by name
    (``COVARIATE_NOT_NUMERIC``) rather than silently encoded. "Numeric" is
    judged the way :func:`_coerce_targets` judges it — by whether coercion loses
    a value, not by dtype — so a column of numeric strings from a CSV is
    accepted and a column of category labels is not. Temporal columns are named
    separately, before that coercion can turn them into nanoseconds.

    ``bool`` is deliberately treated as numeric and coerced to ``0.0``/``1.0``.
    Upstream classes it as *categorical* (``is_numeric_dtype(c) and not
    is_bool_dtype(c)`` at preprocess.py L188), so an unconverted boolean flag
    would take exactly the target-count-dependent path above. Coercing it here
    pins it to the numeric path in both cases, which is also the reading the
    RFC's own covariate example uses (``holiday,0``).

    Nulls and non-finite values are refused for the same reason they are in
    targets: v1 does not impute, and upstream would carry a NaN covariate into
    the forecast without complaint.
    """
    if policy != "strict":
        _fail(
            "TARGET_POLICY_UNSUPPORTED",
            f"covariate coercion policy {policy!r} is not supported in v1; only 'strict' is.",
            policy=policy,
        )
    reason = (
        "upstream would encode a categorical covariate by a route that differs between a "
        "single-target and a multi-target request, so its exported meaning would depend on "
        "how many targets were asked for"
    )
    out = df.copy()
    for col in covariates:
        original = out[col]
        if ptypes.is_bool_dtype(original):
            coerced = original.astype(float)
        elif ptypes.is_numeric_dtype(original):
            coerced = pd.to_numeric(original, errors="coerce")
        elif ptypes.is_datetime64_any_dtype(original) or ptypes.is_timedelta64_dtype(original):
            #: Named before the coercion below, which would turn a temporal column into
            #: nanoseconds-since-epoch and accept it as an ordinary numeric covariate.
            _fail(
                "COVARIATE_NOT_NUMERIC",
                f"{table} covariate column {col!r} has dtype {original.dtype}. A temporal "
                f"column is not a numeric covariate; coercing it would feed the model "
                f"nanoseconds since the epoch. Derive the feature you mean (hour of day, "
                f"days to event) as a numeric column instead.",
                table=table,
                column=col,
                dtype=str(original.dtype),
            )
        else:
            #: Same policy as targets: a column of numeric strings is coerced, a column
            #: that is genuinely categorical is refused. Judged by whether coercion loses
            #: a value, not by dtype, so a CSV read as ``object`` is not refused for it.
            coerced = pd.to_numeric(original, errors="coerce")
            newly_bad = coerced.isna() & original.notna()
            if newly_bad.any():
                offenders = original[newly_bad].unique()[:5].tolist()
                _fail(
                    "COVARIATE_NOT_NUMERIC",
                    f"{table} covariate column {col!r} contains "
                    f"{int(newly_bad.sum())} value(s) that are not numeric-coercible; e.g. "
                    f"{offenders}. Phase 2 supports numeric covariates only: {reason}. "
                    f"Encode it yourself (e.g. 0/1 indicator columns) to pin the "
                    f"representation.",
                    table=table,
                    column=col,
                    dtype=str(original.dtype),
                    n_offending=int(newly_bad.sum()),
                    examples=[str(v) for v in offenders],
                )
        if coerced.isna().any():
            _fail(
                "COVARIATE_MISSING_VALUES",
                f"{table} covariate column {col!r} contains "
                f"{int(coerced.isna().sum())} missing value(s). v1 rejects gaps and nulls "
                f"rather than interpolating them; upstream would carry the NaN into the "
                f"forecast without complaint.",
                table=table,
                column=col,
                n_missing=int(coerced.isna().sum()),
            )
        if not np.isfinite(coerced.to_numpy(dtype=float)).all():
            _fail(
                "COVARIATE_NOT_FINITE",
                f"{table} covariate column {col!r} contains non-finite values (inf/-inf).",
                table=table,
                column=col,
            )
        out[col] = coerced.astype(float)
    return out


# --------------------------------------------------------------------------
# Rules 7-10: regularity, gaps, shared frequency, minimum length
# --------------------------------------------------------------------------


def _fixed_width(offset: Any) -> pd.Timedelta | None:
    """One period of ``offset`` as a ``Timedelta``, or ``None`` if it has no fixed width.

    ``to_offset("h").nanos`` is 3.6e12; ``to_offset("ME").nanos`` raises, because
    a month is not a constant duration. That distinction is the whole Phase-1
    frequency contract.
    """
    try:
        return pd.Timedelta(offset.nanos, unit="ns")
    except (ValueError, AttributeError):
        return None


def _calendar_alias(timestamps: pd.Series) -> str | None:
    """The pandas alias of a series that is calendar-regular but not fixed-width.

    Used only to give monthly/quarterly/business-daily data an honest error
    instead of calling it "irregular" (review R-3). Returns ``None`` for
    anything ``pd.infer_freq`` cannot name, and for anything it names that *is*
    fixed-width (which diff equality would already have accepted).
    """
    try:
        alias = pd.infer_freq(pd.DatetimeIndex(timestamps))
    except (ValueError, TypeError):
        return None
    if alias is None:
        return None
    try:
        offset = pd.tseries.frequencies.to_offset(alias)
    except (ValueError, TypeError):  # pragma: no cover - infer_freq returns parseable aliases
        return None
    return None if _fixed_width(offset) is not None else alias


def _series_frequency(timestamps: pd.Series, series_id: Any, min_observations: int) -> pd.Timedelta:
    """One series' frequency, by explicit diff equality. Rules 7, 9 and 10."""
    n = len(timestamps)
    if n < min_observations:
        _fail(
            "SERIES_TOO_SHORT",
            f"series {series_id!r} has {n} observation(s); at least {min_observations} are "
            f"required before a frequency can be validated.",
            series_id=str(series_id),
            n_observations=n,
            minimum=min_observations,
        )

    diffs = timestamps.diff().dropna()
    distinct = pd.Series(diffs.unique())
    if len(distinct) != 1:
        deltas = sorted(pd.Timedelta(d) for d in distinct)
        calendar = _calendar_alias(timestamps)
        if calendar is not None:
            _fail(
                "CALENDAR_FREQUENCY_UNSUPPORTED",
                f"series {series_id!r} is regular on the calendar frequency {calendar!r}, "
                f"which has no fixed period ({[str(d) for d in deltas]}). Phase 1 supports "
                f"fixed-width frequencies only (e.g. '15min', 'h', 'D', '7D'); calendar "
                f"frequencies such as monthly, quarterly, yearly and business-daily are "
                f"out of scope for v1 and are documented as such in MODEL_CARD.md.",
                series_id=str(series_id),
                inferred_alias=calendar,
                observed_intervals=[str(d) for d in deltas],
            )
        base = deltas[0]
        is_gapped = base > pd.Timedelta(0) and all(
            (d % base) == pd.Timedelta(0) for d in deltas
        )
        if is_gapped:
            _fail(
                "SERIES_GAP",
                f"series {series_id!r} has missing periods: observed intervals "
                f"{[str(d) for d in deltas]} are all multiples of {base}, so timestamps "
                f"are skipped. v1 rejects gaps rather than interpolating them.",
                series_id=str(series_id),
                base_interval=str(base),
                observed_intervals=[str(d) for d in deltas],
            )
        _fail(
            "IRREGULAR_FREQUENCY",
            f"series {series_id!r} is irregular: observed intervals "
            f"{[str(d) for d in deltas]} are not all equal.",
            series_id=str(series_id),
            observed_intervals=[str(d) for d in deltas],
        )

    freq = pd.Timedelta(distinct.iloc[0])
    if freq <= pd.Timedelta(0):
        _fail(
            "IRREGULAR_FREQUENCY",
            f"series {series_id!r} has a non-positive interval {freq}.",
            series_id=str(series_id),
            observed_intervals=[str(freq)],
        )
    return freq


def _shared_frequency(
    df: pd.DataFrame, config: ForecastConfig, limits: ResourceLimits
) -> tuple[pd.Timedelta, int, int]:
    """Per-series frequency (rules 7, 9, 10) plus cross-series agreement (rule 8)."""
    frequencies: dict[Any, pd.Timedelta] = {}
    lengths: list[int] = []
    for series_id, block in df.groupby(config.id_column, sort=True, observed=True):
        stamps = block[config.timestamp_column]
        lengths.append(len(stamps))
        frequencies[series_id] = _series_frequency(stamps, series_id, limits.min_observations)

    distinct = sorted({str(f) for f in frequencies.values()})
    if len(distinct) > 1:
        _fail(
            "MIXED_FREQUENCY",
            f"all series in one request must share a frequency; observed {distinct}.",
            frequencies={str(k): str(v) for k, v in frequencies.items()},
            distinct=distinct,
        )
    return next(iter(frequencies.values())), max(lengths), min(lengths)


def _check_declared_frequency(declared: str | None, observed: pd.Timedelta) -> str | None:
    """A declared ``frequency`` must agree with the data — it may not override it.

    Returns the alias only when it has been affirmatively confirmed equal to
    ``observed``; the caller forwards nothing else to ``predict_df``.

    Every path either confirms or fails. A non-fixed calendar alias ("W", "ME",
    "QS", "B", "YE") used to fall through this function untouched and was then
    handed to ``predict_df``, which builds the horizon index from it without
    checking it against the data — an hourly series declared ``frequency="ME"``
    came back stamped at month ends, with no error (review R-1). Since the
    observed frequency is always a fixed ``Timedelta`` by construction (see
    :func:`_series_frequency`), a non-fixed alias can never agree with it, so
    the answer is always rejection.
    """
    if declared is None:
        return None
    try:
        offset = pd.tseries.frequencies.to_offset(declared)
    except (ValueError, TypeError) as exc:
        _fail(
            "FREQUENCY_UNPARSEABLE",
            f"frequency {declared!r} is not a valid pandas offset alias: {exc}",
            frequency=declared,
        )
    declared_delta = _fixed_width(offset)
    if declared_delta is None:
        _fail(
            "FREQUENCY_NOT_FIXED_WIDTH",
            f"declared frequency {declared!r} is a calendar offset with no fixed period, "
            f"so it cannot agree with the interval observed in the data ({observed}). "
            f"Upstream would use it as-is to lay out the forecast horizon without "
            f"checking it against the data, moving the forecast onto a different time "
            f"axis. Phase 1 supports fixed-width frequencies only, so declare the "
            f"interval the data actually has ({observed}) as a fixed-width alias "
            f"instead (RFC C-3).",
            declared=declared,
            observed_interval=str(observed),
        )
    if declared_delta != observed:
        _fail(
            "FREQUENCY_MISMATCH",
            f"declared frequency {declared!r} ({declared_delta}) does not match the "
            f"interval observed in the data ({observed}). Supplying `frequency` does "
            f"not override the data (RFC C-3).",
            declared=declared,
            declared_interval=str(declared_delta),
            observed_interval=str(observed),
        )
    return declared


# --------------------------------------------------------------------------
# Rules 15-18: the future-covariate table
# --------------------------------------------------------------------------


def _validate_future(
    future_df: pd.DataFrame,
    history: pd.DataFrame,
    config: ForecastConfig,
    frequency: pd.Timedelta,
    series_ids: list[Any],
    covariates: list[str],
    target_policy: str,
) -> tuple[pd.DataFrame, list[str]]:
    _require_columns(
        future_df,
        [config.id_column, config.timestamp_column],
        "FUTURE_MISSING_COLUMNS",
        "future",
    )

    leaked = [c for c in config.target_names if c in future_df.columns]
    if leaked:
        _fail(
            "FUTURE_TARGET_LEAKAGE",
            f"future table contains target column(s) {leaked}. Future target values are "
            f"forbidden as inputs.",
            leaked_columns=leaked,
        )

    extra = [c for c in future_df.columns if c not in history.columns]
    if extra:
        _fail(
            "FUTURE_COLUMN_NOT_IN_HISTORY",
            f"future covariate column(s) {extra} do not exist in the historical table. "
            f"The pinned upstream validator requires future columns to be a subset of "
            f"historical columns (RFC C-4).",
            extra_columns=extra,
            historical_columns=list(history.columns),
        )

    #: Which covariates the future table actually carries — upstream's own rule
    #: (``preprocess.py`` L195). A future table that carries none is a no-op
    #: upstream: every covariate stays past-only and the table changes nothing
    #: but the cost of validating it. Supplying one is a statement that some
    #: covariate is known ahead, so an empty one is refused rather than
    #: silently ignored.
    known_future = [c for c in covariates if c in future_df.columns]
    if not known_future:
        _fail(
            "FUTURE_TABLE_HAS_NO_COVARIATES",
            f"future table carries no covariate column: it has "
            f"{sorted(set(future_df.columns) - {config.id_column, config.timestamp_column})!r} "
            f"beyond the id and timestamp columns, and the historical covariates are "
            f"{covariates!r}. Upstream would treat every covariate as past-only and the "
            f"table would change nothing, so it is refused rather than silently ignored.",
            future_columns=list(future_df.columns),
            historical_covariates=covariates,
        )

    future = future_df.copy()
    future[config.timestamp_column] = _parse_timestamps(future, config.timestamp_column, "future")
    future = _coerce_covariates(future, known_future, target_policy, "future")

    if future[config.id_column].isna().any():
        _fail(
            "NULL_IDS",
            "future table contains null series identifiers.",
            table="future",
        )

    #: Compared as values, not as ``str``. Stringifying made historical id ``1``
    #: and future id ``"1"`` equal, which they are not: upstream joins the two
    #: tables on the raw values and would find no match (review R-13).
    future_ids = set(future[config.id_column].unique())
    history_ids = set(series_ids)
    if future_ids != history_ids:
        future_only = sorted(str(v) for v in future_ids - history_ids)
        history_only = sorted(str(v) for v in history_ids - future_ids)
        _fail(
            "FUTURE_ID_MISMATCH",
            f"future table ids must equal historical ids exactly, compared by value and "
            f"type. Only in future: {future_only[:5]}; only in history: {history_only[:5]}.",
            future_only=future_only[:20],
            history_only=history_only[:20],
        )

    future = future.sort_values(
        [config.id_column, config.timestamp_column], kind="mergesort"
    ).reset_index(drop=True)

    horizon = config.prediction_length
    last_seen = history.groupby(config.id_column, observed=True)[config.timestamp_column].max()

    for series_id, block in future.groupby(config.id_column, sort=True, observed=True):
        if len(block) != horizon:
            _fail(
                "FUTURE_LENGTH_MISMATCH",
                f"future table has {len(block)} row(s) for series {series_id!r}; exactly "
                f"prediction_length={horizon} are required.",
                series_id=str(series_id),
                n_rows=len(block),
                expected=horizon,
            )
        origin = last_seen.loc[series_id]
        expected = pd.date_range(
            start=origin + frequency, periods=horizon, freq=frequency
        )
        observed = pd.DatetimeIndex(block[config.timestamp_column])
        if not observed.equals(expected):
            _fail(
                "FUTURE_TIMESTAMP_MISALIGNED",
                f"future timestamps for series {series_id!r} must start at the forecast "
                f"origin {origin + frequency} and follow the validated frequency "
                f"{frequency} exactly.",
                series_id=str(series_id),
                expected_first=str(expected[0]),
                observed_first=str(observed[0]) if len(observed) else None,
                expected_last=str(expected[-1]),
                observed_last=str(observed[-1]) if len(observed) else None,
            )

    return future, known_future


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def validate_forecast_request(
    history_df: pd.DataFrame,
    config: ForecastConfig,
    *,
    trained_quantiles: tuple[float, ...] | list[float],
    model_context_length: int,
    model_prediction_length: int,
    future_df: pd.DataFrame | None = None,
    limits: ResourceLimits = DEFAULT_LIMITS,
    target_policy: str = "strict",
    allow_unroll: bool = False,
) -> ValidationResult:
    """Run RFC validation rules 1-21 and return the normalised request.

    Raises
    ------
    ValidationError
        On the first rule that fails, carrying a stable ``code``.
    """
    if not isinstance(history_df, pd.DataFrame):
        _fail(
            "HISTORY_NOT_A_DATAFRAME",
            f"history must be a pandas DataFrame, got {type(history_df).__name__}.",
        )

    targets = config.target_names

    # Rule 1 -----------------------------------------------------------------
    _require_columns(
        history_df,
        [config.id_column, config.timestamp_column, *targets],
        "MISSING_COLUMNS",
        "historical",
    )

    # Rule 11 (non-empty) ----------------------------------------------------
    if len(history_df) == 0:
        _fail("EMPTY_HISTORY", "historical table is empty.", n_rows=0)
    if len(history_df) > limits.max_rows:
        _fail(
            "RESOURCE_LIMIT",
            f"historical table has {len(history_df)} rows; the limit is {limits.max_rows}.",
            guard="max_rows",
            observed=len(history_df),
            limit=limits.max_rows,
        )

    history = history_df.copy()

    # Rule 2 -----------------------------------------------------------------
    history[config.timestamp_column] = _parse_timestamps(
        history, config.timestamp_column, "historical"
    )

    # Rule 3 -----------------------------------------------------------------
    if history[config.id_column].isna().any():
        _fail(
            "NULL_IDS",
            f"historical table has {int(history[config.id_column].isna().sum())} null "
            f"value(s) in id column {config.id_column!r}.",
            table="historical",
            column=config.id_column,
        )

    # Rule 4 and rule 19 -----------------------------------------------------
    history = _coerce_targets(history, targets, target_policy)

    # Rule 5 -----------------------------------------------------------------
    #: The frame is wide (one column per target), so a duplicated
    #: (id, timestamp) pair is a duplicated (id, timestamp, target_name)
    #: observation for every target at once.
    dup_mask = history.duplicated(subset=[config.id_column, config.timestamp_column], keep=False)
    if dup_mask.any():
        offenders = (
            history.loc[dup_mask, [config.id_column, config.timestamp_column]]
            .astype(str)
            .drop_duplicates()
            .head(5)
            .to_dict("records")
        )
        _fail(
            "DUPLICATE_OBSERVATIONS",
            f"historical table has {int(dup_mask.sum())} row(s) sharing an "
            f"(id, timestamp) pair; each (id, timestamp, target_name) observation must "
            f"be unique. Examples: {offenders}.",
            n_duplicate_rows=int(dup_mask.sum()),
            examples=offenders,
        )

    # Rule 6 -----------------------------------------------------------------
    history = history.sort_values(
        [config.id_column, config.timestamp_column], kind="mergesort"
    ).reset_index(drop=True)

    series_ids = list(pd.unique(history[config.id_column]))
    covariates = [c for c in history.columns if c not in config.reserved_columns]

    # Rule 20 ----------------------------------------------------------------
    for guard, observed, limit in (
        ("max_ids", len(series_ids), limits.max_ids),
        ("max_targets", len(targets), limits.max_targets),
        ("max_covariates", len(covariates), limits.max_covariates),
    ):
        if observed > limit:
            _fail(
                "RESOURCE_LIMIT",
                f"request exceeds the {guard} guard: {observed} > {limit}.",
                guard=guard,
                observed=observed,
                limit=limit,
            )

    #: Rules 4 and 19 for covariates. After the guards above so that a request
    #: with more covariates than the limit allows is still reported as a
    #: resource-limit breach rather than as whichever of them is first
    #: non-numeric.
    history = _coerce_covariates(history, covariates, target_policy, "historical")

    # Rules 7, 8, 9, 10 ------------------------------------------------------
    frequency, max_len, min_len = _shared_frequency(history, config, limits)
    confirmed_frequency = _check_declared_frequency(config.frequency, frequency)

    # Rule 11 (minimum context) and rule 20 (context guard) -------------------
    if config.context_length is not None:
        if config.context_length > limits.max_context_length:
            _fail(
                "RESOURCE_LIMIT",
                f"context_length {config.context_length} exceeds the DIMER guard "
                f"{limits.max_context_length}.",
                guard="max_context_length",
                observed=config.context_length,
                limit=limits.max_context_length,
            )
        if config.context_length < limits.min_observations:
            _fail(
                "CONTEXT_TOO_SHORT",
                f"context_length {config.context_length} is below the minimum of "
                f"{limits.min_observations} observations.",
                context_length=config.context_length,
                minimum=limits.min_observations,
            )

    # Rule 12 ----------------------------------------------------------------
    if config.prediction_length > limits.max_prediction_length:
        _fail(
            "PREDICTION_LENGTH_LIMIT",
            f"prediction_length {config.prediction_length} exceeds the DIMER guard "
            f"{limits.max_prediction_length}.",
            guard="max_prediction_length",
            observed=config.prediction_length,
            limit=limits.max_prediction_length,
        )
    if config.prediction_length > model_prediction_length and not allow_unroll:
        _fail(
            "PREDICTION_LENGTH_EXCEEDS_MODEL",
            f"prediction_length {config.prediction_length} exceeds the model's native "
            f"prediction length {model_prediction_length}. Upstream would satisfy this by "
            f"autoregressive unrolling, which v1 does not enable by default; pass "
            f"allow_unroll=True to opt in and the flag will be recorded in provenance.",
            requested=config.prediction_length,
            model_prediction_length=model_prediction_length,
        )

    # Rules 13 and 14 --------------------------------------------------------
    #: Hard-fail, with no opt-out. There was an ``allow_out_of_grid`` escape
    #: hatch; it exported a column named for the *requested* level while holding
    #: the substituted one, and recorded ``effective == requested`` in
    #: provenance, so the substitution the RFC exists to surface was invisible
    #: in the export (review R-2). Phase 1 has no consumer for it.
    out_of_grid = quantiles_in_grid(config.quantile_levels, trained_quantiles)
    if out_of_grid:
        hints = {
            str(q): nearest_grid_level(q, trained_quantiles)
            for q in out_of_grid
            if nearest_grid_level(q, trained_quantiles) is not None
        }
        hint = (
            f" Level(s) {list(hints)} are within 1e-6 of grid member(s) "
            f"{list(hints.values())}; membership is exact, so pass the grid value itself."
            if hints
            else ""
        )
        _fail(
            "QUANTILE_NOT_IN_GRID",
            f"requested quantile level(s) {out_of_grid} are outside the grid the pinned "
            f"model was trained on. Upstream would silently substitute the nearest trained "
            f"level, producing a column labelled with a level it does not contain, so this "
            f"request is rejected (RFC C-5).{hint}",
            out_of_grid=out_of_grid,
            trained_quantiles=[float(q) for q in trained_quantiles],
            nearest_grid_levels=hints,
        )

    # Rule 21 ----------------------------------------------------------------
    required_batch = len(targets) + len(covariates)
    if config.batch_size < required_batch:
        _fail(
            "BATCH_SIZE_TOO_SMALL",
            f"batch_size {config.batch_size} is below n_targets + n_covariates = "
            f"{len(targets)} + {len(covariates)} = {required_batch}. Upstream batches count "
            f"every series including covariates (RFC C-9).",
            batch_size=config.batch_size,
            n_targets=len(targets),
            n_covariates=len(covariates),
            required=required_batch,
        )

    # Rules 15-18 ------------------------------------------------------------
    future = None
    known_future_covariates: list[str] = []
    if future_df is not None:
        future, known_future_covariates = _validate_future(
            future_df, history, config, frequency, series_ids, covariates, target_policy
        )

    _ = model_context_length  # recorded by provenance; no DIMER-side rule needs it here

    return ValidationResult(
        history=history,
        future=future,
        frequency=frequency,
        series_ids=series_ids,
        target_names=list(targets),
        covariate_names=covariates,
        n_rows=len(history),
        max_series_length=max_len,
        min_series_length=min_len,
        requested_quantiles=[float(q) for q in config.quantile_levels],
        known_future_covariate_names=known_future_covariates,
        past_covariate_names=[c for c in covariates if c not in known_future_covariates],
        confirmed_frequency=confirmed_frequency,
    )
