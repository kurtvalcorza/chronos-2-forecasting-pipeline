"""Standalone numerical contract for the Philippine reef capstone (CPU only)."""

import io
import json

import numpy as np
import pandas as pd

REEF_COLUMNS = [
    "YYYY",
    "MM",
    "DD",
    "SST_MIN",
    "SST_MAX",
    "SST@90th_HS",
    "SSTA@90th_HS",
    "90th_HS>0",
    "DHW_from_90th_HS>1",
    "BAA_7day_max",
]
REEF_NAMES = [
    "sst_min_c",
    "sst_max_c",
    "representative_sst_c",
    "representative_ssta_c",
    "hotspot_c",
    "dhw_c_weeks",
    "legacy_baa_7day",
]


FORECAST_KEYS = ["region_id", "origin", "arm", "lead"]
QUANTILE_COLUMNS = [
    "model_q10",
    "model_q50",
    "model_q90",
    "quantile_crossing_c",
    "raw_q10",
    "raw_q50",
    "raw_q90",
    "q10",
    "q50",
    "q90",
]
QUANTILE_ARMS = {"chronos", "chronos_180"}


def json_native(value: object) -> object:
    """JSON default for scientific records: native numbers, never blanket strings."""
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return str(value.date()) if value == value.normalize() else value.isoformat()
    raise TypeError(f"Not JSON-serializable: {type(value).__name__}")


def _finite_nonnegative(values: object, label: str) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: numeric values required") from exc
    if not np.isfinite(arr).all() or (arr < 0).any():
        raise ValueError(f"{label}: finite nonnegative values required")
    return arr


def parse_noaa(text: str, region: str) -> pd.DataFrame:
    """Parse exact NOAA RVS schema and its unusual year/day/month metadata."""
    lines = text.splitlines()
    headers = [i for i, line in enumerate(lines) if line.split() == REEF_COLUMNS]
    if len(headers) != 1:
        raise ValueError("Missing or malformed NOAA column header")
    meta = {}
    for label in ["Name", "First Valid DHW Date", "First Valid BAA Date"]:
        matches = [i for i, line in enumerate(lines[: headers[0]]) if line.strip() == label + ":"]
        if len(matches) != 1 or matches[0] + 1 >= headers[0]:
            raise ValueError(f"Missing metadata: {label}")
        meta[label] = lines[matches[0] + 1].strip()
    try:
        valid_dhw = pd.to_datetime(meta["First Valid DHW Date"], format="%Y %d %m")
        valid_baa = pd.to_datetime(meta["First Valid BAA Date"], format="%Y %d %m")
    except ValueError as exc:
        raise ValueError("Malformed validity date; expected year day month") from exc
    body = [line for line in lines[headers[0] + 1 :] if line.strip()]
    if not body or any(len(line.split()) != 10 for line in body):
        raise ValueError("NOAA data rows must contain exactly ten fields")
    raw = pd.read_csv(io.StringIO("\n".join(body)), sep=r"\s+", names=REEF_COLUMNS)
    raw = raw.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(raw.to_numpy(dtype=float)).all():
        raise ValueError("NOAA data contains nonfinite values")
    parts = raw[["YYYY", "MM", "DD"]]
    if not (parts.to_numpy() == parts.to_numpy().astype(int)).all():
        raise ValueError("Date components must be integers")
    dates = pd.to_datetime(
        parts.rename(columns={"YYYY": "year", "MM": "month", "DD": "day"}), errors="raise"
    )
    if dates.duplicated().any():
        raise ValueError(f"Duplicate dates: {dates[dates.duplicated()].astype(str).tolist()[:5]}")
    out = raw.iloc[:, 3:].copy()
    out.columns = REEF_NAMES
    _finite_nonnegative(out[["hotspot_c", "dhw_c_weeks"]], "NOAA HotSpot/DHW")
    if not out["legacy_baa_7day"].isin(range(5)).all():
        raise ValueError("Legacy BAA must be an integer from zero through four")
    if (
        (out.sst_min_c > out.representative_sst_c) | (out.sst_max_c < out.representative_sst_c)
    ).any():
        raise ValueError("Representative SST outside regional bounds")
    out.insert(0, "date", dates)
    out.insert(0, "region_id", region)
    out["raw_dhw_c_weeks"] = out.dhw_c_weeks
    out["raw_legacy_baa_7day"] = out.legacy_baa_7day
    out["dhw_valid"] = out.date >= valid_dhw
    out["baa_valid"] = out.date >= valid_baa
    out.loc[~out.dhw_valid, "dhw_c_weeks"] = np.nan
    out.loc[~out.baa_valid, "legacy_baa_7day"] = np.nan
    out.attrs["metadata"] = meta
    return out.sort_values("date").reset_index(drop=True)


def read_byod_csv(path: object) -> pd.DataFrame:
    """Read BYOD text literally: region IDs such as 001 or NA keep their exact spelling."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    for column in ["hotspot_c", "dhw_c_weeks"]:
        if column in frame:
            if (frame[column].str.strip() == "").any():
                raise ValueError(f"{column}: blank values are not allowed")
            frame[column] = _finite_nonnegative(frame[column], column)
    return frame


def validate_byod(
    frame: pd.DataFrame, units: str = "degC", semantics_confirmed: bool = False
) -> pd.DataFrame:
    """Refuse SST substitutions, timestamps, duplicate/gapped or short contexts."""
    required = {"region_id", "date", "hotspot_c"}
    if not required.issubset(frame.columns) or set(frame.columns) - required - {"dhw_c_weeks"}:
        raise ValueError("Expected region_id,date,hotspot_c and optional dhw_c_weeks only")
    if units != "degC" or not semantics_confirmed:
        raise ValueError("Confirm NOAA threshold-relative HotSpot semantics and degC units")
    out = frame.copy()
    if (
        out.empty
        or out.region_id.isna().any()
        or (out.region_id.astype(str).str.strip() == "").any()
    ):
        raise ValueError("Nonempty region IDs and data required")
    if not out.date.astype(str).str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
        raise ValueError("Dates must be exactly YYYY-MM-DD without a time or timezone")
    out["date"] = pd.to_datetime(out.date, format="%Y-%m-%d", errors="raise")
    if out.duplicated(["region_id", "date"]).any():
        raise ValueError("Duplicate region/date rows")
    for col in ["hotspot_c"] + (["dhw_c_weeks"] if "dhw_c_weeks" in out else []):
        out[col] = _finite_nonnegative(out[col], col)
    for region, group in out.groupby("region_id"):
        group = group.sort_values("date")
        if len(group) < 365 or not group.date.diff().iloc[1:].eq(pd.Timedelta(days=1)).all():
            raise ValueError(f"{region}: at least 365 continuous daily observations required")
        if "dhw_c_weeks" in group:
            reconstructed = reconstruct_dhw(group.set_index("date").hotspot_c)
            if not np.allclose(
                reconstructed.dropna(),
                group.set_index("date").loc[reconstructed.notna(), "dhw_c_weeks"],
                atol=1e-4,
                rtol=0,
            ):
                raise ValueError(
                    f"{region}: DHW disagrees with calendar-aware HotSpot accumulation"
                )
    return out.sort_values(["region_id", "date"]).reset_index(drop=True)


def daily_panel(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Preserve missing calendar days explicitly; never fill numerical outcomes."""
    data = pd.concat(frames, ignore_index=True)
    if data.duplicated(["region_id", "date"]).any():
        raise ValueError("Duplicate region/date in daily panel")
    groups = []
    for region, group in data.groupby("region_id", sort=True):
        group = group.set_index("date").drop(columns="region_id").sort_index()
        group["observed"] = True
        group = group.reindex(
            pd.date_range(group.index.min(), group.index.max(), freq="D", name="date")
        )
        group["observed"] = group.observed.eq(True)
        group["missing_day"] = ~group.observed
        group["region_id"] = region
        groups.append(group.reset_index())
    return pd.concat(groups, ignore_index=True).set_index(["region_id", "date"]).sort_index()


def reconstruct_dhw(hotspot: pd.Series) -> pd.Series:
    """Inclusive 84-calendar-day sum with >=1 degree threshold; gaps remain null."""
    if not isinstance(hotspot.index, pd.DatetimeIndex) or hotspot.index.has_duplicates:
        raise ValueError("Unique DatetimeIndex required")
    hs = hotspot.sort_index().reindex(
        pd.date_range(hotspot.index.min(), hotspot.index.max(), freq="D")
    )
    if (hs.dropna() < 0).any() or np.isinf(hs.to_numpy()).any():
        raise ValueError("Invalid HotSpot")
    heat = hs.where(hs >= 1, 0).where(hs.notna())
    return heat.rolling(84, min_periods=84).sum() / 7


def context_at(panel: pd.DataFrame, region: str, origin: object, length: int = 365) -> pd.Series:
    """Only observations up to origin, with calendar continuity required."""
    origin = pd.Timestamp(origin)
    dates = pd.date_range(origin - pd.Timedelta(days=length - 1), origin)
    context = panel.xs(region).hotspot_c.reindex(dates)
    _finite_nonnegative(context, "context")
    return context


def origin_manifest(panel: pd.DataFrame) -> pd.DataFrame:
    """All planned 1st/15th origins, including split-boundary and gap refusals."""
    rows = []
    for region in panel.index.get_level_values("region_id").unique():
        data = panel.xs(region)
        for year in [2023, 2024, 2025]:
            for month in range(1, 13):
                for day in [1, 15]:
                    origin = pd.Timestamp(year, month, day)
                    targets = pd.date_range(origin + pd.Timedelta(days=1), periods=28)
                    context_dates = pd.date_range(origin - pd.Timedelta(days=364), origin)
                    reasons = []
                    # Each annual test report uses complete horizons within that year.
                    if targets[-1].year != year:
                        reasons.append("outcome_crosses_year_boundary")
                    if data.hotspot_c.reindex(context_dates).isna().any():
                        reasons.append("incomplete_365_day_context")
                    if data.hotspot_c.reindex(targets).isna().any():
                        reasons.append("incomplete_28_day_outcome")
                    if "dhw_c_weeks" not in data or data.dhw_c_weeks.reindex(targets).isna().any():
                        reasons.append("invalid_dhw_outcome")
                    rows.append(
                        dict(
                            region_id=region,
                            origin=origin,
                            year=year,
                            split="validation" if year == 2023 else "test",
                            eligible=not reasons,
                            reason=";".join(reasons),
                            context_start=context_dates[0],
                            target_end=targets[-1],
                        )
                    )
    return pd.DataFrame(rows)


def fit_seasonal(panel: pd.DataFrame) -> pd.DataFrame:
    """Fit only 2017-2022, using month/day rather than shifting leap-year DOY."""
    frame = panel.reset_index()
    frame = frame[frame.date.between("2017-01-01", "2022-12-31")].copy()
    frame["month"] = frame.date.dt.month
    frame["day"] = frame.date.dt.day
    result = frame.groupby(["region_id", "month", "day"], as_index=False).hotspot_c.mean()
    result = result[~((result.month == 2) & (result.day == 29))]
    for region in frame.region_id.unique():
        vals = result[
            (result.region_id == region)
            & (
                ((result.month == 2) & (result.day == 28))
                | ((result.month == 3) & (result.day == 1))
            )
        ].hotspot_c
        if len(vals) != 2 or not np.isfinite(vals).all():
            raise ValueError("Seasonal reference lacks February/March boundary")
        result = pd.concat(
            [
                result,
                pd.DataFrame(
                    [dict(region_id=region, month=2, day=29, hotspot_c=float(vals.mean()))]
                ),
            ],
            ignore_index=True,
        )
    if result.hotspot_c.isna().any() or any(len(g) != 366 for _, g in result.groupby("region_id")):
        raise ValueError("Incomplete seasonal development reference")
    return result.sort_values(["region_id", "month", "day"]).reset_index(drop=True)


def baseline_forecasts(
    context: pd.Series, dates: object, seasonal: pd.DataFrame, region: str
) -> dict:
    hs = _finite_nonnegative(context, "context")
    dates = pd.DatetimeIndex(dates)
    lookup = seasonal[seasonal.region_id == region].set_index(["month", "day"]).hotspot_c
    return {
        "persistence": np.repeat(hs[-1], len(dates)),
        "seasonal": np.array([lookup.loc[(d.month, d.day)] for d in dates], dtype=float),
    }


def compose_dhw(context: object, predictions: object) -> dict:
    """Separate known history from forecast heat, without future observations."""
    history = _finite_nonnegative(context, "context")
    pred = _finite_nonnegative(predictions, "predictions")
    if history.ndim != 1 or pred.ndim != 1 or len(history) < 83:
        raise ValueError("DHW requires a 1D context of at least 83 consecutive days")
    hist_heat = np.where(history >= 1, history, 0)
    pred_heat = np.where(pred >= 1, pred, 0)
    known, new = [], []
    for i in range(len(pred)):
        remaining = max(83 - i, 0)
        known.append(float(hist_heat[-remaining:].sum() / 7) if remaining else 0.0)
        new.append(float(pred_heat[max(0, i - 83) : i + 1].sum() / 7))
    return {
        "dhw": np.array(known) + np.array(new),
        "known": np.array(known),
        "predicted": np.array(new),
    }


def rearrange_quantiles(quantiles: object) -> tuple[np.ndarray, np.ndarray]:
    """Monotone rearrangement: sort each day's q10/q50/q90 and report the largest crossing.

    Chronos-2 predicts its quantiles independently and does not enforce their order, so a
    crossing is a model property, not corrupt output. Sorting never increases pinball loss
    (Chernozhukov, Fernandez-Val and Galichon, 2010); the crossing size is kept for audit.
    """
    q = np.asarray(quantiles, dtype=float)
    if q.ndim != 2 or q.shape[1] != 3 or not np.isfinite(q).all():
        raise ValueError("Quantiles must be finite (H,3) q10,q50,q90")
    pairs = [q[:, 0] - q[:, 1], q[:, 1] - q[:, 2], q[:, 0] - q[:, 2]]
    crossing = np.maximum.reduce([np.zeros(len(q))] + pairs)
    return np.sort(q, axis=1), crossing


def forecast_frame(
    panel: pd.DataFrame,
    region: str,
    origin: object,
    arm: str,
    raw: object,
    split: str,
    quantiles: object = None,
) -> pd.DataFrame:
    """Build 28 daily predictions; observed targets are attached for scoring only."""
    origin = pd.Timestamp(origin)
    values = np.asarray(raw, dtype=float)
    if values.shape != (28,) or not np.isfinite(values).all():
        raise ValueError("Each arm must return exactly 28 finite predictions")
    if quantiles is not None:
        model_q = np.asarray(quantiles, dtype=float)
        if model_q.shape != (28, 3) or not np.isfinite(model_q).all():
            raise ValueError("Quantiles must be finite (28,3) q10,q50,q90")
        if not np.allclose(model_q[:, 1], values, atol=1e-6, rtol=0):
            raise ValueError("Point forecast must equal the model median")
        q, crossing = rearrange_quantiles(model_q)
        # The point forecast is the median of the rearranged (ordered) quantiles.
        values = q[:, 1]
    constrained = np.maximum(values, 0)
    dates = pd.date_range(origin + pd.Timedelta(days=1), periods=28)
    data = panel.xs(region)
    dhw = compose_dhw(context_at(panel, region, origin), constrained)
    out = pd.DataFrame(
        dict(
            region_id=region,
            origin=origin,
            target_date=dates,
            lead=np.arange(1, 29),
            arm=arm,
            split=split,
            raw_hotspot_c=values,
            hotspot_c=constrained,
            dhw_c_weeks=dhw["dhw"],
            dhw_known=dhw["known"],
            dhw_predicted=dhw["predicted"],
            actual_hotspot_c=data.hotspot_c.reindex(dates).to_numpy(),
            actual_dhw_c_weeks=data.dhw_c_weeks.reindex(dates).to_numpy(),
            origin_dhw_c_weeks=float(data.dhw_c_weeks.loc[origin]),
        )
    )
    if quantiles is not None:
        for i, label in enumerate(["q10", "q50", "q90"]):
            out["model_" + label] = model_q[:, i]
        out["quantile_crossing_c"] = crossing
        for i, label in enumerate(["q10", "q50", "q90"]):
            out["raw_" + label] = q[:, i]
            out[label] = np.maximum(q[:, i], 0)
    return out


def validate_forecast_grid(
    frame: pd.DataFrame, plan: pd.DataFrame, arms: object, panel: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Require exactly one row per planned region x origin x arm x lead, with date identity.

    ``plan`` holds the frozen eligible origins (region_id, origin, split). Missing whole arms,
    regions or origins, extra keys, shifted target dates, inconsistent source-derived
    references and absent Chronos quantiles are refused before any score is calculated.
    """
    arms = list(arms)
    needed = FORECAST_KEYS + ["split", "target_date", "hotspot_c", "raw_hotspot_c"]
    needed += ["actual_hotspot_c", "actual_dhw_c_weeks", "origin_dhw_c_weeks", "dhw_c_weeks"]
    missing_columns = [c for c in needed if c not in frame]
    if QUANTILE_ARMS & set(arms):
        missing_columns += [c for c in QUANTILE_COLUMNS if c not in frame]
    if missing_columns:
        raise ValueError(f"Forecast file lacks required columns: {missing_columns}")
    data = frame.copy()
    data["origin"] = pd.to_datetime(data.origin)
    data["target_date"] = pd.to_datetime(data.target_date)
    data["lead"] = pd.to_numeric(data.lead, errors="raise")
    if data.duplicated(FORECAST_KEYS).any():
        raise ValueError("Duplicate forecast rows")
    planned = plan[["region_id", "origin", "split"]].copy()
    planned["origin"] = pd.to_datetime(planned.origin)
    if planned.duplicated(["region_id", "origin"]).any():
        raise ValueError("Duplicate planned origins")
    expected = planned.merge(pd.DataFrame({"arm": arms}), how="cross").merge(
        pd.DataFrame({"lead": np.arange(1, 29)}), how="cross"
    )
    key = FORECAST_KEYS + ["split"]
    joined = expected.merge(data[key], how="outer", on=key, indicator=True)
    absent = joined[joined._merge == "left_only"]
    extra = joined[joined._merge == "right_only"]
    if len(absent):
        groups = absent.drop_duplicates(["region_id", "origin", "arm"])
        raise ValueError(
            f"Missing planned forecasts: {len(absent)} rows in {len(groups)} region-origin-arm "
            f"groups, e.g. {groups.iloc[0][['region_id', 'arm']].tolist()} "
            f"{groups.iloc[0].origin.date()}"
        )
    if len(extra):
        raise ValueError(f"Unplanned forecast rows (key, arm or split): {len(extra)}")
    shifted = data.target_date != data.origin + pd.to_timedelta(data.lead, unit="D")
    if shifted.any():
        raise ValueError(f"target_date must equal origin + lead; {int(shifted.sum())} rows differ")
    model_rows = data[data.arm.isin(QUANTILE_ARMS)]
    if len(model_rows):
        values = model_rows[QUANTILE_COLUMNS].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Required Chronos quantiles are missing or nonfinite")
    if panel is not None:
        idx = pd.MultiIndex.from_arrays([data.region_id, data.target_date])
        origin_idx = pd.MultiIndex.from_arrays([data.region_id, data.origin])
        references = {
            "actual_hotspot_c": panel.hotspot_c.reindex(idx).to_numpy(),
            "actual_dhw_c_weeks": panel.dhw_c_weeks.reindex(idx).to_numpy(),
            "origin_dhw_c_weeks": panel.dhw_c_weeks.reindex(origin_idx).to_numpy(),
        }
        for column, truth in references.items():
            if not np.allclose(data[column].to_numpy(dtype=float), truth, atol=1e-9, rtol=0):
                raise ValueError(f"{column} disagrees with the frozen source data")
    return data


def evaluate_forecasts(frame: pd.DataFrame, expected_arms: object = None) -> dict:
    """Descriptive region/macro metrics; no iid uncertainty or ecological labels."""
    metrics, thresholds, paired = [], [], []
    # Illustrative outlook rows are deliberately not evaluated.
    evaluated = frame[frame.split.isin(["validation", "test"])].copy()
    if evaluated.empty:
        raise ValueError("No validation/test forecasts to evaluate")
    if expected_arms is not None and set(evaluated.arm) != set(expected_arms):
        raise ValueError("Missing or unexpected model arms")
    if evaluated.duplicated(["region_id", "origin", "arm", "lead"]).any():
        raise ValueError("Duplicate forecast rows")
    for _, group in evaluated.groupby(["split", "region_id", "origin", "arm"]):
        if sorted(group.lead.tolist()) != list(range(1, 29)):
            raise ValueError("Missing model outputs or incomplete horizon")
        if (
            not np.isfinite(
                group[["hotspot_c", "dhw_c_weeks", "actual_hotspot_c", "actual_dhw_c_weeks"]]
            )
            .all()
            .all()
        ):
            raise ValueError("Missing/nonfinite model outputs or outcomes")
        if (
            (group[["hotspot_c", "dhw_c_weeks", "actual_hotspot_c", "actual_dhw_c_weeks"]] < 0)
            .any()
            .any()
        ):
            raise ValueError("Negative constrained forecasts or outcomes")
        qcols = ["q10", "q50", "q90"]
        if all(c in group for c in qcols) and group[qcols].notna().any().any():
            qvalues = group[qcols].to_numpy(dtype=float)
            if not np.isfinite(qvalues).all() or (np.diff(qvalues, axis=1) < 0).any():
                raise ValueError("Invalid serialized quantiles")
    for split, sg in evaluated.groupby("split"):
        # Compare whole region-origin sets, so an arm absent from a region cannot hide.
        arm_keys = [set(zip(g.region_id, g.origin, strict=True)) for _, g in sg.groupby("arm")]
        if any(s != arm_keys[0] for s in arm_keys):
            raise ValueError("Models must use identical comparison origins")
        panels = [("full", sg)] + (
            [(str(y), sg[pd.to_datetime(sg.origin).dt.year == y]) for y in [2024, 2025]]
            if split == "test"
            else []
        )
        for period, pg in panels:
            for (region, arm), group in pg.groupby(["region_id", "arm"]):
                for horizon in [7, 14, 28]:
                    prefix = group[group.lead <= horizon]
                    endpoint = group[group.lead == horizon]
                    err = prefix.hotspot_c - prefix.actual_hotspot_c
                    common = dict(
                        split=split,
                        period=period,
                        region_id=region,
                        arm=arm,
                        horizon=int(horizon),
                        origins=int(prefix.origin.nunique()),
                        days=int(len(prefix)),
                    )
                    row = dict(
                        common,
                        hotspot_mae=float(abs(err).mean()),
                        hotspot_rmse=float(np.sqrt((err**2).mean())),
                        dhw_endpoint_mae=float(
                            abs(endpoint.dhw_c_weeks - endpoint.actual_dhw_c_weeks).mean()
                        ),
                    )
                    stress = prefix[prefix.actual_hotspot_c >= 1]
                    row["high_stress_days"] = len(stress)
                    row["high_stress_mae"] = (
                        float(abs(stress.hotspot_c - stress.actual_hotspot_c).mean())
                        if len(stress)
                        else None
                    )
                    if all(c in prefix and prefix[c].notna().all() for c in ["q10", "q50", "q90"]):
                        for level, col in [(0.1, "q10"), (0.5, "q50"), (0.9, "q90")]:
                            residual = prefix.actual_hotspot_c - prefix[col]
                            row[f"pinball_{col}"] = float(
                                np.maximum(level * residual, (level - 1) * residual).mean()
                            )
                        row["coverage_10_90"] = float(
                            (
                                (prefix.actual_hotspot_c >= prefix.q10)
                                & (prefix.actual_hotspot_c <= prefix.q90)
                            ).mean()
                        )
                        row["width_10_90"] = float((prefix.q90 - prefix.q10).mean())
                    metrics.append(row)
                    for threshold in [4.0, 8.0]:
                        for subset in ["all", "new_exceedance"]:
                            eg = (
                                endpoint
                                if subset == "all"
                                else endpoint[endpoint.origin_dhw_c_weeks < threshold]
                            )
                            truth, pred = (
                                eg.actual_dhw_c_weeks >= threshold,
                                eg.dhw_c_weeks >= threshold,
                            )
                            tp, fp, fn, tn = [
                                int(mask.sum())
                                for mask in [
                                    truth & pred,
                                    ~truth & pred,
                                    truth & ~pred,
                                    ~truth & ~pred,
                                ]
                            ]
                            thresholds.append(
                                dict(
                                    common,
                                    threshold=threshold,
                                    subset=subset,
                                    n=len(eg),
                                    tp=tp,
                                    fp=fp,
                                    fn=fn,
                                    tn=tn,
                                    positive_support=tp + fn,
                                    precision=tp / (tp + fp) if tp + fp else None,
                                    recall=tp / (tp + fn) if tp + fn else None,
                                    precision_reason=None if tp + fp else "no_predicted_positives",
                                    recall_reason=None if tp + fn else "no_observed_positives",
                                )
                            )
                    for origin, og in prefix.groupby("origin"):
                        paired.append(
                            dict(
                                common,
                                origin=str(pd.Timestamp(origin).date()),
                                hotspot_mae=float(abs(og.hotspot_c - og.actual_hotspot_c).mean()),
                            )
                        )
            # DHW persistence has no HotSpot score.
            unique = pg.drop_duplicates(["region_id", "origin", "lead"])
            for region, group in unique.groupby("region_id"):
                for horizon in [7, 14, 28]:
                    endpoint = group[group.lead == horizon]
                    metrics.append(
                        dict(
                            split=split,
                            period=period,
                            region_id=region,
                            arm="dhw_persistence",
                            horizon=horizon,
                            origins=int(endpoint.origin.nunique()),
                            days=int(len(endpoint)),
                            dhw_endpoint_mae=float(
                                abs(
                                    endpoint.origin_dhw_c_weeks - endpoint.actual_dhw_c_weeks
                                ).mean()
                            ),
                        )
                    )
                    for threshold in [4.0, 8.0]:
                        for subset in ["all", "new_exceedance"]:
                            eg = (
                                endpoint
                                if subset == "all"
                                else endpoint[endpoint.origin_dhw_c_weeks < threshold]
                            )
                            truth, pred = (
                                eg.actual_dhw_c_weeks >= threshold,
                                eg.origin_dhw_c_weeks >= threshold,
                            )
                            tp, fp, fn, tn = [
                                int(mask.sum())
                                for mask in [
                                    truth & pred,
                                    ~truth & pred,
                                    truth & ~pred,
                                    ~truth & ~pred,
                                ]
                            ]
                            thresholds.append(
                                dict(
                                    split=split,
                                    period=period,
                                    region_id=region,
                                    arm="dhw_persistence",
                                    horizon=horizon,
                                    origins=int(endpoint.origin.nunique()),
                                    days=len(endpoint),
                                    threshold=threshold,
                                    subset=subset,
                                    n=len(eg),
                                    tp=tp,
                                    fp=fp,
                                    fn=fn,
                                    tn=tn,
                                    positive_support=tp + fn,
                                    precision=tp / (tp + fp) if tp + fp else None,
                                    recall=tp / (tp + fn) if tp + fn else None,
                                    precision_reason=None if tp + fp else "no_predicted_positives",
                                    recall_reason=None if tp + fn else "no_observed_positives",
                                )
                            )
    table = pd.DataFrame(metrics)
    numeric = [
        c
        for c in table
        if c.endswith("mae")
        or c.endswith("rmse")
        or c.startswith("pinball_")
        or c in ["coverage_10_90", "width_10_90"]
    ]
    macro = table.groupby(["split", "period", "arm", "horizon"], as_index=False)[numeric].mean()
    support = (
        table.groupby(["split", "period", "arm", "horizon"])
        .agg(regions=("region_id", "nunique"), origins=("origins", "sum"), days=("days", "sum"))
        .reset_index()
    )
    macro = macro.merge(support)
    error_table = pd.DataFrame(paired)
    paired_differences = []
    for keys, group in error_table.groupby(["split", "period", "region_id", "horizon", "origin"]):
        errors = group.set_index("arm").hotspot_mae
        if "chronos" in errors or "chronos2" in errors or "chronos-2" in errors:
            model_arm = next(a for a in ["chronos", "chronos2", "chronos-2"] if a in errors)
            for baseline in ["persistence", "seasonal"]:
                if baseline in errors:
                    paired_differences.append(
                        dict(
                            zip(
                                ["split", "period", "region_id", "horizon", "origin"],
                                keys,
                                strict=True,
                            ),
                            arm=model_arm,
                            baseline=baseline,
                            mae_difference=float(errors[model_arm] - errors[baseline]),
                        )
                    )
    report = dict(
        metrics=json.loads(table.to_json(orient="records")),
        macro=json.loads(macro.to_json(orient="records")),
        thresholds=thresholds,
        per_origin_errors=paired,
        paired_differences=paired_differences,
    )
    # Unavailable metrics are null, never NaN; NumPy scalars become native JSON numbers so a
    # serialized report equals its fresh recomputation under any pandas version.
    return json.loads(json.dumps(report, allow_nan=False, default=json_native))
