# ruff: noqa: F821
"""Standalone stages; core/model names are linked by the notebook builder and tested together."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def reef_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False, default=json_native) + "\n", encoding="utf-8"
    )


def reef_read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def reef_load_data(root: Path):
    manifest = reef_read(root / "dataset_manifest.json")
    raw = (root / manifest["archive"]["filename"]).read_bytes()
    if (
        len(raw) != manifest["archive"]["bytes"]
        or hashlib.sha256(raw).hexdigest() != manifest["archive"]["sha256"]
    ):
        raise ValueError("Frozen NOAA archive integrity failure")
    frames = []
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        expected = {m["filename"] for m in manifest["members"]}
        if len(archive.namelist()) != len(expected) or set(archive.namelist()) != expected:
            raise ValueError("Unexpected or duplicate data members")
        for member in manifest["members"]:
            info = archive.getinfo(member["filename"])
            if info.file_size != member["bytes"]:
                raise ValueError("Unexpected uncompressed size")
            data = archive.read(member["filename"])
            if hashlib.sha256(data).hexdigest() != member["sha256"]:
                raise ValueError("NOAA member integrity failure")
            frames.append(parse_noaa(data.decode("utf-8"), member["region_id"]))
    return daily_panel(frames)


def reef_records(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for name in ("origin", "target_date", "date"):
        if name in frame:
            frame[name] = pd.to_datetime(frame[name])
    return frame


# Stage order and outputs. Restarting a stage removes its own and every later stage's
# records, so a completion file can never describe outputs from an earlier attempt.
REEF_STAGES = {
    "prepare": [
        "origin_manifest.csv",
        "seasonal_reference.csv",
        "experiment_config.json",
        "dataset_audit.json",
        "byod_example.csv",
    ],
    "baselines": ["baselines_forecasts.csv", "baselines_summary.json"],
    "validation": ["validation_forecasts.csv", "validation_summary.json", "validation_parity.json"],
    "activity": ["activity_forecasts.csv", "activity_summary.json"],
    "compare": ["validation_tables.txt"],
    "lock": ["budget_check.json", "experiment_lock.json"],
    "test": ["test_forecasts.csv", "test_summary.json", "test_parity.json"],
    "score": ["metrics.json", "results_tables.txt"],
    "future": ["outlook.csv", "outlook_exclusions.json"],
    "reload": ["reload_verification.json"],
    "report": [
        "forecasts.csv",
        "metrics.csv",
        "outlook.png",
        "failure_examples.png",
        "failure_examples.csv",
        "dhw_components.png",
        "environment.json",
        "run_summary.json",
        "conclusion.md",
        "README.txt",
        "checksums.json",
        "reef_evidence.zip",
    ],
}
# Planned splits and arms of each model-output file; checked against the frozen origins.
REEF_STAGE_ARMS = {
    "baselines": (("validation", "test"), ["persistence", "seasonal"]),
    "validation": (("validation",), ["chronos"]),
    "activity": (("validation",), ["chronos_180"]),
    "test": (("test",), ["chronos"]),
}
# Inputs whose identity the test stage requires to be unchanged since validation.
REEF_LOCKED_FILES = [
    "experiment_config.json",
    "origin_manifest.csv",
    "seasonal_reference.csv",
    "dataset_manifest.json",
    "model_manifest.json",
    "source_identity.json",
    "baselines_forecasts.csv",
    "validation_forecasts.csv",
    "activity_forecasts.csv",
]


def reef_invalidate(root: Path, stage: str) -> None:
    names = list(REEF_STAGES)
    for later in names[names.index(stage) :]:
        for name in REEF_STAGES[later]:
            (root / name).unlink(missing_ok=True)
            (root / name.replace(".csv", ".partial.csv")).unlink(missing_ok=True)


def reef_plan(root: Path) -> pd.DataFrame:
    """Frozen eligible origins, refused if the manifest changed after preparation."""
    config = reef_read(root / "experiment_config.json")
    if reef_sha(root / "origin_manifest.csv") != config["origin_manifest_sha256"]:
        raise ValueError("Origin manifest changed after freezing")
    plan = pd.read_csv(root / "origin_manifest.csv", parse_dates=["origin"])
    return plan[plan.eligible.astype(bool)].reset_index(drop=True)


def reef_stage_forecasts(root: Path, panel: pd.DataFrame, plan: pd.DataFrame, stage: str):
    splits, arms = REEF_STAGE_ARMS[stage]
    frame = reef_records(root / f"{stage}_forecasts.csv")
    try:
        validate_forecast_grid(frame, plan[plan.split.isin(splits)], arms, panel)
    except ValueError as exc:
        raise ValueError(f"{stage}_forecasts.csv: {exc}") from exc
    # Canonical key order: harmless row permutations cannot change floating-point sums.
    return frame.sort_values(FORECAST_KEYS, kind="stable").reset_index(drop=True)


def reef_prepare(root: Path) -> None:
    panel = reef_load_data(root)
    print(f"Loaded {len(panel)} regional daily records from the verified archive.", flush=True)
    origins = origin_manifest(panel)
    origins.to_csv(root / "origin_manifest.csv", index=False)
    seasonal = fit_seasonal(panel)
    seasonal.to_csv(root / "seasonal_reference.csv", index=False)
    eligible = origins[origins.eligible]
    counts = eligible.groupby(["region_id", "split"]).size().unstack(fill_value=0)
    if (counts["test"] < 20).any():
        raise ValueError("Insufficient test origins; revise the design explicitly")
    reef_json(
        root / "experiment_config.json",
        {
            "context_days": 365,
            "activity_context_days": 180,
            "horizons": [7, 14, 28],
            "primary": "14-day HotSpot MAE; equal-region macro mean",
            "validation_years": [2023],
            "test_years": [2024, 2025],
            "seasonal_fit_years": [2017, 2022],
            "seed": 42,
            "availability": "retrospective zero-delay assumption; publication vintages unavailable",
            "pretraining_overlap": "not established absent; no unseen-pretraining claim",
            "origin_manifest_sha256": reef_sha(root / "origin_manifest.csv"),
        },
    )
    audit = {"regions": [], "checks": []}
    for region in panel.index.get_level_values(0).unique():
        frame = panel.xs(region)
        hs = frame.hotspot_c
        heat = hs.where(hs >= 1, 0).where(hs.notna())
        rebuilt = heat.rolling(84, min_periods=84).sum() / 7
        valid = rebuilt.notna() & frame.dhw_c_weeks.notna()
        error = float((rebuilt[valid] - frame.loc[valid, "dhw_c_weeks"]).abs().max())
        if error > 0.0001:
            raise ValueError(f"DHW contract disagreement in {region}")
        audit["regions"].append(
            {
                "region_id": region,
                "missing_days": int(hs.isna().sum()),
                "dhw_compared_days": int(valid.sum()),
                "dhw_max_error": error,
            }
        )
    # Refusal probes run before any model; invalid input must never become a success.
    sample = panel.xs("northern").reset_index()[["date", "hotspot_c"]].tail(400).copy()
    sample.insert(0, "region_id", "example")
    sample["date"] = sample["date"].dt.strftime("%Y-%m-%d")
    sample.to_csv(root / "byod_example.csv", index=False)
    for label, bad in [
        ("duplicate", pd.concat([sample, sample.iloc[-1:]])),
        ("negative", sample.assign(hotspot_c=-1)),
        ("nonfinite", sample.assign(hotspot_c=np.inf)),
    ]:
        try:
            validate_byod(bad, units="degC", semantics_confirmed=True)
        except ValueError:
            audit["checks"].append({"probe": label, "refused": True})
        else:
            raise AssertionError(f"Failed refusal probe: {label}")
    reef_json(root / "dataset_audit.json", audit)
    print(counts.to_string())
    print("DHW reconstructed within 0.0001 °C-weeks. Frozen origins and settings saved.")


def reef_experiment(root: Path, stage: str) -> None:
    start = time.perf_counter()
    panel = reef_load_data(root)
    manifest = origin_manifest(panel)
    config = reef_read(root / "experiment_config.json")
    if reef_sha(root / "origin_manifest.csv") != config["origin_manifest_sha256"]:
        raise ValueError("Origin manifest changed after freezing")
    seasonal = pd.read_csv(root / "seasonal_reference.csv")
    split = "validation" if stage in {"validation", "activity"} else "test"
    selected = manifest[manifest.eligible & (manifest.split == split)]
    if stage == "test":
        lock = reef_read(root / "experiment_lock.json")
        changed = [n for n in REEF_LOCKED_FILES if lock["sha256"].get(n) != reef_sha(root / n)]
        if changed:
            raise ValueError(f"Experiment changed after validation: {changed}")
    model = None
    if stage != "baselines":
        model = reef_load_model(reef_read(root / "model_manifest.json"), root / "model_cache")
    else:
        selected = manifest[manifest.eligible]
    output = []
    for number, row in enumerate(selected.itertuples(), 1):
        origin = pd.Timestamp(row.origin)
        length = 180 if stage == "activity" else 365
        context = context_at(panel, row.region_id, origin, length=length)
        values = np.asarray(context, dtype=float)
        if stage == "baselines":
            dates = pd.date_range(origin + pd.Timedelta(days=1), periods=28)
            predictions = baseline_forecasts(context, dates, seasonal, row.region_id)
            for arm, prediction in predictions.items():
                output.append(
                    forecast_frame(panel, row.region_id, origin, arm, prediction, row.split)
                )
        else:
            q = reef_predict(model, values)
            arm = "chronos_180" if stage == "activity" else "chronos"
            output.append(
                forecast_frame(panel, row.region_id, origin, arm, q[:, 1], row.split, quantiles=q)
            )
            if number == 1:
                reef_json(
                    root / f"{stage}_parity.json",
                    {"history": values.tolist(), "quantiles": q.tolist()},
                )
        # Write progress immediately so an interrupted run never fabricates completion.
        pd.concat(output, ignore_index=True).to_csv(
            root / f"{stage}_forecasts.partial.csv", index=False
        )
        if number % 25 == 0:
            print(f"{stage}: {number}/{len(selected)} origins", flush=True)
    result = pd.concat(output, ignore_index=True)
    result.to_csv(root / f"{stage}_forecasts.csv", index=False)
    summary = {
        "stage": stage,
        "origins": len(selected),
        "seconds": time.perf_counter() - start,
        "complete": True,
        "rows": len(result),
    }
    if "quantile_crossing_c" in result:
        crossed = result.quantile_crossing_c > 0
        summary["quantile_crossing_days"] = int(crossed.sum())
        summary["quantile_crossing_max_c"] = float(result.quantile_crossing_c.max())
    if model is not None:
        import torch

        summary["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
    reef_json(root / f"{stage}_summary.json", summary)
    print(summary)


def reef_lock(root: Path) -> None:
    validation = reef_read(root / "validation_summary.json")
    activity = reef_read(root / "activity_summary.json")
    if not validation["complete"] or not activity["complete"]:
        raise ValueError("Validation and controlled experiment must finish first")
    setup = reef_read(root / "setup_summary.json")
    baseline = reef_read(root / "baselines_summary.json")
    elapsed = setup["seconds"] + baseline["seconds"] + validation["seconds"] + activity["seconds"]
    # Conservative model-stage estimate includes test origins and two extra reloads.
    remaining = validation["seconds"] * (184 + 5) / validation["origins"] + 120
    peak = max(validation.get("peak_gpu_bytes", 0), activity.get("peak_gpu_bytes", 0))
    budget = {
        "elapsed_seconds": elapsed,
        "estimated_remaining_seconds": remaining,
        "estimated_total_seconds": elapsed + remaining,
        "peak_gpu_bytes": peak,
        "limit_seconds": 1800,
        "limit_gpu_bytes": 12 * 1024**3,
        "scope": "validation-based estimate, not measured complete runtime",
    }
    reef_json(root / "budget_check.json", budget)
    if elapsed + remaining > 1800 or peak > 12 * 1024**3:
        raise RuntimeError(
            "Validation exceeds the resource budget; preserve logs and revise design before test"
        )
    panel, plan = reef_load_data(root), reef_plan(root)
    for stage in ("baselines", "validation", "activity"):
        reef_stage_forecasts(root, panel, plan, stage)
    reef_json(
        root / "experiment_lock.json",
        {
            "sha256": {name: reef_sha(root / name) for name in REEF_LOCKED_FILES},
            "selection": "365-day default retained; no test-based tuning",
        },
    )
    print("365-day context retained. Test settings locked.")


def reef_outlook_rows(region, origin, arm, context, raw, quantiles=None) -> list[dict]:
    """Unscored forward rows keep raw, constrained, quantile and DHW-component fields."""
    raw = np.asarray(raw, dtype=float)
    crossing = None
    if quantiles is not None:
        model_q = np.asarray(quantiles, dtype=float)
        quantiles, crossing = rearrange_quantiles(model_q)
        raw = quantiles[:, 1]
    components = compose_dhw(context, np.maximum(raw, 0))
    dates = pd.date_range(pd.Timestamp(origin) + pd.Timedelta(days=1), periods=len(raw))
    rows = []
    for k, day in enumerate(dates):
        row = {
            "region_id": region,
            "origin": pd.Timestamp(origin),
            "target_date": day,
            "lead": k + 1,
            "arm": arm,
            "raw_hotspot_c": float(raw[k]),
            "hotspot_c": float(max(0.0, raw[k])),
            "clipped_to_zero": bool(raw[k] < 0),
            "dhw_c_weeks": components["dhw"][k],
            "known_dhw": components["known"][k],
            "predicted_dhw": components["predicted"][k],
        }
        for i, label in enumerate(["q10", "q50", "q90"]):
            row["model_" + label] = float(model_q[k, i]) if crossing is not None else np.nan
        row["quantile_crossing_c"] = float(crossing[k]) if crossing is not None else np.nan
        for i, label in enumerate(["q10", "q50", "q90"]):
            value = float(quantiles[k, i]) if quantiles is not None else np.nan
            row["raw_" + label] = value
            row[label] = max(0.0, value) if quantiles is not None else np.nan
        rows.append(row)
    return rows


def reef_future(root: Path) -> None:
    panel = reef_load_data(root)
    origin = pd.Timestamp(reef_read(root / "dataset_manifest.json")["coverage_end"])
    seasonal = pd.read_csv(root / "seasonal_reference.csv")
    model = reef_load_model(reef_read(root / "model_manifest.json"), root / "model_cache")
    rows, skipped = [], []
    for region in panel.index.get_level_values(0).unique():
        try:
            context = context_at(panel, region, origin)
        except ValueError as exc:
            skipped.append({"region_id": region, "reason": str(exc)})
            continue
        dates = pd.date_range(origin + pd.Timedelta(days=1), periods=28)
        arms = baseline_forecasts(context, dates, seasonal, region)
        q = reef_predict(model, context)
        arms["chronos"] = q[:, 1]
        for arm, raw in arms.items():
            quantiles = q if arm == "chronos" else None
            rows.extend(reef_outlook_rows(region, origin, arm, context, raw, quantiles))
    pd.DataFrame(rows).to_csv(root / "outlook.csv", index=False)
    reef_json(root / "outlook_exclusions.json", skipped)
    print(
        f"Illustrative origin {origin.date()}; {len(skipped)} regions excluded for missing context."
    )


def reef_same(saved: pd.DataFrame, again: pd.DataFrame) -> None:
    """Numeric parity plus identity: region, arm, split and calendar dates must match."""
    if len(saved) != len(again):
        raise AssertionError("Reloaded row count differs")
    for col in ["region_id", "arm", "split", "origin", "target_date"]:
        if not (saved[col].astype(str).to_numpy() == again[col].astype(str).to_numpy()).all():
            raise AssertionError(f"Reloaded identity column differs: {col}")
    for col in again.select_dtypes(include="number"):
        np.testing.assert_allclose(saved[col], again[col], atol=1e-8, rtol=1e-7, equal_nan=True)


def reef_reload(root: Path) -> None:
    """Fresh CLI process recomputes outputs and reloads real model for one origin."""
    panel, plan = reef_load_data(root), reef_plan(root)
    seasonal = pd.read_csv(root / "seasonal_reference.csv")
    baseline = reef_stage_forecasts(root, panel, plan, "baselines")
    rebuilt = []
    for (region, origin), group in baseline.groupby(["region_id", "origin"]):
        context = context_at(panel, region, origin)
        dates = pd.date_range(origin + pd.Timedelta(days=1), periods=28)
        for arm, pred in baseline_forecasts(context, dates, seasonal, region).items():
            rebuilt.append(forecast_frame(panel, region, origin, arm, pred, group.iloc[0]["split"]))
    recomputed = pd.concat(rebuilt, ignore_index=True)
    keys = ["region_id", "origin", "arm", "lead"]
    original = baseline.sort_values(keys).reset_index(drop=True)
    recomputed = recomputed.sort_values(keys).reset_index(drop=True)
    reef_same(original, recomputed)
    # Recalculate model-derived DHW and metrics from serialized raw predictions.
    for stage in ("validation", "activity", "test"):
        saved = reef_stage_forecasts(root, panel, plan, stage)
        rebuilt_model = []
        for (region, origin, arm), group in saved.groupby(["region_id", "origin", "arm"]):
            group = group.sort_values("lead")
            # Rebuild from the model's own (unsorted) quantiles, as the stage did.
            qcols = ["model_q10", "model_q50", "model_q90"]
            qs = group[qcols].to_numpy() if set(qcols).issubset(group) else None
            raw_col = "model_q50" if qs is not None else "raw_hotspot_c"
            rebuilt_model.append(
                forecast_frame(
                    panel,
                    region,
                    origin,
                    arm,
                    group[raw_col].to_numpy(),
                    group.iloc[0]["split"],
                    quantiles=qs,
                )
            )
        again = pd.concat(rebuilt_model, ignore_index=True).sort_values(keys).reset_index(drop=True)
        reef_same(saved.sort_values(keys).reset_index(drop=True), again)
    if reef_read(root / "metrics.json") != reef_metrics(root):
        raise AssertionError("Fresh-process metric reconstruction mismatch")
    sample = reef_read(root / "validation_parity.json")
    model = reef_load_model(reef_read(root / "model_manifest.json"), root / "model_cache")
    q = reef_predict(model, sample["history"])
    np.testing.assert_allclose(q, sample["quantiles"], atol=1e-5, rtol=1e-5)
    reef_json(
        root / "reload_verification.json",
        {
            "fresh_process": True,
            "baseline_parity": True,
            "derived_dhw_parity": True,
            "model_reload_parity": True,
            "metrics_reload_parity": True,
            "atol": 1e-5,
            "rtol": 1e-5,
        },
    )
    print("Fresh-process baseline, DHW and real-model reload parity passed.")


def reef_joined(root: Path, stages=("baselines", "validation", "activity", "test")):
    panel, plan = reef_load_data(root), reef_plan(root)
    frames = [reef_stage_forecasts(root, panel, plan, stage) for stage in stages]
    return pd.concat(frames, ignore_index=True), plan


REEF_SCOPES = {"all": None, "common_2024": 2024, "available_2025": 2025}


def reef_check_support(report: dict, plan: pd.DataFrame, year: int | None) -> None:
    """Recomputed macro support must equal the frozen plan; nothing is silently dropped."""
    if year is not None:
        plan = plan[plan.origin.dt.year == year]
    for row in report["macro"]:
        subset = plan[plan.split == row["split"]]
        if row["period"] != "full":
            subset = subset[subset.origin.dt.year == int(row["period"])]
        expected = (int(subset.region_id.nunique()), len(subset))
        if (row["regions"], row["origins"]) != expected:
            raise ValueError(
                f"Support mismatch for {row['split']}/{row['period']}/{row['arm']}/"
                f"{row['horizon']}: {(row['regions'], row['origins'])} != planned {expected}"
            )


def reef_metrics(root: Path) -> dict:
    joined, plan = reef_joined(root)
    evaluate_forecasts(
        joined[joined.split == "test"], expected_arms=["persistence", "seasonal", "chronos"]
    )
    reports = {}
    for label, year in REEF_SCOPES.items():
        subset = joined if year is None else joined[joined.origin.dt.year == year]
        reports[label] = evaluate_forecasts(subset)
        reef_check_support(reports[label], plan, year)
    return reports


def reef_score(root: Path) -> None:
    reports = reef_metrics(root)
    reef_json(root / "metrics.json", reports)
    text = reef_results_tables(reports)
    (root / "results_tables.txt").write_text(text, encoding="utf-8")
    print(text)
    print("Saved metrics for independent fresh-process reconstruction.")


def _fmt(frame: pd.DataFrame) -> str:
    return frame.to_string(index=False, float_format=lambda v: f"{v:.3f}", na_rep="n/a")


def _event_reason(n: int, support: int, predicted: int) -> str:
    if n == 0:
        return "no observations"
    if support == 0 and predicted == 0:
        return "no observed or predicted positives"
    if support == 0:
        return "no observed positives: recall undefined"
    if predicted == 0:
        return "no predicted positives: precision undefined"
    return ""


def reef_event_table(records: list[dict], horizons=(14, 28)) -> pd.DataFrame:
    """Pool region-level threshold counts per arm; undefined ratios stay unavailable."""
    frame = pd.DataFrame(records)
    frame = frame[frame.horizon.isin(horizons)]
    keys = ["horizon", "threshold", "subset", "arm"]
    pooled = frame.groupby(keys, as_index=False)[["n", "tp", "fp", "fn", "tn"]].sum()
    pooled["observed_pos"] = pooled.tp + pooled.fn
    predicted = pooled.tp + pooled.fp
    pooled["precision"] = (pooled.tp / predicted).where(predicted > 0)
    pooled["recall"] = (pooled.tp / pooled.observed_pos).where(pooled.observed_pos > 0)
    pooled["threshold"] = pooled.threshold.map(lambda t: f"{t:g} °C-wk")
    pooled["note"] = [
        _event_reason(int(n), int(s), int(p))
        for n, s, p in zip(pooled.n, pooled.observed_pos, predicted, strict=True)
    ]
    return pooled


def reef_results_tables(reports: dict) -> str:
    """Compact learner views of the frozen comparison; complete tables are in metrics.csv."""
    report = reports["all"]
    macro = pd.DataFrame(report["macro"])
    test = macro[macro.split == "test"]
    names = {"full": "Full test (unbalanced)", "2024": "Common 2024", "2025": "2025 (3 regions)"}
    out = []
    rows = []
    for period, label in names.items():
        for arm in ["persistence", "seasonal", "chronos", "dhw_persistence"]:
            sel = test[(test.period == period) & (test.arm == arm)].set_index("horizon")
            if sel.empty:
                continue
            rows.append(
                {
                    "panel": label,
                    "arm": arm,
                    "MAE 7d": sel.hotspot_mae.get(7, np.nan),
                    "MAE 14d": sel.hotspot_mae.get(14, np.nan),
                    "MAE 28d": sel.hotspot_mae.get(28, np.nan),
                    "DHW MAE 14d": sel.dhw_endpoint_mae.get(14, np.nan),
                    "DHW MAE 28d": sel.dhw_endpoint_mae.get(28, np.nan),
                    "regions": int(sel.regions.get(14)),
                    "origins": int(sel.origins.get(14)),
                    "days@14d": int(sel.days.get(14)),
                }
            )
    out += [
        "A. Test HotSpot MAE (°C) and DHW endpoint MAE (°C-weeks); equal-region macro means.",
        "   Support counts show exactly which regions and origins each score covers.",
        "   dhw_persistence carries DHW forward and has no HotSpot score (n/a).",
        _fmt(pd.DataFrame(rows)),
        "",
    ]
    region = pd.DataFrame(report["metrics"])
    region = region[(region.split == "test") & (region.period == "full") & (region.horizon == 14)]
    wide = region[region.arm != "dhw_persistence"].pivot(
        index="region_id", columns="arm", values="hotspot_mae"
    )
    wide["origins"] = region.groupby("region_id").origins.first()
    out += [
        "B. 14-day HotSpot MAE by region, full test panel (°C).",
        _fmt(wide.reset_index()),
        "",
    ]
    chronos = test[test.arm == "chronos"]
    unc = chronos[["period", "horizon", "coverage_10_90", "width_10_90"]].copy()
    for col in ["pinball_q10", "pinball_q50", "pinball_q90"]:
        unc[col] = chronos[col]
    out += [
        "C. Chronos q10–q90 marginal interval: coverage (nominal 0.80), width and pinball loss.",
        _fmt(unc.sort_values(["period", "horizon"])),
        "",
    ]
    metrics = pd.DataFrame(report["metrics"])
    hs = metrics[(metrics.split == "test") & (metrics.period == "full")]
    hs = hs[hs.arm != "dhw_persistence"]
    hs = hs.groupby(["arm", "horizon"], as_index=False).agg(
        high_stress_days=("high_stress_days", "sum"), high_stress_mae=("high_stress_mae", "mean")
    )
    hs["high_stress_days"] = hs.high_stress_days.astype(int)
    out += [
        "D. Errors on observed high-stress days (HotSpot >= 1 °C), full test panel.",
        "   high_stress_mae is the equal-region mean over regions with such days.",
        _fmt(hs),
        "",
    ]
    thresholds = [r for r in report["thresholds"] if r["split"] == "test" and r["period"] == "full"]
    events = reef_event_table(thresholds)
    out += [
        "E. DHW threshold events at the endpoint, pooled over region-origins (full test panel).",
        "   new_exceedance keeps origins that were below the threshold at the origin.",
        "   Unavailable precision/recall is n/a with a reason; it is never perfect performance.",
        _fmt(events),
        "",
    ]
    paired = pd.DataFrame(report["paired_differences"])
    paired = paired[(paired.split == "test") & (paired.horizon == 14)]
    summary = (
        paired.groupby(["period", "baseline"])
        .mae_difference.agg(
            pairs="count",
            mean_difference="mean",
            median_difference="median",
            chronos_better=lambda d: float((d < 0).mean()),
            tie=lambda d: float((d == 0).mean()),
            chronos_worse=lambda d: float((d > 0).mean()),
        )
        .reset_index()
    )
    out += [
        "F. Paired 14-day MAE difference, Chronos minus baseline, per region-origin (°C).",
        "   Negative values favour Chronos. The last three columns are shares of region-origins",
        "   where Chronos was better, tied or worse. Descriptive only; no significance test.",
        _fmt(summary),
        "",
        "Complete tables: metrics.csv (scope/table columns) inside reef_evidence.zip.",
    ]
    return "\n".join(out)


def reef_compare(root: Path) -> None:
    """Validation-only view of the controlled context-length activity, before the test."""
    joined, plan = reef_joined(root, ("baselines", "validation", "activity"))
    validation = joined[joined.split == "validation"]
    report = evaluate_forecasts(validation)
    reef_check_support(report, plan[plan.split == "validation"], None)
    macro = pd.DataFrame(report["macro"])
    cols = ["arm", "horizon", "hotspot_mae", "dhw_endpoint_mae", "regions", "origins", "days"]
    table = macro[cols].sort_values(["horizon", "arm"])
    region = pd.DataFrame(report["metrics"])
    region = region[(region.horizon == 14) & region.arm.isin(["chronos", "chronos_180"])]
    wide = region.pivot(index="region_id", columns="arm", values="hotspot_mae")
    wide["180 minus 365"] = wide.chronos_180 - wide.chronos
    wide["origins"] = region.groupby("region_id").origins.first()
    manifest = pd.read_csv(root / "origin_manifest.csv", parse_dates=["origin"])
    valid = plan[plan.split == "validation"].iloc[0]
    origin = pd.Timestamp(valid.origin)
    rejected = manifest[~manifest.eligible.astype(bool)]
    preferred = rejected[rejected.reason.str.contains("incomplete_365_day_context")]
    reject = (preferred if len(preferred) else rejected).iloc[0]
    lines = [
        "Validation 2023 only; the test set is still unopened.",
        "Macro HotSpot MAE (°C) and DHW endpoint MAE (°C-weeks) with support:",
        _fmt(table),
        "",
        "14-day HotSpot MAE by region: 365-day default (chronos) vs 180-day activity:",
        _fmt(wide.reset_index()),
        "",
        f"Trace a valid context: {valid.region_id}, origin {origin.date()}.",
        f"  365-day context {(origin - pd.Timedelta(days=364)).date()} to {origin.date()};",
        f"  180-day context {(origin - pd.Timedelta(days=179)).date()} to {origin.date()}"
        " (its final 180 days);",
        f"  targets {(origin + pd.Timedelta(days=1)).date()} to "
        f"{(origin + pd.Timedelta(days=28)).date()}.",
        f"Trace a rejected origin: {reject.region_id}, {pd.Timestamp(reject.origin).date()}: "
        f"{reject.reason}.",
        "The 365-day test default stays fixed whichever validation context wins.",
    ]
    text = "\n".join(lines)
    (root / "validation_tables.txt").write_text(text, encoding="utf-8")
    print(text)


def reef_report(root: Path) -> None:
    import importlib.metadata

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    stages = ["baselines", "validation", "activity", "test"]
    joined, _plan = reef_joined(root)
    evaluate_forecasts(
        joined[joined.split == "test"], expected_arms=["persistence", "seasonal", "chronos"]
    )
    joined.to_csv(root / "forecasts.csv", index=False)
    reports = reef_read(root / "metrics.json")
    if reports != reef_metrics(root):
        raise AssertionError("Metrics changed after scoring")
    flat = []
    for scope, report in reports.items():
        for table, records in report.items():
            if isinstance(records, list):
                for row in records:
                    if isinstance(row, dict):
                        flat.append({"scope": scope, "table": table, **row})
    pd.DataFrame(flat).to_csv(root / "metrics.csv", index=False)
    outlook = reef_records(root / "outlook.csv")
    regions = list(outlook.region_id.unique())
    issued = pd.Timestamp(outlook.origin.iloc[0]).date()
    excluded = reef_read(root / "outlook_exclusions.json")

    def dates_axis(ax):
        locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    fig, axes = plt.subplots(len(regions), 1, figsize=(10, 3 * len(regions)), squeeze=False)
    for ax, region in zip(axes[:, 0], regions, strict=True):
        for arm, group in outlook[outlook.region_id == region].groupby("arm"):
            ax.plot(group.target_date, group.dhw_c_weeks, label=arm)
        ax.axhline(4, color="orange", linestyle=":")
        ax.axhline(8, color="red", linestyle=":")
        ax.set_title(f"{region.title()} — DHW from frozen origin {issued}")
        ax.set_ylabel("°C-weeks")
        dates_axis(ax)
        ax.legend()
    note = "Illustrative retrospective outlook issued from frozen origin " + str(issued)
    note += "; not a live warning."
    if excluded:
        note += "\nNot shown (incomplete 365-day context): " + ", ".join(
            e["region_id"] for e in excluded
        )
    fig.suptitle(note, fontsize=10)
    fig.tight_layout()
    fig.savefig(root / "outlook.png", dpi=140)
    plt.close(fig)
    # Predeclared failure selection: worst 14-day Chronos MAE in each region.
    test = joined[(joined.split == "test") & (joined.arm == "chronos")]
    errors = test[test.lead <= 14].copy()
    errors["absolute_error"] = abs(errors.hotspot_c - errors.actual_hotspot_c)
    ranked = errors.groupby(["region_id", "origin"]).absolute_error.mean().reset_index()
    chosen = ranked.sort_values(
        ["absolute_error", "origin"], ascending=[False, True]
    ).drop_duplicates("region_id")
    chosen.to_csv(root / "failure_examples.csv", index=False)
    fig, axes = plt.subplots(len(chosen), 1, figsize=(10, 3 * len(chosen)), squeeze=False)
    for ax, row in zip(axes[:, 0], chosen.itertuples(), strict=True):
        group = test[(test.region_id == row.region_id) & (test.origin == row.origin)]
        ax.plot(group.target_date, group.actual_hotspot_c, label="Observed HotSpot")
        ax.plot(group.target_date, group.hotspot_c, label="Chronos median")
        ax.fill_between(group.target_date, group.q10, group.q90, alpha=0.2, label="q10–q90")
        ax.set_title(f"{row.region_id}: worst 14-day MAE origin {row.origin.date()}")
        ax.set_ylabel("HotSpot °C")
        dates_axis(ax)
        ax.legend()
    fig.tight_layout()
    fig.savefig(root / "failure_examples.png", dpi=140)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    sample = outlook[(outlook.region_id == regions[0]) & (outlook.arm == "chronos")]
    ax.stackplot(
        sample.target_date,
        sample.known_dhw,
        sample.predicted_dhw,
        labels=["Known historical contribution", "New predicted contribution"],
        alpha=0.7,
    )
    ax.set_title(f"{regions[0].title()}: what contributes to DHW after frozen origin {issued}?")
    ax.set_ylabel("°C-weeks")
    dates_axis(ax)
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "dhw_components.png", dpi=140)
    plt.close(fig)
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["torch", "chronos-forecasting", "numpy", "pandas", "transformers"]
        },
    }
    reef_json(root / "environment.json", env)
    reload = reef_read(root / "reload_verification.json")
    summary = {
        "status": "executed_candidate_requires_maintainer_review",
        "scope": "retrospective regional satellite heat stress; not observed bleaching",
        "stages": {s: reef_read(root / f"{s}_summary.json") for s in stages},
        "reload": reload,
        "refusal_probes": reef_read(root / "dataset_audit.json")["checks"],
        "negative_point_predictions_clamped": int((joined.raw_hotspot_c < 0).sum()),
        "quantile_crossing_days": int((joined.quantile_crossing_c.fillna(0) > 0).sum()),
        "quantile_crossing_max_c": float(joined.quantile_crossing_c.fillna(0).max()),
        "source": reef_read(root / "source_identity.json"),
        "model": reef_read(root / "model_manifest.json"),
        "dataset_archive_sha256": reef_read(root / "dataset_manifest.json")["archive"]["sha256"],
        "warnings": [
            "Western/Southern have no eligible 2025 test origins",
            "No confirmed exclusion from foundation-model pretraining",
            "Marginal HotSpot bands are not DHW probability intervals",
            "Crossing Chronos quantiles are sorted (monotone rearrangement) before scoring",
        ],
    }
    reef_json(root / "run_summary.json", summary)
    (root / "conclusion.md").write_text(
        "# Interpretation\n\nUse metrics.csv: compare 14-day HotSpot MAE on identical origins. "
        "Report the winning arm and error, persistence/seasonal errors, common-2024 comparison, "
        "and the sign of the paired difference. A loss or inconclusive result is valid.\n\n"
        "Explain one failure mode, interval coverage and event support. Separate already "
        "accumulated DHW from newly predicted stress. These are regional satellite summaries, "
        "not reef-level observations or bleaching diagnoses. Missing contexts exclude two "
        "regions in 2025; historical availability and pretraining independence are unresolved.\n\n"
        "Personal completion record: task, evidence, baseline, limitation, next experiment. "
        "This record is not a required submission.\n",
        encoding="utf-8",
    )
    (root / "README.txt").write_text(
        "NOAA CRW public-domain data, credited in dataset_manifest.json. "
        "This bundle contains experiment outputs, not model weights. Model identity is pinned. "
        "Candidate evidence requires maintainer review; it is not an operational NOAA product.\n"
    )
    keep = [
        p
        for p in root.iterdir()
        if p.is_file()
        and p.suffix in {".json", ".csv", ".png", ".md", ".txt"}
        and ".partial." not in p.name
        and p.name != "byod_example.csv"
        and p.name != "checksums.json"
    ]
    reef_json(root / "checksums.json", {p.name: reef_sha(p) for p in sorted(keep)})
    bundle = root / "reef_evidence.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for p in keep + [root / "checksums.json"]:
            archive.write(p, p.name)
    print(f"Exported {bundle}. Results are evidence, not release approval.")


def reef_byod(root: Path, source: Path) -> Path:
    """Forward-only BYOD: literal IDs, unscored forecasts and a per-input receipt."""
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    frame = validate_byod(read_byod_csv(source), units="degC", semantics_confirmed=True)
    panel = daily_panel([frame])
    manifest = reef_read(root / "model_manifest.json")
    out = root / "byod" / digest[:16]
    out.mkdir(parents=True, exist_ok=True)
    for stale in ("forecasts.csv", "receipt.json"):
        (out / stale).unlink(missing_ok=True)
    model = reef_load_model(manifest, root / "model_cache")
    rows, inputs = [], []
    for region in panel.index.get_level_values(0).unique():
        origin = panel.xs(region).index.max()
        history = context_at(panel, region, origin)
        q = reef_predict(model, history)
        rows.extend(reef_outlook_rows(region, origin, "chronos", history, q[:, 1], q))
        inputs.append(
            {
                "region_id": region,
                "first_date": str(panel.xs(region).index.min().date()),
                "origin": str(origin.date()),
                "context_days": len(history),
            }
        )
    forecasts = pd.DataFrame(rows)
    forecasts.to_csv(out / "forecasts.csv", index=False)
    identity = root / "source_identity.json"
    reef_json(
        out / "receipt.json",
        {
            "input_file": source.name,
            "input_sha256": digest,
            "input_rows": len(frame),
            "regions": inputs,
            "units": "degC",
            "semantics": "NOAA-compatible threshold-relative HotSpot, confirmed by user",
            "model": {"modelId": manifest["modelId"], "revision": manifest["revision"]},
            "runner_sha256": (
                reef_read(identity).get("runner_sha256") if identity.exists() else None
            ),
            "outputs": {"forecasts.csv": reef_sha(out / "forecasts.csv")},
            "scored": False,
            "scope": "forward inference only; no future outcomes, no accuracy claim",
        },
    )
    print(f"{len(inputs)} BYOD region(s): {[i['region_id'] for i in inputs]}")
    print(f"Saved unscored forecasts and receipt in {out}")
    return out


def reef_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=[
            *REEF_STAGES,
            "byod",
        ],
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()
    # Checkpoint: the child started and its imports loaded (separates launch from stage failures).
    print(f"[runner pid {os.getpid()}] imports loaded; stage {args.stage}", flush=True)
    root = args.root.resolve()
    if args.stage == "byod":
        reef_byod(root, args.csv)
        return
    reef_invalidate(root, args.stage)
    if args.stage in {"baselines", "validation", "activity", "test"}:
        reef_experiment(root, args.stage)
    else:
        {
            "prepare": reef_prepare,
            "score": reef_score,
            "compare": reef_compare,
            "lock": reef_lock,
            "future": reef_future,
            "reload": reef_reload,
            "report": reef_report,
        }[args.stage](root)


if __name__ == "__main__":
    reef_main()
