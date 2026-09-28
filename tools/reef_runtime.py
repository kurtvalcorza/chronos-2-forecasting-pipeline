# ruff: noqa: F821
"""Standalone stages; core/model names are linked by the notebook builder and tested together."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def reef_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8"
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


def reef_prepare(root: Path) -> None:
    panel = reef_load_data(root)
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
        if lock["config_sha256"] != reef_sha(root / "experiment_config.json"):
            raise ValueError("Experiment changed after validation")
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
    reef_json(
        root / "experiment_lock.json",
        {
            "config_sha256": reef_sha(root / "experiment_config.json"),
            "validation_sha256": reef_sha(root / "validation_forecasts.csv"),
            "activity_sha256": reef_sha(root / "activity_forecasts.csv"),
            "selection": "365-day default retained; no test-based tuning",
        },
    )
    print("365-day context retained. Test settings locked.")


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
            components = compose_dhw(context, np.maximum(raw, 0))
            for k, day in enumerate(dates):
                rows.append(
                    {
                        "region_id": region,
                        "origin": origin,
                        "target_date": day,
                        "lead": k + 1,
                        "arm": arm,
                        "hotspot_c": max(0, raw[k]),
                        "dhw_c_weeks": components["dhw"][k],
                        "known_dhw": components["known"][k],
                        "predicted_dhw": components["predicted"][k],
                    }
                )
    pd.DataFrame(rows).to_csv(root / "outlook.csv", index=False)
    reef_json(root / "outlook_exclusions.json", skipped)
    print(
        f"Illustrative origin {origin.date()}; {len(skipped)} regions excluded for missing context."
    )


def reef_reload(root: Path) -> None:
    """Fresh CLI process recomputes outputs and reloads real model for one origin."""
    panel = reef_load_data(root)
    seasonal = pd.read_csv(root / "seasonal_reference.csv")
    baseline = reef_records(root / "baselines_forecasts.csv")
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
    for col in recomputed.select_dtypes(include="number"):
        np.testing.assert_allclose(
            original[col], recomputed[col], atol=1e-8, rtol=1e-7, equal_nan=True
        )
    # Recalculate model-derived DHW and metrics from serialized raw predictions.
    for stage in ("validation", "activity", "test"):
        saved = reef_records(root / f"{stage}_forecasts.csv")
        rebuilt_model = []
        for (region, origin, arm), group in saved.groupby(["region_id", "origin", "arm"]):
            group = group.sort_values("lead")
            qcols = ["raw_q10", "raw_q50", "raw_q90"]
            qs = group[qcols].to_numpy() if set(qcols).issubset(group) else None
            raw_col = "raw_hotspot_c"
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
        saved = saved.sort_values(keys).reset_index(drop=True)
        for col in again.select_dtypes(include="number"):
            np.testing.assert_allclose(saved[col], again[col], atol=1e-8, rtol=1e-7, equal_nan=True)
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


def reef_metrics(root: Path) -> dict:
    frames = [
        reef_records(root / f"{stage}_forecasts.csv")
        for stage in ["baselines", "validation", "activity", "test"]
    ]
    joined = pd.concat(frames, ignore_index=True)
    evaluate_forecasts(
        joined[joined.split == "test"], expected_arms=["persistence", "seasonal", "chronos"]
    )
    return {
        label: evaluate_forecasts(subset)
        for label, subset in [
            ("all", joined),
            ("common_2024", joined[joined.origin.dt.year == 2024]),
            ("available_2025", joined[joined.origin.dt.year == 2025]),
        ]
    }


def reef_score(root: Path) -> None:
    reef_json(root / "metrics.json", reef_metrics(root))
    print("Saved metrics for independent fresh-process reconstruction.")


def reef_report(root: Path) -> None:
    import importlib.metadata

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = ["baselines", "validation", "activity", "test"]
    frames = [reef_records(root / f"{stage}_forecasts.csv") for stage in stages]
    joined = pd.concat(frames, ignore_index=True)
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
    fig, axes = plt.subplots(len(regions), 1, figsize=(10, 3 * len(regions)), squeeze=False)
    for ax, region in zip(axes[:, 0], regions, strict=True):
        for arm, group in outlook[outlook.region_id == region].groupby("arm"):
            ax.plot(group.target_date, group.dhw_c_weeks, label=arm)
        ax.axhline(4, color="orange", linestyle=":")
        ax.axhline(8, color="red", linestyle=":")
        ax.set_title(f"{region.title()} — illustrative regional DHW outlook")
        ax.set_ylabel("°C-weeks")
        ax.legend()
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
    ax.set_title(f"{regions[0]}: what contributes to future DHW?")
    ax.set_ylabel("°C-weeks")
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
        "source": reef_read(root / "source_identity.json"),
        "model": reef_read(root / "model_manifest.json"),
        "dataset_archive_sha256": reef_read(root / "dataset_manifest.json")["archive"]["sha256"],
        "warnings": [
            "Western/Southern have no eligible 2025 test origins",
            "No confirmed exclusion from foundation-model pretraining",
            "Marginal HotSpot bands are not DHW probability intervals",
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
        and p.name != "checksums.json"
    ]
    reef_json(root / "checksums.json", {p.name: reef_sha(p) for p in sorted(keep)})
    bundle = root / "reef_evidence.zip"
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for p in keep + [root / "checksums.json"]:
            archive.write(p, p.name)
    print(f"Exported {bundle}. Results are evidence, not release approval.")


def reef_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=[
            "prepare",
            "score",
            "baselines",
            "validation",
            "activity",
            "lock",
            "test",
            "future",
            "reload",
            "report",
        ],
    )
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.stage in {"baselines", "validation", "activity", "test"}:
        reef_experiment(root, args.stage)
    else:
        {
            "prepare": reef_prepare,
            "score": reef_score,
            "lock": reef_lock,
            "future": reef_future,
            "reload": reef_reload,
            "report": reef_report,
        }[args.stage](root)


if __name__ == "__main__":
    reef_main()
