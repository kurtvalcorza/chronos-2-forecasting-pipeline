"""Standalone source-block forecasting experiment; no DIMER runtime imports."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import time
import zipfile
from pathlib import Path

import dengue_core as core
import dengue_data as data
import dengue_models as models
import numpy as np

STAGES = (
    "prepare",
    "baselines",
    "validation",
    "lock",
    "test",
    "activity",
    "future",
    "reload",
    "report",
)
SYSTEMS = ("persistence", "seasonal", "ridge", "chronos", "mitra_cases", "mitra_weather")
SCOPE = "exploratory_published_source_blocks"
BYOD_FIELDS = ["year", "block", "cases", "rain", "temp"]
DEFAULT_AREA = "Quezon City, Philippines"
DEFAULT_SOURCE = "Zenodo record 21978184 (QC Data sheet), ODC-ODbL 1.0"
RESOURCE_TARGETS = {
    "kind": "targets, not measurements",
    "wall_minutes": 60,
    "free_disk_gib": 20,
    "peak_gpu_allocated_gib": 12,
}
LIMITATIONS = [
    "Published source-block order only: calendar dates and case-weather alignment are unverified",
    "Final data vintages: revised counts and retrospective IMERG/ERA5-Land products do not "
    "reconstruct what was available when each forecast would have been issued",
    "Zero reporting delay is an assumption; the two-block delay activity is illustrative",
    "Foundation-model pretraining overlap with this series is unknown",
    "Two evaluation years (26 origins) limit every comparison; bootstrap intervals are descriptive",
    "Reported counts, not population-adjusted incidence; not an operational warning system",
]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8", newline="\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def output(root: Path) -> Path:
    return root / "results"


def rows_at(root: Path):
    rows = read(root / "data.json")
    core.validate_rows(rows)
    return rows


def identity(root: Path) -> str:
    names = ("data.json", "dataset_audit.json", "model_manifest.json", "source.json")
    identities = {n: sha(root / n) for n in names}
    source = read(root / "source.json")
    for name, checksum in source.get("files", {}).items():
        if Path(name).name != name or sha(root / name) != checksum:
            raise ValueError("Embedded source changed: " + name)
    for name in ("run_config.json", "requirements.txt"):
        if (root / name).exists():
            identities[name] = sha(root / name)
    audit = read(root / "dataset_audit.json")
    if audit.get("derived_data_sha256") and audit["derived_data_sha256"] != sha(root / "data.json"):
        raise ValueError("Derived data differs from audited source")
    for module in (core, data, models):
        identities[Path(module.__file__).name] = sha(Path(module.__file__))
    identities["runtime"] = sha(Path(__file__))
    return digest(identities)


def table(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def selected_window(root: Path) -> int:
    lock = read(output(root) / "experiment_lock.json")
    if lock["identity"] != identity(root):
        raise ValueError("Experiment source changed after lock")
    return lock["window"]


def text_field(value, label: str, limit: int = 160) -> str:
    """Plain one-line provenance text; never used as a predictor."""
    text = str(value or "").strip()
    if not text or len(text) > limit or not text.isprintable():
        raise ValueError(f"{label} must be one printable line of 1-{limit} characters")
    return text


def byod_rows(path: Path) -> list[dict]:
    """Parse the aggregate BYOD CSV, naming the file, line and column of each refusal."""
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header != BYOD_FIELDS:
            raise ValueError(
                f"{path.name}: header must be exactly year,block,cases,rain,temp (found {header}); "
                "reject extra personal fields and any other extra field"
            )
        for line, values in enumerate(reader, start=2):
            where = f"{path.name} line {line}"
            if len(values) != len(BYOD_FIELDS):
                raise ValueError(
                    f"{where}: expected 5 values, found {len(values)}; "
                    "extra or missing values are refused"
                )
            record = {}
            for field, raw in zip(BYOD_FIELDS, values, strict=True):
                text = raw.strip()
                if not text:
                    raise ValueError(
                        f"{where}, column {field}: missing value; missing is never replaced by zero"
                    )
                if field in ("year", "block"):
                    try:
                        record[field] = int(text)
                    except ValueError as exc:
                        raise ValueError(f"{where}, column {field}: must be an integer") from exc
                    continue
                try:
                    number = float(text)
                except ValueError as exc:
                    raise ValueError(f"{where}, column {field}: not a number") from exc
                problem = core.value_problem(field, number)
                if problem:
                    raise ValueError(f"{where}, column {field}: {problem}")
                record[field] = int(number) if field == "cases" else number
            rows.append(record)
    return rows


def planned_contexts(rows, plan):
    """Every support task the model stages can request: partitions, delays, windows, weather."""
    tasks = [("validation", o, delay) for o in plan["validation"] for delay in (0, 2)]
    tasks += [("test", o, 0) for o in plan["test"]]
    tasks += [("future", len(rows) - 1, 0)]
    for partition, origin, delay in tasks:
        for h in range(1, 5):
            for window in plan["windows"]:
                for weather in (False, True):
                    yield partition, origin, h, window, delay, weather


def context_preflight(root, rows, plan) -> None:
    """Refuse an experiment whose mandatory Mitra contexts cannot be fitted, before any model.

    Policy: no origin is dropped, no variance is invented and no fallback is labelled Mitra.
    """
    checked, problems = 0, []
    for partition, origin, h, window, delay, weather in planned_contexts(rows, plan):
        task = core.training_task(rows, origin, h, window=window, delay=delay, weather=weather)
        checked += 1
        problem = models.support_problem(task["X"], task["y"])
        if problem:
            problems.append(
                {
                    "partition": partition,
                    "origin_key": key(rows, origin),
                    "horizon": h,
                    "window": window,
                    "delay": delay,
                    "system": "mitra_weather" if weather else "mitra_cases",
                    "problem": problem,
                }
            )
    write(
        output(root) / "context_preflight.json",
        {
            "policy": "reject before model staging; no dropped origins, invented variance "
            "or relabelled fallback",
            "contexts_checked": checked,
            "contexts_refused": len(problems),
            "refused": problems[:50],
        },
    )
    if problems:
        first = problems[0]
        raise ValueError(
            f"{len(problems)} of {checked} planned Mitra contexts cannot be fitted "
            f"(first: {first['partition']} origin {first['origin_key']}, horizon "
            f"{first['horizon']}, window {first['window']}, delay {first['delay']}: "
            f"{first['problem']}). Both Mitra systems are mandatory, so the experiment stops "
            "before any model is downloaded; see results/context_preflight.json."
        )


def feature_example(rows, plan) -> list[dict]:
    """Trace one weather-feature row and its latest mature label to source blocks."""
    origin, h = plan["test"][0], 1
    values = core.feature_row(rows, origin, h, weather=True)
    trace = core.feature_sources(rows, origin, h, weather=True)

    def span(indices):
        if not indices:
            return "target block calendar position (known in advance)"
        first, last = key(rows, indices[0]), key(rows, indices[-1])
        return first if first == last else f"{first} .. {last}"

    task = core.training_task(rows, origin, h, weather=True)
    label = int(max(task["target_indices"]))
    result = [
        {
            "item": "forecast origin (issued after this block)",
            "value": "",
            "source_blocks": key(rows, origin),
        },
        {
            "item": "target (not observed at issuance)",
            "value": "",
            "source_blocks": key(rows, origin + h),
        },
    ]
    result += [
        {"item": name, "value": round(float(v), 4), "source_blocks": span(indices)}
        for (name, indices), v in zip(trace, values, strict=True)
    ]
    result.append(
        {
            "item": "latest mature training label (cases)",
            "value": rows[label]["cases"],
            "source_blocks": key(rows, label),
        }
    )
    return result


def fit_ridge(X, y, query):
    """Training-only standardisation; fixed L2=1 and unpenalised intercept."""
    X, y, query = np.asarray(X), np.asarray(y), np.asarray(query)
    mean, scale = X.mean(0), X.std(0)
    scale[scale < 1e-12] = 1
    design = np.column_stack([np.ones(len(X)), (X - mean) / scale])
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    state = {"mean": mean.tolist(), "scale": scale.tolist(), "coef": coef.tolist()}
    return ridge_predict(state, query), state


def ridge_predict(state, query):
    q = np.atleast_2d(query)
    design = np.column_stack([np.ones(len(q)), (q - state["mean"]) / state["scale"]])
    return design @ np.asarray(state["coef"])


def point(logvalue: float) -> tuple[float, float]:
    raw = float(np.expm1(logvalue))
    if not np.isfinite(raw):
        raise ValueError("Non-finite inverse transformed forecast")
    return raw, max(0.0, raw)


def key(rows, index):
    # Preserve source block order. These are not reconstructed observation dates.
    year = rows[0]["year"] + index // 52
    return f"{year}-B{index % 52 + 1:02d}"


def prediction(rows, system, origin, h, delay, raw, q=None):
    if not np.isfinite(raw):
        raise ValueError("Non-finite prediction")
    item = dict(
        system=system,
        origin=origin,
        origin_key=key(rows, origin),
        cutoff=origin - delay,
        target_index=origin + h,
        target_key=key(rows, origin + h),
        horizon=h,
        delay=delay,
        reference=rows[origin + h]["cases"] if origin + h < len(rows) else None,
        raw_prediction=float(raw),
        prediction=max(0.0, float(raw)),
        clipped=bool(raw < 0),
        q10=None,
        q50=None,
        q90=None,
    )
    if q is not None:
        if not np.isfinite(q).all() or np.any(np.diff(q) < 0):
            raise ValueError("Invalid/crossing model quantiles")
        item.update(q10=float(q[0]), q50=float(q[1]), q90=float(q[2]))
    return item


def simple(rows, origins, window=104, delay=0, save=False):
    predictions, states = [], {}
    for origin in origins:
        for h in range(1, 5):
            predictions.append(
                prediction(rows, "persistence", origin, h, delay, rows[origin - delay]["cases"])
            )
            seasonal_index = origin + h - 52
            if seasonal_index > origin - delay:
                raise ValueError("Seasonal baseline accesses unavailable value")
            predictions.append(
                prediction(rows, "seasonal", origin, h, delay, rows[seasonal_index]["cases"])
            )
            task = core.training_task(rows, origin, h, window=window, delay=delay)
            estimate, state = fit_ridge(task["X"], task["y"], task["query"])
            raw, _ = point(float(estimate[0]))
            predictions.append(prediction(rows, "ridge", origin, h, delay, raw))
            if save:
                states[str(h)] = {"state": state, "query": np.asarray(task["query"]).tolist()}
    return predictions, states


def free_gpu() -> None:
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def model_forecasts(root, rows, origins, window, system, delay=0, save=False):
    manifest = read(root / "model_manifest.json")
    cache = root / "models"
    state = {"system": system, "window": window, "delay": delay, "origins": {}}
    if system == "chronos":
        model = models.load_chronos(cache, manifest)
    else:
        model = models.load_mitra(cache, manifest)
    predictions = []
    try:
        for number, origin in enumerate(origins):
            print(f"{system}: origin {number + 1}/{len(origins)} {key(rows, origin)}", flush=True)
            if system == "chronos":
                history = [
                    r["cases"] for r in rows[max(0, origin - delay - 103) : origin - delay + 1]
                ]
                quantiles = np.asarray(models.chronos_predict(model, history, 4 + delay))
                if quantiles.shape != (4 + delay, 3):
                    raise ValueError("Unexpected Chronos forecast shape")
                for h in range(1, 5):
                    q = quantiles[h + delay - 1]
                    predictions.append(prediction(rows, system, origin, h, delay, q[1], q))
                if save:
                    state["origins"][str(origin)] = {"history": history}
            else:
                saved = {}
                for h in range(1, 5):
                    task = core.training_task(
                        rows,
                        origin,
                        h,
                        window=window,
                        delay=delay,
                        weather=system == "mitra_weather",
                    )
                    estimate = np.asarray(
                        models.mitra_predict(
                            model, task["X"], task["y"], np.atleast_2d(task["query"])
                        )
                    ).reshape(-1)
                    if len(estimate) != 1:
                        raise ValueError("Expected exactly one direct forecast")
                    raw, _ = point(float(estimate[0]))
                    predictions.append(prediction(rows, system, origin, h, delay, raw))
                    if save:
                        saved[str(h)] = {
                            k: np.asarray(task[k]).tolist() for k in ("X", "y", "query")
                        }
                        saved[str(h)]["issuance_indices"] = np.asarray(
                            task["issuance_indices"]
                        ).tolist()
                        saved[str(h)]["feature_names"] = task["feature_names"]
                if save:
                    state["origins"][str(origin)] = saved
    finally:
        del model
        free_gpu()
    return predictions, state


def scores(predictions):
    result = {}
    for system in sorted({p["system"] for p in predictions}):
        selected = [p for p in predictions if p["system"] == system]
        horizons = {}
        for h in range(1, 5):
            subset = [p for p in selected if p["horizon"] == h]
            horizons[str(h)] = core.metrics(
                [p["reference"] for p in subset], [p["prediction"] for p in subset]
            )
        result[system] = {
            "origins": len(selected) // 4,
            "pairs": len(selected),
            "mae": float(np.mean([v["mae"] for v in horizons.values()])),
            "horizons": horizons,
        }
    if "seasonal" in result:
        for value in result.values():
            value["seasonal_skill"] = core.skill(value["mae"], result["seasonal"]["mae"])
    return result


def figures(root, stage):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = output(root)
    if stage == "prepare":
        rows = rows_at(root)
        plan = read(out / "plan.json")
        spans = [(plan["validation_years"][0], "orange"), (plan["test_years"][0], "green")]
        if any(r.get("covid") for r in rows):
            spans.insert(0, (2020, "gray"))
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        for ax, name in zip(axes, ("cases", "rain", "temp"), strict=True):
            ax.plot([r[name] for r in rows], linewidth=0.8)
            ax.set_ylabel(
                {"cases": "Reported cases", "rain": "Rain (mm/block)", "temp": "Temperature (°C)"}[
                    name
                ]
            )
            for year, color in spans:
                start = (year - rows[0]["year"]) * 52
                if start >= 0:
                    ax.axvspan(start, start + 104, alpha=0.12, color=color)
        axes[-1].set_xlabel("Published source-block index (not verified dates)")
        fig.suptitle(
            f"{plan['area']} · published alignment · orange validation / green test; "
            "gray if COVID flagged"
        )
    else:
        name = {
            "baselines": "baseline_metrics.json",
            "validation": "validation_metrics.json",
            "test": "metrics.json",
            "activity": "delay_metrics.json",
        }[stage]
        values = read(out / name)
        fig, ax = plt.subplots(figsize=(10, 5))
        for system, value in values.items():
            ax.plot(
                range(1, 5),
                [value["horizons"][str(h)]["mae"] for h in range(1, 5)],
                marker="o",
                label=system,
            )
        ax.set(
            xlabel="Forecast horizon (source blocks)",
            ylabel="MAE (cases)",
            xticks=range(1, 5),
            title=stage + " · lower MAE is better",
        )
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"{stage}.png", dpi=130)
    plt.close(fig)
    if stage == "test":
        pred = read(out / "test_predictions.json")
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        for system in ("chronos", "mitra_cases", "mitra_weather"):
            subset = sorted(
                [p for p in pred if p["system"] == system and p["horizon"] == 4],
                key=lambda p: p["target_index"],
            )
            x = [p["target_index"] for p in subset]
            axes[0].plot(x, [p["prediction"] for p in subset], label=system)
            if system == "chronos":
                axes[0].plot(
                    x, [p["reference"] for p in subset], color="black", label="Published counts"
                )
                axes[0].fill_between(
                    x,
                    [p["q10"] for p in subset],
                    [p["q90"] for p in subset],
                    alpha=0.15,
                    label="Chronos nominal 80% interval",
                )
            else:
                axes[1].plot(x, [p["prediction"] - p["reference"] for p in subset], label=system)
        area = read(out / "plan.json")["area"]
        axes[0].set(
            ylabel="Cases",
            title=f"{area} · four-block forecasts · same targets; raw Chronos interval",
        )
        axes[1].axhline(0, color="black", linewidth=0.7)
        axes[1].set(xlabel="Published source-block index", ylabel="Prediction − reference")
        for ax in axes:
            ax.legend()
        fig.tight_layout()
        fig.savefig(out / "test_forecasts.png", dpi=130)
        plt.close(fig)


def largest_misses(predictions) -> list[dict]:
    """Learner view: each miss with its published count, forecast and signed error."""
    misses = sorted(predictions, key=lambda p: abs(p["prediction"] - p["reference"]), reverse=True)[
        :20
    ]
    return [
        {
            "system": p["system"],
            "origin_key": p["origin_key"],
            "target_key": p["target_key"],
            "horizon": p["horizon"],
            "reference": p["reference"],
            "prediction": round(p["prediction"], 3),
            "error": round(p["prediction"] - p["reference"], 3),
            "abs_error": round(abs(p["prediction"] - p["reference"]), 3),
            "raw_prediction": round(p["raw_prediction"], 3),
            "clipped": p["clipped"],
        }
        for p in misses
    ]


def prepare(root):
    # BYOD is selected before this stage and never opens an upload dialog by default.
    config = read(root / "run_config.json") if (root / "run_config.json").exists() else {}
    if config.get("byod_csv"):
        if not config.get("source_blocks_confirmed"):
            raise ValueError("BYOD requires aggregate schema, units and source-block confirmation")
        area = text_field(config.get("area"), "BYOD_AREA", 120)
        citation = text_field(config.get("source_citation"), "BYOD_SOURCE_CITATION")
        rows = byod_rows(Path(config["byod_csv"]))
        core.validate_rows(rows)
        if len(rows) < 6 * 52:
            raise ValueError(
                "Need six complete source years: two history, two validation, two test"
            )
        write(root / "data.json", rows)
        write(
            root / "dataset_audit.json",
            {
                "scope": SCOPE,
                "source": "BYOD",
                "area": area,
                "source_citation": citation,
                "units": core.UNITS,
                "source_sha256": sha(Path(config["byod_csv"])),
                "rows": len(rows),
                "timing_verified": False,
                "weather_alignment_verified": False,
            },
        )
        (root / "DATA_LICENSE.md").write_text(
            "BYOD: retain your source rights and provenance.\n", encoding="utf-8"
        )
    else:
        rows = data.acquire(root)
        area, citation = DEFAULT_AREA, DEFAULT_SOURCE
    core.validate_rows(rows)
    years = sorted({r["year"] for r in rows})
    val_years, test_years = years[-4:-2], years[-2:]
    plan = {
        "scope": SCOPE,
        "validation_years": val_years,
        "test_years": test_years,
        "validation": core.origins(rows, val_years),
        "test": core.origins(rows, test_years),
        "default_delay": 0,
        "windows": [52, 104],
        "chronos_window": 104,
        "high_case_threshold": float(
            np.quantile([r["cases"] for r in rows if r["year"] < val_years[0]], 0.9)
        ),
        "calendar_verified": False,
        "alignment_verified": False,
        "area": area,
        "source_citation": citation,
        "units": core.UNITS,
    }
    if len(plan["test"]) != 26 or len(plan["validation"]) != 26:
        raise ValueError("Expected 26 origins per partition; no silent reduction")
    write(output(root) / "plan.json", plan)
    context_preflight(root, rows, plan)
    table(output(root) / "feature_example.csv", feature_example(rows, plan))
    table(
        output(root) / "origin_manifest.csv",
        [
            {
                "partition": part,
                "origin": o,
                "origin_key": key(rows, o),
                "horizon": h,
                "target_key": key(rows, o + h),
                "cutoff": o,
            }
            for part in ("validation", "test")
            for o in plan[part]
            for h in range(1, 5)
        ],
    )
    write(
        output(root) / "feature_schema.json",
        {
            name: core.feature_names(weather=weather)
            for name, weather in [("cases", False), ("weather", True)]
        },
    )
    audit_rows = []
    for partition in ("validation", "test"):
        for origin in plan[partition]:
            for h in range(1, 5):
                task = core.training_task(rows, origin, h)
                audit_rows.append(
                    {
                        "partition": partition,
                        "origin": origin,
                        "horizon": h,
                        "cutoff": origin,
                        "max_training_target": int(max(task["target_indices"])),
                        "max_training_input": int(max(task["issuance_indices"])),
                        "query_latest_input": origin,
                        "future_input_used": False,
                    }
                )
    table(output(root) / "availability_audit.csv", audit_rows)


def baselines(root):
    rows, plan = rows_at(root), read(output(root) / "plan.json")
    predictions, _ = simple(rows, plan["validation"])
    write(output(root) / "baseline_validation.json", predictions)
    write(output(root) / "baseline_metrics.json", scores(predictions))


def validation(root):
    rows, plan = rows_at(root), read(output(root) / "plan.json")
    candidates = {}
    for window in (52, 104):
        pred, _ = model_forecasts(root, rows, plan["validation"], window, "mitra_cases")
        write(output(root) / f"validation_cases_{window}.json", pred)
        candidates[window] = scores(pred)["mitra_cases"]["mae"]
    chosen = min(candidates, key=lambda w: (candidates[w], w))
    all_predictions = read(output(root) / "baseline_validation.json")
    all_predictions += read(output(root) / f"validation_cases_{chosen}.json")
    for system in ("mitra_weather", "chronos"):
        pred, _ = model_forecasts(root, rows, plan["validation"], chosen, system)
        all_predictions.extend(pred)
    write(output(root) / "selection.json", {"window": chosen, "candidate_mae": candidates})
    write(output(root) / "validation_predictions.json", all_predictions)
    write(output(root) / "validation_metrics.json", scores(all_predictions))


def lock(root):
    write(
        output(root) / "experiment_lock.json",
        {
            "identity": identity(root),
            "window": read(output(root) / "selection.json")["window"],
            "plan_sha256": sha(output(root) / "plan.json"),
            "systems": SYSTEMS,
            "target_transform": "log1p for ridge/mitra; expm1 then nonnegative floor",
            "weather_comparison": "mitra_weather minus mitra_cases",
            "seed": 42,
            "retrospective_delay_assumption": 0,
            "scope": SCOPE,
        },
    )


def test(root):
    rows, plan, window = rows_at(root), read(output(root) / "plan.json"), selected_window(root)
    predictions, _ = simple(rows, plan["test"])
    for system in ("chronos", "mitra_cases", "mitra_weather"):
        pred, state = model_forecasts(root, rows, plan["test"], window, system, save=True)
        predictions.extend(pred)
        # Store just the final completed origin for an independent reload check.
        final = str(plan["test"][-1])
        state["origins"] = {final: state["origins"][final]}
        write(output(root) / f"artifact_test_{system}.json", state)
    last_simple, state = simple(rows, [plan["test"][-1]], save=True)
    write(output(root) / "artifact_test_ridge.json", state)
    write(
        output(root) / "reload_expected_test.json",
        [p for p in predictions if p["origin"] == plan["test"][-1]],
    )
    del last_simple
    assert_cohort(predictions, plan["test"], rows)
    write(output(root) / "test_predictions.json", predictions)
    table(output(root) / "predictions.csv", predictions)
    table(output(root) / "largest_misses.csv", largest_misses(predictions))
    write(output(root) / "metrics.json", scores(predictions))

    def ordered(system):
        return sorted(
            [p for p in predictions if p["system"] == system],
            key=lambda p: (p["origin"], p["horizon"]),
        )

    cases, weather = ordered("mitra_cases"), ordered("mitra_weather")
    ref = np.array([p["reference"] for p in cases]).reshape(-1, 4)
    a = np.array([p["prediction"] for p in cases]).reshape(-1, 4)
    b = np.array([p["prediction"] for p in weather]).reshape(-1, 4)
    write(output(root) / "paired_comparison.json", core.paired_bootstrap(ref, a, b))
    high = {}
    for system in SYSTEMS:
        p = [p for p in ordered(system) if p["reference"] >= plan["high_case_threshold"]]
        high[system] = {
            "n": len(p),
            "threshold": plan["high_case_threshold"],
            "metrics": core.metrics([x["reference"] for x in p], [x["prediction"] for x in p])
            if p
            else None,
            "underprediction_rate": float(np.mean([x["prediction"] < x["reference"] for x in p]))
            if p
            else None,
        }
    write(output(root) / "high_case_metrics.json", high)
    intervals = {}
    for h in range(1, 5):
        p = [p for p in ordered("chronos") if p["horizon"] == h]
        truth = np.array([x["reference"] for x in p])
        q = np.array([[x["q10"], x["q50"], x["q90"]] for x in p])
        error = truth[:, None] - q
        intervals[str(h)] = {
            "n": len(p),
            "raw_coverage_80": float(np.mean((truth >= q[:, 0]) & (truth <= q[:, 2]))),
            "raw_mean_width": float(np.mean(q[:, 2] - q[:, 0])),
            "pinball": np.maximum(
                error * np.array([0.1, 0.5, 0.9]), error * (np.array([0.1, 0.5, 0.9]) - 1)
            )
            .mean(0)
            .tolist(),
        }
    write(output(root) / "interval_metrics.json", intervals)


def assert_cohort(predictions, origins, rows=None):
    expected = {(o, h) for o in origins for h in range(1, 5)}
    matched = {}
    if {p["system"] for p in predictions} != set(SYSTEMS):
        raise ValueError("Unexpected or missing systems")
    for system in SYSTEMS:
        subset = [p for p in predictions if p["system"] == system]
        if (
            len(subset) != len(expected)
            or {(p["origin"], p["horizon"]) for p in subset} != expected
        ):
            raise ValueError(f"Incomplete or duplicated cohort: {system}")
        for p in subset:
            pair = (p["origin"], p["horizon"])
            signature = tuple(
                p[k]
                for k in (
                    "origin_key",
                    "cutoff",
                    "target_index",
                    "target_key",
                    "delay",
                    "reference",
                )
            )
            if pair in matched and matched[pair] != signature:
                raise ValueError("Paired target/cutoff/reference disagreement")
            matched[pair] = signature
            if (
                p["target_index"] != p["origin"] + p["horizon"]
                or p["cutoff"] != p["origin"] - p["delay"]
            ):
                raise ValueError("Invalid forecast timing fields")
            if rows is not None and (
                p["target_key"] != key(rows, p["target_index"])
                or p["reference"] != rows[p["target_index"]]["cases"]
            ):
                raise ValueError("Forecast reference does not match source")


def activity(root):
    rows, plan, window = rows_at(root), read(output(root) / "plan.json"), selected_window(root)
    predictions, _ = simple(rows, plan["validation"], delay=2)
    for system in ("chronos", "mitra_cases", "mitra_weather"):
        p, _ = model_forecasts(root, rows, plan["validation"], window, system, delay=2)
        predictions.extend(p)
    assert_cohort(predictions, plan["validation"], rows)
    table(output(root) / "validation_delay_activity.csv", predictions)
    delayed = scores(predictions)
    write(output(root) / "delay_metrics.json", delayed)
    baseline = {
        (p["system"], p["origin"], p["horizon"]): p
        for p in read(output(root) / "validation_predictions.json")
    }
    paired = []
    order = sorted(
        predictions, key=lambda p: (SYSTEMS.index(p["system"]), p["origin"], p["horizon"])
    )
    for p in order:
        zero = baseline[(p["system"], p["origin"], p["horizon"])]
        if zero["target_index"] != p["target_index"] or zero["reference"] != p["reference"]:
            raise ValueError("Delay activity is not paired with zero-delay validation targets")
        paired.append(
            {
                "system": p["system"],
                "origin_key": p["origin_key"],
                "target_key": p["target_key"],
                "horizon": p["horizon"],
                "reference": p["reference"],
                "prediction_delay_0": round(zero["prediction"], 3),
                "prediction_delay_2": round(p["prediction"], 3),
                "abs_error_delay_0": round(abs(zero["prediction"] - zero["reference"]), 3),
                "abs_error_delay_2": round(abs(p["prediction"] - p["reference"]), 3),
                "abs_error_change": round(
                    abs(p["prediction"] - p["reference"])
                    - abs(zero["prediction"] - zero["reference"]),
                    3,
                ),
            }
        )
    table(output(root) / "delay_comparison.csv", paired)
    zero_scores = read(output(root) / "validation_metrics.json")
    table(
        output(root) / "delay_summary.csv",
        [
            {
                "system": system,
                "pairs": delayed[system]["pairs"],
                "mae_delay_0": round(zero_scores[system]["mae"], 3),
                "mae_delay_2": round(delayed[system]["mae"], 3),
                "change": round(delayed[system]["mae"] - zero_scores[system]["mae"], 3),
            }
            for system in SYSTEMS
        ],
    )


def future(root):
    rows, window = rows_at(root), selected_window(root)
    origin = len(rows) - 1
    predictions, ridge = simple(rows, [origin], save=True)
    write(output(root) / "artifact_future_ridge.json", ridge)
    for system in ("chronos", "mitra_cases", "mitra_weather"):
        p, state = model_forecasts(root, rows, [origin], window, system, save=True)
        predictions.extend(p)
        write(output(root) / f"artifact_future_{system}.json", state)
    write(output(root) / "future_predictions.json", predictions)
    table(output(root) / "future_predictions.csv", predictions)
    by_target = {}
    for p in predictions:
        by_target.setdefault((p["horizon"], p["target_key"]), {})[p["system"]] = round(
            p["prediction"], 2
        )
    table(
        output(root) / "future_forecast_view.csv",
        [
            {"target_key": target, "horizon": h, **{s: values[s] for s in SYSTEMS}}
            for (h, target), values in sorted(by_target.items())
        ],
    )
    artifacts = {p.name: sha(p) for p in output(root).glob("artifact_*.json")}
    write(
        output(root) / "artifact_manifest.json",
        {
            "format": "dimer_dengue_forecast_contexts",
            "format_version": 1,
            "identity": identity(root),
            "files": artifacts,
            "creator_pid": os.getpid(),
            "feature_schema_sha256": sha(output(root) / "feature_schema.json"),
            "experiment_lock_sha256": sha(output(root) / "experiment_lock.json"),
            "target_transform": "log1p for Mitra/Ridge; inverse expm1 then floor at zero",
            "chronos_target": "raw case counts; original quantiles retained",
            "worker_adapter_compatibility": "not claimed",
            "bundle_identity": bundle_identity(
                {n: root / n for n in ("model_manifest.json", "source.json", "dataset_audit.json")},
                output(root),
                sha(root / "data.json"),
            ),
            "retained_data": "Aggregate numeric support contexts only: lagged case counts "
            "(plus lagged rain and temperature for mitra_weather) for the final completed test "
            "origin and the future origin, and fitted Ridge state. No individual records.",
        },
    )


def reload(root):
    manifest, out = read(output(root) / "artifact_manifest.json"), output(root)
    if (
        manifest.get("format") != "dimer_dengue_forecast_contexts"
        or manifest.get("format_version") != 1
    ):
        raise ValueError("Unsupported forecast context artifact format")
    for field, name in (
        ("feature_schema_sha256", "feature_schema.json"),
        ("experiment_lock_sha256", "experiment_lock.json"),
    ):
        if manifest[field] != sha(out / name):
            raise ValueError("Artifact feature schema or experiment lock changed")
    if manifest["identity"] != identity(root) or manifest["creator_pid"] == os.getpid():
        raise ValueError("Reload requires unchanged source and a fresh process")
    for name, checksum in manifest["files"].items():
        if Path(name).name != name or sha(out / name) != checksum:
            raise ValueError("Artifact hash mismatch")
    checked, maximum = reproduce(out, root / "models", read(root / "model_manifest.json"))
    write(
        out / "verification.json",
        {
            "passed": True,
            "predictions_checked": checked,
            "max_absolute_difference": maximum,
            "atol": 1e-3,
            "rtol": 1e-4,
            "fresh_process": True,
            "pid": os.getpid(),
            "scope": "same workspace; see consumer_verification.json for the exported bundle",
        },
    )


def reproduce(out: Path, cache: Path, model_manifest: dict) -> tuple[int, float]:
    """Rebuild every saved context and compare with its recorded prediction."""
    maximum = 0.0
    checked = 0
    for system in ("ridge", "chronos", "mitra_cases", "mitra_weather"):
        model = None
        if system == "chronos":
            model = models.load_chronos(cache, model_manifest)
        elif system.startswith("mitra"):
            model = models.load_mitra(cache, model_manifest)
        try:
            for label, expected_name in [
                ("test", "reload_expected_test.json"),
                ("future", "future_predictions.json"),
            ]:
                artifact = read(out / f"artifact_{label}_{system}.json")
                expected = sorted(
                    [p for p in read(out / expected_name) if p["system"] == system],
                    key=lambda p: p["horizon"],
                )
                if len(expected) != 4:
                    raise ValueError("Missing reload reference")
                if system == "ridge":
                    raw = [
                        point(
                            float(
                                ridge_predict(artifact[str(h)]["state"], artifact[str(h)]["query"])[
                                    0
                                ]
                            )
                        )[0]
                        for h in range(1, 5)
                    ]
                elif system == "chronos":
                    saved = artifact["origins"][str(expected[0]["origin"])]
                    q = np.asarray(models.chronos_predict(model, saved["history"], 4))
                    np.testing.assert_allclose(
                        q, [[p["q10"], p["q50"], p["q90"]] for p in expected], atol=1e-3, rtol=1e-4
                    )
                    raw = q[:, 1]
                else:
                    saved = artifact["origins"][str(expected[0]["origin"])]
                    raw = []
                    for h in range(1, 5):
                        s = saved[str(h)]
                        estimate = models.mitra_predict(
                            model, np.asarray(s["X"]), np.asarray(s["y"]), np.atleast_2d(s["query"])
                        )
                        raw.append(point(float(np.asarray(estimate).reshape(-1)[0]))[0])
                reference = np.array([p["raw_prediction"] for p in expected])
                if not np.allclose(raw, reference, atol=1e-3, rtol=1e-4):
                    raise ValueError(f"Saved {label} {system} context does not reproduce")
                maximum = max(maximum, float(np.max(np.abs(np.asarray(raw) - reference))))
                checked += len(expected)
        finally:
            del model
            free_gpu()
    return checked, maximum


def bundle_identity(provenance: dict, out: Path, data_sha256: str) -> str:
    """Identity a bundle consumer can recompute: no original workspace or run path needed."""
    identities = {name: sha(path) for name, path in provenance.items()}
    for name in ("feature_schema.json", "experiment_lock.json", "plan.json"):
        identities[name] = sha(out / name)
    identities["data_sha256"] = data_sha256
    for module in (core, data, models):
        identities[Path(module.__file__).name] = sha(Path(module.__file__))
    identities["runtime"] = sha(Path(__file__))
    return digest(identities)


def consume(bundle: Path, workdir: Path, cache: Path, code: Path | None = None) -> dict:
    """Reconstruct the exported contexts from results.zip alone, plus trusted code and weights.

    ``code`` holds the notebook's embedded files (source.json, model_manifest.json and the
    modules). The original run directory, its data.json and run_config.json are never read.
    """
    code = Path(code or Path(__file__).resolve().parent)
    target = Path(workdir) / "bundle"
    target.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(bundle) as archive:
        members = archive.infolist()
        names = [m.filename for m in members]
        if len(set(names)) != len(names) or any(
            Path(n).name != n or n in ("", ".", "..") or m.file_size > 100 * 1024**2
            for n, m in zip(names, members, strict=True)
        ):
            raise ValueError("Bundle members must be unique, flat and bounded")
        listed = json.loads(archive.read("checksums.json"))
        if set(names) != {*listed, "checksums.json"}:
            raise ValueError("Bundle members differ from checksums.json")
        for name in names:
            (target / name).write_bytes(archive.read(name))
    for name, checksum in listed.items():
        if sha(target / name) != checksum:
            raise ValueError(f"Bundle member changed: {name}")
    for name in ("source.json", "model_manifest.json"):
        if (target / name).read_bytes() != (code / name).read_bytes():
            raise ValueError(f"Bundle {name} differs from the trusted embedded copy")
    for name, checksum in read(code / "source.json").get("files", {}).items():
        if Path(name).name != name or sha(code / name) != checksum:
            raise ValueError("Trusted embedded source changed: " + name)
    manifest = read(target / "artifact_manifest.json")
    if (
        manifest.get("format") != "dimer_dengue_forecast_contexts"
        or manifest.get("format_version") != 1
    ):
        raise ValueError("Unsupported forecast context artifact format")
    for field, name in (
        ("feature_schema_sha256", "feature_schema.json"),
        ("experiment_lock_sha256", "experiment_lock.json"),
    ):
        if manifest[field] != sha(target / name):
            raise ValueError("Artifact feature schema or experiment lock changed")
    for name, checksum in manifest["files"].items():
        if Path(name).name != name or sha(target / name) != checksum:
            raise ValueError("Artifact hash mismatch")
    expected_identity = bundle_identity(
        {n: target / n for n in ("model_manifest.json", "source.json", "dataset_audit.json")},
        target,
        read(target / "data_manifest.json")["data_sha256"],
    )
    if manifest.get("bundle_identity") != expected_identity:
        raise ValueError("Bundle provenance, configuration or code identity changed")
    checked, maximum = reproduce(target, Path(cache), read(code / "model_manifest.json"))
    result = {
        "passed": True,
        "bundle_sha256": sha(Path(bundle)),
        "predictions_checked": checked,
        "max_absolute_difference": maximum,
        "atol": 1e-3,
        "rtol": 1e-4,
        "original_workspace_read": False,
        "inputs": "results.zip, trusted embedded code/manifests and verified model snapshots",
    }
    write(Path(workdir) / "consumer_verification.json", result)
    return result


def report(root):
    out = output(root)
    predictions = read(out / "test_predictions.json")
    assert_cohort(predictions, read(out / "plan.json")["test"], rows_at(root))
    if digest(scores(predictions)) != digest(read(out / "metrics.json")):
        raise ValueError("Metrics do not reproduce from predictions")
    with (out / "predictions.csv").open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    expected_csv = [{k: "" if v is None else str(v) for k, v in p.items()} for p in predictions]
    if csv_rows != expected_csv:
        raise ValueError("CSV prediction mismatch")
    if not read(out / "verification.json")["passed"]:
        raise ValueError("Reload not verified")
    environment = runtime_environment()
    write(out / "environment.json", environment)
    for name in ("dataset_audit.json", "model_manifest.json", "source.json", "DATA_LICENSE.md"):
        (out / name).write_bytes((root / name).read_bytes())
    write(
        out / "data_manifest.json",
        {
            "data_sha256": sha(root / "data.json"),
            "rows": len(rows_at(root)),
            "source_audit_sha256": sha(root / "dataset_audit.json"),
        },
    )
    receipts = {s: read(out / f"receipt_{s}.json") for s in STAGES[:-1]}
    plan = read(out / "plan.json")
    audit = read(root / "dataset_audit.json")
    write(
        out / "run_summary.json",
        {
            "scope": SCOPE,
            "status": "executed_candidate",
            "area": plan["area"],
            "source_citation": plan["source_citation"],
            "units": plan["units"],
            "timing_verified": False,
            "weather_alignment_verified": False,
            "prospective_validation": False,
            "limitations": LIMITATIONS + list(audit.get("limitations", [])),
            "environment": environment,
            "resource_targets": RESOURCE_TARGETS,
            "resource_measurements": {
                "stage_seconds": {s: r.get("seconds") for s, r in receipts.items()},
                "peak_allocated_gpu_bytes": max(
                    [r.get("peak_allocated_gpu_bytes") or 0 for r in receipts.values()] or [0]
                ),
            },
            "source_audit": "dataset_audit.json",
            "metrics": read(out / "metrics.json"),
            "receipts": receipts,
            "reload": read(out / "verification.json"),
        },
    )
    (out / "conclusion_template.md").write_text(
        "# Evidence-based conclusion\n\nCompare M2 minus M1 by horizon and paired interval. "
        "Quote measured MAE and high-case errors. Explain whether results support adding weather "
        "under this published-block alignment. Describe the calendar conflict, "
        "final-data vintages, unknown pretraining overlap and two-year evaluation limit. "
        "No operational warning claim.\n\n"
        "This reflection is optional and is not a required submission.\n",
        encoding="utf-8",
    )
    files = {
        p.name: sha(p)
        for p in out.iterdir()
        if p.is_file()
        and p.suffix in (".json", ".csv", ".md", ".png")
        and p.name not in ("checksums.json", "receipt_report.json")
    }
    write(out / "checksums.json", files)
    with zipfile.ZipFile(out / "results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted([*files, "checksums.json"]):
            archive.write(out / name, name)
    with zipfile.ZipFile(out / "results.zip") as archive:
        if archive.testzip():
            raise ValueError("Export ZIP CRC failure")
    if (out / "results.zip").stat().st_size > 100 * 1024**2:
        raise ValueError("Export exceeds 100 MiB bound")


def runtime_environment() -> dict:
    """Python, device, precision and the package versions that shape preprocessing/inference."""
    import importlib.metadata

    packages = {}
    for name in (
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "chronos-forecasting",
        "transformers",
        "autogluon.tabular",
        "safetensors",
        "matplotlib",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    device = "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    except (AssertionError, RuntimeError) as exc:
        device = f"unavailable: {exc}"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "precision": "float32 (Chronos dtype; Mitra precision override)",
        "packages": packages,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--consume", type=Path, help="results.zip to reconstruct from alone")
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--models", type=Path)
    args = parser.parse_args()
    if args.consume:
        if not (args.workdir and args.models) or args.root or args.stage:
            parser.error("--consume needs --workdir and --models, and no --root/--stage")
        result = consume(args.consume, args.workdir, args.models)
        print(f"consume: PASS ({result['predictions_checked']} predictions)", flush=True)
        return
    if not (args.root and args.stage):
        parser.error("--root and --stage are required")
    root = args.root.resolve()
    out = output(root)
    out.mkdir(parents=True, exist_ok=True)
    previous = None
    if args.stage != "prepare":
        for stage in STAGES[: STAGES.index(args.stage)]:
            receipt = read(out / f"receipt_{stage}.json")
            if receipt["identity"] != identity(root) or receipt["previous"] != previous:
                raise ValueError("Stale stage receipt; rerun from preparation")
            for name, checksum in receipt["outputs"].items():
                if sha(out / name) != checksum:
                    raise ValueError(f"Changed stage output: {name}")
            previous = sha(out / f"receipt_{stage}.json")
    prior_path = out / f"receipt_{args.stage}.json"
    owned = set(read(prior_path).get("outputs", {})) if prior_path.exists() else set()
    for stage in STAGES[STAGES.index(args.stage) :]:
        (out / f"receipt_{stage}.json").unlink(missing_ok=True)
    before = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    started = time.monotonic()
    gpu_stage = args.stage in ("validation", "test", "activity", "future", "reload")
    if gpu_stage:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("This stage requires the documented hosted T4 runtime")
        torch.cuda.reset_peak_memory_stats()
    globals()[args.stage](root)
    if args.stage in ("prepare", "baselines", "validation", "test", "activity"):
        figures(root, args.stage)
    produced = {
        p.name: sha(p) for p in out.iterdir() if p.is_file() and not p.name.startswith("receipt_")
    }
    write(
        out / f"receipt_{args.stage}.json",
        {
            "stage": args.stage,
            "identity": identity(root),
            "previous": previous,
            "seconds": time.monotonic() - started,
            "pid": os.getpid(),
            "completed": True,
            "peak_allocated_gpu_bytes": int(torch.cuda.max_memory_allocated())
            if gpu_stage
            else None,
            "outputs": {n: h for n, h in produced.items() if before.get(n) != h or n in owned},
        },
    )
    print(f"{args.stage}: PASS", flush=True)


if __name__ == "__main__":
    main()
