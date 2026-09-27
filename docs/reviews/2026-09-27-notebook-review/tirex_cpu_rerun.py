"""Run the real TiRex-2 runner (old joint vs new independent) on CPU using the notebook's own cells.

Usage: python tirex_evidence.py <new notebook> <old notebook> <tirex python> <cache dir> <out json>
"""
import importlib.metadata
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

new_nb, old_nb, tirex_python, cache, out_json = sys.argv[1:6]
work = Path(out_json).parent / "tirex_work"
work.mkdir(exist_ok=True)


def cells(path):
    return {
        "".join(c["source"]).splitlines()[0].removeprefix("# @title "): "".join(c["source"])
        for c in json.loads(Path(path).read_text())["cells"] if c["cell_type"] == "code"
    }


NEW, OLD = cells(new_nb), cells(old_nb)


class Stub:
    def __getattr__(self, name):
        return lambda *a, **k: (self, self) if name == "subplots" else None


pkg = types.ModuleType("matplotlib")
pkg.pyplot = Stub()
sys.modules["matplotlib"], sys.modules["matplotlib.pyplot"] = pkg, pkg.pyplot
real_version = importlib.metadata.version
importlib.metadata.version = lambda n: "3.10.0" if n == "matplotlib" else real_version(n)


def prepare(sample, tag):
    import os
    d = work / tag
    d.mkdir(exist_ok=True)
    os.chdir(d)
    ns = {"display": lambda *a, **k: None}
    for title in ["1.1 Notebook controls", "1.2 Check the notebook runtime", "2.1 Pinned model identities and footprint",
                  "3.1 Generate or load the sample", "3.2 Check the data before any model download",
                  "4.1 Plot the development period and mark the hidden test period",
                  "5.1 Split the data, build the baselines and define the metrics"]:
        src = NEW[title].replace('SAMPLE_DATASET = "SYNTHETIC"  # @param', f'SAMPLE_DATASET = "{sample}"  # @param')
        exec(src, ns)
    exec(NEW["6.1 Build one environment per model"].replace(
        "ensure_uv()\nMODEL_PYTHONS = {name: ensure_environment(name) for name in SELECTED_MODELS}", "MODEL_PYTHONS = {}"), ns)
    exec(NEW["6.2 Write the model runner programs"], ns)
    old_ns = {"WORK_ROOT": d / "old", "hashlib": __import__("hashlib"), "Path": Path}
    (d / "old" / "runners").mkdir(parents=True, exist_ok=True)
    exec(OLD["6.2 Write the model runner programs"], old_ns)
    return ns, d / "old" / "runners" / "tirex_runner.py"


def run(runner, request, horizon, stage, tag):
    output, meta = work / f"{tag}.csv", work / f"{tag}.json"
    env = {**__import__("os").environ, "CUDA_VISIBLE_DEVICES": "", "MPLBACKEND": "Agg"}
    subprocess.run([tirex_python, str(runner), "--input", str(request), "--output", str(output), "--metadata", str(meta),
                    "--cache", cache, "--horizon", str(horizon), "--partition", stage], check=True, env=env,
                   capture_output=True, text=True)
    table = pd.read_csv(output, parse_dates=["timestamp", "forecast_origin"], dtype={"series_id": str}, keep_default_na=False)
    return table, json.loads(meta.read_text())


results = {}
for sample in ("SYNTHETIC", "OPEN_METEO_PH"):
    ns, old_runner = prepare(sample, sample.lower())
    new_runner = ns["RUN_WORK_ROOT"] / "runners" / "tirex_runner.py"
    sample_result = {}
    for stage in ("validation", "test"):
        truth = ns[f"{stage}_truth"]
        seasonal, last = ns[f"{stage}_seasonal"], ns[f"{stage}_last"]
        horizon = ns["VALIDATION_HORIZON"] if stage == "validation" else ns["TEST_HORIZON"]
        per_version = {}
        for version, runner in (("old_joint", old_runner), ("new_independent", new_runner)):
            table, meta = run(runner, ns["REQUEST_PATHS"][stage], horizon, stage, f"{sample}-{stage}-{version}")
            if version == "old_joint":
                table["forecast_origin"] = table["forecast_origin"]
            all_rows = pd.concat([table, last, seasonal], ignore_index=True)
            _, per_series, aggregate = ns["evaluate_forecasts"](
                all_rows, truth, seasonal, ["TiRex-2", "last_value", "seasonal_naive"], stage)
            per_version[version] = {
                "aggregate": aggregate.round(4).to_dict("records"),
                "per_series": per_series.round(4).to_dict("records"),
                "conditioning": meta.get("conditioning"),
                "packages": meta.get("packages"),
            }
        sample_result[stage] = per_version

    # Perturbation: change the last 12 context hours of the second series only.
    request = ns["REQUEST_PATHS"]["validation"]
    frame = pd.read_csv(request, dtype={"series_id": str}, keep_default_na=False)
    ids = ns["panel"]["series_ids"]
    changed = frame.copy()
    tail = changed.index[changed["series_id"] == ids[1]][-12:]
    changed.loc[tail, "target"] = changed.loc[tail, "target"] + 25.0
    perturbed = work / f"{sample}-perturbed.csv"
    changed.to_csv(perturbed, index=False)
    effect = {}
    for version, runner in (("old_joint", old_runner), ("new_independent", new_runner)):
        base, _ = run(runner, request, 24, "validation", f"{sample}-p0-{version}")
        pert, _ = run(runner, perturbed, 24, "validation", f"{sample}-p1-{version}")
        effect[version] = {
            sid: float(np.max(np.abs(
                base[base.series_id == sid]["q0.5"].to_numpy() - pert[pert.series_id == sid]["q0.5"].to_numpy())))
            for sid in ids
        }
    sample_result["perturbation_max_abs_median_change"] = {"changed_series": ids[1], **effect}
    results[sample] = sample_result

Path(out_json).write_text(json.dumps(results, indent=2, default=str))
print(json.dumps({s: r["perturbation_max_abs_median_change"] for s, r in results.items()}, indent=2))
