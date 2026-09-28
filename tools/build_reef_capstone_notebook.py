# ruff: noqa: E501
"""Deterministically assemble a standalone reef capstone with embedded public data."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "DIMER_Philippine_Reef_Heat_Stress_Capstone.ipynb"


def build() -> dict:
    cells = []

    def add(kind: str, text: str, infrastructure: bool = False) -> None:
        cell = {
            "cell_type": kind,
            "id": f"reef-{len(cells):03}",
            "metadata": {"jupyter": {"source_hidden": True}} if infrastructure else {},
            "source": text.strip().splitlines(keepends=True),
        }
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    def md(text: str) -> None:
        add("markdown", text)

    def code(text: str, hidden: bool = False) -> None:
        add("code", text, hidden)

    md("""# DIMER Capstone 7: Philippine Reef Heat-Stress Outlook

**Profile:** TASK-INFERENCE · **Mode:** GUIDED · **Standard:** NOTEBOOK_SPEC 2.2
**Status:** Candidate pending fresh Colab T4 qualification.

Can a pretrained model improve short-horizon forecasts of Philippine regional heat stress over simple baselines?
You will forecast daily HotSpot at 7, 14 and 28 days, calculate accumulated Degree Heating Weeks (DHW),
compare errors, and export evidence. A baseline winning is a valid scientific result.

**Prerequisites:** basic Python and Colab familiarity; no prior ML experience.
Select **Runtime → Change runtime type → T4 GPU**, then **Run all**. No tokens, uploads, repository clone,
manual restart or configuration edits are required. Setup and model downloads need internet access.
The target budget is 30 minutes and 12 GiB GPU memory; these limits are unverified until hosted qualification.
The five NOAA text files are embedded as a verified 1 MB data archive; approximately 0.48 GB of model weights
and a larger Python environment are downloaded. Leave several GB of runtime disk free.

**Input → System → Output:** daily regional HotSpot history → baseline/Chronos forecasting → DHW accumulation
and chronological evaluation → regional outlook, comparison tables and reproducibility bundle.

This is an educational retrospective stress-indicator benchmark, not NOAA's official outlook, a reef-level
measurement, an observed bleaching prediction or an operational warning. Forecasts use a frozen September 2026 snapshot.

**AI Assistance Disclosure:** Generative AI assisted code development and technical writing under maintainer
direction. The maintainer remains responsible for implementation review, validation and release decisions.
AI assistance is not independent verification, provider endorsement or release approval.
""")
    md("""## Roadmap and learning objectives

1. Explain what a satellite-derived regional HotSpot represents.
2. Reconstruct accumulated stress and distinguish known heat from predicted heat.
3. Compare models on identical chronological origins without future-data leakage.
4. Change one setting on validation data and interpret uncertainty and failure cases.
5. Export evidence, verify it in a fresh process, and write a bounded conclusion.

**Core concepts** explain the science; **Evaluation practice** explains comparisons; **Infrastructure** cells
handle environment, data integrity and exports. Infrastructure code is visible but collapsed where supported.
Run it without studying every line. Checkpoint answers and completion notes never block execution.
""")
    md("""## 1. Core concepts: what do the numbers mean?

**SST** is sea-surface temperature. A **HotSpot** is positive temperature excess over a local warm-season reference,
in °C. **DHW** sums HotSpot values at or above 1 °C during the last 84 days and divides by seven, giving °C-weeks.
The value 1.00 contributes; 0.99 does not. An **origin** is the last observed date; **horizon** is the number of future days.

NOAA's regional series uses the pixel at the region's 90th-percentile HotSpot each day. The representative pixel
can change. These are not regional mean temperatures or stationary reef sensors. Do not subtract the header's
regional average climatology from a selected pixel's SST to invent a new HotSpot.

**Predict:** could DHW remain high when today's HotSpot is zero?
<details><summary>Show a sample answer</summary>Yes. DHW retains qualifying heat from the previous 83 days.
Current conditions alone do not describe accumulated exposure.</details>

Source: [NOAA regional methodology](https://coralreefwatch.noaa.gov/product/vs/map.php) and
[DHW method](https://coralreefwatch.noaa.gov/product/5km/methodology.php).
""")
    lock = (ROOT / "tools/reef-requirements.lock").read_text(encoding="utf-8")
    code(
        """# Infrastructure: isolated pinned environment avoids changing the notebook kernel.
import sys, subprocess, pathlib, json, hashlib, base64, os, time
SETUP_STARTED = time.perf_counter()
if sys.version_info[:2] != (3, 12):
    raise RuntimeError("This candidate targets Colab Python 3.12; use a supported runtime.")
RUN_ROOT = pathlib.Path.cwd() / "reef_capstone_run"
RUN_ROOT.mkdir(exist_ok=True)
ENV_ROOT = RUN_ROOT / "environment"
subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "uv==0.10.12"], check=True)
if not (ENV_ROOT / "bin/python").exists():
    subprocess.run([sys.executable, "-m", "uv", "venv", "--python", sys.executable, str(ENV_ROOT)], check=True)
PYTHON = ENV_ROOT / "bin/python"
"""
        + f"REQUIREMENTS = {lock!r}\n"
        + """
(RUN_ROOT / "requirements.lock").write_text(REQUIREMENTS, encoding="utf-8")
subprocess.run([sys.executable, "-m", "uv", "pip", "sync", "--python", str(PYTHON),
                "--require-hashes", str(RUN_ROOT / "requirements.lock")], check=True)
print("Isolated dependencies ready; the notebook kernel does not need a restart.")
(RUN_ROOT / "setup_summary.json").write_text(json.dumps({"seconds": time.perf_counter() - SETUP_STARTED}))
""",
        True,
    )
    md("""## 2. Data provenance and integrity

The immutable sample contains all five Philippine regional records from 1985-01-01 through 2026-09-26.
NOAA makes CRW content public domain and asks for attribution. The archive retains exact source bytes and hashes.
It is embedded data, not executable code. There is no fallback to mutable live data on an integrity failure.

Western and Southern each lack 25 dates. Missing dates remain missing; they are never zero stress.
Early 1985 DHW/BAA zeros precede the fields' declared valid starts and are masked.
The legacy BAA categories are retained for provenance, not treated as the newer official alert system.

[NOAA data directory](https://coralreefwatch.noaa.gov/product/vs/data.php) ·
[Rights and citation](https://coralreefwatch.noaa.gov/satellite/docs/recommendations_crw_citation.php)
""")
    data = ROOT / "tutorials/data/reef"
    manifest = json.loads((data / "manifest.json").read_text())
    archive = (data / manifest["archive"]["filename"]).read_bytes()
    model = json.loads((ROOT / "tools/reef_model_manifest.json").read_text())
    code(
        "# Infrastructure: immutable public NOAA sample, carried inside this notebook.\n"
        + f"DATA_MANIFEST = json.loads({json.dumps(manifest)!r})\n"
        + f"MODEL_MANIFEST = json.loads({json.dumps(model)!r})\n"
        + f"ARCHIVE_B64 = {base64.b64encode(archive).decode()!r}\n"
        + """payload = base64.b64decode(ARCHIVE_B64, validate=True)
if hashlib.sha256(payload).hexdigest() != DATA_MANIFEST["archive"]["sha256"]:
    raise ValueError("Embedded sample integrity failure")
(RUN_ROOT / DATA_MANIFEST["archive"]["filename"]).write_bytes(payload)
(RUN_ROOT / "dataset_manifest.json").write_text(json.dumps(DATA_MANIFEST, indent=2))
(RUN_ROOT / "model_manifest.json").write_text(json.dumps(MODEL_MANIFEST, indent=2))
print("NOAA Coral Reef Watch", DATA_MANIFEST["coverage_start"], "through", DATA_MANIFEST["coverage_end"])
""",
        True,
    )
    md("""## 3. Infrastructure: visible implementation

The following three cells carry the validation/mathematics, pinned model adapter and stage runner as readable
Python source. They are written locally inside this runtime; no DIMER repository source is downloaded or imported.
The core stage helpers expose dates, baselines, DHW accumulation and metrics. Each stage runs in a separate process,
so no hidden kernel model state is required. Chronos is frozen; no fine-tuning is performed.
""")
    sources = []
    hashes = {}
    for name, variable in [
        ("reef_core.py", "CORE_SOURCE"),
        ("reef_models.py", "MODEL_SOURCE"),
        ("reef_runtime.py", "RUNTIME_SOURCE"),
    ]:
        source = (ROOT / "tools" / name).read_text(encoding="utf-8")
        source = source.replace("from __future__ import annotations\n", "")
        if "'''" in source:
            raise ValueError("Use double-quoted docstrings in carried source")
        code(f"# Infrastructure: {name}\n{variable} = r'''\n{source}'''", True)
        sources.append(source)
        hashes[name] = hashlib.sha256(source.encode()).hexdigest()
    identity = {
        "builder": "tools/build_reef_capstone_notebook.py",
        "source_sha256": hashes,
        "requirements_sha256": hashlib.sha256(lock.encode()).hexdigest(),
        "dataset_sha256": manifest["archive"]["sha256"],
        "spec_version": "2.2",
    }
    code(
        """RUNNER_SOURCE = "from __future__ import annotations\\n" + CORE_SOURCE + MODEL_SOURCE + RUNTIME_SOURCE
compile(RUNNER_SOURCE, "reef_runner.py", "exec")
(RUN_ROOT / "reef_runner.py").write_text(RUNNER_SOURCE, encoding="utf-8")
"""
        + f"SOURCE_IDENTITY = {identity!r}\n"
        + """
SOURCE_IDENTITY["runner_sha256"] = hashlib.sha256(RUNNER_SOURCE.encode()).hexdigest()
(RUN_ROOT / "source_identity.json").write_text(json.dumps(SOURCE_IDENTITY, indent=2))
def run_stage(name):
    subprocess.run([str(PYTHON), str(RUN_ROOT / "reef_runner.py"), name, "--root", str(RUN_ROOT)], check=True)
run_stage("prepare")
""",
        True,
    )
    md("""**What to notice:** DHW should reconstruct to within 0.0001 °C-weeks. The origin table should show
23 validation origins per region, 46 test origins for Northern/Central/Eastern, and 23 for Western/Southern.
The latter two have no eligible 2025 origins because their 365-day contexts include gaps.
This loss of coverage is part of the result, not a model failure.

## 4. Evaluation practice: freeze the calendar

Fit the seasonal reference on 2017–2022 only. Use 2023 for validation; test on 2024–2025. Origins are the
1st and 15th of each month, excluding forecasts that cross the year boundary. All arms share identical eligible
origins and 365-day contexts. Score 7-, 14- and 28-day prefixes of the same 28-day prediction.

The primary outcome is equal-region mean 14-day HotSpot MAE (mean absolute error, °C). Show the complete-panel
2024 result alongside the unbalanced full test and three-region 2025 result. Foundation-model pretraining
overlap is unresolved; this is not a claim of a pristine unseen-training benchmark. Reprocessed historical
observations do not prove what was published at each historical origin.

**Predict:** why is randomly shuffling daily observations an invalid forecast experiment?
<details><summary>Show a sample answer</summary>It allows future conditions into development and ignores
the information that would have been available at the forecast origin. Date-aware splits preserve that boundary.</details>
""")
    code("""run_stage("baselines")
print("Baseline forecasts saved. Both arms use exactly the planned contexts and outcomes.")""")
    md("""## 5. Compare a foundation model

**Persistence** repeats the last HotSpot. **Seasonal reference** repeats the development-period mean for each
month/day. **Chronos-2** forecasts from recent history without fitting new weights. Its raw negative predictions
are saved, then constrained to zero consistently with the nonnegative target. Quantile bands are marginal
daily forecasts, not simultaneous guarantees.

**Predict:** will recent history outperform seasonal averages during unusual heat? The experiment may say no.
Run validation first. Model downloads are pinned by immutable revision and verified digests.
""")
    code('run_stage("validation")')
    md("""## 6. Change one thing: how much history?

Predict whether shortening context from 365 to 180 days helps. The next cell changes only context length on the
same validation origins; model, dates, horizon and scoring remain fixed. Compare the `chronos_180` and `chronos`
validation rows in the final results. The 365-day test default remains frozen regardless of this activity.

<details><summary>How to interpret the comparison</summary>A shorter context may respond to recent conditions,
but can lose seasonal information. Use measured errors; a plausible explanation alone is not evidence.</details>
""")
    code('run_stage("activity")\nrun_stage("lock")')
    md("""## 7. Locked test and accumulated stress

Now evaluate once on the held-out origins. For each arm, future DHW combines known past heat with that arm's
predicted HotSpots. The separate DHW-persistence reference simply carries DHW at the origin forward.
Good DHW scores may reflect already accumulated heat; check HotSpot skill and the known/predicted contribution.
The DHW of median HotSpots is not necessarily the median DHW forecast. No DHW probability bands are claimed.
""")
    code('run_stage("test")\nrun_stage("score")')
    md("""## 8. Illustrative regional outlook

Forecast from the frozen snapshot end, 2026-09-26. These dates are explicit: this is not a live warning.
Regions without a complete recent context are listed as excluded; no gap filling is hidden in the chart.
Numerical DHW indicators at 4 and 8 °C-weeks summarize thermal exposure, not field-observed bleaching.
""")
    code('run_stage("future")')
    md("""## 9. Export and independent verification

A fresh process recomputes baseline and DHW values from serialized inputs, then reloads the pinned model for
one origin. Prediction parity is checked with declared tolerances. The bundle records dataset and model identity,
environment, origin exclusions, metrics, forecasts and evidence. A successful run is evidence for maintainer review,
not automatic release approval.
""")
    code('run_stage("reload")\nrun_stage("report")')
    code("""# Read reports with the standard library; no notebook-kernel dependency changes.
reports = json.loads((RUN_ROOT / "metrics.json").read_text())
from IPython.display import display, Markdown, Image, FileLink
rows = reports["all"]["macro"]
table = "| Split | Period | Arm | Horizon | HotSpot MAE | DHW endpoint MAE |\\n|---|---|---|---:|---:|---:|\\n"
for row in rows:
    if row["horizon"] == 14:
        table += "| " + " | ".join(str(row.get(key)) for key in
            ["split", "period", "arm", "horizon", "hotspot_mae", "dhw_endpoint_mae"]) + " |\\n"
display(Markdown(table))
display(Image(filename=str(RUN_ROOT / "outlook.png")))
display(Image(filename=str(RUN_ROOT / "dhw_components.png")))
display(Image(filename=str(RUN_ROOT / "failure_examples.png")))
display(FileLink(str(RUN_ROOT / "reef_evidence.zip")))
print("All mandatory stages completed. Preserve this executed notebook and the evidence bundle.")
""")
    md("""## 10. Interpret the evidence

- Which arm has the lowest 14-day HotSpot MAE? How large is its paired difference from persistence and seasonality?
- Does the common-2024 comparison support the same conclusion as the unbalanced full panel?
- Is good DHW performance mainly inherited from known history? Does the model anticipate new exceedances?
- Inspect q10–q90 coverage and width, and high-stress-day errors. An 80% nominal band need not cover 80% here.
- Report event support. Undefined precision/recall stays unavailable with a reason; it never becomes perfect performance.

<details><summary>Conclusion template</summary>For [regions and dates], [arm] had [MAE] at 14 days versus
[baseline errors]. The common-period comparison [agreed/differed]. A failure case was [example]. These findings
describe a retrospective satellite stress indicator; they do not establish reef-level bleaching, operational
warning skill, or independence from foundation-model training data. My next controlled experiment is [change].</details>

Your completion record is a personal learning aid, **not a required submission**. An inconclusive result is acceptable.
""")
    md("""## 11. Optional: bring your own daily HotSpot data

Disabled by default. Supply `region_id,date,hotspot_c` with ISO dates, °C units, and the same threshold-relative
HotSpot meaning. Raw SST is insufficient. The optional path validates the CSV, then forecasts each region from
its last 365 observed days; it does not pretend unlabelled future dates provide evaluation evidence.
`byod_example.csv` is generated automatically for practicing the interface. A historical backtest requires
separate future outcomes and a newly declared evaluation design.
""")
    code("""USE_BYOD = False # @param {type:"boolean"}
BYOD_CSV_PATH = "" # @param {type:"string"}
HOTSPOT_SEMANTICS_CONFIRMED = False # @param {type:"boolean"}
if USE_BYOD:
    if not HOTSPOT_SEMANTICS_CONFIRMED or not BYOD_CSV_PATH:
        raise ValueError("Supply a CSV path and confirm NOAA-compatible HotSpot semantics in °C.")
    byod_program = "from __future__ import annotations\\n" + CORE_SOURCE + MODEL_SOURCE + r\"\"\"
import json, sys
from pathlib import Path
root, source = Path(sys.argv[1]), Path(sys.argv[2])
frame = validate_byod(pd.read_csv(source), units='degC', semantics_confirmed=True)
panel = daily_panel([frame])
model = reef_load_model(json.loads((root/'model_manifest.json').read_text()), root/'model_cache')
rows=[]
for region in panel.index.get_level_values(0).unique():
    origin=panel.xs(region).index.max()
    history=context_at(panel,region,origin)
    q=reef_predict(model,history)
    point=np.maximum(q[:,1],0)
    dhw=compose_dhw(history,point)
    for i,date in enumerate(pd.date_range(origin+pd.Timedelta(days=1),periods=28)):
        rows.append(dict(region_id=region,origin=str(origin.date()),date=str(date.date()),
                         hotspot_c=point[i],dhw_c_weeks=dhw['dhw'][i]))
pd.DataFrame(rows).to_csv(root/'byod_forecasts.csv',index=False)
print('Saved unscored BYOD forecasts; future outcomes are not available for evaluation.')
\"\"\"
    (RUN_ROOT/"byod_runner.py").write_text(byod_program)
    subprocess.run([str(PYTHON),str(RUN_ROOT/"byod_runner.py"),str(RUN_ROOT),BYOD_CSV_PATH],check=True)
else:
    print("BYOD disabled. The complete default experiment is unchanged.")
""")
    md("""## Troubleshooting, glossary and sources

- **GPU unavailable:** select a Colab T4 before Run all. CPU engineering checks do not qualify model execution.
- **Download/integrity failure:** retain the error and retry the same pinned inputs in a fresh runtime; never bypass a digest.
- **Dependency/runtime mismatch:** this candidate targets Python 3.12 in an isolated environment. Preserve logs for review.
- **Out of memory or excessive runtime:** preserve the failure; do not quietly omit regions, change precision or replace predictions.
- **Missing context:** inspect origin_manifest.csv; missing data is not zero stress.
- **BYOD refusal:** check date uniqueness/continuity, finite nonnegative values, units and target semantics.

**MAE:** mean absolute prediction error. **RMSE:** error emphasizing large misses. **Pinball loss:** quantile error.
**Coverage:** observed fraction inside the stated interval. **Persistence:** repeat the last value. **Macro average:**
equal weight per region, rather than letting more numerous records dominate. **Held out:** not used to tune the experiment.

Credit: NOAA Coral Reef Watch, v3.1 Regional Virtual Station time series, Philippine regions, 1985-01-01–2026-09-26;
accessed 2026-09-28. [Data](https://coralreefwatch.noaa.gov/product/vs/data.php),
[methodology](https://coralreefwatch.noaa.gov/product/5km/methodology.php),
[citation guidance](https://coralreefwatch.noaa.gov/satellite/docs/recommendations_crw_citation.php).
Model: [Amazon Chronos-2](https://huggingface.co/amazon/chronos-2), immutable revision in the export.
Explore related DIMER model pipelines after this activity; the same validation and comparison habits apply.
""")
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
            "dimer": {
                "notebook_spec": "2.2",
                "notebook_profile": "TASK-INFERENCE",
                "notebook_mode": "GUIDED",
                "standalone": True,
                "release_status": "candidate",
                "clean_runtime_evidence": "pending",
                "credentials_required": False,
                "source_identity": identity,
            },
        },
    }


if __name__ == "__main__":
    output = ROOT / "tutorials" / NAME
    output.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output)
