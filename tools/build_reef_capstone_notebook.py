# ruff: noqa: E501
"""Deterministically assemble a standalone reef capstone with embedded public data."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "DIMER_Philippine_Reef_Heat_Stress_Capstone.ipynb"


def _load_carrier():
    """Load tools/notebook_carrier.py by path; tests and validators load generators by path."""
    spec = importlib.util.spec_from_file_location(
        "notebook_carrier", Path(__file__).with_name("notebook_carrier.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


carrier = _load_carrier()


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

    md("""# DIMER Capstone: Philippine Reef Heat-Stress Outlook

**Profile:** TASK-INFERENCE · **Mode:** GUIDED · **Standard:** NOTEBOOK_SPEC 2.2
**Status:** Candidate pending fresh Colab T4 qualification.

Can a pretrained model improve short-horizon forecasts of Philippine regional heat stress over simple baselines?
You will forecast daily HotSpot at 7, 14 and 28 days, calculate accumulated Degree Heating Weeks (DHW),
compare errors, and export evidence. A baseline winning is a valid scientific result.

**Prerequisites:** basic Python and Colab familiarity; no prior ML experience.
Select **Runtime → Change runtime type → T4 GPU**, then **Run all**. Any current Colab Python version works:
setup downloads a pinned, managed Python 3.12.13 into an isolated environment, so you do not need to choose an
older runtime version. No tokens, uploads, repository clone, manual restart or configuration edits are required.
Setup and model downloads need internet access.
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
        """# Infrastructure: isolated pinned environment on a managed Python 3.12; the kernel is unchanged.
import sys, subprocess, pathlib, json, hashlib, base64, os, time, platform, shutil
SETUP_STARTED = time.perf_counter()
TARGET_PYTHON = "3.12.13"  # uv-managed interpreter, independent of the Colab kernel's Python
RUN_ROOT = pathlib.Path.cwd() / "reef_capstone_run"
RUN_ROOT.mkdir(exist_ok=True)
ENV_ROOT = RUN_ROOT / "environment"
PYTHON = ENV_ROOT / "bin/python"
LOG_DIR = RUN_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
def child_environment():
    # Children never inherit the kernel's Python path, startup file, inline plotting backend or
    # Colab's UV_SYSTEM_PYTHON (which makes uv warn that --system has no effect on uv venv).
    dropped = {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "UV_SYSTEM_PYTHON"}
    env = {k: v for k, v in os.environ.items() if k not in dropped}
    env.update(MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
    return env
def resource_note():
    # One-line machine state, printed at each stage start and heartbeat so that a runtime
    # disconnect leaves the last known memory and disk state in the saved notebook.
    try:
        info = {}
        with open("/proc/meminfo") as meminfo:
            for line in meminfo:
                key, value = line.split(":", 1)
                info[key] = int(value.split()[0]) / 1048576
        memory = (f"RAM available {info['MemAvailable']:.1f}/{info['MemTotal']:.1f} GB, "
                  f"unwritten {info['Dirty'] + info['Writeback']:.1f} GB")
    except (OSError, KeyError, ValueError):
        memory = "RAM state unavailable"
    return f"{memory}, disk free {shutil.disk_usage(RUN_ROOT).free / 1e9:.0f} GB"
def run_logged(command, log_name, echo=True):
    # Run a child with its output appended to logs/<name>.log under a per-attempt header, so a
    # retry keeps the earlier attempt's log. Poll it, echo new lines, and print a heartbeat every
    # 30 s so a long silent step is visibly alive; on failure show the log tail. An interrupted
    # cell stops its child instead of leaving it running, and a signal is reported by name.
    import signal
    started = last_beat = time.perf_counter()
    log_path, shown, returncode = LOG_DIR / f"{log_name}.log", 0, None
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"=== attempt started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\\n")
        log.flush()
        offset = log_path.stat().st_size
        def attempt_text():
            return log_path.read_bytes()[offset:].decode("utf-8", errors="replace")
        process = subprocess.Popen([str(part) for part in command], stdout=log,
                                   stderr=subprocess.STDOUT, env=child_environment())
        if echo:
            print(f"  child pid {process.pid} launched", flush=True)
        try:
            while returncode is None:
                try:
                    returncode = process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                text = attempt_text()
                lines = text.splitlines()
                complete = lines if text.endswith("\\n") or returncode is not None else lines[:-1]
                if echo:
                    for line in complete[shown:]:
                        print(line, flush=True)
                shown = max(shown, len(complete))
                if returncode is None and time.perf_counter() - last_beat >= 30:
                    print(f"  ... {log_name} still working, {time.perf_counter() - started:.0f} s"
                          f" ({resource_note()})", flush=True)
                    last_beat = time.perf_counter()
        finally:
            if returncode is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                print(f"  {log_name} interrupted; child pid {process.pid} stopped", flush=True)
        log.write(f"=== exit {returncode} after {time.perf_counter() - started:.0f} s ===\\n")
    if returncode:
        tail = "\\n".join(attempt_text().splitlines()[-30:])
        reason = f"exit code {returncode}"
        if returncode < 0:
            try:
                reason = f"signal {signal.Signals(-returncode).name} ({reason})"
            except ValueError:
                pass
        raise RuntimeError(f"{log_name} failed with {reason}; log: {log_path}\\n{tail}")
    return time.perf_counter() - started
run_logged([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "uv==0.10.12"],
           "setup_uv", echo=False)
def environment_python():
    if not PYTHON.exists():
        return None
    probe = [str(PYTHON), "-c", "import platform; print(platform.python_version())"]
    return subprocess.run(probe, capture_output=True, text=True).stdout.strip() or None
ENV_REUSED = environment_python() == TARGET_PYTHON
if not ENV_REUSED:
    run_logged([sys.executable, "-m", "uv", "venv", "--clear", "--managed-python",
                "--python", TARGET_PYTHON, ENV_ROOT], "setup_python")
if environment_python() != TARGET_PYTHON:
    raise RuntimeError(f"Isolated environment is not Python {TARGET_PYTHON}; preserve this log.")
"""
        + f"REQUIREMENTS = {carrier.carried_literal(lock)}\n"
        + """
(RUN_ROOT / "requirements.lock").write_text(REQUIREMENTS, encoding="utf-8")
print("Installing the hash-locked environment (a few minutes on a fresh runtime)...", flush=True)
seconds = run_logged([sys.executable, "-m", "uv", "pip", "sync", "--python", PYTHON,
                      "--require-hashes", RUN_ROOT / "requirements.lock"], "setup_sync", echo=False)
print(f"Locked packages installed in {seconds:.0f} s (log: {LOG_DIR / 'setup_sync.log'}).")
# Finish writing the ~7 GB environment to disk before any stage starts reading from it.
flush_started = time.perf_counter()
os.sync()
print(f"Installed files flushed to disk in {time.perf_counter() - flush_started:.0f} s ({resource_note()}).")
print(f"Isolated Python {TARGET_PYTHON} dependencies ready (kernel Python {platform.python_version()});"
      " the notebook kernel does not need a restart.")
_ = (RUN_ROOT / "setup_summary.json").write_text(json.dumps({
    "seconds": time.perf_counter() - SETUP_STARTED,
    "kernel_python": platform.python_version(),
    "environment_python": TARGET_PYTHON,
    "environment_reused": ENV_REUSED,
    "resources_after_setup": resource_note(),
}))
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
        + f"DATA_MANIFEST = json.loads({carrier.carried_literal(json.dumps(manifest))})\n"
        + f"MODEL_MANIFEST = json.loads({carrier.carried_literal(json.dumps(model))})\n"
        + f"ARCHIVE_B64 = {carrier.carried_literal(base64.b64encode(archive).decode())}\n"
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
def run_stage(name, *extra):
    print(f"[{time.strftime('%H:%M:%S')}] stage {name} started ({resource_note()})", flush=True)
    # -u and faulthandler: a crash inside the child leaves its Python traceback in the log.
    command = [PYTHON, "-u", "-X", "faulthandler", RUN_ROOT / "reef_runner.py", name,
               "--root", RUN_ROOT, *extra]
    seconds = run_logged(command, f"stage_{name}")
    print(f"[{time.strftime('%H:%M:%S')}] stage {name} finished in {seconds:.0f} s", flush=True)
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
same validation origins; model, dates, horizon and scoring remain fixed. It then prints a validation-only table:
compare the `chronos_180` and `chronos` rows at 7, 14 and 28 days, check the support columns (regions, origins,
days), and read the traced valid and rejected contexts. The 365-day test default remains frozen regardless of
this activity.

<details><summary>How to interpret the comparison</summary>A shorter context may respond to recent conditions,
but can lose seasonal information. Use measured errors; a plausible explanation alone is not evidence.</details>
""")
    code('run_stage("activity")\nrun_stage("compare")\nrun_stage("lock")')
    md("""## 7. Locked test and accumulated stress

Now evaluate once on the held-out origins. For each arm, future DHW combines known past heat with that arm's
predicted HotSpots. The separate DHW-persistence reference simply carries DHW at the origin forward.
Good DHW scores may reflect already accumulated heat; check HotSpot skill and the known/predicted contribution.
The DHW of median HotSpots is not necessarily the median DHW forecast. No DHW probability bands are claimed.

Scoring prints six compact tables, labelled A–F: (A) the full, common-2024 and three-region-2025 comparisons with
region/origin/day support; (B) regional 14-day errors; (C) interval coverage, width and pinball loss;
(D) high-stress-day errors; (E) 4 and 8 °C-week threshold events, including new exceedances; (F) paired
Chronos-minus-baseline differences. Section 10 asks questions you can answer from these tables.
""")
    code('run_stage("test")\nrun_stage("score")')
    md("""## 8. Illustrative regional outlook

Forecast from the frozen snapshot end, 2026-09-26. These dates are explicit: this is not a live warning.
Regions without a complete recent context are listed as excluded; no gap filling is hidden in the chart.
Western and Southern Philippines are expected to be excluded here: their last 365 days contain missing dates.
The outlook file keeps raw and constrained HotSpot, Chronos quantiles and the known/new DHW contributions.
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
keys = ["split", "period", "arm", "hotspot_mae", "dhw_endpoint_mae", "regions", "origins", "days"]
table = ("Primary 14-day summary with support. Tables A–F printed by the scoring cell hold the detail.\\n\\n"
         "| Split | Period | Arm | HotSpot MAE | DHW endpoint MAE | Regions | Origins | Days |\\n"
         "|---|---|---|---:|---:|---:|---:|---:|\\n")
def cell(value):
    return "n/a" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)
for row in rows:
    if row["horizon"] == 14:
        table += "| " + " | ".join(cell(row.get(key)) for key in keys) + " |\\n"
display(Markdown(table))
display(Image(filename=str(RUN_ROOT / "outlook.png")))
display(Image(filename=str(RUN_ROOT / "dhw_components.png")))
display(Image(filename=str(RUN_ROOT / "failure_examples.png")))
display(FileLink(str(RUN_ROOT / "reef_evidence.zip")))
print("All mandatory stages completed. Preserve this executed notebook and the evidence bundle.")
""")
    md("""## 10. Interpret the evidence

- Which arm has the lowest 14-day HotSpot MAE (table A)? How large is its paired difference from persistence
  and seasonality (table F)? Check the regions/origins columns so both scores cover the same cases.
- Does the common-2024 comparison support the same conclusion as the unbalanced full panel (table A)?
- Is good DHW performance mainly inherited from known history (compare `dhw_persistence` in table A and the
  component chart)? Does the model anticipate new exceedances (table E, `new_exceedance`)?
- Inspect q10–q90 coverage and width (table C) and high-stress-day errors (table D). An 80% nominal band need not
  cover 80% here.
- Report event support (table E). "No observations", "no observed positives" and a poor score are different
  findings; undefined precision/recall stays n/a with a reason and never becomes perfect performance.

<details><summary>Conclusion template</summary>For [regions and dates], [arm] had [MAE] at 14 days versus
[baseline errors]. The common-period comparison [agreed/differed]. A failure case was [example]. These findings
describe a retrospective satellite stress indicator; they do not establish reef-level bleaching, operational
warning skill, or independence from foundation-model training data. My next controlled experiment is [change].</details>

Your completion record is a personal learning aid, **not a required submission**. An inconclusive result is acceptable.
""")
    md("""## 11. Optional: bring your own daily HotSpot data

Disabled by default. Supply `region_id,date,hotspot_c` with ISO dates, °C units, and the same threshold-relative
HotSpot meaning. Raw SST is insufficient. Region IDs are read as literal text, so `001` stays `001`.
The optional path validates the CSV, then forecasts each region from its last 365 observed days; it does not
pretend unlabelled future dates provide evaluation evidence. Leave the path empty to practise on the generated
`byod_example.csv`. Each distinct input file gets its own folder with forecasts (raw, constrained, quantiles,
DHW components) and a receipt recording the input digest, units, semantics and model revision.
A historical backtest requires separate future outcomes and a newly declared evaluation design.
""")
    code("""USE_BYOD = False # @param {type:"boolean"}
BYOD_CSV_PATH = "" # @param {type:"string"}
HOTSPOT_SEMANTICS_CONFIRMED = False # @param {type:"boolean"}
if USE_BYOD:
    if not HOTSPOT_SEMANTICS_CONFIRMED:
        raise ValueError("Confirm NOAA-compatible threshold-relative HotSpot semantics in °C.")
    source = pathlib.Path(BYOD_CSV_PATH) if BYOD_CSV_PATH else RUN_ROOT / "byod_example.csv"
    print("BYOD input:", source)
    run_stage("byod", "--csv", str(source.resolve()))
else:
    print("BYOD disabled. The complete default experiment is unchanged.")
""")
    md("""## Troubleshooting, glossary and sources

- **GPU unavailable:** select a Colab T4 before Run all. CPU engineering checks do not qualify model execution.
- **Download/integrity failure:** retain the error and retry the same pinned inputs in a fresh runtime; never bypass a digest.
- **Dependency/runtime mismatch:** setup builds an isolated uv-managed Python 3.12.13 whatever the kernel version.
  If that download fails, retry in a fresh runtime and preserve the log; do not edit the pinned requirements.
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
