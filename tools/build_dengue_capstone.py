"""Generate the self-contained Quezon City source-block forecasting capstone."""

# ruff: noqa: E501 -- notebook prose and embedded cells retain readable paragraph boundaries.
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials/DIMER_Philippine_Dengue_Forecasting_Capstone.ipynb"


def _load_carrier():
    """Load tools/notebook_carrier.py by path; tests and validators load generators by path."""
    spec = importlib.util.spec_from_file_location(
        "notebook_carrier", Path(__file__).with_name("notebook_carrier.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


carrier = _load_carrier()


def carried_files():
    mapping = {
        name: name
        for name in ("dengue_runtime.py", "dengue_core.py", "dengue_data.py", "dengue_models.py")
    }
    mapping.update(
        {
            "model_manifest.json": "dengue_models.json",
            "requirements.txt": "dengue-requirements.lock",
        }
    )
    files = {
        target: (ROOT / "tools" / source).read_bytes().decode("utf-8")
        for target, source in mapping.items()
    }
    files["CODE_LICENSE.txt"] = (ROOT / "LICENSE").read_text(encoding="utf-8")
    files["source.json"] = json.dumps(
        {
            "generator": "build_dengue_capstone/1",
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "files": {
                name: hashlib.sha256(value.encode()).hexdigest() for name, value in files.items()
            },
        },
        indent=2,
    )
    return files


PREFLIGHT = """import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import urllib.request
import uuid
import zipfile
from IPython.display import Image, Markdown, FileLink, display

USE_BYOD = False # @param {type:"boolean"}
BYOD_CSV = "" # @param {type:"string"}
BYOD_SOURCE_BLOCKS_CONFIRMED = False # @param {type:"boolean"}
BYOD_AREA = "" # @param {type:"string"}
BYOD_SOURCE_CITATION = "" # @param {type:"string"}
# Optional BYOD: manually place your aggregate CSV in Colab, then set the controls.
# Exact columns: year,block,cases,rain,temp. At least six complete 52-block years.
# Integer counts, rainfall in mm/block, temperature in Celsius. No individual records.
# Name the area and cite the source: they label outputs and provenance, never predictors.
if USE_BYOD and (not BYOD_SOURCE_BLOCKS_CONFIRMED or not Path(BYOD_CSV).is_file()):
    raise ValueError('Confirm aggregate source-block definitions/units and provide an existing CSV.')
if USE_BYOD and not (BYOD_AREA.strip() and BYOD_SOURCE_CITATION.strip()):
    raise ValueError('Name the BYOD area and cite its source; another area is never relabelled Quezon City.')
if platform.system() != 'Linux' or platform.machine() != 'x86_64':
    raise RuntimeError('Use a fresh Colab Linux x86-64 T4 runtime.')
try:
    GPU = subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).strip()
except (FileNotFoundError, subprocess.CalledProcessError) as exc:
    raise RuntimeError('Select Runtime > Change runtime type > T4 GPU, then Run all.') from exc
if 'T4' not in GPU:
    raise RuntimeError('The canonical target is a fresh T4 runtime.')
ROOT = Path.cwd() / 'outputs' / 'dengue' / uuid.uuid4().hex[:12]
ROOT.mkdir(parents=True)
if shutil.disk_usage(ROOT).free < 20*1024**3:
    raise RuntimeError('Allow at least 20 GiB free disk.')
print('Run directory:', ROOT, '\\nGPU:', GPU)
"""

BOOTSTRAP = """started = time.perf_counter()
UV_URL = 'https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'
with urllib.request.urlopen(UV_URL, timeout=90) as response:
    wheel = response.read(20081405)
if len(wheel) != 20081404 or hashlib.sha256(wheel).hexdigest() != 'aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60':
    raise RuntimeError('Bootstrap hash/size mismatch')
with zipfile.ZipFile(io.BytesIO(wheel)) as archive:
    name = next(n for n in archive.namelist() if n.endswith('.data/scripts/uv'))
    UV = ROOT/'uv'
    UV.write_bytes(archive.read(name))
UV.chmod(0o700)
ENV = dict(os.environ, HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_DISABLE_TELEMETRY='1',
           DO_NOT_TRACK='1', MPLBACKEND='Agg', OMP_NUM_THREADS='2')
ENV.pop('HF_TOKEN', None)
ENV.pop('HUGGING_FACE_HUB_TOKEN', None)
subprocess.run([str(UV),'venv','--managed-python','--python','3.12.12',str(ROOT/'env')],check=True,env=ENV)
PYTHON = ROOT/'env/bin/python'
subprocess.run([str(UV),'pip','install','--python',str(PYTHON),'--require-hashes','--only-binary',':all:',
                '-r',str(ROOT/'requirements.txt')],check=True,env=ENV)
subprocess.run([str(PYTHON),'-c','import torch; assert torch.cuda.is_available(); print(torch.__version__)'],check=True,env=ENV)
(ROOT/'bootstrap.json').write_text(json.dumps({'seconds':time.perf_counter()-started,'gpu':GPU}),encoding='utf-8')

REFUSED_EXIT = 3  # dengue_runtime.py's exit code for an input it refused with a message; not a crash
REFUSED = None  # the stage that refused the input, if any; every later stage is skipped

def run(stage):
    # Returns True when the stage passed. A refused input prints one message, returns False and makes
    # every later stage skip, so Run all ends without a traceback. Any other failure still stops with an error.
    global REFUSED
    if REFUSED:
        print(f'Skipped {stage}: the input was refused in {REFUSED} (message above); nothing else runs.')
        return False
    log = ROOT/(stage+'.log')
    print('Running',stage,'— log:',log,flush=True)
    with log.open('w',encoding='utf-8') as handle:
        process = subprocess.Popen([str(PYTHON),'-u',str(ROOT/'dengue_runtime.py'),'--root',str(ROOT),'--stage',stage],
                                   env=ENV,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        for line in process.stdout:
            handle.write(line); handle.flush()
            if len(line)<2000: print(line.rstrip(),flush=True)
        process.wait()
    if process.returncode == REFUSED_EXIT:
        REFUSED = stage
        print('Nothing was modelled or exported. Fix the file or setting named above, upload it again if needed, '
              'then choose Runtime > Run all (rerunning only this cell would reuse the old settings).')
        return False
    if process.returncode:
        raise RuntimeError(f'{stage} failed. Inspect {log}; do not skip this stage.')
    return True

def record(name, limit=14000):
    # Long files are shown as a labelled excerpt; the linked file is always complete.
    path=ROOT/'results'/name
    text=json.dumps(json.loads(path.read_text()),indent=2)
    if len(text)>limit:
        text=text[:limit]+f'\\n... [truncated excerpt: first {limit:,} of {len(text):,} characters; the full file is linked below]'
    print(text)
    display(FileLink(str(path)))

def figure(name):
    display(Image(filename=str(ROOT/'results'/name)))

def table(name, columns, limit=None):
    # Columns are chosen by meaning; the linked CSV keeps every column and row.
    path=ROOT/'results'/name
    with path.open(newline='',encoding='utf-8') as handle:
        reader=csv.DictReader(handle); rows=list(reader); fields=reader.fieldnames
    missing=[c for c in columns if c not in fields]
    if missing:
        raise KeyError(f'{name} lacks columns {missing}')
    shown=rows if limit is None else rows[:limit]
    clean=lambda value: str(value).replace('|','/').replace('\\n',' ')
    lines=['| '+' | '.join(columns)+' |','| '+' | '.join('---' for _ in columns)+' |']
    lines+=['| '+' | '.join(clean(row[c]) for c in columns)+' |' for row in shown]
    display(Markdown('\\n'.join(lines)))
    print(f'Showing {len(shown)} of {len(rows)} rows. All columns in the CSV:',', '.join(fields))
    display(FileLink(str(path)))

def consume():
    # Rebuild from results.zip in a new directory holding only the embedded code: no data.json,
    # run_config.json or other original-run files. Verified model snapshots are declared inputs.
    consumer=ROOT.parent/(ROOT.name+'-consumer')
    shutil.rmtree(consumer,ignore_errors=True)  # notebook-owned; a re-run starts clean
    consumer.mkdir(parents=True)
    for name, text in FILES.items():
        (consumer/name).write_bytes(text.encode('utf-8'))
    command=[str(PYTHON),'-u',str(consumer/'dengue_runtime.py'),'--consume',str(ROOT/'results/results.zip'),
             '--workdir',str(consumer/'check'),'--models',str(ROOT/'models')]
    result=subprocess.run(command,env=ENV,cwd=consumer,capture_output=True,text=True)
    print(result.stdout[-4000:])
    if result.returncode:
        raise RuntimeError('Bundle reconstruction failed:\\n'+result.stderr[-4000:])
    print(json.dumps(json.loads((consumer/'check/consumer_verification.json').read_text()),indent=2))
"""


def build():
    cells = []

    def md(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip() + "\n"})

    def code(text, metadata=None):
        ast.parse(text)
        cells.append(
            {
                "cell_type": "code",
                "metadata": metadata or {},
                "source": text.strip() + "\n",
                "execution_count": None,
                "outputs": [],
            }
        )

    md("""# Philippine Dengue Forecasting: Do Weather Signals Improve Predictions?

**Exploratory published-source-block benchmark · Revision 0.2.0 · Candidate · CAPSTONE · E2E / GUIDED · NOTEBOOK_SPEC 2.2**

Can past rainfall and temperature improve forecasts of Quezon City's reported dengue counts one to four reporting blocks ahead? You will compare six systems, change one availability assumption, and conclude using measured evidence.

**Input → System → Output:** aggregate case counts and historical weather → source checks, lagged features, Chronos and Mitra → forecasts, errors, model intervals and a reproducible results bundle.

**Important scope:** the publisher gives conflicting week definitions. We preserve its year/block order without inventing calendar dates. Case–weather alignment and real reporting availability are unverified. Results describe this published alignment, not verified weekly forecasting, clinical diagnosis or an operational warning system.

## Problem statement

Dengue programme staff and analysts in Quezon City see reported cases rise and fall with the rainy season, and weather data are easy to obtain. The question this notebook supports is a modelling decision, not an operational one: on the published source blocks, does adding past rainfall and temperature to a forecasting model reduce its error compared with the same model without weather, and does either beat a naive baseline? Answering it honestly needs leakage-free time splits, a naive floor and a paired comparison, which is what you build here.

**How to use:** basic Python/Colab familiarity; choose a fresh **T4 GPU**, keep defaults and **Run all**. No token or manual restart is required. The run downloads about 780 MB of model weights (Chronos-2 478 MB, Mitra 303 MB) plus a pinned Python environment. A recorded fresh Colab T4 run (Colab CLI, 2026-10-03, revision `08b2343`) took 445 seconds end to end, with 0.48 GiB peak GPU allocation; the 60-minute, 20 GiB disk and 12 GiB GPU figures are planning targets. Infrastructure cells handle reproducibility; the numbered sections contain the learning activity.

**Use notice:** this notebook is for research, training and evaluation. It is not validated for production, clinical or safety-critical use, including outbreak warning. Running it downloads third-party model weights and a dataset under their own licences (see the model cards and `DATA_LICENSE.md`), and it uses compute on your own Colab account. Its results are tutorial evidence on one published series, not a benchmark. Everything is supplied "as is", without warranty of any kind, and liability is limited to the extent the licences allow. Optional BYOD data stays in this Colab runtime; you are responsible for having the right to use it.

By the end, identify a forecast origin and horizon, explain time leakage, compare weather against the same model without weather, interpret an interval's empirical coverage, and support a conclusion with errors and limitations.

**AI Assistance Disclosure:** Code and instructional text were developed with generative AI assistance under maintainer direction. The maintainer remains responsible for review, validation and release decisions. AI assistance does not constitute independent verification, provider endorsement or release approval.""")
    code(PREFLIGHT)
    md("""## Infrastructure — reproducible local environment

The next cell is collapsed: it carries the embedded source (about 230,000 characters), checks each file against its SHA-256, and builds an isolated Python environment pinned for both models. Expand it to read the code; you do not need to. It does not fetch DIMER scripts or call workers. All stages run locally in this Colab runtime in separate processes.

Optional BYOD is disabled above. It accepts only aggregate `year,block,cases,rain,temp` data with six complete 52-block years: at least two history years plus two validation and two test years. Cases must be nonnegative integer counts, rainfall nonnegative mm per block and temperature in Celsius (−30 to 60). Every row needs all five values; the notebook names the file, line and column of any refusal. Name the area and cite the source in `BYOD_AREA` and `BYOD_SOURCE_CITATION`: they label captions and provenance and are never predictors. Before any model is downloaded, every Mitra context the experiment will need is checked; a constant-count stretch stops the run there, with the affected contexts listed. No individual health records are accepted.

**Your data stays in this runtime.** The BYOD file is read only inside this Colab runtime; nothing is sent to any other service, and the only copy that leaves is what you download yourself. The results bundle keeps aggregate numeric contexts of your series (`artifact_manifest.json` lists them). Do not use confidential, restricted, personal or regulated data, including any health data under a data-sharing agreement, unless you are authorised to process it in Google Colab.

**How to run BYOD (after or instead of the default run):**
1. Upload the CSV with the **Files** pane (folder icon, left sidebar) and copy its path.
2. In the first code cell set `USE_BYOD`, `BYOD_CSV`, `BYOD_SOURCE_BLOCKS_CONFIRMED`, `BYOD_AREA` and `BYOD_SOURCE_CITATION`.
3. Choose **Runtime > Run all**. Each Run all starts a new run directory, so it rebuilds the isolated environment and downloads the model weights again (about 780 MB; roughly the 7–8 minutes of the default run). Rerunning only the first cell leaves an empty run directory, and rerunning a numbered section alone keeps using the earlier data, so neither switches to your file.

A refused file prints one `Refused:` message naming the file, line and column (or the rule it failed); every later stage then prints `Skipped` and nothing is modelled or exported. Fix the file, upload it again and Run all.""")
    files = carried_files()
    code(
        "# @title Embedded source, integrity check and isolated environment (collapsed; expand to read)\n"
        + "FILES = "
        + carrier.carried_dict(files)
        + """
for name, text in FILES.items():
    (ROOT/name).write_bytes(text.encode('utf-8'))
provenance=json.loads(FILES['source.json'])
for name, checksum in provenance['files'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==checksum
(ROOT/'run_config.json').write_text(json.dumps({'byod_csv': str(Path(BYOD_CSV).resolve()) if USE_BYOD else None, 'source_blocks_confirmed': BYOD_SOURCE_BLOCKS_CONFIRMED, 'area': BYOD_AREA.strip() if USE_BYOD else None, 'source_citation': BYOD_SOURCE_CITATION.strip() if USE_BYOD else None}),encoding='utf-8')
"""
        + BOOTSTRAP,
        {"cellView": "form", "jupyter": {"source_hidden": True}},
    )
    md("""## 1. Know the source before predicting

The pinned [Zenodo release](https://zenodo.org/records/21978184) contains 832 QC rows, 2010–2025, from QCESD/PIDSR counts linked to IMERG rainfall and ERA5-Land temperature. Each row is a published reporting block. Rain describes a centroid grid cell, not every location in the city. Counts are not population-adjusted incidence rates.

The database uses ODC-ODbL 1.0, with source-specific terms retained in `DATA_LICENSE.md`. The notebook validates the full archive and workbook hashes, explicit columns, units, 52-block coverage and missingness. Missing observations never become zero cases.

**Predict:** what could go wrong if an apparent zero actually meant “not reported”? Why might surveillance during 2020–2021 differ?

**What to notice:** 832 rows and 26 origins per evaluation partition are expected for the default source. The audit must explicitly report the unresolved calendar and availability limitations. The context check reports how many planned model contexts it examined.""")
    code(
        "if run('prepare'):\n    plan=json.loads((ROOT/'results/plan.json').read_text())\n    print('Area:',plan['area'],'· Source:',plan['source_citation'])\n    record('plan.json')\n    record('context_preflight.json')\n    table('origin_manifest.csv',['partition','origin_key','horizon','target_key','cutoff'],limit=8)\n    figure('prepare.png')\n    display(FileLink(str(ROOT/'dataset_audit.json')))\n    display(FileLink(str(ROOT/'DATA_LICENSE.md')))"
    )
    md("""## 2. Build a fair chronological comparison

An **origin** is the last block at which a forecast is issued. A **horizon** says how many blocks ahead it predicts. A **lag** is an earlier observed value. **Leakage** occurs when fitting or prediction sees information unavailable at issuance.

Default history: 2010–2021; validation: 2022–2023; test: 2024–2025. Each evaluation year forecasts blocks 1–4, 5–8, …, 49–52 from the preceding origin: 26 origins × four horizons. Older test observations may enter later forecast contexts after becoming observable; test errors never tune settings.

We assume zero reporting delay initially. Event-date truncation cannot reconstruct historical publication vintages: final-run weather and revised counts may not have been available at the time. Foundation-model pretraining overlap is also unknown.

**Trace one row before trusting the rest.** The first table below takes the first test origin, horizon 1, and lists every weather-model feature with its value and the source blocks it reads. Every block is at or before the origin; the target block is not observed at issuance. The last row is the most recent training label that was already mature. The audit table then checks the same rule for every origin.

**Predict:** will persistence or the same block last year work better? Ridge fits a transparent log-count regression using past counts and known target-block seasonality. Negative inverse-transformed estimates are floored at zero and flagged; forecasts are not rounded for scoring.""")
    code(
        "if run('baselines'):\n    table('feature_example.csv',['item','value','source_blocks'])\n    table('availability_audit.csv',['partition','origin','horizon','cutoff','max_training_target','max_training_input','future_input_used'],limit=8)\n    record('baseline_metrics.json')\n    figure('baselines.png')"
    )
    md("""<details><summary>Interpretation checkpoint</summary>
A seasonal baseline can capture recurring patterns but miss changing intensity. Persistence can work over short horizons but miss turning points. Neither guarantees good high-case forecasts. Features and training labels must satisfy their own availability cutoffs, not merely precede the final test year.
</details>

## 3. Compare Chronos and the weather ablation

Chronos-2 reads the last 104 case blocks without gradient training. Mitra performs horizon-specific regression using labelled in-context examples; it does not fine-tune weights here. Validation selects 52 versus 104 issuance blocks for case-only Mitra, with shorter-window ties. Weather Mitra uses that exact setting and the same rows, adding only lagged rainfall and temperature. Fresh context preprocessing uses training examples only.

**Predict:** will weather help every horizon? The controlled question is **Mitra + weather versus Mitra case-only**. Chronos versus Mitra changes the model family too, so it cannot isolate weather's effect.

**Tuning parity:** the window is selected on case-only Mitra and reused for weather Mitra; it is not separately tuned for weather, which if anything favours case-only. Chronos (fixed 104-block context) and Ridge use fixed settings and are not tuned on validation. Read the comparisons with that in mind.

**What the point forecast means:** Chronos's point forecast is its 0.5 quantile, the median of its forecast distribution. Mitra and Ridge are fitted on log1p(count) and back-transformed with expm1, so their point estimates behave like a median on the count scale and sit below the mean when counts are skewed. None of the six systems reports a mean forecast; MAE is the matching error for median-type points.

**What to notice:** all six systems should have 104 validation forecasts, and the progress lines name each model's window and delay. A worse foundation-model score is legitimate evidence, not a notebook error.""")
    code(
        "if run('validation'):\n    record('selection.json')\n    record('validation_metrics.json')\n    figure('validation.png')\nif run('lock'):\n    record('experiment_lock.json')"
    )
    md("""## 4. Evaluate held-out published blocks

Settings are now locked. **MAE** is the average absolute count error; lower is better. **Bias** is prediction minus reference; a negative value means underprediction. Skill compares MAE with seasonal naïve and is undefined if its denominator is zero.

**Which baseline is the floor?** Skill here is relative to seasonal naïve, the weaker of the two simple baselines, so a model can show positive skill and still lose to persistence (the last observed count). Persistence is the naive floor for this conclusion: before calling any model useful, check whether its MAE is below persistence's.

**Predict:** will validation's ranking persist? Read each horizon separately, then the equal-horizon mean. Inspect high-case blocks defined by the development-history 90th percentile; this is not an official outbreak threshold.

The paired weather comparison reports weather MAE minus case-only MAE. Negative favours weather. A bootstrap resamples contiguous groups of four origins, keeping their horizons together; its interval is descriptive with only two evaluation years.

Chronos's nominal 80% model interval is not a guaranteed calibrated interval. Inspect its actual coverage and width. Mitra and Ridge have point forecasts, not borrowed Chronos intervals.

The largest-misses table shows each miss's published count (`reference`), the forecast, the signed error (forecast minus reference; negative means underprediction) and its size. Pick one and explain it: was it a high-case block, and did every system miss it?""")
    code(
        "if run('test'):\n    record('metrics.json')\n    record('paired_comparison.json')\n    record('high_case_metrics.json')\n    record('interval_metrics.json')\n    figure('test.png')\n    figure('test_forecasts.png')\n    table('largest_misses.csv',['system','target_key','horizon','reference','prediction','error','abs_error'],limit=10)"
    )
    md("""<details><summary>Worked interpretation guidance</summary>
If adding weather lowers average MAE but worsens the longest horizon or high-case errors, report that trade-off. An interval containing zero does not establish that the models are identical, and it does not establish a weather gain either: it is compatible with weather helping, hurting or doing nothing on these rows. Compare both Mitra variants with persistence too: if neither beats the naive floor, the weather difference is between two models that are not yet useful here. A consistent gain on these rows still does not validate the source timing, prove causality or establish future operational performance.
</details>

## 5. Change one thing: assume a two-block reporting delay

**Predict → change → run → observe → explain.** The following completed activity changes only the availability cutoff from zero to two blocks on validation origins. Target blocks stay identical; Chronos must forecast two additional steps from its earlier cutoff. Mitra's labels must also be mature by that cutoff.

This two-block delay is illustrative, not an estimate of actual reporting delay. Canonical test outputs and the locked experiment remain unchanged. The first table pairs each system's zero-delay and two-block validation MAE on the same 104 targets; `change` is delay-two minus zero-delay, so a positive value means the delay hurt. The second table shows individual paired forecasts.""")
    code(
        "if run('activity'):\n    table('delay_summary.csv',['system','pairs','mae_delay_0','mae_delay_2','change'])\n    table('delay_comparison.csv',['system','target_key','horizon','reference','prediction_delay_0','prediction_delay_2','abs_error_change'],limit=12)\n    figure('activity.png')\n    record('delay_metrics.json')"
    )
    md("""## 6. Forecast beyond the final source block

Fit/context information now extends through the last published row. The next four block forecasts have no observed reference in this dataset. We export them without an accuracy score or a claim about today's real-world dengue situation.

**What to notice:** the original source year/block labels continue. The first table places all six systems side by side for the four target blocks. The second lists all 24 forecasts: `prediction` is the value used (never below zero), `raw_prediction` and `clipped` show any flooring, and `q10`/`q50`/`q90` are Chronos's nominal quantiles (blank for systems without intervals). There is no reference column because these blocks have not been observed. These are source-block projections, not reconstructed dates.""")
    code(
        "if run('future'):\n    table('future_forecast_view.csv',['target_key','horizon','persistence','seasonal','ridge','chronos','mitra_cases','mitra_weather'])\n    table('future_predictions.csv',['system','target_key','horizon','prediction','raw_prediction','clipped','q10','q50','q90'])"
    )
    md("""## 7. Export, reconstruct and verify

The notebook saves safe numeric context and preprocessing state, pinned model identities and checksums. A fresh process reconstructs both models plus the fitted Ridge state, then reproduces the final completed origin and unscored final inference. It verifies all four horizons and Chronos quantiles with declared tolerances.

Results include predictions, metrics, availability audit, source/licence notices, stage receipts and verification. The ZIP excludes the source archive and model weights. It does retain aggregate numeric support contexts (lagged counts, and rain and temperature for weather Mitra) for two origins; `artifact_manifest.json` says exactly what. The summary records the area, limitations, Python, device, precision and package versions, with the resource figures labelled as targets.

**Reuse the bundle elsewhere.** The last step extracts `results.zip` into a new directory that holds only this notebook's embedded code, then reproduces the same forecasts from the bundle alone. It needs the verified model snapshots, but not the original run directory, its data file or its settings. Elsewhere, run the embedded `dengue_runtime.py --consume results.zip --workdir <new dir> --models <snapshot cache>` next to `source.json` and `model_manifest.json`. Preserve the executed notebook alongside the ZIP for qualification evidence.""")
    code(
        "if run('reload'):\n    record('verification.json')\nif run('report'):\n    record('run_summary.json')\n    display(FileLink(str(ROOT/'results/results.zip')))\n    consume()"
    )
    md("""## 8. Conclude with evidence

Decide first which level of claim your numbers reach. The levels are cumulative; stop at the first one that fails.

1. **Signal: does Mitra + weather beat the naive floor?** Its test MAE must be below persistence's. If it is not, the model is not yet useful on these rows, whatever its skill against seasonal naïve.
2. **Comparator: does weather beat case-only?** Mitra + weather needs a lower MAE than Mitra case-only *and* a paired interval that excludes zero. An interval that includes zero means no weather gain is established.
3. **Operational: would it help a real warning system?** Out of scope here: block timing, case–weather alignment and reporting availability are unverified.

Use your measured results to complete: “On the test years, persistence (the naive floor) had MAE ___; Mitra + weather ___, Mitra case-only ___ and Chronos ___. Mitra + weather ___ [did / did not] beat persistence. Adding historical weather changed Mitra's MAE from ___ to ___ (at horizon 1: ___ to ___). The paired difference was ___, with interval ___, which ___ [excludes / includes] zero. On high-case blocks ___. Under the illustrative delay ___. These results reach the ___ level, for this weather encoding (rain and temperature lags 0–4 and 4- and 8-block means) with a window tuned for case-only Mitra, under the published alignment. They cannot establish ___.”

Keep the unresolved block definition, case–weather alignment, retrospective availability, possible pretraining overlap and limited evaluation span visible. Your completion record is an optional learning aid, not a required submission.

**Continue on DIMER:** explore the Chronos-2 and Mitra regression pipelines for their complete supported workflows. This capstone demonstrates in-context regression; it does not claim gradient fine-tuning or a deployment-ready health service.

## Troubleshooting

| Symptom | What to do |
|---|---|
| `Use a fresh Colab Linux x86-64 T4 runtime` or `Select Runtime > Change runtime type > T4 GPU` | Change the runtime type to T4 GPU, then Runtime > Run all. |
| `Allow at least 20 GiB free disk` | Start a fresh runtime (Runtime > Disconnect and delete runtime); earlier run directories use disk. |
| `Bootstrap hash/size mismatch`, a download or `uv pip install` error | A pinned file did not arrive intact. Rerun after a minute; never remove the hash checks. |
| A model snapshot digest or `Embedded source changed` error | An integrity check failed: start a fresh runtime and Run all; do not edit the carried files. |
| CUDA out of memory | Unexpected at the measured 0.48 GiB peak; restart the runtime so no other notebook shares the GPU. |
| `Refused:` in section 1 (BYOD) | Read the message: it names the file, line and column, or the rule. Fix the file, upload it again and Run all; later stages print `Skipped` until then. |
| `Confirm aggregate source-block definitions` / `Name the BYOD area` | Set the BYOD controls in the first cell, then Run all. |

## References

The two dataset repositories distribute the same workbook; they are not independent replication cohorts.

[1] Amazon, "Chronos-2 (amazon/chronos-2)," Hugging Face model card, revision 95a9710. [Online]. Available: https://huggingface.co/amazon/chronos-2

[2] AutoGluon, "Mitra regressor (autogluon/mitra-regressor)," Hugging Face model card, revision 5f277aa. [Online]. Available: https://huggingface.co/autogluon/mitra-regressor

[3] "Philippine dengue and environmental dataset," v1.1.0, Zenodo, record 21978184, ODC-ODbL 1.0, doi: 10.5281/zenodo.21978184. [Online]. Available: https://zenodo.org/records/21978184

[4] UPRI-NOAH, "dengue-rainfall-dataset," GitHub repository documentation. [Online]. Available: https://github.com/UPRI-NOAH/dengue-rainfall-dataset

[5] pelitro-rcw, "dengue-environmental-dataset," GitHub author repository. [Online]. Available: https://github.com/pelitro-rcw/dengue-environmental-dataset""")
    for i, c in enumerate(cells):
        c["id"] = f"dengue-{i:02d}"
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "accelerator": "GPU",
            "dimer": {
                "notebook_spec": "2.2",
                "notebook_profile": "E2E",
                "notebook_mode": "GUIDED",
                "notebook_kind": "CAPSTONE",
                "status": "Candidate",
                "revision": "0.2.0-candidate",
                "scope": "exploratory_published_source_blocks",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = json.dumps(build(), indent=1, ensure_ascii=False) + "\n"
    if args.check:
        if not NOTEBOOK.exists() or NOTEBOOK.read_bytes() != text.encode():
            raise SystemExit("Dengue notebook generation mismatch")
        print("Dengue notebook parity: PASS")
    else:
        NOTEBOOK.write_text(text, encoding="utf-8", newline="\n")
        print(NOTEBOOK)


if __name__ == "__main__":
    main()
