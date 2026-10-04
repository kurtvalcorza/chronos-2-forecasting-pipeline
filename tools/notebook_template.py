"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.2 §4 standalone carrier, generator /2.1).

Only the task-specific prose and stage cells live here. The isolated runtime, the embedded package
(seven modules, carried verbatim in dependency order), and the model pin/stage/verify cells are
produced by the generator from repository sources so they cannot drift from the package.

Generator keys in use: ``modules`` lists every module of ``src/chronos2_pipeline/`` except
``__init__.py``; ``entry_module`` is ``model.py`` (it holds ``MODEL_ID``/``MODEL_REVISION``/
``MODEL_LICENSE``/``MODEL_KEY``); ``model_load`` is the package's documented loader
``load_pinned_model(weights_dir=WEIGHTS_DIR)`` (there is no ``from_pretrained`` class method in this
package). The default rewrite rule applies: the one ``__file__`` use is ``DEFAULT_WEIGHTS_DIR``.

Review fixes (docs/reviews, CHR-M1..M3, CHR-m1..m5): ``isolated_runtime`` with the fleet uv mechanism
(CHR-M1); a horizon activity at a fixed cut-off and context, a Mode D with/without-covariates comparison
and feasible ranges for ``PREDICTION_LENGTH`` (CHR-M2); the guided layer — audience, how to use, roadmap,
Input → Model → Output, glossary, predictions, What to notice, sample answers, troubleshooting, conclusion
template — and Infrastructure labels (CHR-M3); the seasonal-naive/coverage explanation (CHR-m1); BYOD
path and column fields, a UTF-8 message and a non-overwriting Mode C target (CHR-m2); an optional
real-future forecast (CHR-m3); Mode D provenance export and stale-file removal (CHR-m4).
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    'package': 'chronos2_pipeline',
    'repo_name': 'chronos-2-forecasting-pipeline',
    'stem': 'chronos_2_forecasting',
    'notebook_name': 'chronos_2_forecasting_colab.ipynb',
    'profile': 'TASK-INFERENCE',
    'mode': 'GUIDED',
    'infrastructure_labels': True,
    'isolated_runtime': True,
    # The fleet's uv isolated-environment mechanism (ast-audio-classification-pipeline / bioclip2-biodiversity-pipeline;
    # same uv wheel as bart-mnli-zero-shot-classification-pipeline ee128d2): managed CPython, a size- and SHA-256-verified
    # uv wheel, and a lock compiled from the pyproject pins with
    # `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes
    # --only-binary :all: -o tutorials/requirements-colab.lock.txt`.
    'managed_python': '3.12.12',
    'uv': {
        'version': '0.12.15',
        'url': 'https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl',
        'bytes': 20081404,
        'sha256': 'aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60',
    },
    'lock': 'tutorials/requirements-colab.lock.txt',
    'pipeline_class': 'LoadedModel',
    'weights_key': 'chronos-2',
    'modules': ['config.py', 'errors.py', 'evaluation.py', 'inference.py', 'model.py', 'provenance.py', 'validation.py'],
    'entry_module': 'model.py',
    'model_load': 'load_pinned_model(weights_dir=WEIGHTS_DIR)',
    'runtime_imports': ['torch', 'transformers', 'pandas', 'numpy'],
    'title': 'Chronos-2 Forecasting — DIMER `TASK-INFERENCE` tutorial (standalone)',
    'badges': [
        (
            'GitHub',
            'https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white',
            'https://github.com/kurtvalcorza/chronos-2-forecasting-pipeline',
        ),
        (
            'Open In Colab',
            'https://colab.research.google.com/assets/colab-badge.svg',
            'https://colab.research.google.com/github/kurtvalcorza/chronos-2-forecasting-pipeline/blob/main/tutorials/chronos_2_forecasting_colab.ipynb',
        ),
        (
            'Hugging Face',
            'https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-amazon%2Fchronos--2-ffcc4d?style=flat',
            'https://huggingface.co/amazon/chronos-2',
        ),
        (
            'Upstream',
            'https://img.shields.io/badge/Upstream-amazon--science%2Fchronos--forecasting-181717?style=flat&logo=github&logoColor=white',
            'https://github.com/amazon-science/chronos-forecasting',
        ),
        (
            'arXiv',
            'https://img.shields.io/badge/arXiv-2510.15821-b31b1b.svg',
            'https://arxiv.org/abs/2510.15821',
        ),
    ],
    'capability': (
        'zero-shot probabilistic time-series forecasting (univariate, multi-series, multi-target, optional known-future covariates) with the pinned `amazon/chronos-2` checkpoint'
    ),
    'run_all': (
        'Selecting **Run all** in a fresh supported runtime (Linux x86_64: Google Colab, Kaggle or Linux Jupyter) builds an isolated environment from the hash-locked pins and runs every later cell there, so no restart is needed; it stages and digest-verifies the pinned snapshot, generates the tutorial sample in code, validates it into an input manifest before the model runs, forecasts locally, writes the evaluation report, runs the Section 11 horizon activity, and exports machine-readable outputs with provenance. The default path needs no repository clone, no DIMER worker or service, no credential, no upload dialog and no configuration edit (NOTEBOOK_SPEC 2.2 §5).'
    ),
    'byod': (
        'After the sample workflow completes, bring your own CSV in Section 4: set `BYOD_PATH` to a CSV that is already in this runtime, or tick `USE_BYOD` to upload one; if your column names differ from `series_id` / `timestamp` / `target`, set `ID_COLUMN`, `TIMESTAMP_COLUMN` and `TARGET_COLUMN` in the same cell. Then run Section 4 and every cell after it. Your data passes through the same notebook-local validation, task, evaluation-report and export cells as the sample; the expected input format, the ceilings and the privacy guidance are stated in the Prerequisites and in Section 4, and the file stays inside this runtime. BYOD is optional and never part of the default path.'
    ),
    'intro': (
        'Chronos-2 supplies the pretrained zero-shot forecasting model. This repository supplies DIMER configuration and validation, immutable model pinning/digest checks, normalized outputs, chronological evaluation, baselines, and provenance — all of it carried in this notebook. **No training or fine-tuning occurs**, and **no adaptation occurs:** inference is in-context only, the pinned checkpoint is used as published, and no preprocessing is fitted. The default sample is deterministic synthetic teaching data generated in code (the same bytes the repository checks in as `examples/sample-data/chronos_univariate.csv`); its metrics are tutorial/sanity evidence, not a benchmark claim.\n'
        '\n'
        '### Who this is for\n'
        '\n'
        'Analysts and engineers who know basic Python and pandas and want to forecast a regular time series (hourly, daily, …) with a pretrained model instead of training one. You do not need to know how Chronos-2 works inside; you do need to know what a time series and a forecast are. The glossary before Section 4 defines every other term the notebook uses.\n'
        '\n'
        '### How to use this notebook\n'
        '\n'
        '1. Choose **Runtime → Run all** once and let it finish. The default path needs no edits, no upload and no credential.\n'
        '2. Sections 1–3 are **Infrastructure**: their code is collapsed, and you may run them without reading them. The learning starts in Section 4.\n'
        '3. Before Sections 6, 7 and 9, write down the answer to the **Predict** question; after the cell runs, read **What to notice** and then open the sample answer.\n'
        '4. Form fields (lines ending in `# @param`) are the only lines you are meant to edit. Each instruction names the cells to run after a change.\n'
        '5. Sections 12 and 13 are optional and off by default; Section 11 is a short experiment that runs by default and that you then repeat with your own settings.\n'
        '\n'
        '### Roadmap\n'
        '\n'
        '| Section | What happens | Why it matters |\n'
        '|---|---|---|\n'
        '| 1–3 (Infrastructure) | isolated runtime, carried package, pinned and digest-verified model | the run is reproducible and offline-safe |\n'
        '| 4 | generate the sample (or bring a CSV) | the exact input is known and hashed |\n'
        '| 5 | hold out the last steps, print the ceilings, validate into an input manifest | no future value leaks into the model |\n'
        '| 6 | forecast (predict first) | read medians and quantiles |\n'
        '| 7 | score against naive baselines (predict first) | judge the forecast against something simple |\n'
        '| 8 | plot the held-out window | see the forecast against the truth |\n'
        '| 9 | Mode C: two targets in one call (predict first) | multi-target output |\n'
        '| 10 | export forecasts, metrics and provenance | machine-readable results |\n'
        '| 11 | activity: change the horizon, keep the context | one change, one effect |\n'
        '| 12 (optional) | Mode D: with vs without known-future covariates | what covariates add |\n'
        '| 13 (optional) | forecast beyond the end of the data | a forecast nobody can score yet |\n'
        '\n'
        'Troubleshooting, Interpretation and limits, and a conclusion template close the notebook.\n'
        '\n'
        '### Input → Model → Output\n'
        '\n'
        '- **Input:** one long-format table with a series identifier, a regular timestamp and a numeric target (other numeric columns are covariates). The last `PREDICTION_LENGTH` steps of each series are held out as truth.\n'
        '- **Model:** the pinned `amazon/chronos-2` checkpoint, zero-shot: it reads the history (the *context*) in one pass and returns quantiles for the next steps. Nothing is trained.\n'
        '- **Output:** one row per (series, timestamp, target) with `prediction` (the median) and the quantiles `q0.1`, `q0.5`, `q0.9`; an evaluation report against two naive baselines; an input manifest; and provenance JSON naming the model revision and runtime.'
    ),
    'learning_objectives': (
        'install the pinned runtime; name the public functions of the carried package that the learner cells call; resolve and digest-verify the immutable `amazon/chronos-2` revision; generate the synthetic sample or bring your own CSV; create a leakage-safe chronological holdout and validate the history into an input manifest; run univariate and **multi-target (Mode C)** zero-shot forecasts; interpret median/quantile outputs and the tutorial metrics against naive baselines through an evaluation report; separate the effect of the forecast horizon from the effect of context with a change-one-thing experiment; optionally compare known-future covariates (Mode D) with a run without them, and forecast beyond the end of your data; and export forecasts plus provenance. By the end you can do each of these without the repository being reachable.'
    ),
    'exclusions': (
        'classification, anomaly detection, imputation, embeddings, training/fine-tuning, calibrated prediction intervals, and production-fitness claims. Out of scope: monthly/quarterly/yearly/business-day, irregular, and gappy calendars — only fixed-width regular frequencies are accepted.'
    ),
    'prerequisites': [
        '- **Runtime:** a fresh **Linux x86_64** runtime (Google Colab, Kaggle or Linux Jupyter); Section 1 builds an isolated Python 3.12.12 environment from manylinux wheels and stops with a message on any other platform. CPU is the default path; CUDA is used automatically when available. The locked `torch==2.14.0` install is the largest download of the run, followed by the ~478 MB checkpoint; the forecasts themselves take seconds.',
        '- **Knowledge:** basic Python and pandas; what a time series and a forecast are. The glossary before Section 4 explains quantiles, the chronological holdout and the metrics.',
        '- **Data:** the default sample is a deterministic 96-step hourly series generated in code, so nothing is downloaded and no private data is needed. Optional BYOD is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one UTF-8 CSV with unique headers, including a series-identifier column, a timestamp column and a numeric target column (named `series_id`, `timestamp` and `target` unless you set `ID_COLUMN`, `TIMESTAMP_COLUMN` and `TARGET_COLUMN`); timestamps parseable, regular, and contiguous per series; targets finite numeric; further numeric columns are covariates. BYOD is read locally in the notebook runtime and is not sent to an external inference service. Do not upload confidential or restricted data (personal or otherwise sensitive data included) to a hosted notebook environment unless you are authorized to do so.',
    ],
    "cells": [
        {
            "md": (
                '## Before Section 4: glossary and the functions you will call\n'
                '\n'
                '| Term | Meaning in this notebook |\n'
                '|---|---|\n'
                '| **context** | the history the model reads before forecasting; `effective_context_length` is how many steps it actually used |\n'
                '| **horizon** (`prediction_length`) | how many steps ahead the model forecasts |\n'
                '| **chronological holdout** | the last steps of each series are cut off and kept as truth; the model never sees them |\n'
                '| **cut-off** | the last timestamp the model may see; everything after it is the future |\n'
                '| **quantile** `q0.1` / `q0.5` / `q0.9` | the value the model expects the truth to fall below 10 % / 50 % / 90 % of the time |\n'
                '| **median** (`prediction`) | `q0.5`, the point forecast; not a mean |\n'
                '| **MAE / RMSE** | mean absolute error / root mean squared error of the median (RMSE weights large misses more) |\n'
                '| **pinball loss** | the error of one quantile, scored asymmetrically |\n'
                '| **interval coverage** | the share of held-out truths that fall between `q0.1` and `q0.9` |\n'
                '| **last-value baseline** | repeats the last observed value for every future step |\n'
                '| **seasonal-naive baseline** | repeats the value from one season (24 hours) earlier |\n'
                '| **covariate** | an extra numeric column; *past* covariates are known only up to the cut-off, *known-future* covariates are also known for the forecast window |\n'
                '| **zero-shot** | the model is used as published, with no training on your data |\n'
                '\n'
                'The learner cells call these public functions of the carried package (Section 2): `ForecastConfig`, `chronological_holdout`, `validate_inputs`, `forecast`, `evaluate_forecast`, `last_value_baseline`, `seasonal_naive_baseline` and `evaluation_report`. Section 3 calls `stage_missing_files`, `verify_snapshot` and `load_pinned_model`. You do not need to read their source to use them.'
            ),
        },
        {
            "md": (
                '## 4. Generate the synthetic sample or optional BYOD\n'
                '\n'
                "The default sample is **synthetic**: one hourly series of 96 steps built from a fixed formula (a linear trend of +0.25 per hour plus a 24-step and a 12-step sinusoid, with no noise), rendered to CSV bytes exactly as the repository's `examples/sample-data/generate_samples.py` writes them, so its SHA-256 is asserted against the digest the repository checks in — the notebook proves it is forecasting the very bytes the repository tests. It is deterministic teaching data, not benchmark evidence.\n"
                '\n'
                "**Bring your own data:** set `BYOD_PATH` to a CSV already in this runtime (automation may set the `DIMER_BYOD_PATH` environment variable instead), or tick `USE_BYOD` to upload one CSV. If your columns are not called `series_id`, `timestamp` and `target`, set `ID_COLUMN`, `TIMESTAMP_COLUMN` and `TARGET_COLUMN` to your own names. The file must be UTF-8 with unique headers; duplicate headers are rejected **before pandas can rename them**, and a missing column is reported with the field to set. Additional numeric covariates are permitted by the pipeline. After any change here, run this cell and every cell after it. Look for the input source, its SHA-256 and the first rows."
            ),
            "code": (
                r'''import csv
import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

USE_BYOD = False  # @param {{type:"boolean"}}
BYOD_PATH = ""  # @param {{type:"string"}}
ID_COLUMN = "series_id"  # @param {{type:"string"}}
TIMESTAMP_COLUMN = "timestamp"  # @param {{type:"string"}}
TARGET_COLUMN = "target"  # @param {{type:"string"}}
PREDICTION_LENGTH = 12  # @param {{type:"integer"}}
BYOD_PATH = BYOD_PATH.strip() or os.environ.get("DIMER_BYOD_PATH", "")
SAMPLE_SHA256 = "eff96b1a9bec5a81aa4021b4d308c29e18fc89be0cf8eb54e755c2efe535c7f7"  # examples/sample-data/SHA256SUMS


def read_checked_csv(payload: bytes) -> pd.DataFrame:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("The CSV must be UTF-8 encoded, and this file is not (it may be UTF-16 or a legacy code page). Save it again as 'CSV UTF-8' and run this cell again.") from exc
    rows = csv.reader(io.StringIO(text))
    try:
        header = next(rows)
    except StopIteration as exc:
        raise ValueError("CSV is empty.") from exc
    duplicates = sorted(name for name, count in Counter(header).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate CSV header(s) are ambiguous: {{duplicates}}")
    return pd.read_csv(io.BytesIO(payload))


def synthetic_univariate_csv() -> bytes:
    # The repository's generate_samples.py formula and canonical CSV rendering; no random state.
    n = 96
    step = np.arange(n, dtype=float)
    frame = pd.DataFrame(
        {{
            "series_id": "A",
            "timestamp": pd.date_range("2026-01-01", periods=n, freq="h"),
            "target": 100.0 + 0.25 * step + 7.0 * np.sin(2.0 * np.pi * step / 24.0) + 1.5 * np.cos(2.0 * np.pi * step / 12.0),
        }}
    )
    return frame.to_csv(index=False, date_format="%Y-%m-%dT%H:%M:%S", float_format="%.4f", lineterminator="\n").encode("utf-8")


if BYOD_PATH:
    byod_file = Path(BYOD_PATH)
    if not byod_file.is_file():
        raise FileNotFoundError(f"BYOD_PATH {{BYOD_PATH!r}} is not a file in this runtime. Copy the CSV into the runtime (or tick USE_BYOD and leave BYOD_PATH empty to upload it) and run this cell again.")
    payload = byod_file.read_bytes()
    input_source = f"BYOD path: {{BYOD_PATH}}"
    sample_kind = "BYOD"
elif USE_BYOD:
    from google.colab import files
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise ValueError("Upload exactly one CSV (the upload was empty or cancelled, or held several files); run this cell again.")
    name, payload = next(iter(uploaded.items()))
    input_source = f"BYOD upload: {{name}}"
    sample_kind = "BYOD"
else:
    payload = synthetic_univariate_csv()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != SAMPLE_SHA256:
        raise ValueError(f"Synthetic sample digest mismatch: {{observed}} != {{SAMPLE_SHA256}}")
    input_source = "synthetic sample generated in code (== examples/sample-data/chronos_univariate.csv)"
    sample_kind = "synthetic"

frame = read_checked_csv(payload)
missing_columns = [name for name in (ID_COLUMN, TIMESTAMP_COLUMN, TARGET_COLUMN) if name not in frame.columns]
if missing_columns:
    raise ValueError(f"The CSV has no column named {{missing_columns}}; its columns are {{list(frame.columns)}}. Set ID_COLUMN, TIMESTAMP_COLUMN and TARGET_COLUMN in this cell to your own column names and run this cell again.")
input_sha256 = hashlib.sha256(payload).hexdigest()
print({{"sample_kind": sample_kind, "input_source": input_source, "input_sha256": input_sha256, "rows": len(frame), "columns": list(frame.columns)}})
print(frame.head())'''
            ),
        },
        {
            "md": (
                '## 5. Chronological holdout, ceilings, validate → input manifest\n'
                '\n'
                "The final `PREDICTION_LENGTH` timestamps of each series are held out as truth; future target values never enter model context (chronological holdout — **no random split** or training initialization). Before anything runs, the cell prints the operational ceilings: the pinned model exposes an 8,192-step context and native 1,024-step horizon, and the DIMER request guards (`ResourceLimits`) cap one request at 1,000 series IDs, 64 targets, 64 covariates, 5,000,000 rows, 8,192 context steps, and 4,096 forecast steps; horizons above 1,024 require explicit autoregressive unrolling. `runtime_versions` reports the installed stack. `validate_inputs` is the package's public validation stage: it routes the history through `validate_forecast_request` with exactly the arguments `forecast` passes, so it raises exactly what the forecast call would raise, and returns an **input manifest** naming the schema and ceilings, each series' observed span, the confirmed frequency and the verdict; it is written to `outputs/{stem}_input_manifest.json`. To show what rejection looks like, the cell also validates a deliberately broken copy of the history (one non-finite target) and records the pipeline's own error as a finding.\n"
                '\n'
                "**`PREDICTION_LENGTH` sets two things at once.** It is the number of held-out steps *and* the forecast horizon, so every step you add to it is taken away from the history the model sees: on the 96-step sample, 12 leaves 84 steps of context, 48 leaves 48 and 72 leaves 24. The cell prints how many history rows each series keeps. Values from 1 to 93 run on the sample (the holdout keeps at least 3 history rows); the seasonal-naive comparator needs 24 history rows, so Section 7 skips it above 72. To see the effect of the horizon alone, use the Section 11 activity, which keeps the cut-off and the context fixed. If you change `PREDICTION_LENGTH`, run Section 4 and every cell after it."
            ),
            "code": (
                r'''import os

os.makedirs('outputs', exist_ok=True)
config = ForecastConfig(id_column=ID_COLUMN, timestamp_column=TIMESTAMP_COLUMN, target=TARGET_COLUMN, prediction_length=PREDICTION_LENGTH, quantile_levels=[0.1, 0.5, 0.9])
split = chronological_holdout(frame, config)
for sid in split.history[config.id_column].drop_duplicates():
    h = split.history[split.history[config.id_column] == sid]
    t = split.truth[split.truth[config.id_column] == sid]
    assert h[config.timestamp_column].max() < t[config.timestamp_column].min()
print({{"holdout_and_horizon_steps": PREDICTION_LENGTH, "history_rows_per_series": {{str(k): int(v) for k, v in split.history.groupby(config.id_column, sort=False).size().items()}}}})

versions = runtime_versions()
print({{"runtime_versions": versions}})
print("DIMER request limits:", ResourceLimits())
print("Pinned-model context limit: 8192")
print("Pinned-model native prediction length: 1024")
print({{"ceilings": {{"min_observations_per_series": MIN_OBSERVATIONS, "max_ids": DEFAULT_LIMITS.max_ids, "max_targets": DEFAULT_LIMITS.max_targets, "max_covariates": DEFAULT_LIMITS.max_covariates, "max_rows": DEFAULT_LIMITS.max_rows, "max_context_length": DEFAULT_LIMITS.max_context_length, "max_prediction_length": DEFAULT_LIMITS.max_prediction_length, "model_context_length": pipe.model_context_length, "model_prediction_length": pipe.model_prediction_length}}}})

series_names = [str(sid) for sid in split.history[config.id_column].drop_duplicates()]
input_manifest = validate_inputs(split.history, config, pipe, names=series_names)
# Demonstrate rejection on a history that breaks a rule; the finding is recorded, not swallowed.
broken = split.history.copy()
broken.loc[broken.index[0], config.target] = float("nan")
try:
    validate_inputs(broken, config, pipe)
except ValidationError as exc:
    input_manifest["findings"].append({{"input": "non-finite-target-probe", "verdict": "rejected", "code": exc.code, "message": str(exc)}})
with open("outputs/{stem}_input_manifest.json", "w", encoding="utf-8") as handle:
    json.dump(input_manifest, handle, indent=2, ensure_ascii=False, default=str)
print(json.dumps(input_manifest, indent=2, default=str))'''
            ),
        },
        {
            "md": (
                '## 6. Forecast\n'
                '\n'
                "`forecast(split.history, config, pipe)` is the package's public inference path: it validates again, calls the pinned `chronos-forecasting` package's `predict_df` on the verified snapshot (it does not request model-repository remote code), and normalizes the output. Normalized output fields are `series_id`, `timestamp`, `target_name`, `prediction`, and requested quantiles such as `q0.1`, `q0.5`, `q0.9`. `prediction` is the **median (`q0.5`)**, not a mean. Model quantiles summarize the predictive distribution; they are **not guaranteed frequentist confidence intervals** and are not assumed calibrated on a new domain — the pipeline ships no threshold and no calibration. The effective context and prediction lengths actually used are printed with the first forecast rows; floating-point details can vary across runtime/hardware builds and latency is run-dependent.\n"
                '\n'
                '**Predict first.** For the default sample, how many forecast rows will this cell produce, and how many steps of history (`effective_context_length`) will the model use? Write both numbers down, then run the cell.'
            ),
            "code": (
                r'''result = forecast(split.history, config, pipe)
print(result.forecast.head())
print("effective_context_length:", result.inference["effective_context_length"])
print("effective_prediction_length:", result.inference["effective_prediction_length"])
print({{"n_forecast_rows": len(result.forecast), "quantile_columns": [c for c in result.forecast.columns if c.startswith("q")], "latency_seconds": result.inference.get("latency_seconds")}})'''
            ),
        },
        {
            "md": (
                '#### What to notice (Section 6)\n'
                '\n'
                '- `effective_prediction_length` equals `PREDICTION_LENGTH`, and `effective_context_length` is the history left after the holdout.\n'
                '- In every row `prediction` equals `q0.5`, and `q0.1` ≤ `q0.5` ≤ `q0.9`: the band between `q0.1` and `q0.9` is where the model expects the truth 80 % of the time, not a guaranteed interval.\n'
                '- Nothing here says whether the forecast is good; Section 7 scores it against the held-out truth.\n'
                '\n'
                '<details><summary>Sample answer (default sample)</summary>\n'
                '\n'
                '12 rows (one series × one target × 12 steps) and an effective context of 84 steps (96 − 12). The first rows start at `2026-01-04T12:00`, the hour after the cut-off. If you predicted 96 steps of context, remember that the 12 held-out steps are removed before the model sees anything.\n'
                '\n'
                '</details>'
            ),
        },
        {
            "md": (
                '## 7. Evaluate → evaluation report\n'
                '\n'
                "Metrics use the same chronological holdout: **MAE** is mean absolute error; **RMSE** weights larger misses more; **pinball loss** evaluates a quantile asymmetrically; **empirical interval coverage** is the fraction of held-out truths inside the requested outer quantiles. These are tutorial/sanity metrics, not benchmark evidence. `evaluation_report` is the package's public evaluation stage and always produces a report: with the held-out truth it carries the `evaluate_forecast` metrics for Chronos-2 and for the `last_value_baseline` (and the `seasonal_naive_baseline` when applicable) with the verdict `sample-sanity` — one chronological tail split of one sample, no dispersion estimate; without truth (a forecast of the real future, which the optional Section 13 produces) the verdict is `not-measurable` and the report states what data would make the task measurable. The seasonal-naive comparator runs only for uniformly hourly data when **every series has at least 24 post-holdout history rows**; short but otherwise valid hourly BYOD therefore skips this optional comparator instead of failing. The report is written to `outputs/{stem}_evaluation_report.json`.\n"
                '\n'
                '**Predict first.** The sample is a smooth curve with a steady upward trend and no noise. Which of the three forecasters — Chronos-2, last-value, seasonal-naive — do you expect to have the largest MAE, and why?'
            ),
            "code": (
                r'''evaluation = evaluate_forecast(result.forecast, split.truth, config)
last_value = last_value_baseline(split.history, split.truth, config)
last_value_evaluation = evaluate_forecast(last_value, split.truth, config)

ordered = split.history.sort_values([config.id_column, config.timestamp_column])
steps = ordered.groupby(config.id_column)[config.timestamp_column].diff().dropna()
minimum_history = int(ordered.groupby(config.id_column, sort=False).size().min())
can_use_daily_seasonal = not steps.empty and (steps == pd.Timedelta(hours=1)).all() and minimum_history >= 24

seasonal_evaluation = None
if can_use_daily_seasonal:
    seasonal = seasonal_naive_baseline(split.history, split.truth, config, season_length=24)
    seasonal_evaluation = evaluate_forecast(seasonal, split.truth, config)
seasonal_metrics = None if seasonal_evaluation is None else seasonal_evaluation.aggregate

print("Chronos-2:", evaluation.aggregate)
print("Last-value:", last_value_evaluation.aggregate)
print("Seasonal-naive:", seasonal_metrics)
print("Quantile metrics:")
print(evaluation.quantiles)
print("Per-series metrics:")
print(evaluation.per_series)

report = evaluation_report(result, split.truth, config=config, history_df=split.history, season_length=24 if can_use_daily_seasonal else None, sample_kind=sample_kind)
with open("outputs/{stem}_evaluation_report.json", "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2, ensure_ascii=False, default=str)
print(json.dumps(report, indent=2, default=str))
if report["verdict"] == "not-measurable":
    print("No held-out truth was supplied, so evaluate_forecast is not computed; the forecast above is sanity evidence only.")'''
            ),
        },
        {
            "md": (
                '#### What to notice (Section 7)\n'
                '\n'
                '- **The seasonal-naive error is exactly 6.0, and that is the trend.** Seasonal-naive repeats the value from 24 hours earlier. Both sinusoids in the sample repeat every 24 hours, but the trend adds 0.25 per hour, that is 24 × 0.25 = **6.0 per day**, so every seasonal-naive forecast is exactly 6.0 too low and its MAE and RMSE are both 6.0.\n'
                '- **Coverage of 1.0 here says nothing about calibration.** Twelve points of a noiseless curve all falling inside `q0.1`–`q0.9` is what a wide-enough band gives on easy data; judging whether an 80 % band holds 80 % of the time needs many forecast windows of real, noisy data.\n'
                "- **Chronos-2's much smaller error reflects how easy this sample is**, a smooth deterministic function, not how good Chronos-2 is on your data. That is why the report's verdict is `sample-sanity`.\n"
                '\n'
                '<details><summary>Sample answer (default sample)</summary>\n'
                '\n'
                'Seasonal-naive has the largest MAE (6.0), just above last-value (5.92): last-value repeats one number while the curve keeps moving, and seasonal-naive misses by the full day of trend. Chronos-2 has MAE 0.33 and RMSE 0.35, with coverage 1.0. On a BYOD series with noise, expect the gaps to shrink and coverage to fall below 1.0.\n'
                '\n'
                '</details>'
            ),
        },
        {
            "md": (
                '## 8. Visualize the held-out forecast\n'
                '\n'
                'For multi-series BYOD, this compact SVG deliberately visualizes one series and one target instead of interleaving unrelated lines. The machine-readable exports remain authoritative for all rows. Look for the green truth line between the black `q0.1` and red `q0.9` lines, and the blue median close to it.'
            ),
            "code": (
                r'''from html import escape

plot_series = result.forecast["series_id"].iloc[0]
plot_target = result.forecast["target_name"].iloc[0]
plot_forecast = result.forecast[(result.forecast["series_id"] == plot_series) & (result.forecast["target_name"] == plot_target)].sort_values("timestamp")
plot_truth = split.truth[split.truth[config.id_column] == plot_series].sort_values(config.timestamp_column)

values = plot_forecast["q0.1"].tolist() + plot_forecast["prediction"].tolist() + plot_forecast["q0.9"].tolist() + plot_truth[plot_target].tolist()
low, high = min(values), max(values)
span = high - low or 1.0
width, height = 760, 280
left, right, top, bottom = 48, width - 20, 30, height - 38


def points(series):
    n_steps = max(len(series) - 1, 1)
    return " ".join(f"{{left + (right - left) * i / n_steps:.1f}},{{bottom - (bottom - top) * (float(v) - low) / span:.1f}}" for i, v in enumerate(series))


layers = [("q0.1", plot_forecast["q0.1"].tolist()), ("median", plot_forecast["prediction"].tolist()), ("q0.9", plot_forecast["q0.9"].tolist()), ("truth", plot_truth[plot_target].tolist())]
strokes = ["#111827", "#2563eb", "#dc2626", "#059669"]
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{{width}}" height="{{height}}">']
title = f"{{escape(str(plot_series))}} / {{escape(str(plot_target))}}: held-out future"
svg.append(f'<text x="{{left}}" y="18" font-family="sans-serif" font-size="14">{{title}}</text>')
for idx, (label, series) in enumerate(layers):
    stroke = strokes[idx]
    svg.append(f'<polyline fill="none" stroke="{{stroke}}" stroke-width="2" points="{{points(series)}}"/>')
    svg.append(f'<text x="{{left + 120 * idx}}" y="{{height - 10}}" font-family="sans-serif" font-size="12" fill="{{stroke}}">{{escape(label)}}</text>')
svg.append("</svg>")

svg_path = Path("outputs") / "chronos_forecast.svg"
svg_path.write_text("\n".join(svg), encoding="utf-8")


class InlineSVG:
    """Shows the SVG markup inline; works in the notebook kernel and in the isolated environment."""

    def __init__(self, markup):
        self.markup = markup

    def _repr_html_(self):
        return self.markup


try:
    display(InlineSVG(svg_path.read_text(encoding="utf-8")))
except NameError:  # a plain-Python executor has no display()
    print("Forecast SVG written to", svg_path)'''
            ),
        },
        {
            "md": (
                '## 9. Primary Mode C — multi-target forecasting\n'
                '\n'
                "Multi-target forecasting is a primary user-facing capability, so this path runs by default. For interface demonstration only, a second target `<target>_aux` (for the sample, `target_aux`) is deterministically derived as 0.5 × target + 10; it is **not an independent benchmark variable**. If your CSV already has a column with that name, the derived target is named with `_derived` appended and your column is left unchanged. The assertion proves the normalized output preserves both target names, and the cell prints both medians side by side.\n"
                '\n'
                '**Predict first.** How many forecast rows will Mode C return, and should the `target_aux` median be exactly 0.5 × the `target` median + 10?'
            ),
            "code": (
                r'''aux_target = f"{{config.target}}_aux"
while aux_target in frame.columns:
    aux_target += "_derived"
if aux_target != f"{{config.target}}_aux":
    print(f"The data already has a column named {{config.target}}_aux; the derived target is named {{aux_target}} and your column is left unchanged.")
mode_c_frame = frame.copy()
mode_c_frame[aux_target] = 0.5 * pd.to_numeric(mode_c_frame[config.target]) + 10.0
mode_c_config = ForecastConfig(id_column=ID_COLUMN, timestamp_column=TIMESTAMP_COLUMN, target=[config.target, aux_target], prediction_length=PREDICTION_LENGTH, quantile_levels=[0.1, 0.5, 0.9])
mode_c_split = chronological_holdout(mode_c_frame, mode_c_config)
mode_c_result = forecast(mode_c_split.history, mode_c_config, pipe)
observed_targets = set(mode_c_result.forecast["target_name"].unique())
assert observed_targets == {{config.target, aux_target}}
print("Mode C target names:", sorted(observed_targets), "| rows:", len(mode_c_result.forecast))
mode_c_wide = mode_c_result.forecast.pivot_table(index=["series_id", "timestamp"], columns="target_name", values="prediction")
mode_c_wide["0.5 x target + 10"] = 0.5 * mode_c_wide[config.target] + 10.0
print(mode_c_wide.head().to_string())
print("largest |aux median - (0.5 x target median + 10)|:", round(float((mode_c_wide[aux_target] - mode_c_wide["0.5 x target + 10"]).abs().max()), 4))'''
            ),
        },
        {
            "md": (
                '#### What to notice (Section 9)\n'
                '\n'
                '- One call returns one block of rows per target, distinguished by `target_name`.\n'
                '- The model is not told that the second target is a linear function of the first, yet the last printed line shows how closely its medians follow 0.5 × target + 10.\n'
                '- Because `target_aux` carries no information that `target` does not, this run shows the multi-target interface working, not multivariate forecasting skill.\n'
                '\n'
                '<details><summary>Sample answer (default sample)</summary>\n'
                '\n'
                '24 rows: 12 steps for `target` and 12 for `target_aux`. The `target_aux` medians equal 0.5 × `target` median + 10 to rounding (largest gap 0.0 at four decimals). Chronos-2 scales each target by its own history before forecasting, so an exact rescaled copy of a series looks identical to the model and gets the same forecast, rescaled back. With a second target that carries its own information, the two forecasts would differ.\n'
                '\n'
                '</details>'
            ),
        },
        {
            "md": (
                '## 10. Export outputs and provenance\n'
                '\n'
                "Exports contain the normalized univariate forecast (`outputs/chronos_forecast.csv`), the Mode C multi-target forecast, the per-series evaluation, the aggregate/quantile metrics (`outputs/chronos_evaluation.json`), the pipeline provenance (`outputs/chronos_provenance.json`), and `outputs/{stem}_result.json` with the input manifest, the evaluation report, the sample identity and digest, the notebook's source (repository, revision, embedded module digests, generator), the model identifier, the immutable model revision and licence, the runtime identity (Python, `torch`, `transformers`, `pandas`, `numpy`, device, dtype) and `learner_runs`, which Sections 11–13 fill in (an optional section that is off records `run: false` and removes its files from an earlier run). A BYOD SHA-256 is metadata derived from uploaded bytes and should be retained/disclosed under your data-governance rules. No credentials are recorded."
            ),
            "code": (
                r'''result.forecast.to_csv("outputs/chronos_forecast.csv", index=False)
mode_c_result.forecast.to_csv("outputs/chronos_multitarget_forecast.csv", index=False)
evaluation.per_series.to_csv("outputs/chronos_evaluation_per_series.csv", index=False)

provenance = {{
    **result.provenance,
    "tutorial": {{
        "notebook_profile": "TASK-INFERENCE",
        "notebook_spec": NOTEBOOK_SOURCE["notebook_spec"],
        "notebook_source": NOTEBOOK_SOURCE,
        "input_source": input_source,
        "input_sha256": input_sha256,
        "primary_capability_checks": ["univariate", "multi-target"],
    }},
}}
with open("outputs/chronos_provenance.json", "w", encoding="utf-8") as handle:
    json.dump(provenance, handle, indent=2, default=str)

metrics = {{
    "evidence_scope": "tutorial/sanity; not benchmark or production-fitness evidence",
    "estimation_procedure": "single chronological tail holdout",
    "chronos2": evaluation.aggregate,
    "last_value": last_value_evaluation.aggregate,
    "seasonal_naive_24": seasonal_metrics,
    "quantiles": evaluation.quantiles.to_dict("records"),
}}
with open("outputs/chronos_evaluation.json", "w", encoding="utf-8") as handle:
    json.dump(metrics, handle, indent=2, default=str)

# Sections 11-13 record themselves here; a re-run of this cell keeps their latest records.
LEARNER_RUNS = globals().get("LEARNER_RUNS") or {{"horizon_activity": {{"run": False}}, "mode_d": {{"run": False}}, "future_forecast": {{"run": False}}}}
RESULT_PATH = Path("outputs/{stem}_result.json")


def record_learner_run(name, record):
    """Store a Section 11-13 record in LEARNER_RUNS and in the exported result JSON."""
    LEARNER_RUNS[name] = record
    if RESULT_PATH.exists():
        data = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        data["learner_runs"] = LEARNER_RUNS
        RESULT_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def remove_outputs(paths):
    """Delete files an optional section wrote in an earlier run, so outputs/ matches this run."""
    removed = [str(path) for path in map(Path, paths) if path.exists()]
    for path in removed:
        Path(path).unlink()
    return removed


payload_out = {{
    "forecast_rows": result.forecast.to_dict("records"),
    "evaluation_report": report,
    "input_manifest": input_manifest,
    "sample": {{"kind": sample_kind, "source": input_source, "sha256": input_sha256, "prediction_length": PREDICTION_LENGTH, "columns": {{"id": ID_COLUMN, "timestamp": TIMESTAMP_COLUMN, "target": TARGET_COLUMN}}}},
    'notebook_source': NOTEBOOK_SOURCE,
    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],
    "model_id": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "model_license": MODEL_LICENSE,
    "runtime": {{
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "pandas": pandas.__version__,
        "numpy": numpy.__version__,
        "device": pipe.device,
        "dtype": pipe.dtype,
    }},
    "learner_runs": LEARNER_RUNS,
}}
with RESULT_PATH.open("w", encoding="utf-8") as handle:
    json.dump(payload_out, handle, indent=2, ensure_ascii=False, default=str)
print(sorted(os.listdir("outputs")))'''
            ),
        },
        {
            "md": (
                '## 11. Activity: change the horizon, keep the context\n'
                '\n'
                "Section 5 showed that `PREDICTION_LENGTH` moves the cut-off: a longer horizon also means less history. This activity separates the two. It cuts every series once, at the point that leaves room for the largest horizon in `HORIZONS`, gives the model exactly `FIXED_CONTEXT` steps of history (`ForecastConfig(context_length=...)`), and forecasts each horizon from that same cut-off, scoring each against the first steps of the same held-out window. Only the horizon changes from row to row. It runs by default, then you repeat it:\n"
                '\n'
                '1. **Predict:** as the horizon grows from 12 to 48 steps with the same 48 steps of context, will Chronos-2\'s MAE rise, fall or stay flat? And the last-value baseline\'s?\n'
                '2. **Change one thing:** edit `HORIZONS` (for example `"6, 12, 24, 36, 48"`). Leave `FIXED_CONTEXT` as it is.\n'
                '3. **Run:** run **only this cell**; nothing before it needs to run again.\n'
                '4. **Observe:** `effective_context_length` is the same in every row; compare the MAE columns as the horizon grows.\n'
                '5. **Explain:** in one sentence, why does each forecaster behave as it does? Then compare with the sample answer below.\n'
                '\n'
                'On the 96-step sample, `FIXED_CONTEXT` plus the largest horizon must be at most 96 (for example 48 + 48). If your settings or a shorter BYOD series do not fit, the cell prints "Activity skipped" with the largest horizon that fits, and the rest of the notebook carries on. Its table is written to `outputs/chronos_horizon_activity.csv`.'
            ),
            "code": (
                r'''HORIZONS = "12, 24, 48"  # @param {{type:"string"}}
FIXED_CONTEXT = 48  # @param {{type:"integer"}}

horizons = sorted({{int(value) for value in HORIZONS.replace(",", " ").split()}})
if not horizons or horizons[0] < 1:
    raise ValueError(f"HORIZONS must list positive whole numbers such as '12, 24, 48', got {{HORIZONS!r}}.")
shortest = int(frame.groupby(ID_COLUMN, sort=False).size().min())
activity_rows = []
if FIXED_CONTEXT < MIN_OBSERVATIONS or FIXED_CONTEXT + horizons[-1] > shortest:
    # Not an error: a short BYOD series must not stop Run all here. The message names settings that fit.
    activity_skipped = f"FIXED_CONTEXT={{FIXED_CONTEXT}} plus the largest horizon {{horizons[-1]}} needs {{FIXED_CONTEXT + horizons[-1]}} rows per series, but the shortest series has {{shortest}} (and FIXED_CONTEXT must be at least {{MIN_OBSERVATIONS}}). Use horizons up to {{max(shortest - FIXED_CONTEXT, 0)}}, or lower FIXED_CONTEXT, and run this cell again."
    print("Activity skipped:", activity_skipped)
    record_learner_run("horizon_activity", {{"run": False, "reason": activity_skipped}})
    horizons = []
else:
    activity_config = ForecastConfig(id_column=ID_COLUMN, timestamp_column=TIMESTAMP_COLUMN, target=TARGET_COLUMN, prediction_length=horizons[-1], quantile_levels=[0.1, 0.5, 0.9])
    activity_split = chronological_holdout(frame, activity_config, horizon=horizons[-1])
for horizon in horizons:
    horizon_config = ForecastConfig(id_column=ID_COLUMN, timestamp_column=TIMESTAMP_COLUMN, target=TARGET_COLUMN, prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9], context_length=FIXED_CONTEXT)
    horizon_truth = activity_split.truth.groupby(ID_COLUMN, sort=False).head(horizon)
    horizon_result = forecast(activity_split.history, horizon_config, pipe)
    chronos_metrics = evaluate_forecast(horizon_result.forecast, horizon_truth, horizon_config).aggregate
    last_value_metrics = evaluate_forecast(last_value_baseline(activity_split.history, horizon_truth, horizon_config), horizon_truth, horizon_config).aggregate
    activity_rows.append({{
        "horizon": horizon,
        "effective_context_length": horizon_result.inference["effective_context_length"],
        "chronos2_mae": round(chronos_metrics["mae"], 3),
        "last_value_mae": round(last_value_metrics["mae"], 3),
        "chronos2_coverage": chronos_metrics.get("interval_coverage"),
    }})
if activity_rows:
    activity_table = pd.DataFrame(activity_rows)
    print({{"cut_off": str(activity_split.history[TIMESTAMP_COLUMN].max()), "fixed_context": FIXED_CONTEXT}})
    print(activity_table.to_string(index=False))
    activity_table.to_csv("outputs/chronos_horizon_activity.csv", index=False)
    record_learner_run("horizon_activity", {{"run": True, "horizons": horizons, "fixed_context": FIXED_CONTEXT, "table": "outputs/chronos_horizon_activity.csv", "rows": activity_rows}})
else:
    remove_outputs(["outputs/chronos_horizon_activity.csv"])'''
            ),
        },
        {
            "md": (
                '#### What to notice (Section 11)\n'
                '\n'
                '- The `effective_context_length` column holds one value: the context really is fixed, so the differences between rows come from the horizon alone.\n'
                '- Compare this with changing `PREDICTION_LENGTH` in Section 4, where a longer horizon also shortened the history; that experiment changes two things at once.\n'
                '\n'
                '<details><summary>Sample answer (default sample, HORIZONS "12, 24, 48", FIXED_CONTEXT 48)</summary>\n'
                '\n'
                'Every row reports a context of 48 steps (local CPU run). Chronos-2\'s MAE rises with the horizon — 0.27 at 12 steps, 0.57 at 24 and 2.69 at 48 — because small errors in the extrapolated trend and cycles add up the further ahead it forecasts; it stays well below last-value. Last-value goes up and down instead (6.57, 4.67, 7.15): it repeats one number, so its error depends on which part of the daily cycle the window covers (a full 24-step day averages out part of the swing) as well as on the trend. With `FIXED_CONTEXT = 24` and the same horizons, Chronos-2\'s MAE is 3.50, 4.65 and 7.41 and its coverage falls below 1.0: with only one day of history it cannot see the trend clearly, so less context hurts at every horizon.\n'
                '\n'
                '</details>'
            ),
        },
        {
            "md": (
                '## 12. Optional: Mode D — known-future covariates, with and without\n'
                '\n'
                "Tick `RUN_COVARIATE_DEMO` (automation: set `DIMER_RUN_COVARIATE_DEMO=1`) and run **only this cell**. The deterministic two-series demand table (the repository's `chronos_covariates_history.csv` / `chronos_covariates_future.csv` formulas, built in memory) carries `temperature` and `holiday` as covariates; the future table contains only those two columns for the 24-step horizon, never future `demand`, so a target column there would be refused as leakage. The same formula gives the true future demand, so the cell scores two forecasts from the same history against it: one **with** the known-future covariates and one **without** any covariates (the demand history alone). It writes both forecasts and `outputs/chronos_covariate_provenance.json`, which names the covariates read as known-future and the comparison, and records it in the result JSON. With the box unticked, the cell removes Mode D files left by an earlier run.\n"
                '\n'
                '**Predict first:** the demand formula adds 1.8 × temperature and 10 × holiday. Will knowing the future temperature and holiday flags lower the forecast error?'
            ),
            "code": (
                r'''RUN_COVARIATE_DEMO = False  # @param {{type:"boolean"}}
if os.environ.get("DIMER_RUN_COVARIATE_DEMO") == "1":
    RUN_COVARIATE_DEMO = True
MODE_D_FILES = ("outputs/chronos_covariate_forecast.csv", "outputs/chronos_covariate_no_covariates_forecast.csv", "outputs/chronos_covariate_provenance.json")

if RUN_COVARIATE_DEMO:
    n, horizon = 96, 24
    step = np.arange(n, dtype=float)
    future_step = np.arange(n, n + horizon, dtype=float)
    timestamps = pd.date_range("2026-01-01", periods=n, freq="h")
    future_timestamps = pd.date_range(timestamps[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")
    history_parts, future_parts, truth_parts = [], [], []
    for series_id, base, phase in (("A", 80.0, 0.0), ("B", 110.0, 4.0)):
        temperature = 27.0 + 4.0 * np.sin(2.0 * np.pi * (step + phase) / 24.0)
        holiday = (pd.Series(timestamps).dt.dayofweek >= 5).astype(int).to_numpy()
        demand = base + 1.8 * temperature + 10.0 * holiday + 0.08 * step + 3.0 * np.sin(2.0 * np.pi * step / 12.0)
        history_parts.append(pd.DataFrame({{"series_id": series_id, "timestamp": timestamps, "demand": demand, "temperature": temperature, "holiday": holiday}}))
        future_temperature = 27.0 + 4.0 * np.sin(2.0 * np.pi * (future_step + phase) / 24.0)
        future_holiday = (pd.Series(future_timestamps).dt.dayofweek >= 5).astype(int).to_numpy()
        future_parts.append(pd.DataFrame({{"series_id": series_id, "timestamp": future_timestamps, "temperature": future_temperature, "holiday": future_holiday}}))
        # The same formula on the future steps: the truth both forecasts are scored against (never shown to the model).
        future_demand = base + 1.8 * future_temperature + 10.0 * future_holiday + 0.08 * future_step + 3.0 * np.sin(2.0 * np.pi * future_step / 12.0)
        truth_parts.append(pd.DataFrame({{"series_id": series_id, "timestamp": future_timestamps, "demand": future_demand}}))
    cov_history = pd.concat(history_parts, ignore_index=True)
    cov_future = pd.concat(future_parts, ignore_index=True)
    cov_truth = pd.concat(truth_parts, ignore_index=True)
    assert "demand" not in cov_future.columns
    cov_config = ForecastConfig(target="demand", prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9])
    cov_manifest = validate_inputs(cov_history, cov_config, pipe, future_df=cov_future)
    cov_result = forecast(cov_history, cov_config, pipe, cov_future)
    plain_result = forecast(cov_history[["series_id", "timestamp", "demand"]], cov_config, pipe)
    known_future = cov_result.inference["known_future_covariate_names"]
    print("known-future covariates:", known_future, "| manifest:", cov_manifest["known_future_covariate_names"])
    comparison = [
        {{"run": "with known-future covariates", **evaluate_forecast(cov_result.forecast, cov_truth, cov_config).aggregate}},
        {{"run": "without covariates", **evaluate_forecast(plain_result.forecast, cov_truth, cov_config).aggregate}},
    ]
    print(pd.DataFrame(comparison).to_string(index=False))
    cov_result.forecast.to_csv("outputs/chronos_covariate_forecast.csv", index=False)
    plain_result.forecast.to_csv("outputs/chronos_covariate_no_covariates_forecast.csv", index=False)
    mode_d_provenance = {{
        "request": {{"target": "demand", "prediction_length": horizon, "series": ["A", "B"], "history_rows": len(cov_history), "future_columns": list(cov_future.columns)}},
        "known_future_covariate_names": known_future,
        "past_covariate_names": cov_result.inference["past_covariate_names"],
        "truth_source": "the same deterministic formula evaluated on the future steps; never passed to the model",
        "comparison": comparison,
        "with_covariates": cov_result.provenance,
        "without_covariates": plain_result.provenance,
    }}
    with open("outputs/chronos_covariate_provenance.json", "w", encoding="utf-8") as handle:
        json.dump(mode_d_provenance, handle, indent=2, default=str)
    record_learner_run("mode_d", {{"run": True, "provenance": "outputs/chronos_covariate_provenance.json", "known_future_covariate_names": known_future, "comparison": comparison}})
else:
    removed = remove_outputs(MODE_D_FILES)
    record_learner_run("mode_d", {{"run": False}})
    print("Mode D demo skipped." + (f" Removed files from an earlier run: {{removed}}" if removed else ""))'''
            ),
        },
        {
            "md": (
                '#### What to notice (Section 12, when you ran it)\n'
                '\n'
                '- `known-future covariates` lists `temperature` and `holiday`; without the future table they could only be past covariates.\n'
                '- Both rows are scored on the same 48 future (series, timestamp) pairs, so their MAE values are directly comparable.\n'
                '\n'
                '<details><summary>Sample answer</summary>\n'
                '\n'
                'Yes, by a lot here (local CPU run): MAE 2.46 with the known-future covariates against 9.60 without them, and interval coverage 0.67 against 0.13. The history ends on a weekend (`holiday = 1`, which adds 10 to demand) and the 24 future hours are a Monday (`holiday = 0`); only the run that is told the future holiday flag expects demand to drop. On real data the gain depends on how strongly, and how reliably, the covariate drives the target, and known-future values must really be known in advance.\n'
                '\n'
                '</details>'
            ),
        },
        {
            "md": (
                '## 13. Optional: forecast beyond the end of your data\n'
                '\n'
                "Every section so far holds out the last steps, so the forecast can be scored. To forecast the real future of your own series — the hours after the last row of your CSV — tick `FORECAST_FUTURE` (automation: set `DIMER_FORECAST_FUTURE=1`) and run **only this cell**. It calls `forecast(frame, config, pipe)` on the **full** history, so the forecast starts one step after the last input timestamp. Nobody knows those values yet, so `evaluation_report(..., None, ...)` returns the verdict `not-measurable` and says what data would make it measurable. It writes `outputs/chronos_future_forecast.csv` and `outputs/chronos_future_evaluation_report.json`; with the box unticked, the cell removes them if an earlier run left them."
            ),
            "code": (
                r'''FORECAST_FUTURE = False  # @param {{type:"boolean"}}
if os.environ.get("DIMER_FORECAST_FUTURE") == "1":
    FORECAST_FUTURE = True
FUTURE_FILES = ("outputs/chronos_future_forecast.csv", "outputs/chronos_future_evaluation_report.json")

if FORECAST_FUTURE:
    future_result = forecast(frame, config, pipe)
    future_report = evaluation_report(future_result, None, config=config, sample_kind=sample_kind)
    last_input = pd.to_datetime(frame[config.timestamp_column]).groupby(frame[config.id_column].astype(str)).max()
    first_forecast = pd.to_datetime(future_result.forecast["timestamp"]).groupby(future_result.forecast["series_id"].astype(str)).min()
    print(pd.DataFrame({{"last_input_timestamp": last_input, "first_forecast_timestamp": first_forecast}}).to_string())
    print({{"verdict": future_report["verdict"], "needs": future_report["needs"]}})
    future_result.forecast.to_csv("outputs/chronos_future_forecast.csv", index=False)
    with open("outputs/chronos_future_evaluation_report.json", "w", encoding="utf-8") as handle:
        json.dump(future_report, handle, indent=2, ensure_ascii=False, default=str)
    record_learner_run("future_forecast", {{"run": True, "forecast": "outputs/chronos_future_forecast.csv", "report": "outputs/chronos_future_evaluation_report.json", "verdict": future_report["verdict"]}})
else:
    removed = remove_outputs(FUTURE_FILES)
    record_learner_run("future_forecast", {{"run": False}})
    print("Future forecast skipped." + (f" Removed files from an earlier run: {{removed}}" if removed else ""))'''
            ),
        },
        {
            "md": (
                '## Troubleshooting\n'
                '\n'
                '| What you see | What it means | What to do |\n'
                '|---|---|---|\n'
                '| Section 1 stops with "This notebook needs a Linux x86_64 runtime" | the locked environment holds manylinux x86_64 wheels only | use Google Colab, Kaggle or a Linux x86_64 Jupyter |\n'
                '| Section 1 stops with "failed its size/SHA-256 check" or a download error | the `uv` wheel or a locked package did not arrive intact | run the cell again; if it repeats, the network is altering downloads |\n'
                '| "The isolated environment\'s Python process exited" | usually the runtime ran out of memory | restart the session and choose **Run all** |\n'
                '| Section 3 raises a size or digest mismatch, or a Hub download error | a checkpoint file is incomplete or the Hub was unreachable | run Section 3 again; it fetches only the missing files and re-hashes all of them |\n'
                '| "The CSV has no column named …" | your column names differ from the fields | set `ID_COLUMN`, `TIMESTAMP_COLUMN`, `TARGET_COLUMN` in Section 4 and run from Section 4 |\n'
                '| "The CSV must be UTF-8 encoded" | the file is UTF-16 or a legacy code page | save it again as CSV UTF-8 |\n'
                '| `EVALUATION_SERIES_TOO_SHORT` in Section 5 | a series is shorter than `PREDICTION_LENGTH` + 3 rows | lower `PREDICTION_LENGTH` (at most 93 on the sample) and run from Section 4 |\n'
                '| `SERIES_GAP` or `CALENDAR_FREQUENCY_UNSUPPORTED` | missing timestamps, or a monthly/business-day calendar | fill or resample the gaps; only fixed-width frequencies are supported |\n'
                '| Section 11 prints "Activity skipped" | `FIXED_CONTEXT` + the largest horizon exceeds the shortest series | use the largest horizon the message names, or lower `FIXED_CONTEXT`, and run Section 11 again |\n'
                '\n'
                'Re-running a single cell is safe: it runs in the isolated environment with the variables created so far. To start over, restart the session and choose **Run all**.'
            ),
        },
    ],
    "closing": (
        '## Interpretation and limits\n'
        '\n'
        "A successful default run **proves** that the recorded repository revision's package, carried in this notebook, can install the pinned runtime, acquire and digest-verify the pinned model, preserve the chronological evaluation boundary, validate the demonstrated input, execute the public API for univariate and primary multi-target forecasting, compute the documented tutorial metrics/baselines, and export machine-readable forecasts/provenance in the tested runtime — without the repository being reachable — and no more.\n"
        '\n'
        'It **does not prove** accuracy, calibration, robustness, fairness, safety, or production readiness for your domain. It does **not** establish benchmark superiority, deployment calibration, safety for high-consequence decisions, or production fitness on an unseen domain. The synthetic sample is intentionally simple; its metrics are not benchmark evidence and the evaluation report says `sample-sanity` for that reason. `prediction` is the median of an uncalibrated quantile forecast; the requested quantiles are not calibrated intervals and the pipeline ships no threshold. BYOD requires representative multi-origin backtesting, leakage controls, domain baselines, calibration checks, and operational review.\n'
        '\n'
        'Current limits include fixed-width regular frequencies; monthly/quarterly/yearly/business-day, irregular, and gappy calendars are outside this path, and the pipeline provides no classification, anomaly-detection, imputation, embedding, or training capability.\n'
        '\n'
        '**Next experiments** (each one changes one thing; the cells to run are named):\n'
        '\n'
        '1. **Your own series:** in Section 4 set `BYOD_PATH` (or tick `USE_BYOD`) and, if needed, the column fields, then run Section 4 and every cell after it. Compare the report\'s Chronos-2 MAE with the `last_value_baseline` and `seasonal_naive_baseline` entries; expect coverage below 1.0 on noisy data.\n'
        '2. **Horizon at a fixed context:** in Section 11 change `HORIZONS` (for example `"6, 12, 24, 36, 48"`) and run only Section 11. To see the effect of context instead, keep `HORIZONS` and change `FIXED_CONTEXT` (for example 24, then 48).\n'
        '3. **Known-future covariates:** tick `RUN_COVARIATE_DEMO` in Section 12 and run only Section 12; it prints the MAE with and without the future covariate table.\n'
        '4. **The real future:** tick `FORECAST_FUTURE` in Section 13 and run only Section 13; the forecast starts after your last row and the report verdict is `not-measurable`.\n'
        '\n'
        '## Conclusion template\n'
        '\n'
        'Copy this into your notes and fill in the blanks from your own run:\n'
        '\n'
        '> On **[sample / my data: ___]** with a **[___]**-step horizon and **[___]** steps of context, Chronos-2 had MAE **[___]** against **[___]** for last-value and **[___ / skipped]** for seasonal-naive, with interval coverage **[___]**. The evaluation verdict was **[sample-sanity / not-measurable]**. In the Section 11 activity, MAE went from **[___]** to **[___]** as the horizon grew from **[___]** to **[___]** steps at a fixed context. I would / would not use these numbers to decide **[___]**, because **[one tail split, no noise, …]**. Before relying on it I would need **[more forecast windows / calibration checks / domain baselines]**.\n'
        '\n'
        '## References\n'
        '\n'
        '- Repository README: https://github.com/kurtvalcorza/chronos-2-forecasting-pipeline/blob/main/README.md\n'
        '- Repository model card: https://github.com/kurtvalcorza/chronos-2-forecasting-pipeline/blob/main/MODEL_CARD.md\n'
        '- Sample dataset card: https://github.com/kurtvalcorza/chronos-2-forecasting-pipeline/blob/main/examples/sample-data/DATASET_CARD.md\n'
        '- Upstream model: https://huggingface.co/{MODEL_ID}\n'
        '- Upstream library: https://github.com/amazon-science/chronos-forecasting\n'
        '- Chronos-2: From Univariate to Universal Forecasting: https://arxiv.org/abs/2510.15821'
    ),
}
