"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 1.1 §3.6 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded package
(seven modules, carried verbatim in dependency order), and the model pin/stage/verify cells are
produced by the generator from repository sources so they cannot drift from the package.

Generator /2 keys in use: ``modules`` lists every module of ``src/chronos2_pipeline/`` except
``__init__.py``; ``entry_module`` is ``model.py`` (it holds ``MODEL_ID``/``MODEL_REVISION``/
``MODEL_LICENSE``/``MODEL_KEY``); ``model_load`` is the package's documented loader
``load_pinned_model(weights_dir=WEIGHTS_DIR)`` (there is no ``from_pretrained`` class method in this
package). The default rewrite rule applies: the one ``__file__`` use is ``DEFAULT_WEIGHTS_DIR``.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    'package': 'chronos2_pipeline',
    'repo_name': 'chronos-2-forecasting-pipeline',
    'stem': 'chronos_2_forecasting',
    'notebook_name': 'chronos_2_forecasting_colab.ipynb',
    'profile': 'TASK-INFERENCE',
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
    'intro': (
        'Chronos-2 supplies the pretrained zero-shot forecasting model. This repository supplies DIMER configuration and validation, immutable model pinning/digest checks, normalized outputs, chronological evaluation, baselines, and provenance — all of it carried in this notebook. **No training or fine-tuning occurs**, and **no adaptation occurs:** inference is in-context only, the pinned checkpoint is used as published, and no preprocessing is fitted. The default sample is deterministic synthetic teaching data generated in code (the same bytes the repository checks in as `examples/sample-data/chronos_univariate.csv`); its metrics are tutorial/sanity evidence, not a benchmark claim.'
    ),
    'learning_objectives': (
        'install the pinned runtime; read what the carried package guarantees; resolve and digest-verify the immutable `amazon/chronos-2` revision; generate the synthetic sample or bring your own CSV; create a leakage-safe chronological holdout and validate the history into an input manifest; run univariate and **multi-target (Mode C)** zero-shot forecasts; interpret median/quantile outputs and the tutorial metrics against naive baselines through an evaluation report; optionally run known-future covariates (Mode D); and export forecasts plus provenance. By the end you can do each of these without the repository being reachable.'
    ),
    'exclusions': (
        'classification, anomaly detection, imputation, embeddings, training/fine-tuning, calibrated prediction intervals, and production-fitness claims. Out of scope: monthly/quarterly/yearly/business-day, irregular, and gappy calendars — only fixed-width regular frequencies are accepted.'
    ),
    'prerequisites': [
        '- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). CPU is the default path; CUDA is used automatically when available. The pinned `torch==2.14.0` install is the largest download of the run, followed by the ~478 MB checkpoint.',
        '- **Knowledge:** basic Python and pandas; what a quantile forecast and a chronological holdout are.',
        '- **Data:** the default sample is a deterministic 96-step hourly series generated in code, so nothing is downloaded and no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one UTF-8 CSV with unique headers including `series_id`, `timestamp`, and `target`; timestamps parseable, regular, and contiguous per series; targets finite numeric; further numeric columns are covariates. BYOD is read locally in the notebook runtime and is not sent to an external inference service. Do not upload confidential or restricted data (personal or otherwise sensitive data included) to a hosted notebook environment unless you are authorized to do so.',
    ],
    "cells": [
        {
            "md": (
                '## 4. Generate the synthetic sample or optional BYOD\n'
                '\n'
                "The default sample is **synthetic**: one hourly series of 96 steps built from a fixed formula (a linear trend plus a 24-step and a 12-step sinusoid), rendered to CSV bytes exactly as the repository's `examples/sample-data/generate_samples.py` writes them, so its SHA-256 is asserted against the digest the repository checks in — the notebook proves it is forecasting the very bytes the repository tests. It is deterministic teaching data, not benchmark evidence. BYOD requires unique UTF-8 CSV headers plus `series_id`, `timestamp`, and `target`; duplicate headers are rejected **before pandas can rename them**. Additional numeric covariates are permitted by the pipeline. Set `USE_BYOD=True` in Colab (upload one CSV), or set `DIMER_BYOD_PATH` in automation. Look for the input source, its SHA-256 and the first rows."
            ),
            "code": (
                'import csv\n'
                'import hashlib\n'
                'import io\n'
                'import json\n'
                'from collections import Counter\n'
                'from pathlib import Path\n'
                '\n'
                'import numpy as np\n'
                'import pandas as pd\n'
                '\n'
                'USE_BYOD = False  # @param {{type:"boolean"}}\n'
                'PREDICTION_LENGTH = 12  # @param {{type:"integer"}}\n'
                'BYOD_PATH = os.environ.get("DIMER_BYOD_PATH")\n'
                'SAMPLE_SHA256 = "eff96b1a9bec5a81aa4021b4d308c29e18fc89be0cf8eb54e755c2efe535c7f7"  # examples/sample-data/SHA256SUMS\n'
                '\n'
                '\n'
                'def read_checked_csv(payload: bytes) -> pd.DataFrame:\n'
                '    rows = csv.reader(io.StringIO(payload.decode("utf-8-sig")))\n'
                '    try:\n'
                '        header = next(rows)\n'
                '    except StopIteration as exc:\n'
                '        raise ValueError("CSV is empty.") from exc\n'
                '    duplicates = sorted(name for name, count in Counter(header).items() if count > 1)\n'
                '    if duplicates:\n'
                '        raise ValueError(f"Duplicate CSV header(s) are ambiguous: {{duplicates}}")\n'
                '    return pd.read_csv(io.BytesIO(payload))\n'
                '\n'
                '\n'
                'def synthetic_univariate_csv() -> bytes:\n'
                "    # The repository's generate_samples.py formula and canonical CSV rendering; no random state.\n"
                '    n = 96\n'
                '    step = np.arange(n, dtype=float)\n'
                '    frame = pd.DataFrame(\n'
                '        {{\n'
                '            "series_id": "A",\n'
                '            "timestamp": pd.date_range("2026-01-01", periods=n, freq="h"),\n'
                '            "target": 100.0 + 0.25 * step + 7.0 * np.sin(2.0 * np.pi * step / 24.0) + 1.5 * np.cos(2.0 * np.pi * step / 12.0),\n'
                '        }}\n'
                '    )\n'
                '    return frame.to_csv(index=False, date_format="%Y-%m-%dT%H:%M:%S", float_format="%.4f", lineterminator="\\n").encode("utf-8")\n'
                '\n'
                '\n'
                'if BYOD_PATH:\n'
                '    payload = Path(BYOD_PATH).read_bytes()\n'
                '    input_source = f"BYOD path: {{BYOD_PATH}}"\n'
                '    sample_kind = "BYOD"\n'
                'elif USE_BYOD:\n'
                '    from google.colab import files\n'
                '    uploaded = files.upload()\n'
                '    if len(uploaded) != 1:\n'
                '        raise ValueError("Upload exactly one CSV.")\n'
                '    name, payload = next(iter(uploaded.items()))\n'
                '    input_source = f"BYOD upload: {{name}}"\n'
                '    sample_kind = "BYOD"\n'
                'else:\n'
                '    payload = synthetic_univariate_csv()\n'
                '    observed = hashlib.sha256(payload).hexdigest()\n'
                '    if observed != SAMPLE_SHA256:\n'
                '        raise ValueError(f"Synthetic sample digest mismatch: {{observed}} != {{SAMPLE_SHA256}}")\n'
                '    input_source = "synthetic sample generated in code (== examples/sample-data/chronos_univariate.csv)"\n'
                '    sample_kind = "synthetic"\n'
                '\n'
                'frame = read_checked_csv(payload)\n'
                'input_sha256 = hashlib.sha256(payload).hexdigest()\n'
                'print({{"sample_kind": sample_kind, "input_source": input_source, "input_sha256": input_sha256, "rows": len(frame), "columns": list(frame.columns)}})\n'
                'print(frame.head())'
            ),
        },
        {
            "md": (
                '## 5. Chronological holdout, ceilings, validate → input manifest\n'
                '\n'
                "The final `PREDICTION_LENGTH` timestamps of each series are held out as truth; future target values never enter model context (chronological holdout — **no random split** or training initialization). Before anything runs, the cell prints the operational ceilings: the pinned model exposes an 8,192-step context and native 1,024-step horizon, and the DIMER request guards (`ResourceLimits`) cap one request at 1,000 series IDs, 64 targets, 64 covariates, 5,000,000 rows, 8,192 context steps, and 4,096 forecast steps; horizons above 1,024 require explicit autoregressive unrolling. `runtime_versions` reports the installed stack. `validate_inputs` is the package's public validation stage: it routes the history through `validate_forecast_request` with exactly the arguments `forecast` passes, so it raises exactly what the forecast call would raise, and returns an **input manifest** naming the schema and ceilings, each series' observed span, the confirmed frequency and the verdict; it is written to `outputs/{stem}_input_manifest.json`. To show what rejection looks like, the cell also validates a deliberately broken copy of the history (one non-finite target) and records the pipeline's own error as a finding."
            ),
            "code": (
                'import os\n'
                '\n'
                "os.makedirs('outputs', exist_ok=True)\n"
                'config = ForecastConfig(target="target", prediction_length=PREDICTION_LENGTH, quantile_levels=[0.1, 0.5, 0.9])\n'
                'split = chronological_holdout(frame, config)\n'
                'for sid in split.history[config.id_column].drop_duplicates():\n'
                '    h = split.history[split.history[config.id_column] == sid]\n'
                '    t = split.truth[split.truth[config.id_column] == sid]\n'
                '    assert h[config.timestamp_column].max() < t[config.timestamp_column].min()\n'
                '\n'
                'versions = runtime_versions()\n'
                'print({{"runtime_versions": versions}})\n'
                'print("DIMER request limits:", ResourceLimits())\n'
                'print("Pinned-model context limit: 8192")\n'
                'print("Pinned-model native prediction length: 1024")\n'
                'print({{"ceilings": {{"min_observations_per_series": MIN_OBSERVATIONS, "max_ids": DEFAULT_LIMITS.max_ids, "max_targets": DEFAULT_LIMITS.max_targets, "max_covariates": DEFAULT_LIMITS.max_covariates, "max_rows": DEFAULT_LIMITS.max_rows, "max_context_length": DEFAULT_LIMITS.max_context_length, "max_prediction_length": DEFAULT_LIMITS.max_prediction_length, "model_context_length": pipe.model_context_length, "model_prediction_length": pipe.model_prediction_length}}}})\n'
                '\n'
                'series_names = [str(sid) for sid in split.history[config.id_column].drop_duplicates()]\n'
                'input_manifest = validate_inputs(split.history, config, pipe, names=series_names)\n'
                '# Demonstrate rejection on a history that breaks a rule; the finding is recorded, not swallowed.\n'
                'broken = split.history.copy()\n'
                'broken.loc[broken.index[0], "target"] = float("nan")\n'
                'try:\n'
                '    validate_inputs(broken, config, pipe)\n'
                'except ValidationError as exc:\n'
                '    input_manifest["findings"].append({{"input": "non-finite-target-probe", "verdict": "rejected", "code": exc.code, "message": str(exc)}})\n'
                'with open("outputs/{stem}_input_manifest.json", "w", encoding="utf-8") as handle:\n'
                '    json.dump(input_manifest, handle, indent=2, ensure_ascii=False, default=str)\n'
                'print(json.dumps(input_manifest, indent=2, default=str))'
            ),
        },
        {
            "md": (
                '## 6. Forecast\n'
                '\n'
                "`forecast(split.history, config, pipe)` is the package's public inference path: it validates again, calls the pinned `chronos-forecasting` package's `predict_df` on the verified snapshot (it does not request model-repository remote code), and normalizes the output. Normalized output fields are `series_id`, `timestamp`, `target_name`, `prediction`, and requested quantiles such as `q0.1`, `q0.5`, `q0.9`. `prediction` is the **median (`q0.5`)**, not a mean. Model quantiles summarize the predictive distribution; they are **not guaranteed frequentist confidence intervals** and are not assumed calibrated on a new domain — the pipeline ships no threshold and no calibration. The effective context and prediction lengths actually used are printed with the first forecast rows; floating-point details can vary across runtime/hardware builds and latency is run-dependent."
            ),
            "code": (
                'result = forecast(split.history, config, pipe)\n'
                'print(result.forecast.head())\n'
                'print("effective_context_length:", result.inference["effective_context_length"])\n'
                'print("effective_prediction_length:", result.inference["effective_prediction_length"])\n'
                'print({{"n_forecast_rows": len(result.forecast), "quantile_columns": [c for c in result.forecast.columns if c.startswith("q")], "latency_seconds": result.inference.get("latency_seconds")}})'
            ),
        },
        {
            "md": (
                '## 7. Evaluate → evaluation report\n'
                '\n'
                "Metrics use the same chronological holdout: **MAE** is mean absolute error; **RMSE** weights larger misses more; **pinball loss** evaluates a quantile asymmetrically; **empirical interval coverage** is the fraction of held-out truths inside the requested outer quantiles. These are tutorial/sanity metrics, not benchmark evidence. `evaluation_report` is the package's public evaluation stage and always produces a report: with the held-out truth it carries the `evaluate_forecast` metrics for Chronos-2 and for the `last_value_baseline` (and the `seasonal_naive_baseline` when applicable) with the verdict `sample-sanity` — one chronological tail split of one sample, no dispersion estimate; without truth (a forecast of the real future) the verdict is `not-measurable` and the report states what data would make the task measurable. The seasonal-naive comparator runs only for uniformly hourly data when **every series has at least 24 post-holdout history rows**; short but otherwise valid hourly BYOD therefore skips this optional comparator instead of failing. The report is written to `outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                'evaluation = evaluate_forecast(result.forecast, split.truth, config)\n'
                'last_value = last_value_baseline(split.history, split.truth, config)\n'
                'last_value_evaluation = evaluate_forecast(last_value, split.truth, config)\n'
                '\n'
                'ordered = split.history.sort_values([config.id_column, config.timestamp_column])\n'
                'steps = ordered.groupby(config.id_column)[config.timestamp_column].diff().dropna()\n'
                'minimum_history = int(ordered.groupby(config.id_column, sort=False).size().min())\n'
                'can_use_daily_seasonal = not steps.empty and (steps == pd.Timedelta(hours=1)).all() and minimum_history >= 24\n'
                '\n'
                'seasonal_evaluation = None\n'
                'if can_use_daily_seasonal:\n'
                '    seasonal = seasonal_naive_baseline(split.history, split.truth, config, season_length=24)\n'
                '    seasonal_evaluation = evaluate_forecast(seasonal, split.truth, config)\n'
                'seasonal_metrics = None if seasonal_evaluation is None else seasonal_evaluation.aggregate\n'
                '\n'
                'print("Chronos-2:", evaluation.aggregate)\n'
                'print("Last-value:", last_value_evaluation.aggregate)\n'
                'print("Seasonal-naive:", seasonal_metrics)\n'
                'print("Quantile metrics:")\n'
                'print(evaluation.quantiles)\n'
                'print("Per-series metrics:")\n'
                'print(evaluation.per_series)\n'
                '\n'
                'report = evaluation_report(result, split.truth, config=config, history_df=split.history, season_length=24 if can_use_daily_seasonal else None, sample_kind=sample_kind)\n'
                'with open("outputs/{stem}_evaluation_report.json", "w", encoding="utf-8") as handle:\n'
                '    json.dump(report, handle, indent=2, ensure_ascii=False, default=str)\n'
                'print(json.dumps(report, indent=2, default=str))\n'
                'if report["verdict"] == "not-measurable":\n'
                '    print("No held-out truth was supplied, so evaluate_forecast is not computed; the forecast above is sanity evidence only.")'
            ),
        },
        {
            "md": (
                '## 8. Visualize the held-out forecast\n'
                '\n'
                'For multi-series BYOD, this compact SVG deliberately visualizes one series and one target instead of interleaving unrelated lines. The machine-readable exports remain authoritative for all rows.'
            ),
            "code": (
                'from html import escape\n'
                '\n'
                'plot_series = result.forecast["series_id"].iloc[0]\n'
                'plot_target = result.forecast["target_name"].iloc[0]\n'
                'plot_forecast = result.forecast[(result.forecast["series_id"] == plot_series) & (result.forecast["target_name"] == plot_target)].sort_values("timestamp")\n'
                'plot_truth = split.truth[split.truth[config.id_column] == plot_series].sort_values(config.timestamp_column)\n'
                '\n'
                'values = plot_forecast["q0.1"].tolist() + plot_forecast["prediction"].tolist() + plot_forecast["q0.9"].tolist() + plot_truth[plot_target].tolist()\n'
                'low, high = min(values), max(values)\n'
                'span = high - low or 1.0\n'
                'width, height = 760, 280\n'
                'left, right, top, bottom = 48, width - 20, 30, height - 38\n'
                '\n'
                '\n'
                'def points(series):\n'
                '    n_steps = max(len(series) - 1, 1)\n'
                '    return " ".join(f"{{left + (right - left) * i / n_steps:.1f}},{{bottom - (bottom - top) * (float(v) - low) / span:.1f}}" for i, v in enumerate(series))\n'
                '\n'
                '\n'
                'layers = [("q0.1", plot_forecast["q0.1"].tolist()), ("median", plot_forecast["prediction"].tolist()), ("q0.9", plot_forecast["q0.9"].tolist()), ("truth", plot_truth[plot_target].tolist())]\n'
                'strokes = ["#111827", "#2563eb", "#dc2626", "#059669"]\n'
                'svg = [f\'<svg xmlns="http://www.w3.org/2000/svg" width="{{width}}" height="{{height}}">\']\n'
                'title = f"{{escape(str(plot_series))}} / {{escape(str(plot_target))}}: held-out future"\n'
                'svg.append(f\'<text x="{{left}}" y="18" font-family="sans-serif" font-size="14">{{title}}</text>\')\n'
                'for idx, (label, series) in enumerate(layers):\n'
                '    stroke = strokes[idx]\n'
                '    svg.append(f\'<polyline fill="none" stroke="{{stroke}}" stroke-width="2" points="{{points(series)}}"/>\')\n'
                '    svg.append(f\'<text x="{{left + 120 * idx}}" y="{{height - 10}}" font-family="sans-serif" font-size="12" fill="{{stroke}}">{{escape(label)}}</text>\')\n'
                'svg.append("</svg>")\n'
                '\n'
                'svg_path = Path("outputs") / "chronos_forecast.svg"\n'
                'svg_path.write_text("\\n".join(svg), encoding="utf-8")\n'
                'try:\n'
                '    from IPython.display import SVG, display\n'
                '\n'
                '    display(SVG(filename=str(svg_path)))\n'
                'except ImportError:\n'
                '    print("Forecast SVG written to", svg_path)'
            ),
        },
        {
            "md": (
                '## 9. Primary Mode C — multi-target forecasting\n'
                '\n'
                'Multi-target forecasting is a primary user-facing capability, so this path runs by default. For interface demonstration only, `target_aux` is deterministically derived from `target`; it is **not an independent benchmark variable**. The assertion proves the normalized output preserves both target names.'
            ),
            "code": (
                'mode_c_frame = frame.copy()\n'
                'mode_c_frame["target_aux"] = 0.5 * pd.to_numeric(mode_c_frame["target"]) + 10.0\n'
                'mode_c_config = ForecastConfig(target=["target", "target_aux"], prediction_length=PREDICTION_LENGTH, quantile_levels=[0.1, 0.5, 0.9])\n'
                'mode_c_split = chronological_holdout(mode_c_frame, mode_c_config)\n'
                'mode_c_result = forecast(mode_c_split.history, mode_c_config, pipe)\n'
                'observed_targets = set(mode_c_result.forecast["target_name"].unique())\n'
                'assert observed_targets == {{"target", "target_aux"}}\n'
                'print("Mode C target names:", sorted(observed_targets))'
            ),
        },
        {
            "md": (
                '## 10. Export outputs and provenance\n'
                '\n'
                "Exports contain the normalized univariate forecast (`outputs/chronos_forecast.csv`), the Mode C multi-target forecast, the per-series evaluation, the aggregate/quantile metrics (`outputs/chronos_evaluation.json`), the pipeline provenance (`outputs/chronos_provenance.json`), and `outputs/{stem}_result.json` with the input manifest, the evaluation report, the sample identity and digest, the notebook's source (repository, revision, embedded module digests, generator), the model identifier, the immutable model revision and licence, and the runtime identity (Python, `torch`, `transformers`, `pandas`, `numpy`, device, dtype). A BYOD SHA-256 is metadata derived from uploaded bytes and should be retained/disclosed under your data-governance rules. No credentials are recorded."
            ),
            "code": (
                'result.forecast.to_csv("outputs/chronos_forecast.csv", index=False)\n'
                'mode_c_result.forecast.to_csv("outputs/chronos_multitarget_forecast.csv", index=False)\n'
                'evaluation.per_series.to_csv("outputs/chronos_evaluation_per_series.csv", index=False)\n'
                '\n'
                'provenance = {{\n'
                '    **result.provenance,\n'
                '    "tutorial": {{\n'
                '        "notebook_profile": "TASK-INFERENCE",\n'
                '        "notebook_spec": NOTEBOOK_SOURCE["notebook_spec"],\n'
                '        "notebook_source": NOTEBOOK_SOURCE,\n'
                '        "input_source": input_source,\n'
                '        "input_sha256": input_sha256,\n'
                '        "primary_capability_checks": ["univariate", "multi-target"],\n'
                '    }},\n'
                '}}\n'
                'with open("outputs/chronos_provenance.json", "w", encoding="utf-8") as handle:\n'
                '    json.dump(provenance, handle, indent=2, default=str)\n'
                '\n'
                'metrics = {{\n'
                '    "evidence_scope": "tutorial/sanity; not benchmark or production-fitness evidence",\n'
                '    "estimation_procedure": "single chronological tail holdout",\n'
                '    "chronos2": evaluation.aggregate,\n'
                '    "last_value": last_value_evaluation.aggregate,\n'
                '    "seasonal_naive_24": seasonal_metrics,\n'
                '    "quantiles": evaluation.quantiles.to_dict("records"),\n'
                '}}\n'
                'with open("outputs/chronos_evaluation.json", "w", encoding="utf-8") as handle:\n'
                '    json.dump(metrics, handle, indent=2, default=str)\n'
                '\n'
                'payload_out = {{\n'
                '    "forecast_rows": result.forecast.to_dict("records"),\n'
                '    "evaluation_report": report,\n'
                '    "input_manifest": input_manifest,\n'
                '    "sample": {{"kind": sample_kind, "source": input_source, "sha256": input_sha256, "prediction_length": PREDICTION_LENGTH}},\n'
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                '    "model_id": MODEL_ID,\n'
                '    "model_revision": MODEL_REVISION,\n'
                '    "model_license": MODEL_LICENSE,\n'
                '    "runtime": {{\n'
                '        "python": platform.python_version(),\n'
                '        "torch": torch.__version__,\n'
                '        "transformers": transformers.__version__,\n'
                '        "pandas": pandas.__version__,\n'
                '        "numpy": numpy.__version__,\n'
                '        "device": pipe.device,\n'
                '        "dtype": pipe.dtype,\n'
                '    }},\n'
                '}}\n'
                'with open("outputs/{stem}_result.json", "w", encoding="utf-8") as handle:\n'
                '    json.dump(payload_out, handle, indent=2, ensure_ascii=False, default=str)\n'
                'print(sorted(os.listdir("outputs")))'
            ),
        },
        {
            "md": (
                '## Optional: Mode D known-future covariates\n'
                '\n'
                "Enable `RUN_COVARIATE_DEMO` or set `DIMER_RUN_COVARIATE_DEMO=1`. The deterministic two-series demand table (the repository's `chronos_covariates_history.csv` / `chronos_covariates_future.csv` formulas, built in memory) carries `temperature` and `holiday` as covariates; the future table contains only those two columns for the 24-step horizon, never future `demand`, so a target column there would be refused as leakage. The provenance names which covariates were read as known-future."
            ),
            "code": (
                'RUN_COVARIATE_DEMO = False  # @param {{type:"boolean"}}\n'
                'if os.environ.get("DIMER_RUN_COVARIATE_DEMO") == "1":\n'
                '    RUN_COVARIATE_DEMO = True\n'
                '\n'
                'if RUN_COVARIATE_DEMO:\n'
                '    n, horizon = 96, 24\n'
                '    step = np.arange(n, dtype=float)\n'
                '    future_step = np.arange(n, n + horizon, dtype=float)\n'
                '    timestamps = pd.date_range("2026-01-01", periods=n, freq="h")\n'
                '    future_timestamps = pd.date_range(timestamps[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h")\n'
                '    history_parts, future_parts = [], []\n'
                '    for series_id, base, phase in (("A", 80.0, 0.0), ("B", 110.0, 4.0)):\n'
                '        temperature = 27.0 + 4.0 * np.sin(2.0 * np.pi * (step + phase) / 24.0)\n'
                '        holiday = (pd.Series(timestamps).dt.dayofweek >= 5).astype(int).to_numpy()\n'
                '        demand = base + 1.8 * temperature + 10.0 * holiday + 0.08 * step + 3.0 * np.sin(2.0 * np.pi * step / 12.0)\n'
                '        history_parts.append(pd.DataFrame({{"series_id": series_id, "timestamp": timestamps, "demand": demand, "temperature": temperature, "holiday": holiday}}))\n'
                '        future_temperature = 27.0 + 4.0 * np.sin(2.0 * np.pi * (future_step + phase) / 24.0)\n'
                '        future_holiday = (pd.Series(future_timestamps).dt.dayofweek >= 5).astype(int).to_numpy()\n'
                '        future_parts.append(pd.DataFrame({{"series_id": series_id, "timestamp": future_timestamps, "temperature": future_temperature, "holiday": future_holiday}}))\n'
                '    cov_history = pd.concat(history_parts, ignore_index=True)\n'
                '    cov_future = pd.concat(future_parts, ignore_index=True)\n'
                '    assert "demand" not in cov_future.columns\n'
                '    cov_config = ForecastConfig(target="demand", prediction_length=horizon, quantile_levels=[0.1, 0.5, 0.9])\n'
                '    cov_manifest = validate_inputs(cov_history, cov_config, pipe, future_df=cov_future)\n'
                '    cov_result = forecast(cov_history, cov_config, pipe, cov_future)\n'
                '    known_future = cov_result.inference["known_future_covariate_names"]\n'
                '    print("known-future covariates:", known_future, "| manifest:", cov_manifest["known_future_covariate_names"])\n'
                '    cov_result.forecast.to_csv("outputs/chronos_covariate_forecast.csv", index=False)\n'
                'else:\n'
                '    print("Mode D demo skipped.")'
            ),
        },
    ],
    "closing": (
        '## Interpretation and limits\n'
        '\n'
        "A successful default run **proves** that the recorded repository revision's package, carried in this notebook, can install the pinned runtime, acquire and digest-verify the pinned model, preserve the chronological evaluation boundary, validate the demonstrated input, execute the public API for univariate and primary multi-target forecasting, compute the documented tutorial metrics/baselines, and export machine-readable forecasts/provenance in the tested runtime. Successful execution proves that the recorded repository revision's package, carried in this notebook, can do exactly that — without the repository being reachable — and no more.\n"
        '\n'
        'It **does not prove** accuracy, calibration, robustness, fairness, safety, or production readiness for your domain. It does **not** establish benchmark superiority, deployment calibration, safety for high-consequence decisions, or production fitness on an unseen domain. The synthetic sample is intentionally simple; its metrics are not benchmark evidence and the evaluation report says `sample-sanity` for that reason. `prediction` is the median of an uncalibrated quantile forecast; the requested quantiles are not calibrated intervals and the pipeline ships no threshold. BYOD requires representative multi-origin backtesting, leakage controls, domain baselines, calibration checks, and operational review.\n'
        '\n'
        'Current limits include fixed-width regular frequencies; monthly/quarterly/yearly/business-day, irregular, and gappy calendars are outside this path, and the pipeline provides no classification, anomaly-detection, imputation, embedding, or training capability.\n'
        '\n'
        "**Next experiments:** enable `USE_BYOD` with a multi-series hourly CSV of your own and compare the report's `evaluate_forecast` MAE against the `last_value_baseline` and `seasonal_naive_baseline` entries; raise `PREDICTION_LENGTH` toward the native 1,024-step horizon and watch `effective_prediction_length`; enable `RUN_COVARIATE_DEMO` and compare the Mode D forecast with and without the future covariate table.\n"
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
