# Release verification

`tutorials/chronos_2_forecasting_colab.ipynb` (`TASK-INFERENCE`) is a **release candidate** until
the exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests,
JSON validation, code-cell compilation, and `tools/validate_release_assets.py` are necessary
checks but are **not** runtime evidence under DIMER Notebook Specification 1.1. This file is
the durable release-gate record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no
  persisted outputs or execution counts; no unresolved placeholder markers; every code cell
  is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `TASK-INFERENCE`
  profile, the notebook-spec version and the standalone carrier; `metadata.dimer` declares that profile, spec `1.1`, `standalone: true` and `generated_from` (repository, module commit, the seven carried modules, their combined SHA-256, generator);
- the standalone carrier (ST1–ST6, PAR1–PAR3): no clone, repository install or repository import on the
  primary path; one cell tagged `embedded_module` per module of `src/chronos2_pipeline/` (seven, in dependency order:
  `errors`, `provenance`, `config`, `model`, `validation`, `inference`, `evaluation`), each equal to its module after the
  generator's documented rewrites (the `DEFAULT_WEIGHTS_DIR` rule plus the removal of package-relative imports); the inline
  `MANIFEST` equal to the committed `weights/chronos-2/dimer-base-manifest.json`; the inline `PINS` equal to the
  `pyproject.toml` runtime pins; the notebook byte-identical to `tools/build_notebook.py` output; the pinned-install cell
  with its restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` are bound only in the carried module cells (and repeated in the inline manifest,
  which the notebook asserts against the module before fetching), the revision is a 40-hex immutable commit, and the same
  identity string appears in `README.md` and `MODEL_CARD.md` with no stray revisions;
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`, `load_pinned_model(weights_dir=...)`,
  `chronological_holdout`, `validate_inputs`, `forecast`, `evaluate_forecast`, `last_value_baseline`,
  `seasonal_naive_baseline`, `evaluation_report`), the ceiling print (`ResourceLimits`, `MIN_OBSERVATIONS`, the model's
  8,192-step context and 1,024-step native horizon), the exports, the learner-facing forecasting statements (median point
  forecast, uncalibrated quantiles, no shipped threshold, chronological holdout, MAE/RMSE/pinball/coverage semantics,
  the guarded seasonal-naive comparator, Mode C) and the gated-off BYOD default listed in the validator; forbidden
  patterns (credential-in-URL, any `git clone` / `github.com` / repository import on the primary path, a mutable
  `revision='main'`, direct `chronos` / `BaseChronosPipeline` / `predict_df` / `snapshot_download` /
  `from huggingface_hub import` / `from transformers import` use **outside the carried module cells**, any `worker.run(` /
  `worker_cli(` / `subprocess.run([` outside the generator-owned install cell, `trust_remote_code=True`, `pickle.load`,
  `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no
  document makes an unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter, single H1, required heading order, and the checkpoint provenance section.

CI also installs the locked environment (`uv sync --locked`), runs `ruff`, `tools/build_notebook.py --check`, and the
offline unit suite (`tests/`, including `test_fleet_snapshot.py`, `test_role_helpers.py`, `test_notebook_parity.py`;
no weights, injected downloader). These are source/provenance and unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel | Kaggle CPU kernel, Python 3.12 image | Reproducible clean-room executor of the same class; the notebook is pushed verbatim plus one leading shim cell that provides `google.colab` and chdirs to a scratch directory (no repository checkout is needed — the notebook is standalone) |
| Repository CI integration job (`tools/run_notebook.py`) | GitHub-hosted Ubuntu runner, the locked `uv` environment with `DIMER_NOTEBOOK_CI_PREINSTALLED=1` | Executes the standalone notebook's code cells sequentially against the real pinned weights in a scratch working directory (default path, then the BYOD-shaped and Mode D branches); a **pre-flight** on the locked stack, not a fresh-boundary run of the inline `PINS` and not promotion evidence on its own |
| Local WSL harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU (or CUDA) runtime (Colab, or the Kaggle
   executor above) with **no repository checkout** and a clean model cache;
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their
   defaults for the sample path: `USE_BYOD = False`, `PREDICTION_LENGTH = 12`, `RUN_COVARIATE_DEMO = False`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the module commit recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS` (= `pyproject.toml`);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the seven carried module cells execute (define `load_pinned_model`, `forecast`, `validate_inputs`, `evaluation_report`
     and the rest) with no import of the repository package;
   - pinned `amazon/chronos-2` acquisition at the immutable revision through the package:
     the inline `MANIFEST` is asserted against the module identity and written to `weights/chronos-2/`,
     `stage_missing_files(WEIGHTS_DIR, allow_download=True)` reports all three manifest entries
     (`README.md`, `config.json`, `model.safetensors`) on a clean runtime, `verify_snapshot` returns the manifest dict
     with the digests equal to the pinned constants, and `load_pinned_model(weights_dir=WEIGHTS_DIR)` reports
     `revision_confirmed_against_hub == False` with the manifest named in its confirmation note;
   - synthetic 96-step hourly sample generated in code with SHA-256
     `eff96b1a9bec5a81aa4021b4d308c29e18fc89be0cf8eb54e755c2efe535c7f7` (equal to the repository's checked-in
     `examples/sample-data/chronos_univariate.csv`), the chronological holdout assertion, and the ceilings printed;
   - `validate_inputs` writes `outputs/chronos_2_forecasting_input_manifest.json` (verdict `accepted`, one recorded
     rejection finding from the non-finite-target probe);
   - forecasting through `forecast(split.history, config, pipe)` with `effective_context_length` and
     `effective_prediction_length` printed and the median/quantile columns present;
   - `evaluation_report` writes `outputs/chronos_2_forecasting_evaluation_report.json` with verdict `sample-sanity`,
     `evaluate_forecast` metrics for the model and both baselines (the 96-step hourly sample satisfies the seasonal-naive guard);
   - the SVG, the Mode C multi-target forecast with both target names, and the exports
     (`outputs/chronos_forecast.csv`, `chronos_multitarget_forecast.csv`, `chronos_evaluation_per_series.csv`,
     `chronos_provenance.json`, `chronos_evaluation.json`, `chronos_2_forecasting_result.json`) written with
     `NOTEBOOK_SOURCE`, model revision, model licence, runtime versions, device and dtype;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, transformers, device),
   model identifier and immutable revision, whether the model cache was clean, outcome, produced
   outputs, and any warning or applicable `SHOULD` deviation in the table below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release.

## Recorded executions

Notebook identity is the Git blob id of `tutorials/chronos_2_forecasting_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/chronos_2_forecasting_colab.ipynb`). Wall times, when recorded,
are the sum of per-cell times reported by the executor and include installs and the model download;
they are measurements for the stated runtime, not general estimates.

### Manual clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | `8485f55` / `e06fea3dcb8d` | Kaggle CPU (`kurtvalcorza/dimer-nb2-chronos-2-forecasting` v1) | Default sample path | 236.0 s | **PASSED** — 17/17 ok code cells executed cleanly, 8 files, 478 MB staged |

### Multi-model forecasting workshop

Notebook identity is the Git blob id of `tutorials/DIMER_MultiModel_TimeSeries_Forecasting_Workshop.ipynb`.
This notebook is verified separately from the primary tutorial above.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-26 | `7932079` / `6cdfa5a9d48d` | Google Colab, Tesla T4, Python 3.13.15 kernel (model environments on Python 3.12) | `STANDARD` tier (TiRex-2 on CPU, Chronos-2 on CUDA), synthetic `chronos_multi_series.csv` sample | not recorded | **PASSED** — 18/18 code cells executed without error; test macro MAE TiRex-2 1.147, Chronos-2 0.175; report bundle SHA-256 `518b770c8d92…`. The saved copy differs from the blob only by an empty `# @title` line Colab inserted in one cell. |

Later revisions add the optional `OPEN_METEO_PH` sample and have been run only locally
(`nbclient`, CPU PyTorch); they need their own clean-runtime run before the workshop's evidence
status changes from pending. The `FULL` tier (Toto 2.0) has no recorded run.

## Current status

No clean-runtime execution of the standalone notebook has been recorded yet; the run is **pending** and queued
to the GPU lane. Static validation (`tools/validate_release_assets.py`), nbformat validation, a
`compile()` sweep over every code cell, and the offline unit suite passed on the tutorial source at
the candidate revision, which is necessary but not sufficient. The registry status remains
**Candidate** until a reviewer confirms a recorded run against the notebook blob under review and
an integrator promotes it; promotion is not performed by the builder. Three facts a reviewer should
weigh: `stage_missing_files` was exercised only with an injected downloader in the unit suite (the
real `hf_hub_download` fetch of all three manifest entries into a fresh `weights/chronos-2/` has not been
executed); `load_pinned_model(weights_dir=...)` was exercised only with a stand-in `chronos` module (the real
`BaseChronosPipeline.from_pretrained` on the manifest-described directory has not been executed); and the standalone
carrier itself — executing the seven carried module cells in a runtime that has no repository checkout — has been
validated statically only (parity PASS, carrier probe up to the fetch), never run. The earlier repository-installing
notebook did run in CI against the real weights at `95a9710e2596287d08352589f42634fa5abdf0a7`; that evidence predates the standalone carrier and the
fleet snapshot scheme and does not transfer to it.
