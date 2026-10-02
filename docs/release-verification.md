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
| 2026-09-26 | `129142b` / `13e92f12ef0b` | Google Colab, Tesla T4, Python 3.13.15 kernel (model environments on Python 3.12) | `STANDARD` tier, `SAMPLE_DATASET = "SYNTHETIC"` (default) | not recorded | **PASSED** — 18/18 code cells executed without error, notebook unmodified; test macro MAE TiRex-2 1.147, Chronos-2 0.175 (identical to the `7932079` run); report bundle SHA-256 `4cb835100d1c…` |
| 2026-09-26 | `129142b` / `13e92f12ef0b` | Google Colab, Tesla T4, Python 3.13.15 kernel (model environments on Python 3.12) | `STANDARD` tier, `SAMPLE_DATASET = "OPEN_METEO_PH"` (only that control changed) | not recorded | **PASSED** — 18/18 code cells executed without error; sample digest `74163ee609cd…` verified; test macro MAE TiRex-2 0.463, Chronos-2 0.450, seasonal-naive 0.496; report bundle SHA-256 `2cdac3575695…` |
| 2026-09-26 | `eb426b9` / `40b307491592` | Google Colab, Tesla T4, Python 3.13.15 kernel (model environments on Python 3.12) | `STANDARD` tier, `SAMPLE_DATASET = "SYNTHETIC"` (default), guided-text revision | not recorded | **PASSED** — 18/18 code cells executed without error, notebook unmodified; report bundle SHA-256 `505c03e9f59a…` |
| 2026-09-26 | `eb426b9` / `40b307491592` | Google Colab, Tesla T4, Python 3.13.15 kernel (model environments on Python 3.12) | `STANDARD` tier, `SAMPLE_DATASET = "OPEN_METEO_PH"` (only that control changed) | not recorded | **PASSED** — 18/18 code cells executed without error; sample digest `74163ee609cd…` verified; report bundle SHA-256 `8b02fa737519…` |
| 2026-09-26 | `eb426b9` / `40b307491592` | Google Colab, Tesla T4, Python 3.13.15 kernel (model environments on Python 3.12) | `FULL` tier, `SAMPLE_DATASET = "OPEN_METEO_PH"` | not recorded | **FAILED** — all three environments built and TiRex-2 and Chronos-2 forecast the validation period, but the Toto runner exited while importing `toto2`: Colab's `MPLBACKEND=module://matplotlib_inline.backend_inline` was inherited by the subprocess, and `matplotlib` (loaded through `gluonts` → `lightning` → `torchmetrics`) rejected it. Fixed in the next revision by setting `MPLBACKEND=Agg` for every runner |
| 2026-09-26 | `4199536` / `2213095d1e58` | Google Colab, Tesla T4, Python 3.13.15 kernel (model environments on Python 3.12) | `FULL` tier (TiRex-2 on CPU; Chronos-2 and Toto 2.0 2.5B on CUDA), `SAMPLE_DATASET = "OPEN_METEO_PH"` (only those two controls changed) | not recorded | **PASSED** — 18/18 code cells executed without error; sample digest `74163ee609cd…` verified; Toto loaded its 9,362 MiB checkpoint in 124 s and forecast in 1.4 s; test macro MAE TiRex-2 0.463, Chronos-2 0.450, Toto 2.0 0.494, seasonal-naive 0.496 (TiRex-2 and Chronos-2 identical to the `STANDARD` runs); report bundle SHA-256 `850bda69a927…` |
| 2026-09-26 | `00abb68` / `2213095d1e58` | Google Colab, Tesla T4, Python 3.13.15 kernel (model environments on Python 3.12) | `STANDARD` tier, `SAMPLE_DATASET = "SYNTHETIC"` (default path, notebook unmodified) | not recorded | **PASSED** — 18/18 code cells executed without error; test macro MAE TiRex-2 1.147, Chronos-2 0.175 (identical to every earlier synthetic run); report bundle SHA-256 `c0eb16a1e7ba…` |

Notebook blob `13e92f12ef0b` (from `129142b`) is unchanged at `e7d9cd8`. Both default and optional sample paths of the `STANDARD` tier passed at that blob.

Notebook blob `2213095d1e58` (unchanged from `4199536` to `00abb68`) now has passing Google Colab T4 runs of the default `STANDARD` path with the synthetic sample and of the `FULL` tier with the Open-Meteo sample, which together exercise all three models and both samples; the `STANDARD` + Open-Meteo combination passed at the two preceding blobs. As for the primary tutorial, this table is the evidence record. The notebook's `metadata.dimer.clean_runtime_evidence` stays `pending` as authored, because editing it would change the blob these runs verify.


### Guided learning follow-up — 2026-09-26

The historical runs above remain evidence for notebook blob `2213095d1e58`,
not a claim that the updated notebook has been run in a fresh hosted runtime.
The follow-up changes learner-facing explanations and collapse metadata, fixes
the context-length attribution in the test checkpoint, and adds one optional,
default-off seasonal-period activity. The original 18 code-cell sources are
unchanged. The updated notebook has 19 code cells and remains **Candidate**
pending execution of this exact revised artifact and the remaining BYOD gate.

Local verification (Windows, Python 3.12; no foundation-model inference):

- Baseline: 7/7 existing workshop tests and release-asset validation passed.
- After: 13/13 workshop tests passed, including actual path-mode BYOD loading,
  schema validation, chronological splitting and baseline scoring on a
  separately generated two-series CSV. The path branch bypasses Colab upload.
- Missing targets, duplicate timestamps, irregular timestamps and non-finite
  values were each rejected before model acquisition.
- The optional activity was executed both disabled and enabled. Its two
  conditions score identical validation targets, agree with an independent
  NumPy MAE calculation, do not read test inputs, and preserve canonical data
  and output files.

Reproduce these lightweight checks without installing the model stack:

```shell
python -m pytest --noconftest tests/test_multimodel_forecasting_workshop.py tests/test_forecasting_workshop_learning.py
python tools/validate_release_assets.py
python tools/build_multimodel_forecasting_workshop.py --check
```

`--noconftest` isolates these notebook-cell checks from the repository-wide
model fixtures; it does not stand in for the full unit/integration suite.

**Remaining REL12 procedure (not claimed complete):**

1. In a fresh Colab T4 runtime, save the exact revised notebook and record its
   Git commit/blob, runtime versions and controls. Run the default path first
   and retain the executed notebook and report ZIP SHA-256.
2. In a separate fresh runtime, place a representative user-owned compatible
   CSV at `/content/byod.csv`, set `BYOD_CSV_PATH` to that location and run all
   with the other default controls. Verify `sample_kind=BYOD`, validation,
   both foundation-model validation forecasts, freeze, test evaluation, future
   forecasts and export. Save the input digest and report ZIP digest. A built-in
   `OPEN_METEO_PH` run alone does not exercise this file-reading branch.
3. In a separate negative run, remove the `target` column from a copy of that
   CSV. Use that path and confirm `Missing required columns` before any model
   download. Retain the rejection output and input digest.
4. Optionally enable the seasonal-period activity in the successful run after
   writing a prediction, and save its separate comparison CSV and conclusion.
   Do not change or retune the frozen test comparison using that activity.

The local positive/negative checks establish the input and baseline boundary;
they do **not** yet establish BYOD foundation-model execution through export.
Promote only after the exact-revision hosted and BYOD evidence is recorded.

## Current status

Use the per-notebook, per-blob execution tables above for historical outcomes;
the earlier blanket statement that no clean-runtime execution had been recorded
was stale. The revised multi-model notebook's hosted rerun and full BYOD gate
are **pending** as described above. The registry status remains **Candidate**
until a reviewer confirms a recorded run against the notebook blob under review and
an integrator promotes it; promotion is not performed by the builder. Three facts a reviewer should
weigh: `stage_missing_files` was exercised only with an injected downloader in the unit suite (the
real `hf_hub_download` fetch of all three manifest entries into a fresh `weights/chronos-2/` has not been
executed); `load_pinned_model(weights_dir=...)` was exercised only with a stand-in `chronos` module (the real
`BaseChronosPipeline.from_pretrained` on the manifest-described directory has not been executed); and the standalone
carrier itself — executing the seven carried module cells in a runtime that has no repository checkout — has been
validated statically only (parity PASS, carrier probe up to the fetch), never run. The earlier repository-installing
notebook did run in CI against the real weights at `95a9710e2596287d08352589f42634fa5abdf0a7`; that evidence predates the standalone carrier and the
fleet snapshot scheme and does not transfer to it.


## Maintainer-supplied Colab execution — 2026-09-26

The maintainer reported that this notebook passed an end-to-end Colab run and authorized merging its open PR. The supplied [executed notebook](execution-evidence/2026-09-26/DIMER_MultiModel_TimeSeries_Forecasting_Workshop.ipynb) is preserved byte-for-byte as evidence.

- Reviewed source commit: `9e10f240fd76eb6799eded892776a60ec21539c8`.
- Executed-file SHA-256: `b84f24987b939b073b3147535491c7399a37b9df1d5fe5f4a119b82e2124ac9e`.
- Independently inspected: 19 executed code cells; zero saved error outputs; terminal completion and exports present.
- Configuration/source comparison: STANDARD tier with SAMPLE_DATASET=OPEN_METEO_PH. This is the supported weather sample route, not evidence for the default SYNTHETIC setting. Other code is unchanged.
- Evidence boundary: saved outputs were inspected; execution was not independently repeated. This submission establishes the recorded path, not optional FULL/BYOD paths. Fresh-runtime/restart details beyond the maintainer's explicit prior confirmations are not inferred.

This record supersedes the pending rerun item for the source/configuration above. It does not promote the whole pipeline or close untested optional-path qualification.


## Multi-model forecasting workshop: Notebook Review Framework v1 findings — revision 0.2.0-candidate (2026-09-27)

A review under the Notebook Review Framework v1 examined commit `e13b362` (notebook blob `8d65d164`) and concluded **Needs revision**. It reported six major and four minor findings. The review and its probes are archived in [`reviews/2026-09-27-notebook-review/`](reviews/2026-09-27-notebook-review/). Revision `0.2.0-candidate` (notebook blob `9b13733993da`) addresses every finding.

`tests/test_forecasting_workshop_review_fixes.py` executes the notebook's own cells end to end, with pretrained models replaced by a small fake runner program that has the same command-line and file contract. It uses the notebook's real `run_model`, freeze, test and export cells. All 39 of its tests fail on the reviewed revision and pass on 0.2.0. These are orchestration and contract checks, not model runs.

| Finding | Correction in 0.2.0 | Acceptance check |
|---|---|---|
| **TS-R01** (major): pre-model EDA plotted and summarised the test targets | The test boundary is fixed before any plot. §4.1 plots and summarises only development rows, draws the test interval as an empty band, and fits the z-scores on development rows. The complete series is plotted in §9.2, after scoring | For the synthetic sample, the Open-Meteo sample and BYOD, every plotted timestamp precedes the test start and the summary counts only development rows; the chronological-split check is kept |
| **TS-R02** (major): TiRex received one joint multivariate item, while Chronos and Toto forecast each series independently | One policy, `independent-per-series`: TiRex gets one univariate `TimeseriesType` per series; Chronos keeps `cross_learning=False`; Toto keeps a distinct `series_id` per series. The pinned `toto-2==2.0.0` source confirms that distinct IDs mask cross-series attention. Every runner reports its conditioning, and `run_model` refuses any other report | In-process adapter tests show TiRex receiving one `(1, T)` item per series and Chronos called with `cross_learning=False`. A runner reporting joint groups is refused. Real-model perturbation (below) |
| **TS-R03** (major): numeric-looking IDs passed validation, then broke scoring after a CSV round trip | IDs are read as text from the first read (`dtype=str`, with missing-value parsing limited to `target`). Every runner and the forecast reader do the same. The evaluator rejects a non-text ID instead of repairing it | IDs `001`, `1`, `101` and `A` survive acquisition, requests, forecasts, scoring, the forecast beyond the data, and export; series named `NA` and `null` are kept |
| **TS-R04** (major): the evaluator accepted missing rows or series and crossed or missing quantiles | `validate_forecast_table` requires the exact model × series × timestamp grid with correct steps and origins. It also requires finite values, the point equal to q0.5, and q0.1 ≤ q0.5 ≤ q0.9. Crossing is rejected rather than sorted. The same contract applies to the forecast beyond the data | A missing row, missing series, unexpected series, duplicate, shifted timestamp, renumbered step, non-finite or missing quantile, crossed bounds, median ≠ point, or missing model are each rejected. A complete table keeps its metrics |
| **TS-R05** (major): the freeze did not bind the staged request bytes | The freeze records request-file digests, split boundaries, runner digests, environment pins and signatures, context limits and evaluation rules. §9.1 recomputes all of these from the current files and refuses on any difference, and `run_model` rechecks the input digest immediately before each call. A freeze is never overwritten, and a frozen test runs once | A changed request value or timestamp, runner, seasonal period, environment or context limit each stops §9.1 before any model call. An unchanged re-freeze keeps the file; a second test is refused |
| **TS-R06** (major): a report could contain an earlier run's files | Each run of §1.1 creates `outputs/<experiment ID>/`. Model environments and checkpoints are shared under `work/`, but results are not. The ZIP holds only the current experiment and lists every file in `provenance/inventory.json` | FULL on Open-Meteo with the optional activity, then STANDARD on the synthetic sample: the second report contains no Toto file or activity, and the first report is unchanged. A re-export flags an activity as optional |
| **TS-R07** (minor): "real future" for dates that have already happened | §11 is now "Forecast beyond the supplied dataset", with status "not evaluated: matching future targets were not provided", and it explains prospective forecasting | Status and wording checks; the synthetic forecast starts 2026-01-05 |
| **TS-R08** (minor): macro MAE presented as scale normalisation; the overlay called min–max | Pooled and macro scores are described as target-unit aggregation. Equal pooled and macro MAE on complete grids is explained, skill is the scale-free comparison, and a new `skill_series` count is reported. The overlay is described as z-scores | The complete-grid test asserts pooled MAE = macro MAE; wording checks |
| **TS-R09** (minor): opaque long runs; FULL GPU checked after a 9.8 GB download | §1.2 checks the GPU, its memory and free disk for the tier before any download. Runners print `[stage]` lines, which `run_model` streams along with a 30-second heartbeat and a full log. Toto checks CUDA before staging. Runtime text says the timings compare configurations, not architectures | FULL without a GPU stops in §1.2; stage lines appear during a run; static order check on Toto |
| **TS-R10** (minor): incomplete reproducibility record | The manifest records the notebook file and revision (`0.2.0-candidate`), run controls, parent versions against a tested range (warns, never blocks), each environment's intended pins and the package versions observed in it, device, precision, and request digests | Manifest and freeze content checks |
| Optional: first result before the long install | §5.1 shows both baselines' validation scores immediately; test baselines stay unscored until §9 | The first table holds validation scores only (seasonal naive 3.60, last value 5.53) |

The generator source and the notebook had drifted since `e13b362` because the AI-disclosure edit was made to the notebook only. As a result, `test_generator_parity` failed on `main`. The disclosure now lives in `tools/multimodel_forecasting_workshop_source.py`, and the notebook is regenerated from it, so parity passes again.

### Real-model check of the conditioning fix (CPU, 2026-09-27)

The pinned TiRex-2 runner was executed on CPU with `tirex-2==0.2.1`, `torch==2.8.0+cpu`, `numpy==2.3.3`, `pandas==2.3.3`, `huggingface-hub==0.36.2` and `triton==3.4.0`. It used the real checkpoint at revision `05e5b26d`, fed with the notebook's own request files. TiRex runs on CPU in Colab as well. The script and full results are in [`reviews/2026-09-27-notebook-review/`](reviews/2026-09-27-notebook-review/) (`tirex_cpu_rerun.py`, `tirex_cpu_rerun_0.2.0.json`).

- **Environment fidelity:** the reviewed revision's joint runner reproduced the recorded Colab figures exactly:
  - synthetic validation MAE 3.28 (`A`) and 2.34 (`B`);
  - synthetic test MAE 1.76 and 0.54;
  - Open-Meteo validation skill +0.05, with Manila at −0.09.
- **Perturbation:** only the second series' last 12 context hours were changed.

  | Runner | Change in the unchanged series' q0.5 | Change in the edited series' q0.5 |
  |---|---|---|
  | Joint (reviewed revision) | synthetic `A` 0.54; Open-Meteo Cebu 0.032, Manila 0.035 | 43.1 (synthetic `B`), 25.6 (Davao) |
  | Independent (0.2.0) | **0.0 exactly** for every unchanged series | 43.7 (`B`), 25.5 (Davao) |

  The joint runner let one series change another's forecast. The independent runner does not.
- **TiRex-2 figures after the fix, used in the regenerated sample answers:**

  | Sample | Period | Macro MAE | Macro skill | Coverage | Width |
  |---|---|---|---|---|---|
  | Synthetic | Validation | 2.44 (`A` 2.83, `B` 2.04) | 0.32 | 0.65 | 5.00 |
  | Synthetic | Test | 1.16 (`A` 1.84, `B` 0.49) | 0.70 | 0.94 | 4.63 |
  | Open-Meteo | Validation | 0.53 | +0.085 (Manila +0.004) | 0.75 | 1.73 |
  | Open-Meteo | Test | 0.47 | −0.066 | 0.81 | 1.57 |

  The Chronos-2 input, settings and runner semantics did not change, so its quoted figures are unchanged. The EDA answers now quote development-period statistics.

Code cells changed, so no earlier hosted run describes this revision. **Status: Candidate.** The following exact-revision evidence is required:

- a fresh Colab T4 `STANDARD` run on the synthetic sample;
- a `STANDARD` run on the Open-Meteo sample;
- a `FULL` Open-Meteo run on the intended GPU (this also covers Toto under the new contract);
- a BYOD run with text, numeric and leading-zero IDs through export;
- the corrected rejection paths;
- a sample or tier switch within one session, checking that each report holds only its own experiment;
- the optional activity.

The review's learner-observation recommendation is not addressed by code and remains open.

### Maintainer-supplied Colab execution of revision 0.2.0 — 2026-09-28

The maintainer supplied an executed copy of revision `0.2.0-candidate`. It is preserved byte-for-byte as [evidence](execution-evidence/2026-09-28/DIMER_MultiModel_TimeSeries_Forecasting_Workshop.ipynb).

- **Source:** branch `fix/forecasting-review-findings` at `ff27feb`, notebook blob `9b13733993da`. All 50 cell IDs and sources match exactly.
- **Executed-file SHA-256:** `206cf627caeb8b38f15e16002f2be64fa1291ee897317ef556e105b39a6ec140`.
- **Runtime:** Colab Tesla T4, host Python 3.13.15. Host packages: NumPy 2.1.3, pandas 2.2.3, matplotlib 3.10.0, packaging 26.3, all within the tested range. Free disk was 202 GB.
- **Path:** `STANDARD` tier, `SYNTHETIC` sample, default controls.
- **Execution:** 19 of 19 code cells ran in order, with no error outputs. The completion summary reports revision `0.2.0-candidate`, experiment `20260928T012724Z-3dd52a`, and the status "not evaluated: matching future targets were not provided".
- **Behaviour of the fixes in the saved outputs:**
  - **TS-R01:** EDA summarises only the 144 development rows (means 126.39 and 159.26).
  - **TS-R02:** every runner reports "forecasting 2 series independently", and the conditioning check passed.
  - **TS-R04:** the complete-grid contract passed for all models, and `skill_series` is 2.
  - **TS-R05:** the freeze was followed by a verified test run.
  - **TS-R06:** the report covers only its own experiment folder (36 files, with inventory); its SHA-256 is `f1f54a446c0049dbb26af7f3e8621fbcc44ea117eb7738fdd9b997b0f9082382`.
  - **TS-R07:** the forecast beyond the data covers 2026-01-05.
  - **TS-R09:** stage lines and a heartbeat were printed while the models ran.
  - **Optional improvement:** the early baseline table shows last value 5.53 and seasonal naive 3.60.
- **Results:**

  | Model | Period | Macro MAE (A / B) | Skill | Coverage | Width |
  |---|---|---|---|---|---|
  | TiRex-2 | Validation | 2.438 (2.83 / 2.04) | 0.317 | 0.646 | 5.00 |
  | TiRex-2 | Test | 1.164 (1.84 / 0.49) | 0.702 | 0.9375 | 4.63 |
  | Chronos-2 | Validation | 0.267 (0.29 / 0.25) | 0.924 | 1.00 | 3.87 |
  | Chronos-2 | Test | 0.175 | 0.952 | 1.00 | 1.98 |

  The TiRex-2 figures equal the CPU rerun that regenerated the sample answers, to the displayed precision. The Chronos-2 figures equal the earlier recorded reference.
- **Evidence boundary:** the saved outputs were inspected, but the execution was not repeated independently.

This satisfies the fresh `STANDARD` synthetic item for revision 0.2.0. The following items remain open:

- `STANDARD` on the Open-Meteo sample;
- `FULL` on a GPU;
- BYOD with numeric and leading-zero IDs through export;
- the rejection paths;
- a sample or tier switch within one session;
- the optional activity.

**Status: Candidate.**

## Philippine reef heat-stress capstone: maintainer-supplied Colab execution — 2026-09-28

Notebook identity is the Git blob of `tutorials/DIMER_Philippine_Reef_Heat_Stress_Capstone.ipynb`.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-28 | `f3c4a1e` / `61fae866df91` | Google Colab, Tesla T4, Python 3.13.15 kernel (isolated environment on uv-managed Python 3.12.13) | Default Run all | not recorded | **INCOMPLETE** — the runtime disconnected during the first stage. The notebook reports no error. |

- **File.** The [executed notebook](execution-evidence/2026-09-28/DIMER_Philippine_Reef_Heat_Stress_Capstone_f3c4a1e.ipynb)
  is preserved byte for byte. SHA-256 `e15148da3bdcc54a92ea62caa34e28690d6e720d43109454ff2202cb7a9d9a3d`.
- **Source match.** The source equals revision `f3c4a1e` except for:
  - the maintainer's title edit, which removes "7" from "DIMER Capstone 7";
  - empty `# @title` lines that Colab inserted in seven collapsed cells. These are not substantive.
- **Executed cells.** Cells `reef-003` to `reef-009` ran, with execution counts 1–5.
  - Setup printed `Isolated Python 3.12.13 dependencies ready (kernel Python 3.13.15)`. This is
    hosted evidence that the R1 interpreter fix works on a Python 3.13 Colab kernel.
  - The embedded NOAA archive verified.
  - `reef-010` (runner assembly and the `prepare` stage) has no execution count or output. No later
    cell ran.
- **Cause.** Not determined from the notebook: a disconnect leaves no traceback in the saved file.
  The same cells ran offline without error from a non-3.12 kernel, with the full hash-locked
  environment on managed Python 3.12.13:
  - setup took 71 s and produced a 6.8 GB environment;
  - `prepare` took 3 s;
  - baselines took 29 s over 299 origins.

  They were also run with Colab's `PYTHONPATH` and inline `MPLBACKEND` set. Colab's runtime log
  would be needed to identify the cause.
- **Follow-up in the next revision.** Every child process now:
  - runs with `PYTHONPATH`, `PYTHONHOME` and `PYTHONSTARTUP` removed and `MPLBACKEND=Agg`;
  - writes a log under `reef_capstone_run/logs/`;
  - echoes progress into the cell, with a heartbeat every 30 s during silent steps.

  A failure shows the log tail. Setup no longer prints a stray `116`.
- **Evidence boundary.** Saved outputs were inspected; the hosted execution was not repeated. No
  model stage ran, so this run is not model or forecast evidence.

**Status: Candidate.** A complete fresh-T4 Run all is still required.

### Maintainer-supplied Colab execution of `2de49e1` — 2026-09-28

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-28 | `2de49e1` / `9a4d35edf291` | Google Colab, Tesla T4, Python 3.13.15 kernel (isolated environment on uv-managed Python 3.12.13) | Default Run all with `USE_BYOD` and `HOTSPOT_SEMANTICS_CONFIRMED` ticked, path empty | about 3 min to the failure | **FAILED** — the 180-day context activity stopped with `ValueError: Crossing quantiles`. |

- **File.** The [executed notebook](execution-evidence/2026-09-28/DIMER_Philippine_Reef_Heat_Stress_Capstone_2de49e1_byod.ipynb)
  is preserved byte for byte. SHA-256 `dcad18fcca01df7a014af9aa782e6527d4021b8e9e90f3a1140d2b792ea6a88d`.
- **Source match.** The source equals `2de49e1` except for:
  - the two BYOD `# @param` toggles;
  - empty `# @title` lines that Colab inserted in collapsed cells. These are not substantive.
- **Executed cells.** Execution counts run 1–9 in order, with no disconnect.

  | Step | Result |
  |---|---|
  | Setup | 80 s for the locked sync |
  | `prepare` | 6 s; the origin table matches the plan |
  | Baselines | 43 s, 299 origins |
  | Validation, real Chronos-2 | 40 s, 115 origins, peak GPU 513,587,712 bytes |
  | `activity` | Failed after origin 25 |

  The stage log and the error were shown in the cell, as the previous revision intended.
- **Cause.** Chronos-2 2.3.1 returns the trained quantile slices without enforcing their order, so a
  small crossing is a model property. The adapter treated it as corrupt output. A CPU probe with the
  pinned weights (`95a9710e…`) over all 115 validation origins found exactly one crossing day:
  - context: 180 days;
  - region: eastern;
  - origin: 2023-05-01;
  - q10 exceeds q50 by 0.000023 °C, with both about −0.0038 °C.

  No 365-day forecast crossed. The CPU probe is not hosted evidence; GPU values can differ slightly.
- **Fixed in the next revision.** Chronos quantiles are sorted before use (monotone rearrangement).
  The point forecast is the rearranged median. The model's unsorted values (`model_q10`–`model_q90`)
  and the crossing size (`quantile_crossing_c`) are kept in every forecast file and checked at reload.
  Stage summaries and `run_summary.json` count crossing days.
- **Not assessed in this run.** Lock, test, scoring tables, outlook, reload, report, BYOD.
- **Evidence boundary.** Saved outputs were inspected; the hosted execution was not repeated.

**Status: Candidate.** A complete fresh-T4 Run all is still required.

### Maintainer-supplied Colab execution of `6ad2ac4` — 2026-09-28

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-28 | `6ad2ac4` / `2675506667d0` | Google Colab, Tesla T4, Python 3.13.15 kernel (isolated environment on uv-managed Python 3.12.13) | Default Run all, notebook unmodified | about 2 min to the disconnect | **INCOMPLETE** — the runtime disconnected at the start of `prepare`. |

- **File.** The [executed notebook](execution-evidence/2026-09-28/DIMER_Philippine_Reef_Heat_Stress_Capstone_6ad2ac4.ipynb)
  is preserved byte for byte. SHA-256 `06008642482e0a196493c28c3c5e8da0c482626cd726f9ceb697096127d0c6f4`.
- **Source match.** The source is identical to `6ad2ac4`. The BYOD toggles were not ticked.
- **Executed cells.** Setup completed: the locked sync took 93 s, and uv no longer warned about
  `--system`. Cells up to `reef-009` ran, with execution counts 1–5. `reef-010` printed
  `[16:17:13] stage prepare started` and nothing after it, not even the first 30 s heartbeat.
- **Pattern.** This matches the `f3c4a1e` run: both disconnected seconds after a fresh 6.8 GB install,
  when the first stage started. The `2de49e1` run passed the same point in 6 s.
- **Offline measurement.** Local, not Colab. A fresh hash-locked install peaked at:
  - 0.14 GB uv resident memory;
  - 0.73 GB total memory in use;
  - 1.5 GB of dirty and writeback page cache.

  `os.sync()` then took 2 s. Neither memory nor the install explains a disconnect on a 12.7 GB
  runtime. A disk-writeback stall on Colab's slower disk is possible but unconfirmed. The cause needs
  Colab's runtime log.
- **Changes in the next revision.**
  - Setup calls `os.sync()` after installing, so the environment is on disk before any stage reads it.
  - Every stage start, every heartbeat and `setup_summary.json` print available RAM, unwritten page
    cache and free disk. A further disconnect will leave the last machine state in the saved notebook.
- **Evidence boundary.** Saved outputs were inspected; the hosted execution was not repeated.

**Status: Candidate.** A complete fresh-T4 Run all is still required.

### Maintainer-supplied Colab execution of `77a116d` — 2026-09-28

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-28 | `77a116d` / `85835a14445b` | Google Colab, Tesla T4, Python 3.13.15 kernel (isolated environment on uv-managed Python 3.12.13) | Default Run all, notebook unmodified | about 2 min to the disconnect | **INCOMPLETE** — the runtime disconnected at the start of `prepare`. |

- **File.** The [executed notebook](execution-evidence/2026-09-28/DIMER_Philippine_Reef_Heat_Stress_Capstone_77a116d.ipynb)
  is preserved byte for byte. SHA-256 `03d7222959a2033340b14aeb6f5d668d4b552cddb781ce55e0c424434e811658`.
- **Source match.** Identical to `77a116d`. The BYOD toggles were not ticked.
- **Executed cells.** Execution counts 1–5 (`reef-003` to `reef-009`). Setup: locked sync 94 s, flush
  1 s, 11.4/12.7 GB RAM available, 195 GB disk free. `reef-010` holds one line and no execution count:
  `[22:03:49] stage prepare started (RAM available 11.6/12.7 GB, unwritten 0.0 GB, disk free 195 GB)`.
- **Reading.** A child that exits non-zero raises `RuntimeError` with its log tail, and a healthy
  `prepare` prints its origin table within 3–6 s. Neither appeared, and neither did the 30 s heartbeat.
  So the kernel's output stopped within about 30 s of launching the child, which points at the Colab
  session rather than the stage. This is an inference: the child's own log did not survive the session.
- **Evidence boundary.** Saved outputs were inspected; the hosted execution was not repeated.

### Kaggle T4 execution of `77a116d` with BYOD — 2026-09-28

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-28 | `77a116d` / `85835a14445b` | Kaggle, Tesla T4 (`kurtvalcorza/dimer-nb2-philippine-reef-heat-stress-capstone` v1), fresh nbclient kernel on Python 3.12.13 | Run all with `USE_BYOD` and `HOTSPOT_SEMANTICS_CONFIRMED` ticked, path empty | 339.4 s | **PASSED** — 14/14 code cells executed in order, no errors. |

- **Files.** Both are preserved byte for byte:
  - the [executed notebook](execution-evidence/2026-09-28/DIMER_Philippine_Reef_Heat_Stress_Capstone_77a116d_kaggle_byod.ipynb),
    SHA-256 `5bba7e4c914fec48350b3e8a6e47133a7dd439bbe9138c74ca371fe770032c6c`;
  - the executor's [run summary](execution-evidence/2026-09-28/DIMER_Philippine_Reef_Heat_Stress_Capstone_77a116d_kaggle_run_summary.json),
    SHA-256 `2bba78f4800efb7f5271d10125bb14f841e847dafa63ff9f9456ccd1565be2d4`.
- **Source match.** The executor fetched the notebook from `raw.githubusercontent.com` at the full
  commit SHA and verified blob `85835a14445bfad8efc0a7df17affe1b21567f36` before running. It then
  changed only the two `# @param` toggle lines (input blob `b50e783e8fd5`). `inspect_run.py` confirms
  that is the only source difference and that cell ids and order are unchanged.
- **Runtime.**
  - Kaggle GPU image `sha256:37c64f7dd9c5…`; Tesla T4, driver 580.159.04; 31.3 GB RAM.
  - The isolated environment reports Python 3.12.13, torch 2.9.1, chronos-forecasting 2.3.1,
    NumPy 2.3.5, pandas 2.3.3 and transformers 4.57.6.
  - The Hugging Face cache was empty at start.
- **Stages.**

  | Stage | Time | Result |
  |---|---|---|
  | Setup | 79 s sync, 8 s flush | Passed |
  | `prepare` | 3 s | Passed. Test/validation origins: 46/23 for central, eastern and northern; 23/23 for southern and western |
  | Baselines | 49 s | Passed; 299 origins |
  | Validation | 34 s model time | Passed; 115 origins, 0 crossing days, peak GPU 0.51 GB |
  | 180-day activity | 24 s model time | Passed; 1 crossing day of 0.000023 °C, rearranged |
  | compare / lock | — | Passed; 365-day context retained |
  | test / score | 39 s / 15 s | Passed; 184 origins, tables A–F printed |
  | future | 9 s | Passed; origin 2026-09-26, 2 regions excluded for missing context |
  | reload / report | 30 s / 24 s | Passed. Fresh-process baseline, DHW and real-model parity held; `reef_evidence.zip` exported (SHA-256 `06e593e41d4c…`) |
  | BYOD | 9 s | Passed. `byod/bae569f03fe0b80f/` holds `forecasts.csv` and `receipt.json` (400 rows, 1 region, unscored) |

- **Headline test result (14-day HotSpot MAE, °C).**

  | Period | Chronos | Persistence | Seasonal |
  |---|---|---|---|
  | Full test | 0.153 | 0.152 | 0.213 |
  | Common 2024 | 0.157 | 0.153 | 0.237 |
  | 2025 (3 regions) | 0.137 | 0.139 | 0.150 |
- **Kaggle-specific observations.**
  - uv reported `Using CPython 3.12.13 interpreter at: /usr/bin/python3`. On this image the kernel's
    Python is already 3.12.13, so the environment was built on it rather than on a downloaded Python.
    The R1 path from a non-3.12 kernel is covered by the Colab runs above, not by this one.
  - Every stage printed `Error in sitecustomize; ... No module named 'wrapt'`. The image's
    `sitecustomize` is visible to an environment built on the image's Python. The message is not a
    notebook error and did not affect any result.
- **Journeys.**

  | Journey | Verdict |
  |---|---|
  | Default Run all, all mandatory stages | **Passed** |
  | Real Chronos-2 validation, activity and test | **Passed** |
  | Fresh-process reload parity | **Passed** |
  | BYOD with the generated example | **Passed** |
  | Entry from a non-3.12 kernel | Not assessed in this run |
  | Repeated Run all in a warm runtime | Not assessed in this run |

- **Evidence boundary.** The Kaggle executor ran the committed bytes; the outputs, run summary and run
  files were inspected. Kaggle is not Colab, so this run does not explain the Colab disconnects.

### Agent-driven Colab T4 execution of `77a116d` (not preserved) — 2026-09-29

Two further Colab T4 attempts were driven cell by cell through the Colab MCP bridge. Each fetched the
committed bytes at the full SHA, verified the blob and ran the cells in the notebook kernel with
`get_ipython().run_cell`.

| Attempt | Setup | `prepare` | Model stages | test / score / future / reload / report |
|---|---|---|---|---|
| First | 87 s | 4 s | Completed | Not observed |
| Second | 82 s | 4 s | Completed | Completed; reload parity held |

Metrics equal the Kaggle run to the printed precision.

In both attempts the bridge went silent at the cell that displays the three figures (`reef-023`), and
the reconnected runtime was a new VM. No executed notebook or run files survived, so these attempts are
recorded as observations, not evidence. They do show `prepare` completing twice on a Colab T4 runtime.

**Status: Candidate.** The `77a116d` Kaggle run is complete hosted evidence for every mandatory stage
and for BYOD. A saved Colab Run all is still needed if the release gate requires Colab.

### Kaggle T4 execution of `d55d51c` with BYOD — 2026-09-29

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-29 | `d55d51c` / `238285bda22a` | Kaggle, Tesla T4 (`kurtvalcorza/dimer-nb2-philippine-reef-heat-stress-capstone` v2), fresh nbclient kernel on Python 3.12.13 | Run all with `USE_BYOD` and `HOTSPOT_SEMANTICS_CONFIRMED` ticked, path empty | 363.8 s | **PASSED** — 14/14 code cells executed in order, no errors. |

- **Files.** Both are preserved byte for byte:
  - the [executed notebook](execution-evidence/2026-09-29/DIMER_Philippine_Reef_Heat_Stress_Capstone_d55d51c_kaggle_byod.ipynb),
    SHA-256 `11c1341f306fb5b32ee5bde6819b0e8f4962ec45bcae6e59617830d56e641946`;
  - the executor's [run summary](execution-evidence/2026-09-29/DIMER_Philippine_Reef_Heat_Stress_Capstone_d55d51c_kaggle_run_summary.json),
    SHA-256 `1c773bdff499f999af8ce135c9c30fa5ba9375ab16622891490c3799b1964a19`.
- **Source match.** The executor fetched the notebook at the full commit SHA and verified blob
  `238285bda22a83a12238b2a0001f96396ea47144`. It then changed only the two `# @param` toggle lines
  (input blob `8fe90c98f4ce`). `inspect_run.py` confirms that is the only source difference and that
  execution counts run 1–14 in order.
- **Runtime.** Same Kaggle GPU image as the `77a116d` run (`sha256:37c64f7dd9c5…`); Tesla T4, driver
  580.159.04. The Hugging Face cache was empty at start.
- **Diagnostics from `d55d51c`, observed working.**
  - `prepare` printed `child pid 149 launched`, `[runner pid 149] imports loaded; stage prepare` and
    `Loaded 76220 regional daily records from the verified archive.`
  - Every stage log opens with `=== attempt started … ===` and closes with `=== exit 0 after … s ===`.
- **Parity with the `77a116d` run.** All model and data outputs are byte-identical to the `77a116d`
  Kaggle run: the baseline, validation, activity and test forecasts, `forecasts.csv`, `metrics.json`,
  `metrics.csv`, `outlook.csv`, the origin manifest, the seasonal reference, `failure_examples.csv`
  and the BYOD forecasts. The files that differ hold only:
  - stage timings;
  - the runner and `reef_runtime.py` digests, which the logging change was expected to alter;
  - checksums and bundle entries derived from those.
- **Journeys.**

  | Journey | Verdict |
  |---|---|
  | Default Run all, all mandatory stages | **Passed** |
  | Real Chronos-2 validation, activity and test | **Passed** (1 activity crossing day of 0.000023 °C, rearranged) |
  | Fresh-process reload parity | **Passed** |
  | BYOD with the generated example | **Passed** (`byod/bae569f03fe0b80f/`) |
  | Stage diagnostics (PID, checkpoints, per-attempt logs) | **Passed** |
  | Entry from a non-3.12 kernel | Not assessed in this run |
  | Repeated Run all in a warm runtime | Not assessed in this run |

- **Evidence boundary.** The Kaggle executor ran the committed bytes; the outputs, run summary and run
  files were inspected and compared with the `77a116d` run. Kaggle is not Colab.

**Status: Candidate.** This is complete hosted evidence for the current head on Kaggle T4. A saved
Colab Run all is still needed if the release gate requires Colab.

### Maintainer-supplied Colab execution of `d55d51c` — 2026-09-29

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-29 | `d55d51c` / `238285bda22a` | Google Colab, Tesla T4, Python 3.13.15 kernel (isolated environment on uv-managed Python 3.12.13) | Default Run all; BYOD toggles not ticked | 84 s locked sync, then 229 s of stages (05:42:39–05:46:28) | **PASSED** — 14/14 code cells executed in order, no errors. |

- **File.** The [executed notebook](execution-evidence/2026-09-29/DIMER_Philippine_Reef_Heat_Stress_Capstone_d55d51c.ipynb)
  is preserved byte for byte. SHA-256 `38155442baac3b3eb42cf8bfd93de8423afeae62e2ed9c974d7238aa9f5e9106`.
- **Source match.** The source equals `d55d51c` except for empty `# @title` lines that Colab inserted at
  the top of seven collapsed cells (`reef-003` to `reef-010`, `reef-016`). They are not substantive. Cell
  ids and order are unchanged, and execution counts run 1–14.
- **Runtime.** Colab `gpuType` T4; 12.7 GB RAM; 195 GB disk free after setup. Setup printed
  `Isolated Python 3.12.13 dependencies ready (kernel Python 3.13.15)`: hosted evidence of the R1
  entry path from a non-3.12 kernel on a complete run.
- **Stages.** Every stage printed its child PID and the runner's `imports loaded` checkpoint.

  | Stage | Time | Result |
  |---|---|---|
  | `prepare` | 7 s | Passed; 76,220 records; origin table as planned |
  | Baselines | 41 s | Passed; 299 origins |
  | Validation | 45 s | Passed; 115 origins, 0 crossing days, peak GPU 0.51 GB |
  | 180-day activity | 21 s | Passed; 1 crossing day of 0.000023 °C, rearranged |
  | compare / lock | — | Passed; 365-day context retained |
  | test / score | 36 s / 14 s | Passed; 184 origins, tables A–F printed |
  | future | 9 s | Passed; origin 2026-09-26, southern and western excluded |
  | reload / report | 28 s / 23 s | Passed; fresh-process baseline, DHW and real-model parity; `reef_evidence.zip` exported |
  | Summary (`reef-023`) | — | Passed; the 14-day table and three figures displayed |

- **Results.** The printed validation and test tables and the primary 14-day summary equal the Kaggle
  runs of `77a116d` and `d55d51c` to the printed precision. The full-test 14-day HotSpot MAE is 0.153
  for Chronos, 0.152 for persistence and 0.213 for seasonal.
- **Figures.** All three figures were inspected: the outlook, the DHW components and the failure
  examples. The outlook title names the frozen origin and the excluded regions. In the Eastern and
  Northern panels the Chronos DHW line lies under the seasonal line, because its forecast HotSpot stays
  near zero; this is model behaviour, not a plotting defect.
- **Journeys.**

  | Journey | Verdict |
  |---|---|
  | Entry from a non-3.12 kernel (R1) | **Passed** |
  | Default Run all, all mandatory stages | **Passed** |
  | Real Chronos-2 validation, activity and test | **Passed** |
  | Fresh-process reload parity | **Passed** |
  | Stage diagnostics (PID and checkpoint lines) | **Passed** |
  | BYOD | Not assessed in this run (toggles not ticked); passed on Kaggle T4 at this head |
  | Export re-run, repeated Run all | Not assessed in this run |

- **Evidence boundary.** Saved outputs were inspected; the hosted execution was not repeated. The
  evidence bundle itself was not supplied, so its contents were not inspected for this run.

**Status: Candidate.** Hosted evidence at `d55d51c` now covers every mandatory stage on both Colab T4 and
Kaggle T4, the R1 entry path on Colab, and BYOD on Kaggle. Promotion is the maintainer's decision.

### Notebook source layout change (2026-10-02)

The generator now writes the carried requirements lock, the two manifests and the base64 NOAA archive
as parenthesised runs of short string pieces (`tools/notebook_carrier.py`) instead of single
`repr` lines; the longest source line fell from 1,387,428 to 1,006 characters. Python joins the
pieces back into identical text, so the carried text is unchanged: `REQUIREMENTS`, `DATA_MANIFEST`,
`MODEL_MANIFEST` and `ARCHIVE_B64` equal the previous notebook's values byte for byte, and the
archive SHA-256 check is the same. Only cells `reef-003` and `reef-005` changed. The notebook blob
changes from `238285bda22a` to `90dfa21f20e2`. Every hosted run recorded above was of blob
`238285bda22a` or earlier; a hosted re-run of the new blob is pending. Status is unchanged.
