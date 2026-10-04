# Chronos-2 Forecasting TASK-INFERENCE Notebook — Review

**Verdict: Needs revision**  
**Review date:** 2 October 2026  
**Repository:** `kurtvalcorza/chronos-2-forecasting-pipeline`  
**Notebook:** `tutorials/chronos_2_forecasting_colab.ipynb`  
**Reviewed commit:** `d661102411f93e34fe8d8c6d343c4e95cdd11a6a` (`main`, confirmed with `gh api repos/kurtvalcorza/chronos-2-forecasting-pipeline/commits/main`)  
**Notebook Git blob:** `e06fea3dcb8da7bca1e8c663e11a57225b7cedc6`. This blob was last changed in `8485f55` and is the blob executed in the recorded Kaggle run of 2026-09-14. The recorded generating revision `d26fe8e` is an ancestor of `main`. Since then the carried modules, generator and template are unchanged; `pyproject.toml` gained two lines for the reef capstone (`dcd3d90`). Generator `--check` and `tools/validate_release_assets.py` both exit 0 at the reviewed commit.  
**Finding prefix:** `CHR`

## Executive assessment

The engineering on the default path is careful and the run reproduces. The notebook carries the package's seven modules verbatim, asserts the inline manifest against the module identity, fetches only the missing files of the pinned `amazon/chronos-2` revision, re-hashes all three, and loads without remote code. It regenerates the synthetic sample in code and asserts its SHA-256 against the checked-in digest. It holds out the last 12 hours, prints the operational ceilings, writes an input manifest with a recorded rejection, forecasts, scores Chronos-2 against last-value and seasonal-naive baselines, and exports nine files with provenance. The quantile and median semantics are stated correctly, and so are the limits of the evidence.

A direct CPU run of all 17 code cells, with the install skipped and the real checkpoint, reproduced the recorded numbers:

| Measure | This review (CPU, torch 2.11.0+cpu) | Kaggle CPU record (blob `e06fea3d`) |
|---|---|---|
| Chronos-2 MAE / RMSE (12-step holdout) | 0.33015 / 0.34603 | 0.33015 / 0.34603 |
| Last-value MAE / RMSE | 5.91665 / 6.69292 | 5.91665 / 6.69292 |
| Seasonal-naive (24) MAE / RMSE | 6.0 / 6.0 | 6.0 / 6.0 |
| Interval coverage q0.1–q0.9 | 1.0 | 1.0 |
| Effective context / prediction length | 84 / 12 | 84 / 12 |
| Input manifest | `accepted` + 1 rejection (`TARGET_MISSING_VALUES`) | same |
| Exports | 9 files | 9 files |

Three problems stand in the way of `Ready for intended use`:

1. **No one-pass `Run all` (CHR-M1).** The only recorded hosted run of this blob stopped in cell 3 with the restart `RuntimeError` (`numpy: loaded=2.0.2, installed=2.5.3`) after 181.8 s and passed only on a second attempt. `docs/release-verification.md` records it as "**PASSED** — 17/17 ok code cells executed cleanly", with no mention of the restart. No Colab run is recorded.
2. **The documented experiments are not working change-one-thing activities (CHR-M2).** `PREDICTION_LENGTH` sets the holdout as well as the horizon, so raising it also shrinks the context (84 → 72 → 48 → 24 → 3 steps). Verified: Chronos-2 falls behind last-value at 72 and 93 steps, coverage swings 1.0 → 0.39 → 0.03 → 1.0, and the seasonal comparator disappears above 72. Any value from 94 up stops in cell 23. "Toward the native 1,024-step horizon" would need 1,027 rows. The Mode D "with and without the future covariate table" comparison has no code path.
3. **Guided layer missing (CHR-M3).** The notebook is declared `GUIDED`, but it has no audience statement, how-to-use, roadmap, input→output contract, glossary, prediction, checkpoint, troubleshooting section or conclusion template. The 3,210 carried lines in Section 2 are not labelled or collapsed as infrastructure, and the forecast, metric and Mode C outputs are printed without interpretation.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `TASK-INFERENCE` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, opening cell) |
| Declared spec | DIMER Notebook Specification **2.0** (metadata, opening cell, `NOTEBOOK_SOURCE`) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** (2026-09-26), `ml-worker` `origin/main` |
| Intended audience | Not stated. Prerequisites: "basic Python and pandas; what a quantile forecast and a chronological holdout are" |
| Supported runtime | "Google Colab or Jupyter, Python 3.12"; CPU default, CUDA used when available; the `torch==2.14.0` install and the ~478 MB checkpoint are the largest downloads |
| Promised outcomes | One-pass `Run all` with no configuration edit; pinned install; seven carried modules; digest-verified pinned snapshot; synthetic sample digest-asserted; chronological holdout; ceilings; input manifest with a rejection; univariate and Mode C forecasts; median/quantile interpretation; metrics against naive baselines through an evaluation report; optional Mode D; exports with provenance; BYOD "through the same notebook-local validation, task, evaluation-report and export cells" |
| Generator | `tools/build_notebook.py` (`build_notebook.py/2`) + `tools/notebook_template.py`; recorded revision `d26fe8e` |

### Evidence actually obtained

- **Source inspection.** All 37 cells (17 code, 7 of them carried modules). Also read: the generator, the template, `tools/validate_release_assets.py`, `tools/run_notebook.py`, `README.md`, `STATUS.md`, `tutorials/README.md`, `docs/release-verification.md` and `examples/sample-data/DATASET_CARD.md`. `docs/execution-evidence/` holds only the workshop and capstone notebooks, not this one.
- **Documented execution evidence.**
  - Sources: `docs/release-verification.md`, plus the archived executor output `.agent/backups/kaggle-pass-2026-09-14/out/dimer-nb2-chronos-2-forecasting/v1/evidence/` (`run_summary.json`, `executed-pass1.ipynb`, `executed.ipynb`, workspace).
  - The run: Kaggle CPU, 2026-09-14, on **blob `e06fea3d`, the reviewed blob** (`fetched_blob_verified: true`, clean HF cache, image numpy 2.0.2 and torch 2.10.0+cpu).
  - Attempt 1 failed in cell 3 with the restart `RuntimeError` after 181.8 s.
  - Attempt 2 ran 17/17 cells in 54.0 s (`restarted_after_install_cell: true`).
  - No Colab run, no Mode D run and no BYOD run of this blob are recorded.
- **Direct execution (this review).**
  - **Environment:** `run_probes.py` on Windows, CPU only, in the repository's existing `.venv`: Python 3.12.12, numpy 2.5.3, pandas 3.0.5, transformers 4.57.6, and **torch 2.11.0+cpu, not the notebook's `torch==2.14.0` pin**. Nothing was installed.
  - **Install skipped:** cell 3 ran with `DIMER_NOTEBOOK_CI_PREINSTALLED=1`, the notebook's own executor hook.
  - **Not a clean runtime:** the pinned `model.safetensors` (sha256 `ddcda3c7…`, re-hashed) was hard-linked into a scratch working directory. The notebook's `stage_missing_files` fetched `README.md` and `config.json` from the Hub at the pinned revision, and `verify_snapshot` passed on all three files.
  - **Execution method:** every code cell ran **verbatim from the notebook JSON** in one namespace. Only form-field literals were substituted (the way an EXE6 executor does), and BYOD was driven through `DIMER_BYOD_PATH`. After each change the cells were re-run from cell 21, as the notebook instructs.
  - **Runs:** one probe run, about 2 minutes, PID 9838, exited on its own.

## 2. Separate judgments

- **Technical correctness:** sound on the default path. Identity, digest and holdout guards all hold. Validation errors from the carried package are specific (`SERIES_GAP`, `CALENDAR_FREQUENCY_UNSUPPORTED`, `EVALUATION_BAD_TARGET`, `EVALUATION_SERIES_TOO_SHORT`). The install cell is the one technical defect (CHR-M1). There is also stale-export friction (CHR-m4).
- **Promise fulfilment:** every engineering promise of the opening is met, except one-pass `Run all` (CHR-M1). The learning promise to "interpret … the tutorial metrics against naive baselines" is only partly met (CHR-m1, CHR-M3). The documented experiments do not work as described (CHR-M2).
- **Learner experience:** this is a well-documented reference notebook, not a guided one (CHR-M3). The prose is accurate but dense. A learner who reaches Section 7 sees the numbers and no explanation of what they mean on this sample.
- **Spec conformance (2.2):** applicable `MUST`s not met: RUN1, RUN10, ENV6, REL2 and REL11 (CHR-M1). GDL1–GDL15, UX5 and UX8 are `SHOULD`s and are largely unmet, without a durable record of the deviation (§2). The record drift in CHR-m5 touches REL10. Met: ST1–ST8, RUN2–RUN9, RUN11–RUN14, ENV1–ENV5, MOD1–MOD9, DAT1–DAT18 and DAT19 in part (CHR-m2), VAL1–VAL8, SPL4, the §21.3 forecasting list, UNC5, INF3, OUT1, OUT2, OUT4–OUT6, OUT8–OUT10 (OUT3/OUT7 in part; see CHR-m4), SRC1–SRC12 (notebook source; documentation drift is CHR-m5).

## 3. Promise and objective tracing

| Claim (opening / section) | Implementation | Observable result | Learner interpretation |
|---|---|---|---|
| `Run all` in a fresh runtime installs and completes with no intervention | cell 3 in-kernel `pip install` + stale-import guard | Kaggle attempt 1 `RuntimeError`; attempt 2 after a restart | **Not delivered** (CHR-M1) |
| Pinned, digest-verified model | cell 19 manifest assert, `stage_missing_files`, `verify_snapshot`, `load_pinned_model` | `verified_files: 3`, revision `95a9710e…`, `source: local-snapshot` | clear |
| Synthetic sample equals the checked-in bytes | cell 21 digest assert | `input_sha256 eff96b1a…` | clear |
| Leakage-safe chronological holdout | cell 23 `chronological_holdout` + assert | context 84, truth 12 | clear |
| Ceilings and input manifest with a rejection | cell 23 | ceilings printed; `TARGET_MISSING_VALUES` finding recorded | clear |
| Median/quantile semantics | cell 24 prose, report `score_semantics` | `prediction == q0.5` | clear |
| Interpret metrics against naive baselines | cell 27 | 0.33 vs 5.92 vs 6.0; coverage 1.0 | **No interpretation given**; the comparison is degenerate on a noiseless series (CHR-m1) |
| Mode C multi-target | cell 31 | both target names asserted | output shown but not interpreted |
| Optional Mode D | cell 35 | runs; known-future `['temperature','holiday']` | exports partial (CHR-m4) |
| BYOD through the same cells | cell 21 upload / `DIMER_BYOD_PATH` | 2-series hourly, daily, 30-row hourly and BOM-prefixed CSVs reach export; gaps, monthly, non-numeric and short series are refused with coded messages | backtest only (CHR-m3); friction (CHR-m2) |
| "Next experiments" | cell 36 prose | horizon sweep confounded; Mode D comparison absent | **Not working as stated** (CHR-M2) |

| Objective | Learner activity | Evidence it was exercised |
|---|---|---|
| install the pinned runtime | run cell 3 | needs a restart (CHR-M1) |
| read what the carried package guarantees | none; 3,210 lines, no guidance on what to read | none (CHR-M3) |
| resolve and digest-verify the revision | run cell 19 | printed dict |
| generate the sample or bring a CSV | toggle `USE_BYOD`, or set an env var | works (CHR-m2 friction) |
| create the holdout and validate | run cell 23 | manifest |
| run univariate and Mode C forecasts | run cells 25 and 31 | tables and assertion |
| interpret median/quantiles and metrics against baselines | none; numbers are printed with no prompt or note | none (CHR-M3, CHR-m1) |
| optionally run Mode D | flip a flag | runs; no comparison (CHR-M2) |
| export forecasts plus provenance | run cell 33 | 9 files |

## 4. Journeys

- **First-time learner (source inspection).** The opening is informative and honest about scope. After it, the learner meets 3,210 lines of carried package code before any model runs. Sections 6, 7 and 9 print results with no "what to notice", prediction or checkpoint, and nothing explains why the seasonal-naive error is exactly 6.0 or why coverage is 1.0 (CHR-M3, CHR-m1).
- **Clean default.**
  - Documented: Kaggle CPU, 2026-09-14, blob `e06fea3d`. Attempt 1 failed in cell 3 with the restart `RuntimeError` after 181.8 s; attempt 2 ran 17/17 cells in 54.0 s (CHR-M1).
  - Direct: local CPU, install skipped, torch 2.11.0+cpu, real checkpoint. 17/17 cells ran in 7.7 s and the metrics match the Kaggle record (table above).
  - No Colab run.
- **Active learning (direct).** I re-ran from cell 21 with `PREDICTION_LENGTH` set to 24, 48, 72, 73, 93, 94 and 1024, and I ran Mode D on (CHR-M2). The Mode D "with and without covariates" comparison has no code path, so it was not run.
- **Reuse and recovery (direct, `DIMER_BYOD_PATH` only).**
  - Accepted and carried through export: a two-series hourly CSV with a numeric covariate, a daily CSV (seasonal comparator skipped as documented), a 30-row hourly CSV (seasonal comparator skipped) and a BOM-prefixed CSV.
  - Refused with coded, actionable messages: a 14-row series, a gap, a monthly calendar and a non-numeric target.
  - Duplicate headers were refused in cell 21 with a clear message.
  - Rough edges: a target column named `value` was refused only by the missing-`target` rule, a UTF-16 file raised a bare `UnicodeDecodeError`, and a user `target_aux` column was silently overwritten by Mode C (CHR-m2).
  - The Colab upload widget was not verified.

## 5. Findings

### Major

#### CHR-M1 — `Run all` needs a manual restart after the install cell, and the release record reports a clean pass

- **Cell/section:** Section 1, cell 3 (generated by the `_INSTALL_GUARD` block in `tools/build_notebook.py` lines 46–70 and the Section 1 prose around line 453). Release records in `docs/release-verification.md` (*Manual clean-runtime evidence*, line 109) and `tutorials/README.md` ("verified — clean-runtime `Run all` execution recorded").
- **Observed issue:** cell 3 pip-installs eight pins (`numpy==2.5.3`, `pandas==3.0.5`, `torch==2.14.0`, …) into the running kernel. In a hosted image that has already imported a different NumPy, it raises `RuntimeError: Core dependencies changed while older modules were loaded … Restart the runtime, then rerun from the top.` The Section 1 prose describes this stop as designed behaviour.
- **Consequence:** the opening promises that `Run all` completes with "no configuration edit" and no intervention. In the supported runtime class it does not. The repository record calls the run "PASSED — 17/17 ok code cells executed cleanly". The workspace ledger for the same run says "17/17 ok (1 restart after install cell)".
- **Evidence (documented):** `run_summary.json` for blob `e06fea3d`:
  - attempt 1 `ok: false` after 181.8 s, with `numpy: loaded=2.0.2, installed=2.5.3`;
  - attempt 2 `ok: true` after 54.0 s;
  - `restarted_after_install_cell: true`.
- **Recommended correction:** adopt the fleet's **uv isolated-environment pattern**, which is how the capstone and newer workshop notebooks already run in one pass. The setup cell:
  - bootstraps uv;
  - creates an isolated managed interpreter (`uv venv --managed-python --python 3.12.12 <ROOT>/env`);
  - installs a hash-locked `requirements.txt` compiled with `uv pip compile` (`uv pip install --require-hashes --only-binary :all:`);
  - runs the pinned stages in that environment.

  The kernel's preloaded NumPy/torch are then never replaced, so no restart can be required. Reference implementations on `main`: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` and `bioclip2-biodiversity-pipeline/tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb`. This repository's own reef capstone already provisions an isolated environment (`tools/reef_runtime.py`, `tools/reef-requirements.lock`).

  Do not add another in-kernel install guard or loosen pins to dodge the restart. Implement the pattern in the repository's notebook generator, regenerate, re-qualify with a one-pass hosted Run all, and correct the release record so a restart-dependent run is not reported as a `Run all` PASS.
- **Acceptance check:** a hosted Colab or Kaggle `Run all` of the regenerated blob completes all code cells in **one pass** (`restarted_after_install_cell: false`), starting from a fresh runtime with an empty cache. The release records name that blob and no longer report a restart-dependent run as a clean pass.
- **Spec:** RUN1, RUN10, ENV6, REL2, REL11.

#### CHR-M2 — The documented experiments are not working "change one thing" activities

- **Cell/section:** cell 21 (`PREDICTION_LENGTH` form field), cell 23 (`chronological_holdout`), cell 35 (Mode D), and the "Next experiments" paragraph of cell 36 (template `tools/notebook_template.py` around line 412).
- **Observed issue:** the notebook's only learner experiments are the three "Next experiments". Two of them do not work as written.
  - *Horizon sweep.* "Raise `PREDICTION_LENGTH` toward the native 1,024-step horizon and watch `effective_prediction_length`." `PREDICTION_LENGTH` is also the holdout length, so each increase removes the same number of steps from the context. On the 96-step sample, the maximum is 93; every value from 94 up stops in cell 23 with `EVALUATION_SERIES_TOO_SHORT`, and 1,024 would need 1,027 rows. The seasonal comparator silently drops out above 72. Nothing in the notebook says that horizon and context are coupled.
  - *Mode D comparison.* "Compare the Mode D forecast with and without the future covariate table." Mode D builds and forecasts only the with-covariate case. It has no without-covariate variant and no future truth, so no comparison is possible without writing new code.
  - No experiment asks for a prediction first or says which cells to re-run.
- **Consequence:** the learner changes what looks like one variable and gets several confounded effects. A natural reading is "Chronos-2 gets worse at long horizons", but at 72 steps it has only 24 steps of context. Following the 1,024 instruction literally ends in an error.
- **Evidence (direct, CPU, re-run from cell 21):**

  | `PREDICTION_LENGTH` | context | Chronos-2 MAE | last-value MAE | seasonal MAE | coverage |
  |---|---|---|---|---|---|
  | 12 | 84 | 0.330 | 5.917 | 6.0 | 1.0 |
  | 24 | 72 | 0.294 | 4.668 | 6.0 | 1.0 |
  | 48 | 48 | 2.694 | 7.153 | 9.0 | 1.0 |
  | 72 | 24 | 10.568 | 9.981 | 12.0 | 0.389 |
  | 73 | 23 | 11.228 | 12.031 | skipped | 0.027 |
  | 93 | 3 | 9.342 | 8.860 | skipped | 1.0 |
  | 94, 1024 | — | `EVALUATION_SERIES_TOO_SHORT` in cell 23 | | | |

  Mode D (direct): runs, prints `['temperature', 'holiday']`, writes one CSV, and computes no comparison.
- **Recommended correction:** in the template:
  - decouple the experiment from context, either with a fixed-context holdout (for example, hold out a fixed final window and forecast a horizon of at most that size from the same cut-off) or with a second field `CONTEXT_LENGTH` passed to the forecast;
  - state the feasible range for the sample (at most 93, and at most 72 for the seasonal comparator), replacing "toward 1,024";
  - give Mode D a without-covariates call on the same history and its truth (the formula yields future demand), so the comparison exists;
  - phrase at least one experiment as Predict → Change one thing → Run → Observe → Explain, with the exact cells to re-run.
- **Acceptance check:**
  - following each "Next experiment" exactly as written (field values and re-run cells) completes without error on the sample;
  - the horizon experiment prints an unchanged `effective_context_length` across the values it suggests;
  - the Mode D experiment prints metrics for both the with-covariates and without-covariates runs.
- **Spec:** GDL7, GDL10, UX5, UX7, ENV8.

#### CHR-M3 — Declared `GUIDED`, but the guided layer is largely absent

- **Cell/section:** opening cells 0–1; Section 2 (cells 4–17); Sections 6, 7 and 9; cell 36. Generator `tools/build_notebook.py` (opening and Section 2 prose, lines ~364–440) and `tools/notebook_template.py`.
- **Observed issue:** markdown search: 0 hits for "how to use", "roadmap", "glossary", "check your", "troubleshoot", "expected result", "what to notice", "conclusion" and "audience", and 2 hits for "Look for" (Sections 1 and 4 only).
  - The task is never stated as an Input → Model → Output contract.
  - No result section has a prediction prompt, interpretation note or checkpoint.
  - The seven carried module cells (3,210 lines) are introduced as "Pipeline code". They are not labelled **Infrastructure** and not collapsed (`cellView` unset), and they sit between the install and the first model step.
  - The objective "read what the carried package guarantees" has no guidance on what to read.
  - The interpretation cell states the same "proves" sentence twice in succession.
- **Consequence:** a learner new to forecasting cannot tell which code they need to study, what normal output looks like, or how to turn the printed metrics into a conclusion. The notebook is a good `REFERENCE` notebook labelled `GUIDED`.
- **Evidence (source inspection):** static probe in `results.json` (`probes.static`).
- **Recommended correction:** in the generator and template, add:
  - the GDL1–GDL4 opening (audience, how to use, roadmap, Input → Model → Output);
  - `# @title Infrastructure: …` with `cellView: form` on cell 3 and the seven carried cells, plus a one-line "you may run this without reading it";
  - a short glossary (context, horizon, quantile, pinball loss, coverage, seasonal-naive);
  - a prediction and a "What to notice" note around Sections 6, 7 and 9, with collapsible sample answers;
  - a troubleshooting list (install/restart, Hub download, digest mismatch, BYOD refusals);
  - a conclusion template;
  - removal of the duplicated "proves" sentence.

  The fleet reference for the layer is `prithvi-flood-segmentation-pipeline/tutorials/DIMER_Philippines_Flood_Mapping_Capstone.ipynb`.
- **Acceptance check:** the regenerated notebook contains:
  - a How to use section and a roadmap;
  - an Input → Model → Output statement;
  - a glossary;
  - a prediction prompt before, and a "What to notice" note after, each of Sections 6, 7 and 9;
  - at least one collapsible sample answer;
  - a troubleshooting section and a conclusion template;
  - every carried/install cell titled `Infrastructure:` and set to `cellView: form`.
- **Spec:** GDL1–GDL15, UX4, UX8.

### Minor

#### CHR-m1 — The baseline comparison on a noiseless sample is degenerate and unexplained

- **Cell/section:** Section 7, cell 27 and its prose (template around line 202).
- **Observed issue:** the sample contains no random draw (DATASET_CARD: "no random draw"). It is a trend of 0.25 per step plus two sinusoids, so:
  - the 24-step seasonal-naive forecast misses by exactly the trend, 24 × 0.25 = 6.0, giving MAE = RMSE = 6.0;
  - interval coverage of 1.0 over 12 points of a noiseless curve says nothing about interval quality;
  - Chronos-2's MAE of 0.33, against 5.92 and 6.0, reflects how easy a smooth deterministic function is to extrapolate.

  The prose labels the metrics "sanity" but does not explain any of this.
- **Consequence:** a learner can read an 18× advantage and perfect coverage as evidence about Chronos-2, which the limits section disclaims only in general terms.
- **Evidence:** direct (the table above); source (`examples/sample-data/DATASET_CARD.md` line 7).
- **Recommended correction:** add a "What to notice" note that explains the exact 6.0 (seasonal-naive repeats yesterday, and the trend adds 6 per day) and why coverage 1.0 on 12 noiseless points is uninformative. Optionally, see CHR-S2.
- **Acceptance check:** the Section 7 prose names the trend as the source of the seasonal-naive error and states that coverage on this sample does not assess calibration.
- **Spec:** EVAL3, EVAL15, DAT8.

#### CHR-m2 — BYOD friction: env-var-only path, fixed column names, a bare decode error and a silent column overwrite

- **Cell/section:** cell 21 (template lines 75–135) and cell 31.
- **Observed issue:**
  - The non-interactive BYOD location is only the environment variable `DIMER_BYOD_PATH`; there is no form field.
  - The column names `series_id` / `timestamp` / `target` are fixed, with no field. A CSV whose target is named `value` is refused in cell 23 by the missing-`target` rule, without saying that the column must be renamed.
  - A non-UTF-8 file raises a bare `UnicodeDecodeError` in cell 21.
  - Mode C overwrites a user column named `target_aux` without notice.
- **Consequence:** users must rename columns by hand, and a common encoding mistake gives a traceback instead of a contract message.
- **Evidence:** direct: `target_named_value` gives `EVALUATION_MISSING_COLUMNS` in cell 23; `non_utf8_bytes` gives `UnicodeDecodeError` in cell 21; `has_target_aux_column` runs, with the column replaced.
- **Recommended correction:**
  - add form fields `BYOD_CSV_PATH`, `ID_COLUMN`, `TIMESTAMP_COLUMN` and `TARGET_COLUMN`, used by `ForecastConfig` (keeping the env var as an optimisation);
  - catch `UnicodeDecodeError` in `read_checked_csv` with "file must be UTF-8 CSV";
  - have Mode C refuse, or rename around, an existing `target_aux`.
- **Acceptance check:**
  - a CSV with a differently named target runs once `TARGET_COLUMN` is set;
  - a UTF-16 file raises a `ValueError` naming the UTF-8 requirement;
  - a BYOD frame with `target_aux` is not silently altered.
- **Spec:** DAT19, UX10, EXE1, EXE2, EXE5.

#### CHR-m3 — BYOD can only backtest; the described `not-measurable` path is unreachable

- **Cell/section:** cells 23–27; Section 7 prose.
- **Observed issue:** every path holds out the last `PREDICTION_LENGTH` rows. A user can never forecast beyond the end of their own data. The Section 7 prose describes the `not-measurable` verdict for "a forecast of the real future", but no notebook control reaches it.
- **Consequence:** the practical transfer step, forecasting the next hours of one's own series, requires writing code.
- **Evidence:** source; direct (`forecast_timestamps_within_input: true` for every accepted BYOD case).
- **Recommended correction:** add an optional `FORECAST_FUTURE` branch that calls `forecast(frame, config, pipe)` on the full history, writes `evaluation_report(result, None, …)` and exports it.
- **Acceptance check:** with the branch on, the exported forecast timestamps start one step after the last input timestamp and the report verdict is `not-measurable`.
- **Spec:** INF2, UX9.

#### CHR-m4 — Mode D exports are partial and outlive the run that made them

- **Cell/section:** cell 35 and its prose; cell 33.
- **Observed issue:**
  - The prose says "the provenance names which covariates were read as known-future". The Mode D provenance is only printed; only `chronos_covariate_forecast.csv` is written, and neither `chronos_2_forecasting_result.json` nor `chronos_provenance.json` describes the Mode D run.
  - After a run with Mode D on, re-running with it off leaves the earlier `chronos_covariate_forecast.csv` in `outputs/`.
- **Consequence:** `outputs/` can hold a file from a previous configuration, and nothing records which run produced it.
- **Evidence:** direct: P4 and P5 (`stale_covariate_csv_present: true`).
- **Recommended correction:** write the Mode D provenance and request to `outputs/chronos_covariate_provenance.json` and reference it from the result JSON. Clear or overwrite Mode D files when the branch is off, or write each run to its own folder.
- **Acceptance check:** after on-then-off, `outputs/` contains no Mode D file. With Mode D on, the exported provenance lists `temperature` and `holiday` as known-future covariates.
- **Spec:** OUT3, OUT7.

#### CHR-m5 — Release records and docs disagree with the notebook and with each other

- **Cell/section:** `STATUS.md`; `README.md` line 56; `docs/release-verification.md` lines 3–6, 16 and the *Current status* paragraph (~188–204); `tutorials/README.md` promotion sentence.
- **Observed issue:**
  - `STATUS.md` and `README.md` describe the notebook under Spec 1.1, and `STATUS.md` says it is "awaiting clean-runtime execution".
  - `release-verification.md` says the validator checks spec `1.1`, but `tools/validate_release_assets.py:125` checks `2.0`.
  - Its *Current status* paragraph says the real Hub fetch and the carrier "have never been executed". The Kaggle row of 2026-09-14 on the same page contradicts this.
  - `tutorials/README.md` gates promotion on "Specification 1.1 MUST requirements".
  - The run's wall time and file count ("8 files") omit the restart and the SVG.
- **Consequence:** a reviewer cannot tell from the records which spec governs, or whether the carrier has run.
- **Evidence:** source inspection; `tools/validate_release_assets.py` exit 0.
- **Recommended correction:** make all four documents name the same spec version and the same evidence. Rewrite the *Current status* paragraph against the recorded run, and record the restart (see CHR-M1).
- **Acceptance check:**
  - `grep -n "1\.1" STATUS.md README.md tutorials/README.md docs/release-verification.md` finds no Notebook Specification reference for this notebook;
  - the *Current status* paragraph cites the recorded run.
- **Spec:** REL10, SRC10.

### Suggestions

- **CHR-S1 — Update the declared spec.** The notebook, generator (`NOTEBOOK_SPEC = "2.0"`, `tools/build_notebook.py:31`) and validator declare 2.0; the current baseline is 2.2.
- **CHR-S2 — Make the sample informative.** Add seeded noise, or a second real series such as the bundled Open-Meteo data, or a multi-origin backtest, so that MAE gaps and coverage carry information.
- **CHR-S3 — Show the context in the plot.** The SVG draws only the 12-step horizon, without the history or axes. Including the last 24–48 context steps would let the learner see what the model extrapolated from.
- **CHR-S4 — Record per-stage wall times in `result.json`.** The install and the 478 MB download dominate the run, and per-stage times would make the record's 236 s interpretable.

## 6. Readiness

**Needs revision.** Three Major findings are open, and the applicable `MUST`s RUN1, RUN10, ENV6, REL2 and REL11 (CHR-M1) are not met. The recorded execution evidence is for the reviewed blob, but it documents a restart, so it does not establish a one-pass `Run all`. The guided-layer and experiment defects (CHR-M2, CHR-M3) are `SHOULD`-level for the spec but Major for the declared `GUIDED` audience.

Remaining gates after the fixes:
- a one-pass hosted `Run all` of the regenerated blob, recorded in `docs/release-verification.md`;
- a hosted BYOD run with one representative CSV and one clear refusal (REL12);
- each "Next experiment" re-run exactly as written.

## 7. Verified versus inferred

- **Verified by direct execution (CPU, torch 2.11.0+cpu, not a clean runtime):**
  - the default-path metrics and exports;
  - the horizon/context coupling and the sweep table;
  - the cell 23 failure from 94 steps up;
  - Mode D behaviour and the stale CSV;
  - the BYOD acceptances, refusals and rough edges.
- **Verified from documented evidence:** the restart in the recorded Kaggle run of the reviewed blob.
- **Inferred from source:** Colab behaviour of the install cell (the same guard, and Colab preloads NumPy 2.0.x); GPU numerics; the upload widget path.
- **Not verified:** any Colab run; the `torch==2.14.0` pin itself; learner understanding.
- **Finding most likely to be wrong:** CHR-M2's severity. The coupling is certain, but a maintainer could argue that the experiment is meant to show the context/horizon trade-off. If the prose said so, the defect would reduce to the infeasible "toward 1,024" wording and the missing Mode D comparison, which is Minor.
