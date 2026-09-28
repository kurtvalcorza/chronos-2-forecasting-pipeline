# DIMER multi-model time-series forecasting notebook review

**Verdict: Needs revision**  
**Review date:** 27 September 2026  
**Framework:** DIMER Notebook Review Framework v1 — executable software, instructional content, learner journeys, promise fulfillment, and relevant specification requirements.

## 1. Review contract and evidence boundary

| Field | Reviewed value |
|---|---|
| Repository | `kurtvalcorza/chronos-2-forecasting-pipeline` |
| Notebook | `tutorials/DIMER_MultiModel_TimeSeries_Forecasting_Workshop.ipynb` |
| Repository commit | `e13b36249cb3b9e4cd46993ccb581dbe2d6b94e0` |
| Notebook Git blob | `8d65d164525e46e81649c581d4983812cfa67e41` |
| Declared profile / mode | `TASK-INFERENCE` / `WORKSHOP` |
| Declared notebook specification | 2.1 |
| Current fleet specification checked | 2.2, dated 26 September 2026 |
| Audience | Readers able to run Colab and read basic Python, but new to forecasting / ML |
| Recommended host | Colab T4; separate Python 3.12 model environments |
| Default | `STANDARD`, `SYNTHETIC`, 24-step validation and test horizons, seasonal period 24 |
| Models | TiRex-2 and Chronos-2 by default; Toto 2.0 2.5B in `FULL` |
| Optional journeys | Embedded Open-Meteo sample, file/upload BYOD, other accepted horizons/frequencies, seasonal-period activity |
| Review authorization | Read and review only. No repository edits, PRs, issues, or releases were created. |

Sources: [reviewed notebook][notebook], [release verification at the reviewed commit][release], [notebook-specific design proposal][design], [fleet specification][fleet]. The design document is explicitly a revised proposal; where it conflicts with the released learner workflow, this review identifies the conflict rather than silently treating every proposal detail as a release requirement.

### What was actually checked

The review used the notebook source, its release-verification record, and relevant upstream API definitions. Eleven local checks exercised transcribed source-boundary logic, controlled forecast fixtures, and arithmetic. The local environment was Python 3.13.5, NumPy 2.3.5, pandas 2.2.3, and Matplotlib 3.10.8.

**The complete original notebook was not executed. No foundation-model weights were downloaded and no pretrained forecasting inference was run.** The probe script contains manually transcribed, explicitly scoped source excerpts; it is not a byte-identical notebook executor. Where forecasts were needed to exercise the scorer, controlled tables stood in for model outputs. A reproduced fault-injection case does not establish that the upstream models generated that fault in a recorded run.

The pre-model EDA plot was reproduced locally and visually inspected. Saved plots from the maintainer's complete Colab runs were not independently rendered during this review. Learner-experience judgments are expert inspection, not measured learning effectiveness.

### Existing execution evidence

The release record contains historical passing Colab runs for `STANDARD` with both built-in samples, and `FULL` with the weather sample. It also records the earlier Toto `MPLBACKEND` failure and its subsequent correction; that already-fixed issue is not reported here as a new defect.

The latest maintainer-supplied execution entry identifies source commit `9e10f240fd76eb6799eded892776a60ec21539c8` and executed-file SHA-256 `b84f24987b939b073b3147535491c7399a37b9df1d5fe5f4a119b82e2124ac9e`. According to that entry, all 19 code cells executed with no saved errors and terminal/export outputs were present. Its configuration was **STANDARD + OPEN_METEO_PH**, not the default synthetic setting. This entry supersedes the earlier pending-rerun statement for that source/configuration. The release record leaves full downstream BYOD qualification open.

These are documented executions, not independent executions by this reviewer. They should be retained and associated with their exact identities rather than discarded because an authored metadata field still says `pending`.

## 2. Executive assessment

The notebook is a substantive guided tutorial, not merely a model launcher. It explains forecast origins, advancing history, baselines, point forecasts, quantiles, coverage, and uncertainty. It implements a real post-dataset forecast rather than only relabeling an existing prediction table. Its ready-made seasonal-period activity changes a meaningful variable and uses the same validation targets in both conditions.

The principal problems are inconsistencies between the lesson and its execution boundaries:

- Test targets are shown during development despite an unseen-test lesson.
- TiRex and Chronos use different cross-series conditioning modes without that distinction being taught.
- Numeric-looking BYOD identifiers pass initial validation but fail after CSV round-tripping.
- The shared evaluator accepts incomplete forecast grids and invalid quantile outputs.
- The freeze does not bind the actual staged model inputs, and a later report can include older artifacts.

These findings do not establish that the documented default runs failed. They do establish that a successful default run alone is insufficient to sign off the instructional and interactive contract.

## 3. Major findings

### TS-R01 — Pre-model EDA reveals the test targets

**Severity:** Major  
**Dimensions:** Scientific validity, learner progression, promise fulfillment  
**Location:** Section 4.1, cell `dimer-ts-workshop-16`; contradictory guidance in Sections 8 and 9  
**Evidence:** Source inspection, local plot reproduction, visual inspection; probe P02

The first EDA plot iterates over the full `frame` and plots every target. The normalized overlay likewise uses all target values, and the per-series summary covers the complete dataset. Split-boundary lines annotate these observations; they do not conceal the test segment.

The synthetic sample therefore exposes all 24 test targets for A and all 24 for B before model comparison and freezing. The same code exposes the final weather day. Section 9 nevertheless says that the test values stay hidden until scoring, and Section 8 asks learners not to look at them while making choices.

**Consequence:** A learner can follow every instruction and still see the held-out pattern before selecting settings. This undermines the lesson about protecting an independent test and can allow test-informed choices. It is not evidence that future targets were passed into the model histories: the chronological split itself was correct in the local check. The distinction is between model-input leakage and development/selection exposure.

**Correction:** Establish the test boundary before EDA. Plot and summarize only the permitted development portion, fit any display normalization on that portion, and mark the later test interval without revealing its values. Show complete overlays only after freezing/evaluation. An explicitly exploratory full-series view is possible, but it must invalidate an untouched-test claim rather than coexist with that claim.

**Acceptance:** On both samples and BYOD, no pre-freeze plot or data-dependent summary exposes test targets. Tests should inspect plotted data and table inputs, not merely check that a boundary line exists. Keep the positive chronological-split check.

**Specification relevance:** Split/selection integrity, SPL6–SPL7 and the forecasting task requirements; UX1–UX4. The notebook-specific design itself says test values remain hidden, while its EDA section is not sufficiently explicit about truncation. Clarify the design as well as the implementation.

### TS-R02 — The purported common-input comparison mixes joint and independent forecasting

**Severity:** Major  
**Dimensions:** Experimental validity, technical content, promise fulfillment  
**Location:** Section 6.2, cell `dimer-ts-workshop-24`, TiRex and Chronos runner construction  
**Evidence:** Notebook configuration and upstream API semantics; numerical impact not measured

TiRex receives a single `TimeseriesType` with `target` shaped `(number_of_series, history_length)`, wrapped in a one-element list. The [upstream TiRex guide][tirex-guide] describes this as forecasting multiple target variates jointly. A batch of independent series is represented by a list of separate `TimeseriesType` objects.

Chronos instead receives the long-form table with separate `series_id` values, `target=["target"]`, and `cross_learning=False`. Its [pinned v2.3.1 implementation][chronos-source] prepares separate inputs by item and documents cross-learning as the mechanism for sharing information across those inputs. This configuration is one univariate task per item, not the same joint multivariate grouping supplied to TiRex.

**Consequence:** The models receive the same collection of historical values but not the same configured conditioning structure. The lesson frames the comparison as differing models under a common two-channel input contract without explaining this additional experimental difference. It can lead learners to attribute an effect of grouping or information sharing solely to model identity.

This is not a claim that joint conditioning must improve accuracy, or that it caused any particular recorded ranking. Such a comparison can be useful when labeled honestly. It is not the strict matched-mode experiment described in the notebook.

**Correction:** Choose and document a canonical grouping policy. Either forecast each series independently for every model, or supply an equivalent joint target group to each compatible model. Keep independent-versus-joint inference as a separately labeled experiment. Do not assume that simply passing the same CSV guarantees equivalent conditioning. Verify Toto's `series_ids`/attention grouping as part of the same contract; this finding does not assume its behavior solely from tensor shape.

**Acceptance:** Record per-model target groups and cross-series conditioning settings in run metadata. Add adapter tests for the intended grouping and a controlled perturbation experiment: change one series while holding another fixed and examine whether cross-series dependence matches the declared mode. Then rerun the numerical comparison and regenerate any affected reference answers.

**Specification relevance:** Honest experiment and evaluation semantics, EVAL5/EVAL8; common-input design sections 6 and 10; GDL4/GDL7.

### TS-R03 — Numeric-looking series IDs pass validation but fail downstream

**Severity:** Major  
**Dimensions:** Technical correctness, BYOD transfer, recovery  
**Location:** Sections 3.1–3.2 and 7.1; cells `dimer-ts-workshop-12`, `-14`, and `-26`  
**Evidence:** Transcribed CSV/validation/evaluation path with generated forecasts; probe P04

The initial reader uses `pd.read_csv` without an explicit identifier dtype. Validation converts `series_id` to strings. Model runners also convert IDs to strings, then serialize forecasts. The parent `run_model` reads those forecast CSVs without a string dtype, allowing numeric IDs to become integers again.

With IDs `101` and `202`, the local input check accepted both series and retained string IDs in the validated truth. Reloaded forecast IDs became `int64`. Concatenating these forecasts with the string-ID baselines and evaluating produced:

```text
ValueError: Forecast rows do not align exactly to held-out truth
```

Alphabetic IDs passed the positive control. No pretrained models were executed; a well-formed forecast fixture reproduced the exact serialization and scoring boundary.

**Consequence:** A user may supply a valid, regularly sampled CSV, wait for model setup/inference, and then receive an alignment error caused by the notebook's own identifier handling. Leading-zero identifiers are additionally vulnerable to loss of identity at initial CSV import.

**Correction:** Treat identifiers as strings from the first read, not only after inference has already occurred. Preserve that contract in every history/forecast CSV reader and writer. Make missing-ID handling explicit, including any valid literal IDs that pandas would otherwise interpret as missing.

**Acceptance:** Exercise IDs `A`, `101`, `001`, and mixed forms through acquisition, history staging, forecast reload, evaluation, future output, and export. Preserve leading zeros and distinguish identifiers such as `001` from `1`. Complete hosted BYOD qualification after the local boundary tests.

**Specification relevance:** DAT13/DAT15/DAT19, VAL2–VAL4, INF4, OUT4.

### TS-R04 — The common evaluator accepts missing forecast rows and invalid quantiles

**Severity:** Major  
**Dimensions:** Technical correctness, experimental validity, interpretation  
**Location:** Section 5.1, cell `dimer-ts-workshop-19`, `evaluate_forecasts`  
**Evidence:** Fault-injection fixtures through the transcribed evaluator; probes P05–P06

The evaluator rejects duplicate forecast keys and forecasts with no matching truth. It does not enforce the reverse condition: that every model supplies every required `(series_id, timestamp)` in the holdout. A left join cannot establish that completeness on its own.

A controlled 48-row forecast fixture had one error of 100 units. Its full pooled MAE was 2.083333. Removing that one forecast row left 47 rows, which the evaluator accepted and scored at MAE 0. Omitting a whole series was also accepted.

The same evaluator accepted reversed q0.1/q0.9 bounds and reported mean width −2.0. It also accepted missing lower quantiles and turned their coverage comparisons into zero coverage rather than rejecting an invalid forecast. Point-equals-median assertions in the runners do not validate the lower/upper quantiles, and shape checks in some adapters do not replace validation of the shared table.

**Consequence:** Missing difficult predictions can improve a score unnoticed, and invalid uncertainty outputs can look like poor calibration rather than a contract failure. These are demonstrated gaps in fault detection—not evidence that recorded upstream forecasts contained such faults.

**Correction:** Before scoring or plotting, validate the exact expected model × series × timestamp key set; unique step numbering and origins; finite numeric predictions/quantiles; median identity; and ordered quantile bounds. Treat known quantile crossing with an explicit, justified policy rather than silently sorting outputs. Apply the same output contract to unscored future forecasts.

**Acceptance:** Reject a missing horizon row, missing series, unexpected series, duplicate row, shifted timestamp, non-finite quantile, and crossed bounds. A complete valid table must pass and retain the existing metric values.

**Specification relevance:** EVAL2/EVAL5/EVAL8, INF3, OUT4, UNC1; notebook-specific design section 13 expressly prohibits silent row dropping and horizon truncation.

### TS-R05 — Freeze checks verify declared state, not the actual staged inputs

**Severity:** Major  
**Dimensions:** Experiment integrity, technical correctness, reproducibility  
**Location:** Sections 5.1, 8.1, and 9.1; cells `dimer-ts-workshop-19`, `-31`, and `-33`  
**Evidence:** Source inspection and controlled staged-file mutation; probe P07

Section 5.1 writes shared request files such as `work/requests/test_history.csv`. Section 9.1 checks the original input digest variable, tier, selected models, quantiles, test horizon, seasonal period, model revision, and runner hash. It does not verify the bytes of the request CSV that is then supplied to the model.

The local probe changed a target in the staged test-history CSV while leaving the recorded original-input digest and the checked configuration unchanged. All visible freeze assertions passed. The test request had different bytes, but the freeze did not detect that difference. This was a controlled test, not an observation that a recorded Colab run was contaminated.

The freeze also omits the environment pins that its preceding prose says it records, and recorded context/validation-horizon fields are not all rechecked. Reexecuting the freeze cell overwrites the same file; there is no durable evaluated-state guard.

**Consequence:** Learners may reasonably interpret “frozen” as stronger protection than the notebook implements. A changed or stale staged request can be used under an unchanged declaration of the original data. A filename and a cached digest variable are not sufficient evidence of the input actually consumed.

**Correction:** Create an experiment identity that includes the source digest, actual stage-input digests, split boundaries, context settings, model/runner identity, environment configuration, and evaluation configuration. Derive final requests from verified frozen state and rehash the actual files before execution. Preserve completed freezes; make new experiments explicit. State clearly that these controls do not prevent a human from having seen test information.

**Acceptance:** Independently modify a stage-input value, timestamp, model code, context configuration, and environment signature. Every mismatch should be detected before a new final-model call. An unchanged frozen rerun should have explicitly documented behavior rather than silently creating a new claim.

**Specification relevance:** ENV9, SPL6–SPL7, OUT6–OUT8, and the notebook's explicit freeze promise. Not every suggested state-machine detail is a fleet MUST; the observed issue is the gap between claimed and implemented experimental identity.

### TS-R06 — A new report can contain an older run's files

**Severity:** Major  
**Dimensions:** State management, completion/transfer, reproducibility  
**Location:** Sections 1.1 and 13.1; cells `dimer-ts-workshop-05` and `-42`  
**Evidence:** Controlled filesystem reproduction of the directory/archive logic; probe P08

The notebook uses fixed default `outputs/` and `work/` locations, creates directories with `exist_ok=True`, and archives everything under `OUTPUT_DIR`. It neither creates a fresh experiment output root nor restricts archive membership to the current run.

If a FULL run is followed by STANDARD in the same workspace, older `test/toto.csv` and Toto plots remain. They can be packaged into a new ZIP whose current summary lists only TiRex and Chronos. An optional seasonal-activity CSV from an earlier run can likewise be swept into a later report, even though the activity text says it is separate from the earlier export.

The probe created the same path layout and a current STANDARD summary, then used the notebook's archive call. The ZIP included the old Toto forecast, old Toto plot placeholder, and old activity file. It did not execute either model tier.

**Consequence:** A learner receives an internally inconsistent experiment record and can accidentally share an older dataset's results as part of a new run. Running the two samples sequentially is explicitly encouraged, so workspace reuse is a normal learner journey rather than an exotic threat model.

**Correction:** Use separate, clearly named output directories per experiment and an explicit current-run file inventory. Keep reusable model caches separate from result artifacts. Preserve earlier reports; do not resolve this by blindly deleting an arbitrary user-specified output directory.

**Acceptance:** Run FULL then STANDARD, and one sample then another, with an optional activity in between. Each report must contain only its own selected models, data identity, figures, and intentionally included activities. Earlier reports must remain available.

**Specification relevance:** OUT6–OUT9, GDL14, UX3/UX10; fulfillment of the current-experiment export promise.

## 4. Smaller content and learner-experience issues

### TS-R07 — “Real future” confuses a dataset boundary with the present date

**Severity:** Minor, but conceptually important. Section 11 does run new inference on observations beyond the supplied data. It is not merely displaying old test predictions. The problem is its claim that those timestamps have not happened yet.

With the fixed synthetic sample, the forecast covers 5 January 2026. With the fixed weather sample it covers 9 September 2026. Both precede this review date. For synthetic data, future values could also be generated from the declared formula, although they are not supplied to this notebook's evaluator.

Prefer **“Forecast beyond the supplied dataset”** and **“Not evaluated: matching future targets were not provided.”** Describe genuinely prospective operation as using recently acquired observations up to an actual current forecast origin. Preserve the valuable distinction between held-out evaluation and unscored inference.

### TS-R08 — Macro MAE is not scale normalization; the overlay is a z-score

**Severity:** Minor. Section 5 contrasts scale-dependent pooled error with macro error that gives each series equal weight. Both MAE quantities remain in target units. Because this notebook requires the same grid and horizon for every series, pooled MAE and macro MAE are mathematically equal when all forecasts are complete. Averaging per-series errors does not eliminate differences in units or scale. The local arithmetic check gave 50.5 for both with equal-length series having errors 1 and 100.

Explicitly distinguish aggregation from normalization. Per-series skill is a separate, baseline-relative quantity, with undefined cases when the seasonal baseline has zero error; show how many valid series contribute. Do not describe a large numerical level alone as making MAE large: shifting truth and forecast together does not change the error.

Section 4 also describes the overlay as putting series on the “same range,” but implements mean subtraction and standard-deviation division. Say **standardized scale / z-scores**, not a shared min–max range.

### TS-R09 — Slow stages are opaque, and FULL hardware checks arrive late

**Severity:** Minor. The runner captures subprocess output and prints only a tail after completion. The learner sees the model name but little progress during a long first download. Toto's CUDA check occurs after checkpoint staging, so a missing accelerator can be discovered after attempting a 9.8 GB download.

Preflight the selected tier's accelerator and practical disk/memory requirements before expensive acquisition. Provide stage-level progress and elapsed-time feedback; keep technical logs available without overwhelming the teaching outputs. Treat the current CPU-versus-GPU runtime comparison as a comparison of the demonstrated configurations, not an intrinsic architecture speed ranking.

### TS-R10 — The exported reproducibility record is incomplete

**Severity:** Minor for immediate interaction; relevant to specification conformance. The notebook displays host package versions but does not verify a tested range for the parent stack. Its model environments have principal package pins, yet the exported manifest primarily records the top-level runtime package and generic workshop revision. The freeze text promises environment pins that the freeze object does not contain.

Export exact notebook identity, intended environment pins, effective installed package versions, material precision/device settings, and the actual run configuration. Distinguish controlled settings from observed versions. A generic `0.1.0-candidate` label is not a unique notebook revision.

### Optional improvement — Give the learner a result before the long installation

Section 5 builds baselines but waits until Section 7's completed model runs to show their combined scores. Display the inexpensive baseline scores immediately after constructing them, alongside the learner's prediction. This shortens the wait for the first interpretable result without changing the experiment.

## 5. Strengths and verified positive behavior

- **Appropriate profile:** This is genuinely zero-shot `TASK-INFERENCE`. Artificial fine-tuning or an adaptation artifact is not required merely to mimic E2E symmetry.
- **Concrete workflow:** Input/output semantics, forecast origin, context, horizon, and model quantiles are explained. Validation observations legitimately become historical context at the later test origin.
- **Useful baselines:** The local generator reproduced the published synthetic digest. The split produced 48/72/96 context steps for validation/test/post-dataset inference. No held-out target appeared in its corresponding forecast history. Seasonal-naive macro MAE was 3.60, versus approximately 5.5348 for last value; these agree with the worked baseline explanation.
- **Meaningful activity:** The optional 12-versus-24-period comparison used identical 48 validation targets and changed only the baseline period. The local per-series MAEs were 7.037883/7.376817 for period 12 and 4.32/2.88 for period 24. This supports the intended change-one-thing lesson.
- **Useful negative validation:** Missing targets, duplicate timestamps, a gap, and non-finite targets were rejected in the local checks. Numeric identifiers expose a different downstream boundary not covered by those successes.
- **Careful uncertainty language in the main lesson:** It distinguishes median from mean and quantiles from guaranteed coverage. Chronos's pinned implementation confirms that its generically named point output is actually the 0.5 quantile; this should not be falsely flagged as a mean/median mismatch.
- **Real inference stage:** Section 11 executes a new forecast beyond the supplied dataset. Its temporal wording needs correction, not its existence.
- **Recorded execution discipline:** The release record separates notebook variants, samples, tiers, and local versus hosted evidence. Preserve that granularity.

## 6. Promise-to-evidence assessment

| Promise | Assessment |
|---|---|
| Run pretrained models locally without training or DIMER services | Implemented in source; supported by documented Colab runs, not independently rerun here |
| Compare models under one common input contract | Same history timestamps and target values, but conditioning/grouping mismatch needs correction or explicit reframing |
| Keep test information separate until final evaluation | Model histories are chronological; learner-facing pre-freeze EDA violates the unseen-test promise |
| Score every model on the same holdout | Intended, but shared evaluator does not enforce complete key coverage |
| Reuse the workflow with a compatible user CSV | Real path exists; numeric identifiers fail after serialization; full hosted BYOD qualification remains open |
| Perform one bounded active-learning experiment | Ready-made seasonal-period activity passed the local control checks |
| Forecast beyond the observed sample | Implemented; “real future” wording is inaccurate for fixed historical samples |
| Export the current reproducible experiment | Detailed outputs exist, but current-run membership and complete provenance are not reliably enforced |

## 7. Framework judgments and specification treatment

| Dimension | Judgment |
|---|---|
| Promise fulfillment | Needs revision: hidden-test, common conditioning, current-run export |
| Technical correctness | Needs revision: identifiers, output contract, staged-state verification |
| Scientific/experimental validity | Needs revision: development exposure and model grouping |
| Learner orientation/progression | Strong structure; unsupported temporal wording and delayed first results need attention |
| Explanations/result interpretation | Substantive; several explanations overstate the controls or conflate quantities |
| Meaningful learner activity | Positive local evidence for the bounded seasonal-period task |
| Interaction/pacing/recovery | Needs improvement for BYOD failures, multiple runs, and long first downloads |
| Completion/transfer | New forecasts are real; clean current-run exports and durable identity need work |

The notebook declares specification 2.1, whereas the current fleet document is 2.2. Requirement mappings above are selected applicable references, not a claim of exhaustive compliance against either version. Known mandatory conformance failures cannot be averaged away by good pedagogy. Conversely, a later spec version alone is not evidence that the original notebook claimed compliance with it.

## 8. Prioritized acceptance plan

1. **Restore the experiment's meaning:** conceal test targets during development, choose a consistent conditioning policy, and regenerate affected worked guidance from the corrected run.
2. **Protect data and output boundaries:** preserve string identifiers across files; reject incomplete, non-finite, or incoherent forecast tables; bind actual staged inputs to the freeze.
3. **Isolate runs and exports:** use experiment-specific result roots, explicit artifact membership, and complete revision/environment metadata.
4. **Re-execute supported journeys:** fresh STANDARD synthetic; STANDARD weather; FULL weather on the intended hardware; string/numeric/leading-zero BYOD; corrected rejection paths; sample/tier switching; and the optional activity. Record exact source/configuration identities and export digests.
5. **Observe a representative learner:** ask them to identify what is hidden, explain the shared-input contract, interpret coverage versus width, make one controlled change, recover from an input error, and locate the correct report without instructor intervention.

**Release decision:** Needs revision. Historical passing runs remain valid evidence for the configurations they exercised. The local probes neither replace hosted verification nor erase those runs. No source-system changes were made.

## 9. Probe receipt

| Probe | Scope / outcome |
|---|---|
| P01 | Canonical digest, chronological splits, baseline arithmetic: positive controls passed |
| P02 | Reproduced 24 test targets per synthetic series in pre-model EDA |
| P03 | Missing target, duplicate time, gap, infinity: negative controls passed |
| P04 | Numeric-ID CSV round-trip: accepted input, downstream alignment error; alphabetic control passed |
| P05 | Missing forecast row and missing whole series accepted; duplicate row correctly rejected |
| P06 | Crossed and missing quantiles accepted by shared evaluator |
| P07 | Changed staged test-history bytes pass visible freeze assertions |
| P08 | Shared output-root archive includes older Toto and activity files |
| P09 | Equal-grid pooled and macro MAE equality; no scale normalization |
| P10 | Validation-only seasonal-period activity uses identical targets; arithmetic verified |
| P11 | Post-dataset forecast dates precede the review date |

The companion ZIP contains `review_probes.py`, `probe_results.json`, the locally reproduced EDA PNG, the activity CSV, and a README with scope and reproduction instructions. The script uses controlled fixtures and does not require model weights. The file `probe_results.json` is the machine-readable record of the checks performed during this review.

[notebook]: https://github.com/kurtvalcorza/chronos-2-forecasting-pipeline/blob/e13b36249cb3b9e4cd46993ccb581dbe2d6b94e0/tutorials/DIMER_MultiModel_TimeSeries_Forecasting_Workshop.ipynb
[release]: https://github.com/kurtvalcorza/chronos-2-forecasting-pipeline/blob/e13b36249cb3b9e4cd46993ccb581dbe2d6b94e0/docs/release-verification.md
[design]: https://github.com/kurtvalcorza/chronos-2-forecasting-pipeline/blob/e13b36249cb3b9e4cd46993ccb581dbe2d6b94e0/docs/multimodel-forecasting-workshop-spec.md
[fleet]: https://github.com/kurtvalcorza/ml-worker/blob/main/integrations/dimer/fleet-specs/NOTEBOOK_SPEC.md
[tirex-guide]: https://github.com/NX-AI/tirex-2/blob/main/docs/how-to/forecasting.md
[chronos-source]: https://github.com/amazon-science/chronos-forecasting/blob/7dc4435706a4454feb79df44ca9f33631f3027bf/src/chronos/chronos2/pipeline.py
