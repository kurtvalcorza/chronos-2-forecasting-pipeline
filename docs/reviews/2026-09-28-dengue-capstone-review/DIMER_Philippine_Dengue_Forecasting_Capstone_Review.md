# Philippine Dengue Forecasting Capstone — Notebook and PR Review

**Verdict: Needs revision. Recommended PR disposition: request changes.**

The central six-system forecasting experiment is substantially implemented. The main reproduced problems concern the learner-facing results, the consistency of BYOD admission with the downstream models, and reconstruction from the exported evidence bundle. The targeted checks supported important availability, target-maturity, pairing, interval-validation and activity-isolation controls. This review does not establish that the default real-model workflow fails.

## 1. Review contract and evidence boundary

| Item | Reviewed value |
|---|---|
| Pull request | [chronos-2-forecasting-pipeline #20](https://github.com/kurtvalcorza/chronos-2-forecasting-pipeline/pull/20) |
| Head commit | `209d8f5c8499949c8f78413d2cba3601e111c45b` |
| Notebook Git blob | `66309b6a5585a41d69f2e3e91915fd6c18c0ce03` |
| Notebook SHA-256 | `91295a5b2362af0f97dfc671ad1a1e1412a47fc2ca9788222304cf243cc9cd16` |
| Notebook | `tutorials/DIMER_Philippine_Dengue_Forecasting_Capstone.ipynb` |
| Declared standard/profile/mode | Notebook Specification 2.2; `E2E`; `GUIDED`; Candidate |
| Scientific scope | Exploratory published-source-block forecasting, not verified weekly or operational forecasting |
| Intended learner | Basic Python/Colab familiarity; forecasting terminology introduced in the notebook |
| Canonical runtime | Fresh Linux x86-64 Colab T4; isolated Python 3.12.12 environment |
| Local review runtime | Python 3.13.5, NumPy 2.3.5, CPU PyTorch 2.10.0+cpu; not the notebook's complete locked environment |
| Review method | Source inspection; 14 grouped offline probes; exact notebook-embedded modules; synthetic inputs; actual NumPy Ridge fitting; deterministic model substitutes; separate-process reconstruction; file and rendering inspection |

The uploaded notebook matches the PR blob exactly. It has 19 cells, including nine code cells, with no saved outputs. Notebook schema and all code-cell compilation checks passed. The seven files bound by its embedded source manifest matched their SHA-256 values.

**Not executed or independently verified:** the real Zenodo workbook, publisher alignment/calendar evidence, pretrained Chronos or Mitra weights, AutoGluon's real fitted preprocessor, complete dependency installation, a Colab/T4 run, real-model latency/memory, model accuracy, true model-backed reload parity, or representative learner behavior. No repository code, PR discussion, or release status was modified.

The local probes run exact extracted code rather than reimplementing the forecasting core. The complete synthetic stage exercise replaces both pretrained-model adapters and bypasses the GPU gate; missing upstream package versions are explicitly marked as review doubles. These substitutions test orchestration and contracts, not model validity or performance. The separate reload process also uses those declared substitutes.

### Repository execution evidence

The PR description reports 453 passing tests and 22 deliberately deselected live integration tests after rebasing. The inspected CI run, `36376224136`, passed lint/contract tests, release validation and dengue generation parity. Its live pinned-weight tutorial job was skipped. A separate board-management workflow failed; that is not a failed forecasting test.

The [capstone build record](https://github.com/kurtvalcorza/chronos-2-forecasting-pipeline/blob/209d8f5c8499949c8f78413d2cba3601e111c45b/docs/dengue-capstone.md) explicitly leaves actual pretrained-model results, full-model reconstruction and fresh hosted execution pending. It reports an earlier 413-test local run and actual-source CPU baseline results; those records are not this reviewer's execution evidence. No scientific gain from weather should be inferred from CI or from the synthetic probes supplied with this review.

## 2. Scope and specification amendments

The attached specification requires stopping verified-timing performance claims when the reporting calendar cannot be resolved, and permits a separately approved exploratory source-block design. The [PR build record](https://github.com/kurtvalcorza/chronos-2-forecasting-pipeline/blob/209d8f5c8499949c8f78413d2cba3601e111c45b/docs/dengue-capstone.md) documents that approval on 2026-09-27. The notebook preserves source year/block order and expressly does not establish actual observation dates, case–weather alignment, historical publication availability, or an uncontaminated pretraining-independent benchmark.

This is a documented limitation, not evidence that the calendar problem has been solved. Keep that narrower scope in figures, exports and conclusions. In particular, “weather improved predictions under the published alignment” is distinct from “weather improved a prospectively available weekly warning system.”

Two other deviations are already documented: BYOD requires **six**, not the draft's five, complete years, and this version admits only complete selected fields rather than implementing median imputation. Six years supplies at least two history years before two validation and two test years. These are explicit design changes, not hidden failures. Record them in a versioned amendment rather than treating the original specification as unchanged.

## 3. Judgments by review dimension

| Dimension | Judgment within the checked scope |
|---|---|
| Promise fulfillment | Six systems, paired weather ablation, delay activity and future inference are implemented. Display and portable-consumption gaps remain. |
| Technical correctness | Selected deterministic helpers and synthetic orchestration pass; BYOD admission and bundle reconstruction need revision. |
| Experimental validity | Availability, target maturity and pairing passed targeted checks. Source timing/alignment and real-model behavior remain unqualified. |
| Learner orientation/progression | The introductory scope and forecast terminology are useful. The long carrier cell needs collapsing and clearer separation. |
| Explanation/interpretation | Scope and metric caveats are substantive, but some visible tables omit the actual forecast/error quantities. |
| Meaningful activity | The two-block delay comparison is real and isolated; a paired zero-versus-two-delay view would make it easier to interpret. |
| Interaction/recovery | Constant-target inputs fail late; some malformed BYOD values are admitted or yield low-context errors. |
| Completion/transfer | Machine-readable outputs and same-workspace reload exist. Archive-only reconstruction with the supplied loader fails. |
| Specification conformance | Not certified. Maintain Candidate until corrections and hosted/BYOD gates are complete. |

## 4. Major findings

### DENGUE-01 — Forecast and error tables omit the quantities learners need

**Severity:** Major — promise fulfillment and learner interpretation.  
**Location:** `tools/build_dengue_capstone.py`, `BOOTSTRAP.table`; notebook Sections 4–6.  
**Evidence:** exact helper execution, captured Markdown, synthetic complete-workflow exports; probe 09.

The [table helper](https://github.com/kurtvalcorza/chronos-2-forecasting-pipeline/blob/209d8f5c8499949c8f78413d2cba3601e111c45b/tools/build_dengue_capstone.py) always selects the first nine columns and the first twelve rows. Forecast records are ordered as:

```text
system, origin, origin_key, cutoff, target_index, target_key,
horizon, delay, reference, raw_prediction, prediction, clipped, q10, q50, q90
```

The visible columns therefore stop at `reference`. The forecast, nonnegative effective value, clipping flag and quantiles are omitted. In the final forecast table, every displayed reference is empty because the target is unobserved: the learner sees identifiers and empty ground truth, not a forecast.

The synthetic run produced all **24 future rows**: six systems times four horizons. The helper displayed only the first twelve, which contain persistence, seasonal naive and Ridge. Chronos and the two Mitra systems were absent from the preview. A complete CSV link is present and the numeric forecasts are actually saved; this is not a missing-inference finding.

The same fixed column selection weakens the largest-misses and delay-activity views. Those tables inherit prediction metadata before the result columns, so the quantities the surrounding questions ask learners to compare are outside the visible preview.

**Consequence:** “Forecast beyond the final source block” can complete without visibly telling the learner what any model predicts. Reading a linked CSV can recover the values, but the notebook's principal result display should not require that detour.

**Correction:** Choose columns by task meaning, not ordinal position. For future output show system, target block, horizon, effective prediction, raw prediction/clipping and applicable quantiles. Show all six systems or pivot four target blocks against six systems. For misses show reference, prediction, signed/absolute error and target; for the activity show paired zero-delay and two-delay values and differences.

**Acceptance check:** Normal notebook output exposes all 24 future predictions with no fabricated targets, and lets a learner explain a largest miss and one delay comparison. Complete machine-readable outputs retain all metadata. Include both default and BYOD display paths.

### DENGUE-02 — Accepted BYOD series can violate model or source-data requirements

**Severity:** Major — input contract, recovery, and experimental validity.  
**Location:** `tools/dengue_runtime.py:360–460`, `tools/dengue_core.py:validate_rows`, `tools/dengue_models.py:168–195`, and the stricter default validator in `tools/dengue_data.py:140–161`.  
**Evidence:** actual preparation and Ridge/baseline execution on synthetic CSVs; actual Mitra adapter preflight; probes 05, 06 and 14.

#### Constant support is discovered after preparation

A six-year source-block CSV with zero reported cases throughout passes preparation, produces the planned 26 origins per partition, and completes the simple validation baselines. Their MAE is zero and seasonal skill is correctly undefined because the comparator error is zero.

The first validation Mitra task has 51 mature support targets in its 52-issuance-block context. All are zero. The actual [Mitra wrapper](https://github.com/kurtvalcorza/chronos-2-forecasting-pipeline/blob/209d8f5c8499949c8f78413d2cba3601e111c45b/tools/dengue_models.py#L168-L195) then raises:

```text
ValueError: Mitra requires nonconstant support targets and at least one varying feature
```

The refusal occurs before the wrapper imports AutoGluon. It is thus reproducible without downloading weights. The runtime [loads the model before constructing each forecast task](https://github.com/kurtvalcorza/chronos-2-forecasting-pipeline/blob/209d8f5c8499949c8f78413d2cba3601e111c45b/tools/dengue_runtime.py#L186-L224), so the complete real path would encounter this incompatibility only after earlier stages and model acquisition/loading.

The all-zero fixture isolates the problem; a locally constant support window within a changing series has the same target-variation issue. This is not evidence that the pinned default QC data cause that failure.

#### Numeric and CSV checks differ between default and BYOD

The default source validator enforces integer nonnegative case counts and a declared temperature range. BYOD's shared row validator checks finite numeric values and nonnegative cases/rain, but lacks those additional checks.

| Fixture | BYOD preparation | Default-source numeric validator |
|---|---|---|
| Case count `0.5` | Accepted | Rejects fractional count |
| Temperature `500` in the declared Celsius field | Accepted | Rejects its declared bound |
| Missing case value | Raw conversion error, without row/field identification | Missingness rejected explicitly |
| Additional unnamed field after the five CSV fields | Accepted and discarded | Not applicable to the workbook parser |
| Negative cases | Rejected | Rejected |
| Duplicate/mismatched header | Rejected | Not applicable |

The extra-field probe used a harmless sentinel; no patient data were supplied. Rejecting extra header names does not also reject extra row values.

**Consequence:** Data can pass intake yet fail late inside a mandatory comparison, or contradict the stated count/unit semantics without rejection. Corrected data and errors are harder for a learner to interpret when validation is inconsistent.

**Correction:** Share applicable semantic checks while keeping source-specific cohort/year checks separate. Validate row widths, types, count meaning and units with file/row/column diagnostics. Preflight every planned origin/horizon/window/delay support task before model staging. Define the constant-target policy explicitly: either support it using a documented, labelled special case or reject the experiment early with its affected contexts. Do not perturb zeros, invent variance, silently drop failing origins or call a fallback a Mitra prediction.

**Acceptance check:** An ordinary varying series completes; supported zero counts remain valid; constant support is dealt with explicitly before expensive execution; malformed/extra values fail actionably; missing observations never become zero; M1 and M2 keep the same cohort. The established six-year and complete-case policies remain documented.

### DENGUE-03 — The downloadable context bundle still depends on the original workspace

**Severity:** Major — artifact completion and transfer.  
**Location:** `tools/dengue_runtime.py:identity`, `future`, `reload` and `report`.  
**Evidence:** real JSON state serialization, separate-process synthetic reload, archive membership and digest checks, fresh-directory reconstruction probe; probes 08, 11 and 12.

The notebook stores useful state: fitted Ridge parameters, per-horizon Mitra support/query arrays, Chronos history, feature schema, model identity and hashes. The existing fresh-process reload genuinely exercises that state. It is not merely a printed success message.

However, [reload checks the original execution identity](https://github.com/kurtvalcorza/chronos-2-forecasting-pipeline/blob/209d8f5c8499949c8f78413d2cba3601e111c45b/tools/dengue_runtime.py#L654-L671). The [identity function](https://github.com/kurtvalcorza/chronos-2-forecasting-pipeline/blob/209d8f5c8499949c8f78413d2cba3601e111c45b/tools/dengue_runtime.py#L62-L79) requires the original root's `data.json` and hashes `run_config.json` when present. Neither file is included in the downloadable result archive. The data manifest records the prepared-data digest, not the missing prepared-data bytes. These dependencies are not addressed by an archive-consumption recipe.

**Local result:** the complete synthetic workflow reloaded in a separate OS process and matched 32 point predictions, plus its Chronos quantiles. The generated ZIP contained 57 members, and every listed member hash verified. After extracting it into a fresh workspace and supplying the trusted code/manifests from the distributed notebook, the supplied reload function failed first on missing `data.json`. The original run configuration would also need to match the frozen execution identity; an absolute BYOD path is not portable configuration by itself.

This does **not** mean the saved support arrays are corrupt or that same-workspace parity failed. It means that this parity check does not establish reconstruction from the delivered bundle. A knowledgeable user could supply additional files or reconstruct state manually; the notebook does not provide that complete handoff.

**Correction:** Define an explicit artifact-consumer entry point and the necessary dependencies. Prefer verifying model identity, feature/target semantics and the exported context/fitted state without requiring an entire old working directory merely for provenance. Alternatively, document and provide an authorized reconstruction mechanism for every required external data/configuration dependency. Do not automatically bundle private BYOD data to make a test pass; minimize and disclose the aggregate support values retained in the artifact.

**Acceptance check:** With declared base assets available, extract the bundle into a clean directory and reproduce the fixed completed origin and final unscored inference without accessing the original root. Verify keys, point values, quantiles and transformations, and reject mutated context/configuration. Keep same-workspace stage receipts as a separate integrity control.

## 5. Smaller findings and improvements

### DENGUE-04 — BYOD provenance lacks area identity

The BYOD audit records `source: BYOD`, source digest, row count and timing/alignment flags, but no area identifier, descriptive source citation or persistent declared-units metadata. The notebook's static title still names Quezon City. This review did not find an output that explicitly relabelled another area as Quezon City; the problem is that the bundle does not establish what area its otherwise valid aggregate series represents.

Add explicit non-personal area/source/units metadata and use it in output captions and provenance. Keep these fields outside the numeric predictors. This supports the specification's instruction not to reinterpret another area as Quezon City.

### DENGUE-05 — Interpretation would benefit from less raw infrastructure

The embedded setup/carrier cell has approximately 226,000 characters and no collapse metadata. It is preceded by an infrastructure heading, so its purpose is labelled, but it still creates substantial visual bulk. Collapse it by default and retain an accessible source explanation.

The notebook exports an availability audit, but does not display the actual values of one named feature row alongside their source blocks. Add that example so the learner can trace a lag and the latest mature label without decoding the carrier. For the delay activity, show a paired zero-versus-two-block comparison rather than making the learner reconcile two separated metric dictionaries. The existing activity is valid; this recommendation improves interpretation rather than replacing it.

### DENGUE-06 — Carry fuller limitations and runtime identity into the result summary

The summary correctly sets timing, alignment and prospective-validation flags to false. The conclusion scaffold contains final-data-vintage and pretraining caveats. The machine-readable main summary does not spell those out, and its environment file lists only four package versions, omitting Python, the actual device and several material preprocessing/runtime libraries.

Include explicit availability/vintage/pretraining limitations, area identity, Python/device/precision and effective upstream preprocessing package versions. Attach the source audit and resource targets as targets, not measurements. This is a reporting improvement; it does not imply the current notebook claims prospective validation.

## 6. Positive controls and evidence worth preserving

**Availability and maturity.** Sixteen combinations of two delays, four horizons and two windows retained identical case-only/weather issuance rows and targets. Training targets never exceeded the current cutoff. Poisoning all observations after that cutoff left support features, support targets and query features unchanged. December/January source-key mapping was correct.

**Delay mapping.** The exact Chronos adapter was exercised with a CPU tensor-returning upstream test double. With delay two, the runtime requested six forecast steps and retained steps three through six for the original four target blocks. The history still contained 104 blocks and ended two positions earlier. Non-finite, crossing and malformed quantile outputs were rejected.

**Metrics.** Hand-computed MAE, RMSE, signed bias, inverse transforms and clipping agreed. Zero comparator error yielded undefined skill. A paired bootstrap fixture retained all four horizons in each sampled origin group and reported the expected weather-minus-case difference; the final shorter group was retained. This validates arithmetic and grouping mechanics, not the statistical adequacy of seven source-origin groups for the real data.

**Cohort integrity.** Missing, duplicated and changed-reference prediction records were rejected. The complete six-system synthetic test contained 624 rows: 104 origin/horizon pairs per system. Future inference contained 24 rows and no observed references.

**Activity isolation.** Re-executing the delay activity against synthetic model substitutes kept all canonical test-prediction, metric and experiment-lock digests unchanged. Its 624 validation keys matched the original keys, with all cutoffs shifted by two blocks.

**Real fitted state and guarded export.** Ridge was actually fitted on synthetic numerical data and reconstructed from serialized JSON state. The separate-process reload checked 32 point predictions with zero difference under the substitutes. Artifact tampering and a changed CSV prediction identifier were refused. The existing identity and receipt controls should not be removed while repairing portable consumption.

**Visual check.** The renderer's `test_forecasts.png` was inspected using the synthetic run: forecast/reference curves, the nominal interval band and signed-error panel were visible and labelled. This is a renderer check, not an inspection of real QC forecasts. The missing table quantities remain distinct from the fact that this forecast plot is displayed.

**Scope discipline.** Keep the distinction among Ridge fitting, Mitra in-context conditioning and Chronos zero-shot forecasting. Gradient fine-tuning is not claimed and should not be added merely to satisfy a superficial reading of E2E. Keep M2 minus M1 as the weather question; comparisons with another model family answer a different question.

## 7. Acceptance sequence

1. Correct the semantic display columns, BYOD semantic/context preflight, and archive-consumer boundary. Update the generator and regenerate the notebook; preserve the exact source identities and original tutorials.
2. Add regression tests for the reproduced defects, retaining the positive availability, maturity, cohort, invalid-quantile, activity-isolation and mutation controls.
3. Reconcile the scope, six-year minimum and complete-case policy in a versioned specification amendment. Resolve constant-support behavior explicitly, without opportunistic cohort changes.
4. Run the exact revised notebook with real pretrained models in a fresh supported T4 environment. Preserve all six systems, 26 origins per partition, 104 pairs per system, actual resource measurements, final forecasts, artifacts and true reconstruction evidence.
5. Exercise the complete revised BYOD workflow with authorized representative aggregate data and separate invalid-input cases. Reconstruct the exported context bundle outside the original workspace.
6. Observe a learner explaining one availability-bound feature, the matched weather comparison, a high-case miss, the delay activity and a final forecast. They should be able to retrieve the correct outputs and explain the scope without inspecting the embedded carrier.

**Readiness decision:** Request changes on the reviewed PR, retain Candidate status, and treat hosted/real-model qualification as pending rather than failed. No favorable weather or foundation-model result is an acceptance requirement. Invalid inputs, invisible principal results and an incomplete artifact handoff are correction requirements.

## 8. Probe inventory

All 14 grouped probes completed. “Completed” means the review check ran, including expected defect reproductions; it does not mean the notebook passed every check.

| Probe | Scope |
|---|---|
| 01 | Notebook identity, schema, compilation and embedded-file digests |
| 02 | Availability, maturity, feature pairing and future-value poisoning |
| 03 | Metrics, clipping, intervals and paired block bootstrap |
| 04 | Chronos output contract and delay-to-horizon alignment with a test double |
| 05 | Constant BYOD acceptance followed by actual Mitra preflight refusal |
| 06 | BYOD semantic values, CSV width, headers and minimum years |
| 07 | Complete common cohort and rejection of missing/duplicate/changed-reference records |
| 08 | Nine-stage synthetic integration, actual Ridge fitting and separate-process reload |
| 09 | Exact learner table helper and hidden forecast quantities |
| 10 | Canonical result preservation during delay activity |
| 11 | Context-artifact and prediction-CSV tamper rejection |
| 12 | Archive member hashes and failed clean-directory reconstruction |
| 13 | Main-summary caveats and BYOD area-provenance inspection |
| 14 | Stricter default-source numeric validation positive/negative controls |

The companion ZIP contains the review scripts, exact embedded source, input notebook, captured probe observations and selected synthetic rendering evidence. It contains no pretrained weights, patient data, real QC source table or font files. Read its README before treating any contained metrics as evidence.

**No repository changes, PR comments, review submissions or release-status mutations were made.**
