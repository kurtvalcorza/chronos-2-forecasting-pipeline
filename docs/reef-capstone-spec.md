# Philippine Reef Heat-Stress Outlook — Capstone 7

**Status:** Implementation specification, 2026-09-28. Notebook not built or hosted-qualified.

**Implementation update:** Built locally on 2026-09-28; hosted qualification remains pending.
The remotely hosted immutable sample proposed below is implemented as an embedded, hash-verified
data-only ZIP so the notebook works before publication. Original NOAA bytes and attribution are
preserved. Optional delay sensitivity is deferred. See `reef-capstone.md` and
`reef-local-validation.md` for implementation decisions and measured local evidence.
**Proposed host:** `kurtvalcorza/chronos-2-forecasting-pipeline`.
**Notebook:** `tutorials/DIMER_Philippine_Reef_Heat_Stress_Capstone.ipynb`.
**Standard:** DIMER NOTEBOOK_SPEC v2.2; `TASK-INFERENCE`, `GUIDED`.
**Placement:** Scientific applications capstone after time-series forecasting.
**Default runtime:** Fresh Google Colab T4; no authentication, cloning, DIMER services, manual restart, or cell edits. CPU-only local engineering checks.

## 1. Question and outcome

Can a pretrained forecasting model improve short-horizon forecasts of Philippine regional coral heat stress over persistence and seasonal baselines?

Learners compare 7-, 14-, and 28-day forecasts, distinguish instantaneous from accumulated stress, detect leakage, and explain what a regional satellite indicator can establish. Basic Python and Colab familiarity are sufficient; explain forecasting and marine terms at first use.

**Input → System → Output:** daily NOAA regional HotSpot history → validation, chronological forecasting, deterministic DHW calculation, baseline comparison → regional stress trajectories, evaluation tables, and an evidence-based conclusion.

The task is retrospective forecasting of satellite-derived heat-stress indicators. It does not predict observed bleaching, coral mortality, reef condition, or fisheries losses. It is not NOAA's operational outlook, an official warning, or a replacement for field surveys. Model improvement is a research question, not a release requirement.

## 2. Verified data and rights

Use NOAA Coral Reef Watch v3.1 Regional Virtual Station TXT records. Audit retrieved 2026-09-28, covering 1985-01-01 through 2026-09-26:

| Region | Rows | Missing calendar days | Official file |
|---|---:|---:|---|
| Northern Philippines | 15,244 | 0 | https://coralreefwatch.noaa.gov/product/vs/data/northern_philippines.txt |
| Central Philippines | 15,244 | 0 | https://coralreefwatch.noaa.gov/product/vs/data/central_philippines.txt |
| Western Philippines | 15,219 | 25 | https://coralreefwatch.noaa.gov/product/vs/data/western_philippines.txt |
| Eastern Philippines | 15,244 | 0 | https://coralreefwatch.noaa.gov/product/vs/data/eastern_philippines.txt |
| Southern Philippines | 15,219 | 25 | https://coralreefwatch.noaa.gov/product/vs/data/southern_philippines.txt |

Total: 76,170 rows, 6,324,302 bytes. No duplicate dates or nonfinite numeric entries were found in existing rows. Western and Southern omit 2025-01-05, 2025-01-13, 2025-10-02 through 2025-10-23, and 2026-03-24. These are observations about this snapshot, not promises about later downloads.

The audit and five source files reside in `C:/Users/Kurt Valcorza/Downloads/reef-data-audit-2026-09-28/`. `audit.json` contains per-file SHA-256, sizes, ranges, missing dates, and calendar-aware DHW checks. Verify those hashes against the files before packaging.

NOAA states that CRW website content is public domain and freely distributable; retain NOAA Coral Reef Watch attribution, source URLs, product version, dates used, retrieval date, and its recommended citation. Keep data attribution separate from the notebook/code license. Source: https://coralreefwatch.noaa.gov/satellite/docs/recommendations_crw_citation.php

### Immutable sample acquisition

Package the five verified TXT files plus provenance in a small, versioned public data asset at implementation/publication time. Pin its exact URL, byte size, SHA-256, and member hashes in notebook metadata and cells. A commit-addressed data-only asset is acceptable; fetching executable repository source is not.

Default Run all downloads this frozen sample, not NOAA's growing daily files. No silent fallback to live files after a checksum failure. An optional refresh path may download NOAA directly, generate a new audit, and mark the result a different dataset version. Publication URL is an implementation dependency; do not invent a working URL or leave a placeholder in a released notebook.

## 3. Scientific data contract

These are satellite-derived regional summaries, not fixed sensors. Each day's representative SST, SSTA, and HotSpot correspond to the pixel at the region's 90th-percentile HotSpot. Its location may change daily. Header coordinates locate the region, not a sampled reef. Do not present a regional value as an individual reef measurement, spatial mean, or percentage of bleached coral. Source: https://coralreefwatch.noaa.gov/product/vs/map.php

Canonical fields:

| Source field | Canonical name | Meaning / use |
|---|---|---|
| YYYY, MM, DD | date | Daily observation date, no fabricated timestamp |
| SST_MIN, SST_MAX | sst_min_c, sst_max_c | Regional bounds; contextual plots only |
| SST@90th_HS | representative_sst_c | SST at the daily selected pixel |
| SSTA@90th_HS | representative_ssta_c | Its published SST anomaly |
| 90th_HS>0 | hotspot_c | Primary forecast target, nonnegative °C |
| DHW_from_90th_HS>1 | dhw_c_weeks | Published accumulated stress; derived forecast target |
| BAA_7day_max | legacy_baa_7day | Retain for provenance, not primary target |

Do not derive HotSpot from representative SST minus the header's regionally averaged maximum monthly mean: these are not matched pixel quantities. Use the published HotSpot.

Mask DHW before 1985-03-25 and BAA before 1985-03-31, based on the header's declared valid starts. The header dates use year/day/month ordering; parse explicitly. Early numeric zeros do not imply valid measurements.

Despite the TXT column's `>1` wording, accumulate HotSpot **at or above 1 °C**. Define `g(x) = x if x >= 1 else 0`; then `DHW(d) = sum(g(HS(s)), s=d-83,...,d) / 7`. It uses 84 calendar days including d. The audit reproduced published DHW within 0.000043 °C-weeks on complete windows. Verify within 0.0001, excluding incomplete/invalid windows. Source: https://coralreefwatch.noaa.gov/product/5km/methodology.php

Reindex all regions to a daily calendar. Never collapse missing dates, zero-fill them, interpolate evaluation outcomes, or treat row number as elapsed days. Retain raw and validated columns plus missing/validity flags.

Legacy BAA values are not interchangeable with the newer alert scheme. V1 reports numerical DHW threshold indicators, not reconstructed official alert levels. Source: https://coralreefwatch.noaa.gov/product/vs/data.php

## 4. Frozen experiment design

Use all five regions. Split by calendar time, identically across regions:

| Period | Role |
|---|---|
| 1985–2016 | Provenance and optional historical visualization only |
| 2017–2022 | Development history; fit seasonal reference |
| 2023 | Validation; freeze implementation and runtime choices |
| 2024–2025 | Locked retrospective test |
| 2026 through snapshot end | Final illustrative forecast origin; no test tuning |

The model remains pretrained and frozen. Earlier observations in a held-out year may enter later rolling-origin contexts once their dates are past; outcomes after an origin may never enter its input. No random train/test split. A historical holdout does not establish absence of foundation-model pretraining overlap: inspect the pinned model card and explicitly report any unresolved training-data overlap.

For validation and test, use origins on the 1st and 15th of every month. Retain origins only when the entire 28-day outcome lies within the assigned split. Record planned origins and all exclusions before model inference. For every origin, require 365 consecutive observed HotSpot context days, complete 28-day HotSpot outcomes, and valid DHW outcomes. The same eligible origins apply to every model and all three horizons.

The 365-day context intentionally causes a missing day to exclude subsequent origins whose input would contain it. Report these losses by region/year; do not conceal them by evaluating only surviving high-stress periods. Acceptance requires at least 20 eligible test origins per region and at least one per calendar quarter across the two-year test, or a documented design revision before performance claims. Excluded origin counts must be computed, not inferred from raw missing-date counts.

Verified against the pinned local files on 2026-09-28: each region has 23 eligible validation origins in 2023 and 23 test origins in 2024. Northern, Central and Eastern each retain 23 in 2025; Western and Southern retain **zero** in 2025 under this strict context rule. Total: 115 validation and 184 test region-origin pairs. The test is therefore an unbalanced retrospective panel, not two years of evaluation for every region. Show a mandatory common-period 2024 comparison alongside the full-panel macro score, and a separate 2025 table for the three eligible regions. Never describe the full-panel score as performance across all five regions in 2025. These exclusions are a deliberate no-imputation design choice; changing that policy is a new experiment.

Primary comparison uses a 365-day context. A required bounded controlled experiment compares 180 versus 365 days on the same eligible validation origins, with all other settings fixed. Keep 365 as the preregistered test default; any change needs a versioned spec amendment before inspecting test results.

### Availability assumption

Baseline experiment assumes observations through origin t are available. NOAA's reprocessed archive does not establish publication-time availability; describe this as retrospective, not a historically deployable simulation.

An optional delay sensitivity hides the last two days. It forecasts 30 days from t-2, discards the first two forecast targets for scoring, and still scores t+1 through t+28. DHW must use forecasts for the two hidden days too. Never splice withheld observations into the delayed accumulator.

## 5. Models and composed forecast

Required arms:

1. **Persistence:** repeat the last observed HotSpot for all 28 days.
2. **Seasonal reference:** mean HotSpot by month/day, independently per region, fitted only on 2017–2022 and frozen. For February 29 use the mean of February 28 and March 1 references. Export the fitted table.
3. **Chronos-2:** pinned `amazon/chronos-2`, univariate zero-shot forecasting independently per region/origin. Use the upstream library directly; no DIMER runtime source, fine-tuning, or hidden cross-region context.

Resolve and record the actual model commit, allowed model files and hashes, package versions, device, precision, seed, and effective context/horizon during implementation. Verify the existing pipeline's pin rather than assuming a README value is current. Reference: https://huggingface.co/amazon/chronos-2

Forecast all 28 daily steps once per origin and score prefixes of 7/14/28 days. For point predictions use the model median, clamp negative HotSpot predictions to zero, and export both raw and constrained outputs plus the number changed. Apply identical nonnegative semantics to all arms. No upper cap chosen from test data.

For each future date d, compute DHW from observed history up to t plus that arm's predicted daily HotSpots after t. Carry the known historical contribution and newly predicted contribution as separate output fields. Use no future observed DHW, SST, HotSpot, or legacy alert as a predictor.

Also show **DHW persistence** (repeat DHW(t)) as a separate derived-target benchmark. This makes the large contribution of already accumulated stress visible. Do not give it a HotSpot score or imply that good DHW persistence demonstrates skill at predicting new heat.

The plug-in DHW of median HotSpots is a deterministic trajectory, not necessarily the median DHW forecast. Do not sum marginal quantile curves and call them calibrated DHW intervals; temporal dependence matters. V1 offers probabilistic evaluation for HotSpot only. Coherent DHW uncertainty is deferred.

## 6. Evaluation and interpretation

Predeclare **14-day HotSpot MAE**, averaged equally across regions, as the primary outcome. Report 7/14/28-day MAE and RMSE per region, then equal-region macro averages. Prefix metrics use all daily steps within the horizon. Report number of origins and scored days beside every table.

For DHW, report endpoint MAE at t+7/t+14/t+28 for each composed arm and DHW persistence. Supplement with trajectory plots separating known and predicted contributions.

For predicted endpoint DHW >=4 and >=8 °C-weeks, report TP/FP/FN/TN, precision, recall, and positive support against the published DHW indicator. Undefined precision/recall remains null with a reason. These are stress-indicator thresholds, not observed bleaching labels or full official alert classes. Add the subset with DHW(t) below the threshold to distinguish new exceedances from ongoing accumulated stress; empty subsets remain visible.

For Chronos HotSpot forecasts, report pinball loss and empirical coverage/width of the 10th–90th percentile interval, by horizon and region. Validate quantile ordering and document any constraints. Do not promise 80% empirical coverage or calibrated event probabilities.

Show paired error differences on identical origins. Avoid naive iid confidence intervals: forecast windows overlap and regions share weather. V1 reports descriptive differences, not significance or causal claims. Separate high-stress target days (observed HS >=1) as a predefined diagnostic, reporting support; do not use that diagnostic to select origins/models.

The conclusion must address seasonality, missing-data exclusions, retrospective availability, moving regional pixels, model pretraining uncertainty, and the distinction between thermal exposure and ecological damage. Retain negative or inconclusive results.

## 7. Guided notebook sequence

1. Orientation: question, prerequisites, T4 selection, Run all, expected resource budget, research scope.
2. Input → forecasting → accumulated stress → evidence diagram; define SST, HotSpot, DHW, origin, horizon, baseline.
3. Infrastructure: pinned installation and runtime/model identity, clearly labelled and collapsed where supported.
4. Acquire immutable data; verify hashes, parse metadata, show source credit and calendar gaps.
5. Reconstruct DHW; ask why identical current HotSpot values can accompany different accumulated stress.
6. Freeze splits/origins; illustrate a valid context and a rejected gap-containing window.
7. Predict which baseline will be hardest to beat; run baselines and Chronos.
8. Controlled validation experiment: change only context length; compare and explain.
9. Locked test: display errors, support, uncertainty diagnostics, failure examples chosen by a declared rule.
10. Regional outlook panel: illustrative forecast from the common frozen snapshot cutoff, actual origin date prominent. No claim of a live advisory.
11. Export and independent verification; optional BYOD and delay sensitivity.
12. Evidence-based conclusion, optional personal completion record, glossary, troubleshooting, citations, AI disclosure.

Every principal stage needs a prediction/question, What to notice guidance, and a collapsible worked interpretation. Learner responses never block Run all. Completion records are personal learning aids, not required submissions.

Use regional small-multiple plots rather than reef-level heat maps. Region markers, if added, must be labelled as regional reference points. Do not invent station polygons or local reef measurements.

## 8. BYOD, exports, and reload

Optional BYOD is disabled by default. Accept CSV fields `region_id,date,hotspot_c`, plus optional published DHW for consistency checks. Require explicit units and confirmation that HotSpot semantics match NOAA's threshold-relative definition; arbitrary SST alone is insufficient. Validate unique dates, finite nonnegative values, daily continuity, at least 365 observed context days and outcome coverage for backtesting. Invalid inputs explain the offending rows. A generated schema example exercises this path without private data.

Export a portable bundle containing:

- `dataset_manifest.json`: source URLs, retrieval time, rights, hashes, product metadata.
- `dataset_audit.json` and `origin_manifest.csv`: validity, gaps, split, cutoff, eligibility/rejection reason.
- `experiment_config.json`, `environment.json`, model revision/file manifest, and seasonal reference table.
- `forecasts.csv`: region, origin, target date, lead, arm, raw/constrained HotSpot and quantiles where applicable, DHW components, split and run identity.
- `metrics.json`, `metrics.csv`, threshold confusion counts/support, figures and `conclusion.md`.
- `run_summary.json`: executed stages, source/model/notebook identity, counts, timings, peak GPU memory, warnings, refusal probes and verification results.
- File checksum manifest and bundle README distinguishing data, code, pretrained model identity, and predictions. Do not redistribute model weights by default.

Verify in a fresh process with explicit serialized inputs: recompute baselines, DHW and metrics, and compare against exports at declared tolerances. Reload the pinned model in fresh state for one fixed origin and check prediction parity; release the earlier model first to bound memory. No reliance on hidden notebook globals or unsafe pickle deserialization. Saved predictions alone do not establish model reload parity.

## 9. Acceptance and qualification

Required CPU checks:

- Exact schema, malformed header/date, duplicate date, nonfinite value and negative HotSpot refusals.
- Missing-day reindexing, leap day, invalid early DHW masking, immutable checksum failure.
- Hand-calculated DHW fixtures including HS=0.99, 1.00, 1.01 and exact 84-day boundary expiration.
- Future-data perturbation leaves all origin-t inputs/predictions unchanged; altered future targets affect only evaluation.
- Identical comparison origins, split boundary enforcement, seasonal reference isolation, empty-event metrics.
- Quantile dimensions/order, honest missing model outputs, finite JSON, and independent export reload.
- Notebook format validation, syntax parsing, standalone source scan, shared-namespace execution checks, and applicable NOTEBOOK_SPEC conformance report.

Required hosted evidence: one fresh Colab T4 default Run all with no restart, reruns, credentials, or intervention; preserve executed notebook, exact source revision/hash, outputs and run summary. Target budget is <=30 minutes and <=12 GiB GPU peak; these are design targets, not measured claims. Validate runtime cost on validation data before locked test execution. If the budget fails, revise the documented design rather than silently truncating regions/test origins or substituting mock outputs.

Release status remains **Candidate pending hosted qualification** until evidence exists. Release is allowed when the experiment runs correctly even if Chronos loses. No minimum winning score. No commit, push, PR or merge is implied by this specification request.

## 10. Deliverables and implementation decisions

Build the standalone notebook, deterministic builder, focused tests, immutable data manifest/asset, dataset card, evidence note, and repository/tutorial registry entry. Proposed companion paths are `docs/reef-capstone.md`, `docs/datasets/philippine-reef-heat-stress.md`, and `tools/build_reef_capstone_notebook.py`; reconcile with the host's conventions before editing.

Implementation must resolve the public immutable asset location, verify the pinned Chronos package/model files and output API, calculate exact eligible origin counts, and measure hosted runtime. These are verification tasks, not assumed completed facts. Stop for a design revision if source semantics fail, required origin support is insufficient, or a mandatory default dependency requires authentication.

Deferred: individual reef/pixel forecasts, gridded maps, in-situ bleaching validation, ocean/weather covariates, fine-tuning, coherent DHW uncertainty, and prospective operational evaluation. Each requires separate data and evaluation work.

**AI Assistance Disclosure:** This notebook and its documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and release decisions. AI assistance is not independent verification, provider endorsement, or release approval.
