# Philippine dengue capstone — implementation and evidence

Status: **Candidate**, revision `0.2.0-candidate` (see the review section at the end). Fresh Colab
T4 Run all, actual pretrained-model performance, full-model reload and bundle-reconstruction
verification remain pending. No GPU or pretrained weights were used locally for this build.

## Approved interpretation

The maintainer approved an exploratory source-block benchmark on 2026-09-27 after the pinned
source's own README and validation script confirmed inconsistent calendar descriptions.
We retain the published `(YR, WN)` ordering and do not reconstruct actual dates or claim verified
weekly timing or case–weather alignment. The zero-reporting-delay default is a retrospective
assumption; the two-block sensitivity is illustrative. Final weather products and revised
counts do not establish historical publication availability. Pretraining overlap is unknown.

## Source audit

Zenodo release: https://zenodo.org/records/21978184 (v1.1.0), ODC-ODbL 1.0 with original source
terms and citation retained. Both author and UPRI-NOAH repositories distribute the same workbook;
they are not independent replication datasets. Download verifies the full archive and workbook
bytes/hashes before parsing. `tools/dengue_dataset_audit.json` contains the measured inventory,
dictionary mapping, year-level missingness/flags, ranges and source evidence.

The QC subset has 832 unique rows, 2010–2025, with complete selected case/rain/temperature fields.
Counts range 1–697. The output `DATA_LICENSE.md` preserves the full original publisher licence
and README. Source photos, health records and spatial archives are not needed by this capstone.

## Experiment and implementation choices

- Initial history 2010–2021; validation 2022–2023; test 2024–2025.
- Each evaluation partition has 26 origins and 104 origin/horizon pairs, with common source keys.
- Persistence, seasonal naïve, fixed Ridge, Chronos-2, Mitra case-only and Mitra plus weather.
- Mitra uses labelled in-context conditioning, not gradient fine-tuning. Each query creates a
  fresh support-only preprocessor; safe support arrays reconstruct the state at reload.
- Validation chooses a 52/104-block issuance window using case-only Mitra, ties favour 52.
  Both Mitra variants use the same window/rows; Chronos and Ridge retain fixed 104-block windows.
- A target is eligible for context only once mature at the current cutoff. The optional delay
  activity uses validation targets only and leaves the canonical experiment intact.
- Raw and nonnegative point forecasts are exported; no rounding before evaluation. Chronos raw
  model quantiles are explicitly uncalibrated. Paired bootstrap groups contiguous forecast origins.
- Fresh-process reload checks the final completed origin and unscored final inference.
  Receipt hashes bind source, dataset, policy and stage outputs; CSV verification checks all fields.

The implementation requires six complete years for optional BYOD, correcting the draft spec's
five-year minimum: two history years are needed before the four validation/test years with
52-block lags. It rejects extra columns and requires aggregate source-block/unit confirmation.
Default selected fields have no missing values; this version rejects missing BYOD fields rather
than silently imputing them. All compared systems therefore share the same complete cohort.

The capstone environment is isolated from the repository's runtime. Its fully hashed lock uses
AutoGluon 1.5.0-compatible Torch 2.9.1, NumPy 2.3.5 and pandas 2.3.3. Snapshot manifests retain
the existing immutable Chronos/Mitra revisions and independently verified file digests.

## Verification boundary

The original offline repository suite passed before integration (344 tests). Local CPU validation
includes source parsing, feature/label availability, future-data poisoning, exact paired cohorts,
baseline math, bootstrap/interval math, download/cache tampering, synthetic stage integration,
safe artifacts and reload refusals. Synthetic model outputs are software fixtures, not performance
evidence. The model adapter also received a tiny random-weight upstream CPU interface check.

Final full offline run: **413 passed, 22 live integration tests deselected**. Ruff, all three
notebook generation checks and release-asset validation passed. The generated notebook contains
19 cells with empty execution outputs and passes nbformat and code parsing checks. All 69 new
tests passed; live integration tests were deliberately excluded because they require real weights.

Actual pinned-source preparation and all three simple validation baselines have run from the
notebook's embedded source. Equal-horizon validation MAE: persistence **16.9327**, seasonal naïve
**43.1731**, Ridge **75.1282**. These are retrospective source-block results; Ridge's poor result
is retained, not tuned away. Foundation-model scores and gains from weather remain unmeasured.

Keep the generated notebook output-free for distribution. For qualification preserve the executed
notebook, exact source revision, data/model manifests, experiment lock, predictions, metrics,
receipts, resource measurements, `verification.json` and `results.zip`. The 60-minute/20-GiB/12-GiB
resource targets remain unverified. Optional BYOD needs a hosted representative run too.

```text
python tools/build_dengue_capstone.py
python tools/build_dengue_capstone.py --check
python tools/validate_release_assets.py
pytest -m "not integration"
ruff check src tests tools
```

No commit, push or publication is included in this local build.

## Revision 0.2.0-candidate — Notebook Review Framework v1 findings (2026-09-28)

A review of PR #20 at `209d8f5` (notebook blob `66309b6a`) concluded **Needs revision** and
recommended requesting changes. It reported three major findings (DENGUE-01 to 03) and three
smaller ones (DENGUE-04 to 06). It also recorded that the availability, maturity, pairing,
interval, cohort and activity-isolation controls passed its checks. Those controls are unchanged.
The review and its probe ZIP are archived in
[`reviews/2026-09-28-dengue-capstone-review/`](reviews/2026-09-28-dengue-capstone-review/).

`tests/test_dengue_review_fixes.py` adds 30 tests. They execute the notebook's own table helper
and section cells, plus the embedded runtime, on synthetic series with model doubles. All 30 fail
on `209d8f5`.

| Finding | Correction in 0.2.0 | Acceptance check |
|---|---|---|
| **DENGUE-01** (major): tables showed the first nine columns and twelve rows, so the forecasts, clipping, quantiles and three systems were hidden | `table(name, columns, limit)` selects columns by name and refuses a missing column. The runtime writes learner views: `future_forecast_view.csv` (four target blocks by six systems), `largest_misses.csv` (reference, prediction, signed and absolute error), `delay_summary.csv` (paired zero-delay and two-block MAE with the change) and `delay_comparison.csv` (paired forecasts). Full CSVs keep every column | The section-6 cell shows the four-by-six view and all 24 forecasts, with no reference column. Each largest miss shows `error = prediction − reference`. The delay summary equals both metric files over 104 pairs per system |
| **DENGUE-02** (major): BYOD accepted fractional counts, a temperature of 500 and extra row values; a missing value gave a raw error; constant support failed only inside Mitra, after the models were loaded | `core.value_problem` is shared by the default and BYOD paths: nonnegative integer counts, nonnegative rain, and Celsius from −30 to 60. The BYOD parser names the file, line and column, and refuses wrong-width rows. `prepare` checks all 1,264 planned Mitra contexts (both partitions, both windows, delays 0 and 2, both systems, and the future origin) with the adapter's own `support_problem` rule. It stops before any model is staged and writes `context_preflight.json` | Six malformed-value cases are refused with line and column. Zero counts in a varying series pass. An all-zero series is refused (1,264 of 1,264 contexts) without loading a model. A locally constant stretch is named by origin, horizon, window and delay. The real pinned source checks 1,264 contexts and refuses none |
| **DENGUE-03** (major): reload needed the original `data.json` and `run_config.json`, which the ZIP lacks | A new `dengue_runtime.py --consume results.zip --workdir DIR --models CACHE` entry point extracts the bundle safely and verifies every member hash. It requires `source.json` and `model_manifest.json` to equal the trusted embedded copies, and it checks the lock and feature-schema hashes and a `bundle_identity`. That identity covers the provenance, plan, lock, recorded data digest and code, but no run path. It then reproduces the final test origin and the future forecasts. Section 7 runs it from a new directory holding only the embedded code. The artifact manifest discloses the aggregate support data it retains. Same-workspace `reload` is unchanged | After the original run directory is deleted, the bundle reproduces all 32 predictions. A changed lock, area, plan or data digest, an extra member, a different model manifest, and a changed support context (even with updated hashes) are each refused. The reviewer's ZIP reconstructs in a separate process (`consumer_verification_0.2.0.json`) |
| **DENGUE-04**: BYOD provenance had no area | New `BYOD_AREA` and `BYOD_SOURCE_CITATION` controls are required for BYOD. They go into `plan.json`, `dataset_audit.json`, figure titles and `run_summary.json`, with the units; they are never predictors. The default path records Quezon City | A missing or multi-line area is refused; the recorded values are checked |
| **DENGUE-05**: long carrier cell; no traced feature row; delay comparison split across two files | The carrier cell is collapsed (`cellView: form`, `source_hidden`) with a title and an explanation above it. `feature_example.csv` traces every weather-model feature of the first test origin to its source blocks, plus the latest mature label. Section 5 shows the paired delay view | Metadata check; each traced block is at or before the origin, and each value equals `feature_row` |
| **DENGUE-06**: the summary lacked limitations and runtime identity | `run_summary.json` carries the limitations (calendar, final data vintages, delay, pretraining overlap, evaluation span), area, source, units and environment. The environment covers Python, platform, device, precision, and NumPy, pandas, scikit-learn, torch, Chronos, Transformers, AutoGluon, safetensors and Matplotlib versions. Resource figures are labelled as targets, next to the measured stage seconds and peak GPU allocation | Field check on a complete synthetic run |

### Specification amendment A1 (applies from revision 0.2.0)

These choices were already documented above; they are now recorded as a versioned amendment to
the capstone specification:

1. **Scope:** an exploratory published-source-block benchmark (approved 2026-09-27). Performance
   claims about verified weekly timing, alignment or prospective availability are out of scope.
2. **BYOD minimum:** six complete 52-block years (the draft said five): two history years, then
   two validation and two test years.
3. **Complete cases only:** missing selected values are refused, never imputed or set to zero.
   The draft's median imputation is not implemented.
4. **Constant support:** when any planned Mitra context has a constant target or no varying
   feature, the whole experiment stops before model staging. The affected contexts are listed.
   No origin is dropped, no variance is invented, and no fallback is reported as a Mitra forecast.
5. **Units and area:** counts are nonnegative integers, rain is nonnegative mm per block and
   temperature is Celsius from −30 to 60, on both paths. BYOD names its area and source.

### Verification of revision 0.2.0 (not hosted execution evidence)

- **Repository checks:** `ruff check src tests tools`, both notebook generation checks and
  release-asset validation pass. `pytest -m "not integration"` gives 482 passed, 22 deselected:
  the earlier 453 plus the 29 new tests.
- **Reviewer's harness, re-run against the regenerated notebook's embedded source**
  (`revised_probe_results_0.2.0.json`). The only harness change adds the new BYOD area fields.
  - Probes 05 and 09 now stop at the corrected behaviour (constant BYOD refused during
    preparation; `table()` needs named columns).
  - Probe 06 records every malformed value refused with line and column.
  - The other probes complete as before, including the synthetic 624-row run and the 32-prediction
    same-workspace reload.
- **Bundle consumer:** `consume_fixture.py` ran in a separate process on the harness's synthetic
  ZIP, from a fresh code directory, with the original run directory moved away. It reproduced 32
  predictions with a maximum difference of 0.0 (`consumer_verification_0.2.0.json`). Its model
  adapters were review doubles.
- **Real source on CPU:** `prepare` and `baselines` ran on the SHA-verified Zenodo archive
  (`real_source_cpu_check_0.2.0.json`). The preflight checked 1,264 contexts and refused none. The
  validation MAE is unchanged: persistence 16.9327, seasonal naïve 43.1731, Ridge 75.1282.

Still required at this exact revision (the first item is closed by the run recorded below):

- a fresh Colab T4 Run all with the real models, including the bundle-consumer step and the
  actual resource figures;
- a representative authorized BYOD run with separate invalid-input cases;
- the learner observation described in the review.

Status stays **Candidate**.

## Maintainer-supplied Colab execution of revision 0.2.0 — 2026-09-28

The maintainer supplied an executed Colab copy of revision `0.2.0-candidate`, preserved
byte-for-byte as
[evidence](execution-evidence/2026-09-28/DIMER_Philippine_Dengue_Forecasting_Capstone_0.2.0.ipynb).

- **Source:** commit `bd35d0d`, notebook blob `a8a4348b`. All 19 cell ids and sources match the
  committed notebook exactly, with no Colab `# @title` or parameter edits. Default settings were
  used, with BYOD off.
- **Executed-file SHA-256:** `85efcb065e402f934be98ff03e778bb55995d19728ab2c037badb2d9b101f838`.
- **Runtime:** Google Colab, Tesla T4. The isolated environment ran Python 3.12.12 with torch 2.9.1,
  chronos-forecasting 2.3.1, autogluon.tabular 1.5.0, transformers 4.57.6, NumPy 2.3.5, pandas
  2.3.3, scikit-learn 1.7.2, safetensors 0.8.0 and Matplotlib 3.10.8, all at float32.
- **Execution:** 9/9 code cells ran with counts 1–9 in order, and there are no saved errors.
  Every stage printed `PASS`: prepare, baselines, validation, lock, test, activity, future,
  reload and report. The last cell printed `consume: PASS (32 predictions)`.
- **Resources (measured):**
  - Stage times: prepare 34.1 s, baselines 1.0 s, validation 118.7 s, lock 0.0 s, test 66.8 s,
    activity 72.6 s, future 15.9 s, reload 18.4 s; 327.9 s in total. That excludes the
    environment build and the model downloads, whose time the saved outputs do not print.
  - Peak allocated GPU memory: 0.48 GiB (512,901,632 bytes), against the 12 GiB target.
  - The 60-minute and 20-GiB targets were not measured end to end.

| Journey or check | Result |
|---|---|
| Source and context preflight | 832 rows; 26 validation and 26 test origins; 1,264 of 1,264 planned Mitra contexts admitted (DENGUE-02); area "Quezon City, Philippines" printed and shown in figure titles (DENGUE-04) |
| Traced feature row (DENGUE-05) | First test origin 2023-B52 → target 2024-B01. All 27 features read only blocks at or before 2023-B52 (the two seasonal terms read none); the latest mature label is 2023-B52 (72 cases) |
| Validation (zero delay, 104 pairs each) | Equal-horizon MAE: persistence 16.93, Chronos 18.63, Mitra + weather 19.61, Mitra case-only 20.50, seasonal naïve 43.17, Ridge 75.13. The Mitra window selected was 52 (MAE 20.50 vs 21.01 at 104) |
| Held-out test (104 pairs each) | MAE: Chronos 57.70, persistence 59.05, Mitra + weather 64.79, Mitra case-only 71.41, seasonal naïve 109.83, Ridge 111.20 |
| Weather ablation | Mitra + weather minus case-only: −6.61 MAE, bootstrap interval [−21.48, 6.44] over 7 origin groups. The interval contains zero; descriptive only |
| High-case blocks (35 test pairs above 242.4) | MAE: Chronos 90.62, persistence 99.31, Mitra case-only 111.26, Mitra + weather 111.66, seasonal 195.60, Ridge 204.67 |
| Chronos nominal 80% interval | Coverage 0.69 / 0.65 / 0.62 / 0.54 for horizons 1–4, below nominal |
| Largest misses (DENGUE-01) | 10 of 20 shown with reference, prediction and signed error. The largest is Ridge at 2025-B10: 1,229 vs 321 |
| Delay activity (DENGUE-01/05) | Paired zero-delay vs two-block MAE change: persistence +6.04, seasonal 0.00, Ridge +33.92, Chronos +5.16, Mitra case-only +3.03, Mitra + weather +5.06 |
| Future forecasts (DENGUE-01) | All 24 shown: the four-by-six view plus the detail table, with Chronos q10/q50/q90. No flooring occurred. No reference column |
| Same-workspace reload | 32 predictions, max difference 0.0, fresh process |
| Bundle reconstruction (DENGUE-03) | `consume: PASS (32 predictions)`, max difference 0.0, `original_workspace_read: false`, bundle SHA-256 `bd49e267…` |
| Run summary (DENGUE-06) | Limitations, area, units, environment and resource targets present, with the targets labelled as not measurements |
| BYOD, invalid inputs | Not assessed in this run (BYOD off by default) |

These are retrospective source-block results under the published alignment. They are not
verified weekly or operational forecasts. The paired weather interval includes zero, so no
weather gain is claimed.

**Evidence boundary:** saved outputs were inspected; execution was not independently repeated.
The exported files and `results.zip` were not supplied separately, so the bundle's bytes were not
inspected. In `test_forecasts.png`, the systems' colours differ between the two panels; each
legend is correct.

**Still open:**

- a representative authorized BYOD run with invalid-input cases;
- the learner observation;
- an end-to-end wall-clock and disk measurement against the 60-minute and 20-GiB targets.

Status stays **Candidate**.

## Notebook source layout change (2026-10-02)

The generator now writes each carried file in the `FILES` cell as a parenthesised run of short
string pieces (`tools/notebook_carrier.py`) instead of one `repr(files)` line; the longest source
line fell from 242,899 to 754 characters. Python joins the pieces back into identical text, so the
carried text is unchanged: every carried file equals the previous notebook's byte for byte, except
that `source.json` records the generator's own SHA-256 (`generator_sha256`), which changes with any
generator edit. The per-file hashes in `source.json` and the runtime integrity check are the same.
Only cell `dengue-03` changed. The notebook blob changes from `a8a4348b` to `de007324`. The hosted
run recorded above was of blob `a8a4348b`; the new blob was re-run on 2026-10-03 (below). Status is
unchanged.

### Colab CLI execution of `08b2343` (blob `de007324`) — 2026-10-03

- **File:** [`execution-evidence/2026-10-03/DIMER_Philippine_Dengue_Forecasting_Capstone_08b2343_colab-cli-t4.ipynb`](execution-evidence/2026-10-03/DIMER_Philippine_Dengue_Forecasting_Capstone_08b2343_colab-cli-t4.ipynb), SHA-256 `4357fba57a290c097ccfe6a264bbfc78412f62705c916f0f7633293428af3b28`, a byte-for-byte copy of the CLI's output notebook.
- **Executor:** Google Colab CLI 0.7.4 on a fresh Colab Tesla T4 session via the workspace `colab-cli-serial-test-suite` (`colab new --gpu T4`, `colab exec -f`, `colab stop`). Code cells ran in order in one kernel; this is not a browser Run all, and the CLI records no execution counts, so order is evidenced by its `Executing cell k/N` log. The notebook was downloaded from GitHub at `08b2343` and its git blob checked before the session was created.
- **Result:** **PASSED**: 9/9 code cells, no error output, 444.9 s wall. Default path.
- **Equivalence:** compared with the 2026-09-28 Colab run of revision 0.2.0 (blob `a8a4348b`). Of the 38 output files whose SHA-256 the notebook prints, 35 are byte-identical, including `future_predictions.json`/`.csv`, the metrics, selections and figures. Two differ as expected: `experiment_lock.json` and `artifact_manifest.json` store the run `identity`, which hashes the carried `source.json`, whose `generator_sha256` changed with the generator; the derived `identity` and `previous` chain hashes differ for the same reason. The third, `artifact_future_chronos.json` (the Chronos state saved by the future stage), differs for a reason not determined here: its sibling `artifact_test_chronos.json` is identical and the forecasts it produced are byte-identical. This is recorded as an open observation, not a regression.
- **Boundary:** saved outputs were inspected; BYOD was not exercised. Status remains Candidate.
