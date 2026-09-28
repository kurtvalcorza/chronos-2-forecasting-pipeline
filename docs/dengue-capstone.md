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

Still required, at this exact revision:

- a fresh Colab T4 Run all with the real models, including the bundle-consumer step and the
  actual resource figures;
- a representative authorized BYOD run with separate invalid-input cases;
- the learner observation described in the review.

Status stays **Candidate**.
