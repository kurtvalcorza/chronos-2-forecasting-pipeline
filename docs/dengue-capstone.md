# Philippine dengue capstone — implementation and evidence

Status: **Candidate**. Fresh Colab T4 Run all, actual pretrained-model performance and full-model
reload verification remain pending. No GPU or pretrained weights were used locally for this build.

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
