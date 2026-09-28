# Reef capstone — fixes for the 2026-09-28 notebook review

**Status: Candidate. Verification pending; no hosted run of this revision exists yet.**

The review of PR #21 at `dcd3d905f3b3e7f807168bc3022db454817d8049` (notebook blob
`d49580720502f1f25dc9e4b41ef29f54695f38ed`) found three major issues (R1–R3), three smaller
issues (R4–R6) and one verification gap (V1). It recommended changes while keeping the scientific
design. This note describes each fix and the evidence behind it, and marks which checks were
executed and which were only inspected.

All edits are in the generator (`tools/build_reef_capstone_notebook.py`) and in the three carried
sources (`tools/reef_core.py`, `tools/reef_models.py` unchanged, `tools/reef_runtime.py`). The
notebook is regenerated, never hand-edited.

## Per finding

### V1 — the fresh-process reload failed under the exact lock

**Reproduced before fixing.** The unmodified revision was run under Python 3.12 with the notebook
lock's pandas 2.3.3 and NumPy 2.3.5. `test_export_reload_integration_with_explicit_model_double`
failed with `Fresh-process metric reconstruction mismatch`. So the failure also affects the locked
runtime, not only the reviewer's pandas 2.2.3. It passes in the repository's development environment
(pandas 3.0.5), which is why CI was green.

**Cause.** A NumPy integer horizon in the paired-difference records was written as a string by
`json.dumps(..., default=str)`, then compared with a freshly computed integer.

**Fix.**
- `evaluate_forecasts` casts horizons to `int`.
- The whole report is normalised through `json_native`, which converts NumPy and pandas scalars to
  native JSON numbers or dates and rejects anything else.
- `reef_json` uses `json_native` in place of the blanket `default=str`.

**Evidence.** The complete reef suite (56 tests, including the unmodified integration test) passes
under the exact lock with no review JSON shim. The CPU harness replaces two things, both labelled:
`torch` (absent from a CPU environment) is stubbed, and the package-version metadata lookups are
stubbed. New tests check that metrics round-trip through strict `json.dumps` without a string
fallback, and that every horizon is an `int`.

### R1 — the documented runtime entry path was incomplete

**Fix.**
- Setup no longer rejects kernels that are not Python 3.12.
- It installs the pinned `uv==0.10.12` and creates the isolated environment with
  `uv venv --clear --managed-python --python 3.12.13`. That is the version of Colab's own Python 3.12
  runtime.
- An existing environment is reused only if its interpreter reports 3.12.13; otherwise it is rebuilt.
- Setup then refuses to continue unless the environment interpreter is exactly 3.12.13.
- `setup_summary.json` records the kernel Python, the environment Python and whether the environment
  was reused.
- The opening instructions and the troubleshooting entry now say that any current Colab Python works.
- The no-restart contract and the hash-locked requirements are unchanged.

**Evidence (offline).** The pinned uv 0.10.12, run from a Python 3.11 host, lists
`cpython-3.12.13-linux-x86_64-gnu` and creates a working 3.12.13 environment. It also rebuilds an
existing 3.11 environment in place with `--clear`. The full locked dependency sync (torch, CUDA
wheels) was not repeated here. **A hosted T4 Run all is still required.**

### R2 — forecast validation did not bind the complete frozen comparison

**Fix.** `validate_forecast_grid` (core) builds the expected grid from the frozen eligible-origin
manifest and each stage's planned arms:

| Stage file | Planned origins × arms |
|---|---|
| baselines | validation and test × persistence, seasonal |
| validation | validation × chronos |
| activity | validation × chronos_180 |
| test | test × chronos |

The check then requires:
- exactly one row per key, with no missing and no extra keys (arm or split);
- `target_date = origin + lead`;
- `actual_hotspot_c`, `actual_dhw_c_weeks` and `origin_dhw_c_weeks` equal to the snapshot;
- every raw and constrained Chronos quantile column present and finite.

The origin manifest is re-hashed against the frozen configuration first. The check runs at lock,
score, reload and report. The comparison inside `evaluate_forecasts` now compares whole
region-origin sets across arms, so an arm missing from one region is refused even without the plan.
`reef_check_support` requires recomputed macro support to equal the plan:

| Scope | Regions | Origins |
|---|---:|---:|
| Full test | 5 | 184 |
| Common 2024 | 5 | 115 |
| 2025 | 3 | 69 |
| Validation | 5 | 115 |

Stage files load in canonical key order, so harmless row permutations give identical metrics.
Reload now also compares region, arm, split, origin and target-date identity, not only numeric
columns.

**Evidence.** The review's fault injections are now tests, and each is refused before any score:
- Central Chronos test rows deleted;
- one region-origin deleted from every arm;
- one lead row deleted;
- all target dates shifted by 17 days, both at scoring and at reload;
- a perturbed reference;
- NaN `q10`;
- a dropped `raw_q90` column;
- a duplicate row;
- a renamed arm;
- the seasonal arm deleted.

A shuffled test file scores identically. The Western/Southern 2025 exclusions remain excluded.

### R3 — the learner was asked to interpret diagnostics that were not surfaced

**Fix.**
- A `compare` stage runs after the context activity. On validation data only, it prints:
  - macro 7/14/28-day HotSpot and DHW errors for all arms, with region/origin/day support;
  - a per-region 365-versus-180-day table;
  - a traced valid context (365- and 180-day windows, target window) and a traced rejected origin
    with its reason.
- Scoring prints six tables with `to_string`, never truncated:
  - **A.** Full, common-2024 and 2025 macro errors, with support.
  - **B.** Regional 14-day errors.
  - **C.** q10–q90 coverage, width and pinball loss.
  - **D.** High-stress-day errors.
  - **E.** Pooled 4 and 8 °C-week endpoint events (all and new exceedances), with counts,
    precision and recall. Undefined ratios show `n/a` with a reason: "no observations", "no observed
    positives" or "no predicted positives".
  - **F.** Paired Chronos-minus-baseline 14-day differences, with the better/tie/worse shares.
- The final summary table adds regions, origins and days.
- Section 10's prompts name the table that answers each one.

**Evidence.** A CPU harness ran the full pipeline on the real NOAA snapshot with a model double.
It printed and saved every table, and tests assert their presence and content. Tables from a model
double show no Chronos skill (it repeats the last value) and are **not forecast evidence**.

### R4 — recorded hashes were not fully enforced

**Fix.**
- `experiment_lock.json` records SHA-256 digests of:
  - the configuration, origin manifest and seasonal reference;
  - the dataset and model manifests;
  - the source identity;
  - the baseline, validation and activity forecasts.
- The test stage refuses to run if any of them changed.
- When the notebook starts a stage, the runner removes that stage's records and those of every later
  stage, so a completion file cannot describe an earlier attempt.

**Evidence.** Tests change validation, activity, model-manifest and seasonal-reference files after
the lock and show that the test stage refuses before any model load. They also show that restarting
validation removes the lock, test outputs, metrics and outlook, and keeps preparation and baselines.

### R5 — forward and BYOD exports discarded fields

**Fix.**
- **Outlook.** `outlook.csv` keeps raw and constrained HotSpot, a clipping flag, raw and constrained
  q10/q50/q90 for Chronos (null for baselines), and the known and new DHW contributions.
- **BYOD stage.** BYOD is now a runner stage, `byod --csv`, instead of an inline program.
  - It reads the CSV as literal text, so `001` and `NA` stay identifiers.
  - It refuses blank or non-numeric values.
  - It writes `byod/<input sha256[:16]>/forecasts.csv` with the same diagnostic fields, plus a
    `receipt.json`. The receipt holds the input digest, row count, per-region context, units,
    confirmed semantics, model revision, runner digest and output digest, and marks the run as
    unscored.
- **Empty path.** With the path empty, BYOD practises on the generated `byod_example.csv`.

**Evidence.** Tests cover:
- literal `001` and `NA` identifiers carried through the forecasts and receipt;
- distinct output folders per input;
- refusal of blank, non-numeric, negative, short and gapped inputs.

### R6 — figures

**Fix.**
- The outlook figure has a suptitle naming the frozen issuance origin (2026-09-26). It says the
  outlook is not a live warning and lists the regions excluded for incomplete context.
- Each panel and the component chart name the origin.
- All time axes use concise automatic date ticks.

**Evidence.** Rendered with the model double and inspected; a test asserts the titles. The
rendering is not a real forecast.

## User-visible changes

- Setup downloads a managed Python 3.12.13 (about 30 MB) instead of using the kernel interpreter.
- New printed tables after the context activity and after scoring; the new `compare` stage adds
  `validation_tables.txt`, and scoring adds `results_tables.txt`. Both are in the bundle.
- Scoring, reload and report now refuse incomplete or misidentified model outputs that were
  previously accepted.
- `experiment_lock.json` schema: a `sha256` map replaces the three separate digest fields.
- `outlook.csv` has additional columns.
- BYOD outputs moved from `byod_forecasts.csv` to `byod/<digest>/forecasts.csv` plus `receipt.json`;
  an empty `BYOD_CSV_PATH` now uses the generated example.
- The obsolete test that compiled the inline BYOD program is replaced by stage tests.

## Verification summary

| Check | Environment | Result |
|---|---|---|
| Full non-integration suite | repo dev lock (Python 3.12.11, pandas 3.0.5) | 440 passed, 22 integration deselected (baseline 407) |
| Reef suite, unmodified revision | notebook lock (pandas 2.3.3, NumPy 2.3.5) | 1 failed (V1 reproduced), 22 passed |
| Reef suite, this revision | notebook lock, with torch and package-metadata stubs | 56 passed |
| `ruff check src tests tools`, release-asset validator, `build_notebook.py --check` | repo dev lock | pass |
| Managed interpreter provisioning with pinned uv 0.10.12 | Python 3.11 host | 3.12.13 created; stale environment rebuilt |

Not performed:
- Colab or GPU execution.
- The full hosted dependency sync.
- Chronos weight download, real-model forecasts or real-model reload.
- Observation of learners.

Model outputs in every integration check come from a deterministic test double.

## Hosted run of `f3c4a1e` — 2026-09-28

A maintainer Colab T4 Run all confirmed R1 on a Python 3.13.15 kernel: the isolated environment
was built on Python 3.12.13. The runtime then disconnected during the first stage, `prepare`,
without an error in the notebook. The run is recorded in
[release-verification.md](release-verification.md). Offline, the same cells completed: setup in
71 s, `prepare` in 3 s and baselines in 29 s. The next revision streams every stage into the cell
through a per-stage log file, prints a 30 s heartbeat, and gives child processes a clean
environment. It also drops the "7" from the title, as the maintainer edited it.

## Remaining gates

1. Fresh Colab T4 default Run all on this head, recording the kernel and environment interpreters,
   setup and stage timings, peak GPU memory, all planned origins, exports and real-model reload.
2. The BYOD journey on the generated example and on one invalid input.
3. Maintainer review of the executed notebook and evidence bundle. The status stays Candidate until
   then.
