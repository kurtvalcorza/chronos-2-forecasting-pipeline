# Philippine Reef Heat-Stress Outlook

**Status: Candidate — a Kaggle T4 Run all with BYOD passed at `77a116d` and at `d55d51c`; a saved Colab T4 Run all is pending.**

The standalone guided notebook compares Chronos-2, persistence and a seasonal reference
on five NOAA Coral Reef Watch Philippine regional HotSpot series. It forecasts 28 daily
values, scores 7/14/28-day prefixes and derives DHW from known history plus predicted heat.
It does not predict observed coral bleaching or act as an operational NOAA advisory.

Notebook: [`DIMER_Philippine_Reef_Heat_Stress_Capstone.ipynb`](../tutorials/DIMER_Philippine_Reef_Heat_Stress_Capstone.ipynb).
Dataset: [source, rights and exact snapshot](datasets/philippine-reef-heat-stress.md).
Specification: [implementation contract](reef-capstone-spec.md).

## Build and validation

Run `python tools/build_reef_capstone_notebook.py` after editing the three readable source
files `tools/reef_core.py`, `tools/reef_models.py`, and `tools/reef_runtime.py`, or the builder's
guided text. The notebook carries these sources visibly and never downloads DIMER source.
The pinned requirements lock is embedded too. Notebook format, deterministic rebuild and
the assembled shared namespace are checked by `tests/test_reef_notebook.py`.

CPU checks: set `PYTHONPATH=src`, then run
`python -m pytest tests/test_reef_core.py tests/test_reef_notebook.py`.
The tests execute the actual NOAA parsing, audit, baselines and metrics. A separately labelled
integration test uses an explicit model double to exercise serialization, report charts,
refusal of altered metrics and resource limits. **That test is not real-model evidence.**

Upstream configuration hash and safetensors digest were checked against immutable HF revision
`95a9710e2596287d08352589f42634fa5abdf0a7`; no local GPU inference was performed.

## Scientific and implementation decisions

- Primary: equal-region macro 14-day HotSpot MAE. All model comparisons share origins.
- 115 validation and 184 test region-origin pairs. 2024 includes all five regions;
  2025 includes only Northern, Central and Eastern under the strict 365-day context rule.
- DHW adds values **>=1 °C** over 84 calendar days. Missing dates are never collapsed or zero-filled.
- Median-HotSpot-derived DHW is not necessarily median DHW; no DHW probability intervals are claimed.
- Raw and nonnegative-constrained forecasts are retained. No test-chosen upper clipping threshold.
- Test defaults remain fixed after the validation-only 180-versus-365-day comparison.
- The frozen snapshot outlook excludes Western/Southern because recent contexts contain gaps.
- Historical availability and foundation-model pretraining independence remain unverified.

The spec's proposed remotely hosted immutable data asset is replaced by an **embedded,
hash-verified data-only ZIP**. This avoids a publication dependency and makes the local notebook
portable. The original source TXT bytes, provenance and attribution are retained. No mutable
live download fallback exists. Publishing a new snapshot requires rebuilding and reviewing it.

Optional delay sensitivity is deferred; the implemented default records its zero-delay
retrospective assumption. BYOD offers validation and forward inference, without claiming
evaluation when future observations are unavailable. It is disabled by default.

### Dispositions after the 2026-09-28 notebook review

The review of revision `dcd3d90` (findings R1–R6, V1) led to these contract decisions; details and
evidence are in [reef-review-fixes.md](reef-review-fixes.md).

- **Interpreter.** Setup provisions a uv-managed CPython **3.12.13** for the isolated environment
  (`uv venv --managed-python`), whatever the Colab kernel's Python. Learners do not select an older
  runtime version. The kernel and environment interpreter versions are recorded in `setup_summary.json`.
- **Output completeness.** Every model-output file must contain exactly the frozen plan's
  region × origin × arm × lead grid, with `target_date = origin + lead`, source-derived references
  equal to the snapshot and all Chronos quantiles present. Scoring, lock, reload and report refuse
  anything else. Recomputed support must equal the plan (184 test, 115 common-2024, 69 eligible-2025).
- **Presentation.** The notebook prints a validation-only comparison after the context activity and
  six compact result tables (A–F: support, regional errors, interval diagnostics, high-stress errors,
  4/8 °C-week events, paired differences) after scoring. Complete tables remain in `metrics.csv`.
- **Lock.** The test stage requires unchanged configuration, frozen origins, seasonal reference,
  dataset/model manifests, source identity and baseline/validation/activity forecasts. Restarting a
  stage removes its own and all later stage records.
- **Forward outputs.** The outlook keeps raw and constrained HotSpot, Chronos quantiles, clipping and
  DHW components. BYOD reads region IDs literally and writes each input's forecasts and receipt to
  `byod/<input sha256 prefix>/`.
- **Serialization.** JSON records use native numeric types; the blanket `default=str` fallback is removed.
- **Quantile order.** Chronos-2 can emit slightly crossing quantiles. They are sorted before use
  (monotone rearrangement), and the point forecast is the rearranged median. The unsorted model values
  and the crossing size are exported.

## Qualification procedure

Open the notebook in a fresh Colab T4 runtime (the default runtime version) and use default Run all.
Preserve the executed notebook and `reef_evidence.zip`. Confirm no restart, rerun, manual edit,
authentication, or private data was needed. Record the exact notebook SHA-256 and Git revision
externally alongside the run; source manifests inside the notebook identify its generated code.
Do not treat CPU mocks, source validation, or baseline-only checks as hosted qualification.

The default creates an isolated Python environment. The validation budget check considers setup,
baseline and model-stage wall time including weight acquisition/loading; remaining time is an
estimate. Exceeding 30 minutes estimated total or 12 GiB observed peak blocks test unlocking.
Record actual complete-run time too: the estimate is not a timing guarantee.

The export includes forecasts, origins/exclusions, metrics/support, raw quantiles, data/model/source
identity, package versions, figures, conclusion prompts, checksums, and fresh-process baseline,
DHW, metric and real-model reload parity. Release promotion requires review of actual hosted outputs.

## NOTEBOOK_SPEC 2.2 conformance map

| Area | Implementation / evidence |
|---|---|
| Profile and guided layer | TASK-INFERENCE/GUIDED metadata; predictions, worked answers, controlled experiment, glossary |
| Standalone | Visible carried code; upstream library and embedded data; no workers/clone/source download |
| Default sample | Five-region immutable ZIP, member hashes, manifest, public-domain attribution |
| Runtime/model identity | Hash-locked isolated install; immutable model revision and file hashes |
| Validation | Strict source/BYOD parsing, calendar gaps, DHW reconstruction, refusal probes |
| Evaluation | Frozen chronological origins; same-grid baselines; quantile/support/paired diagnostics |
| New-data inference | Frozen-cutoff outlook and optional BYOD forecast |
| Export/reload | Fresh processes and explicit inputs; metric and weight reload checks |
| Fine-tuning/trained artifact | Not applicable: frozen pretrained task inference |
| External artifact inference | Not applicable: not ARTIFACT-INFERENCE |
| Hosted Run all | **Pending** — mandatory release evidence not yet available |

AI assistance supported development and technical writing under maintainer direction. Review,
validation and release decisions remain the maintainer's responsibility; assistance is not
independent verification, provider endorsement or release approval.
