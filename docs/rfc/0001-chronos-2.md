<!-- Source issue: https://github.com/kurtvalcorza/chronos-2-forecasting-pipeline/issues/1 -->
<!-- Captured at: 2026-09-08T04:34:04Z (last issue update) -->
<!-- Verbatim copy of the issue body; the issue is mutable, this file is the anchor. -->

## Status

**RFC v2 / Integrator-amended implementation proposal** — revised after Agent Relay review. This version is the builder contract pending one final reviewer pass.

## Summary

DIMERify **Amazon Chronos-2** as an open-weight, pretrained **zero-shot time-series forecasting** pipeline.

The intended UX mirrors the existing Mitra/TabICL pattern: users bring a valid dataset, identify the time-series schema, configure a forecast, and run pretrained inference without training a task-specific model.

Chronos-2 is a **forecasting specialist**, not a generic time-series model.

### v1 capabilities

- univariate forecasting;
- multiple independent series;
- multivariate / multi-target forecasting where supported by the pinned upstream contract;
- past covariates;
- known-future covariates;
- configurable horizon;
- probabilistic quantile forecasts;
- CPU/GPU inference with explicit runtime provenance;
- optional chronological evaluation against held-out future truth;
- BYOD + bundled reproducible sample datasets.

---

## Integrator decisions from RFC review

The following review findings are accepted into this RFC:

- **C-1:** preserve `target_name` in multi-target output and explicitly map upstream raw columns to DIMER-normalized columns.
- **C-2:** treat upstream `predictions` as the **median / q0.5 point forecast** for the pinned version, not a statistical mean; add a discriminating oracle.
- **C-3:** DIMER validates regularity/gaps/same-frequency **before** calling `predict_df`, even when the user explicitly supplies `frequency`.
- **C-4:** known-future covariates must also exist in the historical/context table because the pinned upstream validator requires future columns to be a subset of historical columns.
- **C-5:** requests outside the model's trained quantile grid are rejected by default rather than silently clamped.
- **C-6:** provenance records requested **and effective** context/horizon semantics and whether autoregressive unrolling occurred.
- **C-7:** the immutable Hugging Face revision SHA is the **primary supply-chain invariant**; weight/config digests are secondary file-level assertions.
- **C-8:** fully lock the resolved runtime stack, not only `chronos-forecasting`.
- **C-9:** resource guards account for targets + covariates when validating batch size.
- **C-10:** `cross_learning=false` remains the v1 default and is not exposed as an ordinary beginner knob.
- **C-11:** the model identity is now concretely pinned below; weight/license metadata has been independently rechecked.
- **X-1:** tests must be discriminating oracles, not presence-only assertions.
- **X-2:** the first implementation PR must copy this RFC into version control under `docs/rfc/0001-chronos-2.md` and link the issue to the resulting commit.

---

## Upstream model identity and supply chain

### Model

- **Model:** `amazon/chronos-2`
- **Family:** Chronos
- **Task:** zero-shot time-series forecasting
- **Architecture:** T5-family Chronos-2 model
- **Parameters:** ~119.5M
- **License:** Apache-2.0
- **Weight format:** `safetensors`
- **Pinned Hugging Face revision:** `95a9710e2596287d08352589f42634fa5abdf0a7`
- **`model.safetensors` SHA-256:** `ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42`
- **`model.safetensors` size:** 477,930,472 bytes

### Supply-chain invariants

The implementation MUST:

1. fetch only the pinned official Hugging Face revision in the tutorial/production-facing path;
2. reject mutable `main`, `latest`, arbitrary local model paths, or `s3://` sources in that path;
3. verify the model revision resolved to the pinned commit;
4. verify `model.safetensors` SHA-256 and byte size;
5. calculate and record `config.json` SHA-256 for the pinned revision before Phase 1 merge;
6. record the license file and source URL at that revision;
7. use `safetensors`; no silent fallback to pickle-based weights;
8. record revision + config digest + weight digest in `MODEL_CARD.md` and exported provenance.

The **revision SHA is the primary integrity anchor** because it covers repository configuration as well as weights.

---

## Runtime/dependency reproducibility

Pin a tested `chronos-forecasting` release/source revision **and the complete resolved environment**.

Required:

- lock `chronos-forecasting`, PyTorch, Transformers, Accelerate, NumPy, pandas, einops, Hugging Face Hub, and transitive runtime dependencies;
- commit a reproducible lock (`uv.lock`, fully pinned requirements with hashes, or equivalent);
- CI and Colab install from the same lock;
- export actual resolved `chronos-forecasting`, `torch`, `transformers`, Python, device, and dtype versions in inference metadata.

Do not rely on broad upstream dependency ranges in a production-facing notebook.

---

## Scope

### In scope for v1

- pinned Chronos-2 base model;
- zero-shot inference only — **no fine-tuning**;
- long-format CSV/Parquet/DataFrame workflow;
- one or multiple series IDs;
- one or multiple targets where supported;
- past covariates;
- known-future covariates;
- quantile forecasting;
- forecast visualization;
- chronological evaluation with optional held-out truth;
- naive benchmark(s) on the identical test window;
- sample datasets + dataset cards;
- fresh-Colab BYOD tutorial;
- reusable Python validation/model/inference API outside the notebook;
- portable result/provenance export.

### Explicitly out of scope for v1

- fine-tuning or pretraining Chronos;
- online/streaming weight updates;
- causal interpretation of covariates;
- silent interpolation of irregular/gappy input;
- calibration guarantees;
- Chronos-Bolt / older Chronos checkpoints;
- DIMER backend/server implementation beyond a stable portable inference contract;
- `cross_learning=true` as a default/tutorial workflow.

---

## Canonical input contract

### Historical/context table

Required columns:

```text
<id column>
<timestamp column>
<target column or columns>
```

Example:

```csv
series_id,timestamp,target,temperature,holiday
A,2026-01-01T00:00:00,102.3,27.4,0
A,2026-01-01T01:00:00,108.7,27.1,0
```

Additional historical columns may be used as upstream-supported covariates.

### Future-covariate table

Optional. Used only for known-future covariates.

Must contain:

```text
<id column>
<timestamp column>
<future covariate columns...>
```

Contract:

- future target values are forbidden as inputs;
- future item IDs must match historical IDs;
- exactly `prediction_length` future rows per item;
- **every future covariate column must also exist in the historical/context table**;
- future timestamps must start exactly at the forecast origin and follow the validated frequency.

### User parameters

```yaml
id_column: series_id
timestamp_column: timestamp
target: target
prediction_length: 24
quantile_levels: [0.1, 0.5, 0.9]
batch_size: 256
context_length: null
frequency: null
device: auto
cross_learning: false
```

`cross_learning` remains `false` in v1 tutorials. If exposed later, its value and `batch_size` MUST be exported because results depend on both.

---

## DIMER-side validation contract

Validation runs **before `predict_df` and independently of upstream validation**.

Required checks:

1. required columns exist;
2. timestamp parses successfully;
3. IDs are non-null;
4. targets are numeric/coercible under an explicit policy;
5. no duplicate `(id, timestamp, target_name)` logical observations;
6. deterministic ordering;
7. each series/target has a regular frequency;
8. all series in one request share the required frequency semantics;
9. gaps are detected and rejected in v1 — no silent interpolation;
10. series with fewer than 3 observations are rejected for ordinary forecasting because upstream frequency inference cannot validate them robustly;
11. history/context is non-empty and meets configured minimum context requirements;
12. `prediction_length > 0` and within DIMER resource limits;
13. quantiles are unique, sorted numerics strictly within the **trained quantile grid supported by the pinned model**;
14. outside-grid quantiles hard-fail by default; no silent upstream clamping;
15. future covariate timestamps align exactly;
16. future table length equals horizon per ID;
17. future columns are a subset of historical columns;
18. future target leakage is rejected;
19. missing-value handling follows an explicit documented policy;
20. resource guards cover number of IDs, targets, covariates, context, horizon, and batch size;
21. enforce `batch_size >= n_targets + n_covariates` where required by the pinned upstream execution path.

Supplying `frequency=` does **not** bypass rules 7–10.

---

## Forecasting modes

### Mode A — univariate

One target per series. Default beginner tutorial.

### Mode B — multi-series

Many independent IDs sharing the same target schema.

### Mode C — multivariate / multi-target

Multiple jointly forecast target variables where supported by the pinned Chronos-2 API.

Documentation MUST distinguish Mode B from Mode C.

### Mode D — covariate-informed

Explicitly distinguish:

- targets;
- past-only covariates;
- known-future covariates.

No future target leakage.

---

## Raw and normalized output contracts

### Upstream raw `predict_df` schema

For the pinned upstream behavior, expect:

```text
<id_column>
<timestamp_column>
target_name
predictions
<quantile columns named as strings, e.g. "0.1", "0.5", "0.9">
```

For multi-target output, `target_name` is required to disambiguate rows.

### DIMER normalized schema

Normalize explicitly to:

```text
series_id
timestamp
target_name
prediction
q0.1
q0.5
q0.9
```

The rename map must be deterministic and tested.

### Point-forecast semantics

For the pinned implementation, upstream `predictions` is the **median / 0.5 quantile** even though upstream documentation may call it a mean.

DIMER MUST:

- label it `prediction` / `median_point_forecast`, never `mean`;
- with `0.5` requested, assert that upstream `predictions` equals raw quantile column `"0.5"` exactly;
- fail CI if a future upstream version changes this behavior.

---

## Quantile semantics

The model's trained quantile support must be read from the pinned config and recorded in provenance.

DIMER behavior:

- default `[0.1, 0.5, 0.9]`;
- reject requested quantiles outside the trained grid;
- do not export labels such as `q0.01` when upstream has silently substituted an edge quantile;
- record requested and effective quantile levels.

---

## Context and horizon semantics

Upstream may clamp context to model limits and may autoregressively unroll horizons beyond native prediction capacity.

DIMER provenance must therefore record:

```text
requested_context_length
effective_context_length
model_context_length
requested_prediction_length
effective_prediction_length
model_prediction_length
autoregressive_unrolled: true|false
```

Default tutorials should stay within native tested limits. If unrolling is permitted, surface an explicit warning and preserve the flag in exported metadata.

---

## Evaluation contract

Evaluation is optional for genuine future inference.

When held-out truth is available:

- split **chronologically**, never randomly;
- never expose future targets as covariates;
- derive preprocessing/statistics from historical context only;
- report per-series and aggregate metrics;
- evaluate point and probabilistic forecasts separately;
- compare baselines on the identical horizon/window.

Recommended metrics:

- MAE;
- RMSE;
- sMAPE or WAPE where mathematically valid;
- pinball/quantile loss;
- empirical interval coverage.

Include at least last-value and/or seasonal-naive baseline where appropriate.

Do not claim tutorial benchmark performance is unbiased if the dataset may overlap upstream pretraining/benchmark corpora.

---

## Sample datasets and notebook UX

Provide at least:

1. univariate sample;
2. multiple-series sample;
3. covariate-informed sample.

Each sample requires:

- `DATASET_CARD.md`;
- source/provenance;
- verified redistribution/license basis;
- rows, IDs, frequency, targets, covariates;
- preprocessing disclosure;
- deterministic archive generation if zipped;
- SHA-256 verification on remote resolution.

Fresh-Colab tutorial sequence:

1. install locked environment;
2. select sample or BYOD;
3. validate schema/frequency;
4. inspect/plot history;
5. configure forecast;
6. resolve and verify pinned Chronos-2 model;
7. run forecast;
8. visualize median + intervals;
9. optionally evaluate held-out future + baseline;
10. export forecast + complete provenance.

---

## Runtime / serving requirements

- `device=auto|cpu|cuda` where supported;
- no hard CUDA requirement in tests;
- record actual device/dtype;
- warm up before latency measurement;
- report latency with IDs, targets/covariates, context, horizon, batch size, device;
- explicit OOM/resource guards;
- reusable loader/inference path outside notebooks;
- no raw user-series logging by default.

---

## Security requirements

- immutable official model revision only;
- `safetensors` only in standard model path;
- revision + config + weight integrity checks;
- safe archive extraction;
- file type/size limits;
- no arbitrary code execution from datasets;
- preserve provenance in exports.

---

## Required tests

### Contract / unit tests

- valid univariate/multi-ID/multi-target schemas;
- duplicate timestamps rejected;
- irregular and mixed frequency rejected **even with explicit `frequency`**;
- `<3`-point series rejected;
- bad/out-of-grid quantiles rejected;
- future covariate subset/alignment errors;
- future target leakage rejected;
- `batch_size` resource guard;
- revision/config/weight integrity failures;
- deterministic output rename map.

### Discriminating upstream oracles

- multi-target raw result contains `target_name`;
- with q0.5 requested: `raw["predictions"] == raw["0.5"]`;
- an irregular fixture with explicit frequency is rejected by DIMER **before** `predict_df`;
- model resolution proves exact revision and config/weight digests.

### Integration

- real CPU tiny-sample smoke inference;
- GPU smoke optional when available;
- sample resolver/hash path;
- BYOD path;
- chronological evaluation path.

Static notebook JSON validation alone is insufficient.

---

## Repository structure

```text
chronos-2-forecasting-pipeline/
├── README.md
├── MODEL_CARD.md
├── pyproject.toml / requirements lock
├── docs/rfc/0001-chronos-2.md
├── src/chronos2_pipeline/
│   ├── config.py
│   ├── validation.py
│   ├── model.py
│   ├── inference.py
│   ├── evaluation.py
│   └── provenance.py
├── tutorials/
├── examples/sample-data/
├── tests/
└── .github/workflows/ci.yml
```

---

## Implementation phases

### Phase 1 — foundation

- version this RFC in-repo;
- lock environment;
- pinned, verified loader;
- model card;
- DIMER-side validation;
- univariate inference;
- raw/normalized output contract tests;
- real CPU smoke test.

### Phase 2 — full Chronos-2 inference

- multiple series;
- multivariate/multi-target;
- past/future covariates;
- quantiles;
- context/horizon effective-state provenance.

### Phase 3 — DIMER tutorial

- deterministic sample datasets + cards;
- BYOD;
- Colab;
- visualization;
- optional evaluation + naive baseline;
- export.

### Phase 4 — serving readiness

- stable reusable inference API;
- resource limits;
- latency instrumentation;
- packaging contract for later DIMER backend integration.

---

## Acceptance criteria

- [ ] This RFC is copied to `docs/rfc/0001-chronos-2.md` and commit-linked from this issue.
- [ ] HF revision `95a9710e2596287d08352589f42634fa5abdf0a7` is verified.
- [ ] Weight SHA-256 and byte size above are verified.
- [ ] `config.json` SHA-256 is recorded and verified.
- [ ] Apache-2.0 model provenance is recorded.
- [ ] Complete runtime environment is locked reproducibly.
- [ ] Loader rejects mutable/unapproved model sources in standard tutorial path.
- [ ] DIMER frequency/gap validation cannot be bypassed with explicit `frequency`.
- [ ] Future-covariate subset and leakage rules are tested.
- [ ] Out-of-grid quantiles fail clearly.
- [ ] Raw multi-target output preserves `target_name`.
- [ ] Normalized DIMER schema is deterministic.
- [ ] q0.5/`predictions` median oracle passes.
- [ ] Requested/effective context and horizon are recorded.
- [ ] Autoregressive unrolling is explicitly flagged.
- [ ] CPU smoke inference passes.
- [ ] Bundled sample + BYOD Colab runs top-to-bottom.
- [ ] Evaluation is chronological and leakage-safe.
- [ ] Sample artifacts are deterministic and hash-verified.
- [ ] `MODEL_CARD.md` documents capabilities, limits, upstream references, training/benchmark-overlap caveat, and supply-chain pins.
- [ ] CI exercises runtime/sample-resolution paths rather than static notebook structure only.

## Integrator disposition

**Architecture approved. Builder may start only after the final reviewer confirms this amended RFC contract.**
