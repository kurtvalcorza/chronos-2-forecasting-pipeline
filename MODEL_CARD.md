---
license: apache-2.0
pipeline_tag: time-series-forecasting
tags:
  - time-series-forecasting
  - time-series-foundation-model
  - zero-shot
base_model: amazon/chronos-2
---

# Chronos-2 — DIMER profile

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-amazon%2Fchronos--2-ffcc4d?style=flat)](https://huggingface.co/amazon/chronos-2)
[![Upstream](https://img.shields.io/badge/Upstream-amazon%2Fchronos--forecasting-181717?style=flat&logo=github&logoColor=white)](https://github.com/amazon-science/chronos-forecasting)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Summary

Chronos-2 is Amazon's pretrained time-series forecasting foundation model. This DIMER pipeline exposes **zero-shot probabilistic forecasting** only: no task-specific training or fine-tuning occurs in the standard path.

The v1 tutorial/developer-preview capability set supports:

- one or many independent series;
- one or multiple target columns;
- numeric past-only covariates;
- numeric known-future covariates;
- probabilistic quantile forecasts;
- chronological holdout evaluation and naive baselines;
- deterministic sample and BYOD tutorial workflows;
- forecast and runtime/model provenance export.

It is a forecasting specialist. This repository does not present Chronos-2 as a classifier, anomaly detector, imputer, or generic representation model.

## Model details

| Field | Value |
|---|---|
| Model identifier | `amazon/chronos-2` |
| Developer | Amazon |
| Family | Chronos |
| Task | zero-shot probabilistic time-series forecasting |
| Architecture | T5-family Chronos-2 patch-based model |
| Approx. parameters | ~119.5M |
| Input patch size / stride | 16 / 16 |
| Output patch size | 16 |
| Maximum native output patches | 64 |
| Model context length | 8,192 timesteps |
| Native prediction length | 1,024 timesteps |
| Weight format | `model.safetensors` |
| Model-repository license metadata | Apache-2.0 |

## Checkpoint and runtime provenance

The public loader accepts only the approved model identity and immutable revision.

| Item | Pin / assertion |
|---|---|
| Model | `amazon/chronos-2` |
| Hugging Face revision | `95a9710e2596287d08352589f42634fa5abdf0a7` |
| `model.safetensors` SHA-256 | `ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42` |
| `model.safetensors` size | 477,930,472 bytes |
| `config.json` SHA-256 | `ef1143bfdc9c0376d9a056eefca46cb4b1ec3d0ffacd541ff56feb40fb708031` |
| Upstream runtime package | `chronos-forecasting==2.3.1` |
| Reviewed upstream tag / commit | `v2.3.1` / `7dc4435706a4454feb79df44ca9f33631f3027bf` |
| Full environment | `uv.lock` + `requirements.lock.txt` |
| Python | `>=3.12,<3.13` |

The revision SHA is the primary repository identity. The loader additionally verifies the model and configuration file digests and the expected model byte size. Mutable refs such as `main` or `latest`, arbitrary local paths, and unapproved sources are rejected in the standard path.

The Hugging Face model repository at the pinned revision does **not** contain a separate `LICENSE` file. The Apache-2.0 model-license statement therefore rests on the model-card/repository metadata at that immutable revision; exported provenance records that basis rather than implying a file that is not present.

## Forecast modes

### Mode A — univariate

One target per series. This is the default live tutorial path.

### Mode B — multiple independent series

Many series IDs can be forecast in one request while retaining independent series identity in the output.

### Mode C — multi-target

A request can contain several target columns. The normalized output preserves `target_name`, and the pipeline asserts the upstream row layout so values cannot silently shift to a neighbouring series or target if upstream ordering changes.

### Mode D — covariate-informed

The pipeline distinguishes:

- **past-only covariates** — available through the forecast origin;
- **known-future covariates** — supplied across the requested horizon.

Both sets are recorded separately in provenance. Future targets are refused as leakage.

## Covariate policy

v1 accepts **numeric covariates only**. Numeric strings that coerce without data loss are accepted; booleans are normalized to numeric 0/1. Categorical/string and temporal payload columns are refused unless the user explicitly derives a stable numeric feature.

This is deliberate. In the reviewed upstream path, categorical covariate encoding can depend on unrelated request structure such as the number of targets. DIMER refuses a representation whose meaning could change across otherwise similar exports.

Covariates are predictive inputs, not evidence of causal effect.

## Quantile semantics

Chronos-2 was trained on the fixed grid:

```text
0.01 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50
0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 0.99
```

The pipeline rejects requested levels outside this trained grid rather than allowing upstream to substitute a nearby quantile under the requested column name. Requested and effective quantiles therefore remain self-describing in exported results.

The default tutorial requests `[0.1, 0.5, 0.9]`.

## Point-forecast semantics

The normalized `prediction` column is the **median / q0.5**, not a statistical mean.

For `chronos-forecasting==2.3.1`, upstream `predictions` is populated from the 0.5 quantile. Every request that includes q0.5 asserts this equality before export. A future upstream change causes an `UpstreamContractError` rather than silently changing the meaning of the field.

## Context and horizon

The pinned model exposes:

- model context length: 8,192 timesteps;
- native prediction length: 1,024 timesteps.

Requests beyond the native prediction length are rejected by default. `allow_unroll=True` explicitly permits upstream autoregressive unrolling within DIMER resource limits, and provenance records `autoregressive_unrolled: true` together with requested/effective/model context and horizon values.

The default live tutorial stays within native limits.

## Frequency and gaps

DIMER validates frequency **before** `predict_df`, including when the caller explicitly supplies a frequency. Upstream's `freq` argument is not treated as an independent validator because it can be used as-is to construct a horizon.

v1 supports fixed-width intervals such as:

- `15min`
- `h`
- `D`
- `7D`

Calendar-dependent offsets such as month-end, quarter-end, year-end, and business-day aliases remain outside the current v1 contract. Regular calendar data is rejected explicitly rather than mislabeled as ordinary irregularity. This is deferred serving/capability work, not unfinished Phase-2 inference.

Gaps and missing target values are rejected in v1; the pipeline does not silently interpolate them.

## Output contract

Normalized forecasts use:

```text
series_id
timestamp
target_name
prediction
q<level>...
```

The upstream-to-normalized rename map is deterministic. The pipeline also asserts one row per `(series, target, forecast step)` in the reviewed upstream order so positional changes cannot silently relabel forecasts.

Every forecast travels with structured provenance including model pins, runtime versions, device/dtype, requested/effective context and horizon, quantiles, target/covariate counts, past vs known-future covariate names, batch size, cross-learning state, latency metadata, and observed frequency.

## Evaluation

Phase 3 evaluation is implemented and **chronological only**. The public path includes:

- chronological holdout construction;
- MAE;
- RMSE;
- quantile pinball loss;
- interval coverage when an appropriate lower/upper interval is available;
- last-value baseline;
- explicit seasonal-naive baseline when a valid season length is supplied;
- per-series and aggregate results.

The helper does not expose a random-split evaluation path. Baselines are scored on the identical holdout horizon.

### Evaluation caveat

Do not treat the deterministic tutorial's metric values as an unbiased estimate of performance on a real application. Tutorial data is synthetic and designed to exercise contracts. For external data, use a genuinely held-out chronological future window and interpret results against domain-relevant baselines.

More generally, benchmark or user datasets may overlap corpora seen during upstream development/pretraining; this repository does not claim to prove absence of such overlap.

## Live tutorial

The v1 tutorial is:

[`tutorials/chronos_2_forecasting_colab.ipynb`](tutorials/chronos_2_forecasting_colab.ipynb)

It covers:

1. reproducible runtime bootstrap;
2. deterministic bundled sample or BYOD;
3. DIMER schema/frequency validation;
4. history inspection/visualization;
5. forecast configuration;
6. pinned model resolution and integrity verification;
7. forecast execution;
8. median + interval visualization;
9. optional chronological evaluation + baselines;
10. forecast CSV and provenance JSON export.

CI executes the notebook's actual code cells against the pinned real weights on `main` and manual workflow dispatch; static notebook JSON validation alone is not considered live-readiness evidence.

## Intended use

Appropriate uses include exploratory and operational zero-shot forecasting where:

- the input satisfies the explicit regular fixed-width time-series contract;
- probabilistic quantiles are useful;
- users can evaluate performance on their own chronological holdout before relying on forecasts;
- numeric covariates are predictive inputs rather than causal claims.

## Limitations

- No fine-tuning/pretraining workflow in v1.
- Fixed-width frequency support only; calendar offsets are deferred.
- Missing targets/gaps are rejected rather than interpolated.
- Categorical/string covariates are not encoded by the public path.
- Quantiles are not guaranteed calibrated for a user's domain.
- Horizons beyond the native model limit require explicit autoregressive-unroll opt-in.
- `cross_learning=true` is not a beginner/tutorial default and makes results depend on batch composition.
- The release boundary is a tutorial/developer preview; stable production-serving request limits, SLO/latency instrumentation, and DIMER backend packaging remain follow-on work.

## License

Pipeline code in this repository is Apache-2.0 licensed. The Chronos-2 model-repository metadata at the pinned revision declares Apache-2.0; model and wrapper provenance are recorded separately.

## References

- Model: `amazon/chronos-2` at `95a9710e2596287d08352589f42634fa5abdf0a7`
- Upstream implementation: `amazon-science/chronos-forecasting`, release `v2.3.1`
- DIMER contract: [`docs/rfc/0001-chronos-2.md`](docs/rfc/0001-chronos-2.md)
- Release-completion tracking: issue #5
