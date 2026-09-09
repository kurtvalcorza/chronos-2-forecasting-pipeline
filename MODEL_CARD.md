---
license: apache-2.0
model_card_spec: "1.0"
pipeline_tag: time-series-forecasting
tags:
  - time-series-forecasting
  - time-series-foundation-model
  - zero-shot
base_model: amazon/chronos-2
---

# Chronos-2 (v1.0)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-amazon%2Fchronos--2-ffcc4d?style=flat)](https://huggingface.co/amazon/chronos-2)
[![GitHub](https://img.shields.io/badge/GitHub-amazon--science%2Fchronos--forecasting-181717?style=flat&logo=github&logoColor=white)](https://github.com/amazon-science/chronos-forecasting)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

###### Description

Chronos-2 is a pretrained time-series foundation model from Amazon (`amazon/chronos-2`, pinned revision `95a9710e2596287d08352589f42634fa5abdf0a7`), packaged by this repository for zero-shot probabilistic forecasting. Built on the T5 encoder-decoder architecture (`model_type: t5`, ~119.5M parameters), it processes univariate or multi-target historical observations as patched tokens (16-timestep patches at stride 16, hidden dimension 768) and directly generates predictive distributions across a fixed 21-level quantile grid. Output generation operates purely via in-context conditioning without parameter fine-tuning or gradient steps. This repository establishes an immutable, production-grade DIMER packaging layer around the upstream checkpoint: verified `model.safetensors` loading, strict SHA-256 digest and byte-size supply chain gating, total rejection of unsafe pickle checkpoints, rigorous input validation (diff-based regularity, fixed-width frequency enforcement, non-numeric covariate refusals), an output row-layout oracle, and tamper-evident provenance generation.

#### Intended Use and Limitations

###### Primary Intended Uses

The primary machine learning task is zero-shot, univariate, multi-target, and covariate-informed probabilistic time-series forecasting. The pipeline ingests regular, contiguous tabular historical time series (Pandas DataFrame) and optional future covariate tables, and emits structured forecasts containing median point predictions and 21 calibrated quantiles (from 0.01 to 0.99) spanning up to 1,024 future steps. Concrete application domains include electric grid load prediction, retail inventory demand planning, data center server telemetry forecasting, financial market indicators, macroeconomic time series, and environmental sensor monitoring. Within larger enterprise systems, the pipeline functions as an ultra-reliable, zero-configuration baseline, a standalone microservice, or an automated forecasting engine in the DIMER platform registry requiring zero training overhead.

###### Primary Intended Users

Primary intended users are machine learning engineers, time-series data scientists, quantitative analysts, and infrastructure automation developers deploying forecasting models in enterprise, research, or mission-critical data environments. Users are assumed to possess competencies in time-series data preparation—specifically understanding the critical distinctions between point forecasts (median vs. mean), recognizing that statistical covariates do not imply causal intervention, acknowledging the implications of patch-based tokenization, and understanding regular interval alignment versus calendar-based offset schedules.

###### Out-of-scope use cases

1. **Capability boundaries:** Chronos-2 is strictly a forecasting model. It cannot perform time-series classification, missing-value imputation, or representation learning (for anomaly detection or imputation, use the companion `moment-pipeline`). Pretraining and fine-tuning are not supported.
2. **Input boundaries:** Inputs must have fixed-width frequencies (`15min`, `h`, `D`, `7D`). Calendar frequencies (`ME`, `YE`, `W`, `B`), irregular time steps, missing target values, non-numeric or categorical covariates, and series shorter than 3 observations are strictly rejected with explicit errors. Context is capped at 8,192 steps; prediction length is capped natively at 1,024 steps, and requests exceeding 1,024 require explicit `allow_unroll=True` up to a hard system limit of 4,096 steps.
3. **Decision boundaries:** Autonomous, unmonitored decision-making in high-consequence environments—such as automated high-frequency financial trading, autonomous clinical dosing, or critical utility shutdowns without human operator review—is strictly out of scope.

---

#### Factors

###### Groups

Chronos-2 is an abstract numerical sequence model and is not human-centric in architecture or design. Its inputs are continuous floating-point arrays and timestamps representing machine, physical, or business metrics. However, because upstream pretraining utilized diverse public and synthetic time-series corpora whose individual datasets were not demographically audited by Amazon, the model may reflect structural biases present in underlying societal or economic data. When operators deploy this pipeline on data derived from or impacting human populations (e.g., individual healthcare vitals, household credit scoring, or smart utility billing), the operator is obligated to perform independent fairness, subgroup performance parity, and disparate-impact audits across relevant demographic categories.

###### Instrumentation

Data consumed by Chronos-2 originates from diverse upstream instrumentation and logging systems, including IoT telemetry sensors, smart utility meters, transactional relational databases, API event streams, and environmental weather stations. Key instrumentation characteristics that materially alter data quality include sampling precision, clock synchronization, analog-to-digital converter resolution, and network latency. The pipeline defensively detects and rejects non-monotonic timestamps, duplicate entries, null values, and sampling irregularities. However, upstream instrumentation drift, sensor calibration decay, or silent schema changes propagate directly into the forecast horizon as structural prediction errors that the model cannot internally diagnose.

###### Environment

1. **Operating environment:** Execution requires Python 3.12, PyTorch >=2.1.2, and `transformers`. The pipeline operates fully deterministically on CPU using ~478 MB of memory for model weights. Half-precision (`float16`, `bfloat16`) is refused on CPU to prevent numerical corruption, defaulting to standard `float32`. GPU execution is supported via `device="cuda"`.
2. **Data environment:** The pipeline mathematically assumes regular, contiguous, stationary or smoothly evolving time series within the 8,192-step context window. Accuracy degrades sharply under sudden regime shifts, black-swan macroeconomic disruptions, catastrophic physical sensor failures, or unobserved external shocks that violate historical context patterns.

---

#### Metrics

###### Performance Measures

Standard performance measures evaluated in this pipeline include Continuous Ranked Probability Score (CRPS) across the full predictive distribution, Mean Absolute Scaled Error (MASE) relative to a naive persistence baseline, and Weighted Absolute Percentage Error (WAPE). CRPS measures probabilistic sharpness and calibration without imposing distributional shape assumptions; MASE provides scale-independent benchmarking across heterogeneous series; WAPE summarizes total aggregate volume error. In Phase 1 and Phase 2 zero-shot inference, raw quantile arrays and median predictions are emitted; downstream task-specific business loss metrics (e.g., inventory stockout penalty) must be computed by the caller.

###### Decision thresholds

The pipeline establishes a strict default point forecast rule: the emitted `prediction` field is mathematically identical to the median quantile `q0.5` (`raw["predictions"] == raw["0.5"]`), explicitly rejecting the upstream labeling of the median as a mean. Decision thresholds for forecast horizons enforce a strict native ceiling of 1,024 timesteps, requiring `allow_unroll=True` for horizons between 1,025 and 4,096, and refusing requests above 4,096 (`PREDICTION_LENGTH_LIMIT`). Quantiles requested outside the trained 21-level grid are refused outright rather than clamped. Alerting or operational intervention thresholds must be calibrated by the downstream operator according to asymmetric domain costs (e.g., the differential cost of over-generating power versus experiencing a brownout).

###### Approaches to uncertainty and variability

Predictive uncertainty is quantified non-parametrically across the fixed 21-level quantile grid (`0.01` through `0.99`) derived directly from the trained predictive head. Inference is completely deterministic on CPU under standard runtime execution, with zero stochastic sampling or Monte Carlo simulation variability. Quantiles represent the model's uncalibrated predictive distribution conditioned on context. Downstream operators requiring guaranteed empirical coverage or conformal validity must apply conformal prediction intervals or empirical calibration against their own held-out historical validation windows.

---

#### Ethical considerations and biases

###### Data

Chronos-2 was pretrained by Amazon on an extensive combination of synthetic time series generated via Gaussian processes and diverse public datasets from the Monash Time Series Repository. Amazon's official documentation does not fully enumerate every proprietary dataset utilized in pretraining, and the exact data mix cannot be audited by inspection. The repository distributes code, pipeline interfaces, and tests; model weights are fetched from Hugging Face Hub and cached locally, never committed to version control. Downstream operators are strictly obligated to audit their own inference data for proprietary, personally identifiable, or sensitive demographic attributes before submitting payloads to the pipeline.

###### Human Life

Chronos-2 is not designed, certified, or intended for use in life-critical decision-making or domains affecting human life, physical safety, or fundamental liberties. This includes medical diagnostics, patient telemetry monitoring, surgical guidance, criminal justice sentencing, autonomous vehicle navigation, emergency dispatch, or life-support infrastructure. No clinical or safety certifications exist for this pipeline. Deployment in any sensitive domain is admissible only when subject to rigorous external clinical or domain validation, fail-safe redundant hardware controls, and mandatory human expert oversight.

###### Mitigations

Implemented mitigations include:
1. **Supply-chain integrity:** Enforcing pinned immutable revision `95a9710e…`, SHA-256 verification of `config.json` (`ef1143bf…`) and `model.safetensors` (`ddcda3c7…`), byte-size validation (`477,930,472`), rejection of unapproved URI schemes or mutable refs, and absolute rejection of pickle-format files (`.bin`, `.pt`, `.pkl`).
2. **Input validation:** Rejection of irregular timestamps via pairwise diff equality, fixed-width frequency validation (`CALENDAR_FREQUENCY_UNSUPPORTED`), NaN/null rejection, non-numeric covariate refusal (`COVARIATE_NOT_NUMERIC`), and future-table covariate consistency checks (`FUTURE_TABLE_HAS_NO_COVARIATES`).
3. **Alignment oracles:** Verifying exact row layout `(series, target, step)` alignment and asserting that point predictions match `q0.5`.
4. **Reproducibility:** Locking the complete dependency tree in `uv.lock`, executing deterministic inference, and exporting detailed runtime provenance with every forecast.

###### Risks and harms

Key operational risks include:
1. **Automation bias:** Downstream operators accepting uncalibrated quantile forecasts as absolute certainty, leading to catastrophic inventory misallocation or financial loss.
2. **Distributional shift:** Significant forecast degradation during abrupt macroeconomic or physical regime shifts where historical context no longer represents future dynamics.
3. **Covariate fallacy:** Operators inferring causal relationships from predictive covariates, incorrectly assuming that manipulating a covariate will predictably alter the forecast.
4. **Autoregressive error compounding:** When horizons exceed native capacity (>1,024 steps), unrolled errors compound exponentially, presenting significant drift risks.

###### Use cases

Prohibited use cases include:
1. Deceptive, manipulative, or predatory applications, including algorithmic market manipulation, predatory subprime lending, or dynamic price gouging during emergencies.
2. Mass surveillance, individual behavioral tracking, or predictive social scoring.
3. Automated lethal or weaponized systems, or hazardous physical infrastructure control without human-in-the-loop overrides.
4. Any deployment violating the upstream Apache-2.0 license or regional regulatory frameworks (such as the EU Artificial Intelligence Act).

---

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