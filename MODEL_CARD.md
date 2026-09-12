---
license: apache-2.0
model_card_spec: "1.1"
pipeline_tag: time-series-forecasting
tags:
  - time-series-forecasting
  - time-series-foundation-model
  - zero-shot
base_model: amazon/chronos-2
---

# Chronos-2 (v1.0) — Time-Series Foundation Model (Zero-Shot Forecasting)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-amazon%2Fchronos--2-ffcc4d?style=flat)](https://huggingface.co/amazon/chronos-2)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-amazon--science%2Fchronos--forecasting-181717?style=flat&logo=github&logoColor=white)](https://github.com/amazon-science/chronos-forecasting)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2510.15821-b31b1b.svg)](https://arxiv.org/abs/2510.15821)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Pipeline](https://img.shields.io/badge/Pipeline-chronos--2--forecasting--pipeline-2ea44f?style=flat&logo=github)](https://github.com/kurtvalcorza/chronos-2-forecasting-pipeline)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This pipeline provides a ready-to-run interactive Google Colab notebook that exercises the repository's public API end to end — bootstrap a fresh runtime, resolve and verify the pinned upstream revision, validate an input, run the task, and inspect and export the outputs:

- **Task Inference Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/chronos-2-forecasting-pipeline/blob/main/tutorials/chronos_2_forecasting_colab.ipynb) [`chronos_2_forecasting_colab.ipynb`](https://github.com/kurtvalcorza/chronos-2-forecasting-pipeline/blob/main/tutorials/chronos_2_forecasting_colab.ipynb)  
  *Validate bundled or your own CSV series, build a leakage-safe chronological holdout, verify the pinned `amazon/chronos-2` checkpoint, run univariate and multi-target zero-shot forecasts, and read median/quantile outputs against baselines; no training occurs.*

---

###### Description

Chronos-2 is a pretrained time-series foundation model from Amazon (`amazon/chronos-2`, pinned revision `95a9710e2596287d08352589f42634fa5abdf0a7`) packaged by this repository for zero-shot probabilistic forecasting. The pinned checkpoint uses the T5-family Chronos-2 architecture (`model_type: t5`, ~119.5M parameters), with 16-timestep input/output patches, stride 16, hidden dimension 768, and a fixed 21-level trained quantile grid. Inference is in-context only: this repository performs no parameter training or fine-tuning.

The DIMER wrapper adds immutable checkpoint selection, file-integrity verification, rejection of pickle-style weights, request validation, normalized output semantics, row-layout oracles, chronological evaluation helpers, naive baselines, and runtime/inference provenance. This packaging layer is a tutorial/developer-preview boundary; it is not a claim of production-serving stability.

#### Intended Use and Limitations

###### Primary Intended Uses

The primary machine-learning capability is zero-shot probabilistic time-series forecasting across four supported request modes: univariate, multiple independent series, multi-target, and numeric covariate-informed forecasting. The pipeline accepts regular contiguous tabular histories and optional future covariate tables and emits median point forecasts plus requested quantiles selected from the model's fixed 21-level trained grid (0.01 through 0.99). **The emitted quantiles are model quantiles and are not claimed to be calibrated prediction or confidence intervals for a new domain.**

Representative application areas include demand, telemetry, environmental, operations, and other regularly sampled numerical time series. The pipeline is suitable as a zero-shot baseline or forecasting component when users can validate performance on representative chronological holdouts before relying on forecasts.

###### Primary Intended Users

Primary users are ML engineers, time-series data scientists, quantitative analysts, researchers, and application developers. The envisioned deployment setting is internal enterprise or research use through the DIMER platform in a tutorial/developer-preview boundary; the card does not claim a stable production-serving contract. Users are assumed to understand temporal holdout design, the difference between a point forecast and a quantile grid, the fixed-width frequency constraint the pipeline enforces, and the difference between a predictive covariate and a causal intervention. A user who would read the `q0.1`/`q0.9` columns as a guaranteed interval, or who cannot run a chronological backtest on their own history, is outside the assumed competency.

###### Out-of-scope use cases

1. **Capability boundaries:** classification, anomaly detection, missing-value imputation, representation learning, model pretraining, and model fine-tuning are outside this repository's public forecasting path.
2. **Input boundaries:** fixed-width regular frequencies are required. Calendar frequencies such as month-end/year-end/business-day schedules, irregular or gappy series, missing target values, non-numeric covariates, and histories shorter than the validator minimum are rejected.
3. **Resource boundaries:** model context is capped at 8,192 timesteps; native prediction length is 1,024. Requests above 1,024 require explicit `allow_unroll=True` and remain bounded by the DIMER 4,096-step request guard.
4. **Decision boundaries:** autonomous high-consequence decisions without independent domain validation, monitoring, and human review are out of scope.

---

#### Factors

###### Groups

Chronos-2 operates on numerical time-series rather than human demographic attributes. Deployments using data about or affecting people can nevertheless inherit structural bias from the deployment data or upstream pretraining mix. Such deployments require independent subgroup, fairness, and disparate-impact assessment appropriate to the domain.

###### Instrumentation

Forecast quality depends on upstream measurement and logging systems. Sampling precision, clock synchronization, calibration drift, schema changes, missing events, and latency can alter the effective data-generating process. The wrapper rejects several detectable structural problems—such as duplicate timestamps, gaps, irregular intervals, null targets, and unsupported covariates—but cannot diagnose all instrumentation drift.

###### Environment

The repository targets Python `>=3.12,<3.13`, with a fully locked runtime exported through `uv.lock` and `requirements.lock.txt`. CPU is the tutorial default; CUDA is supported by the library. Runtime provenance records the actual Python/package versions, device, and dtype used for a forecast.

---

#### Metrics

###### Performance Measures

The evaluation helpers currently implemented by this repository are:

- **MAE** for point forecasts;
- **RMSE** for point forecasts;
- **per-quantile pinball loss** for requested forecast quantiles;
- **empirical interval coverage** when at least two requested quantiles define an outer interval;
- **last-value** naive baseline; and
- **seasonal-naive** baseline when the caller supplies a valid season length.

Evaluation returns per-series and pooled aggregate point metrics. Aggregate MAE/RMSE are scale-dependent and can be dominated by larger-magnitude series or targets, so heterogeneous workloads should also inspect per-series results and add domain-appropriate scale-aware metrics. CRPS, MASE, and WAPE are **not** currently implemented by this repository's evaluator; callers may compute them downstream when mathematically and operationally appropriate.

Tutorial metrics use deterministic synthetic data and a single chronological tail holdout. They are sanity/evidence-of-execution metrics, not benchmark or production-fitness claims.

###### Decision thresholds

The normalized `prediction` field is required to equal the median (`q0.5`) for the pinned upstream version; it is not labeled as a statistical mean. Quantile requests must belong to the trained 21-level grid and are refused rather than silently clamped or substituted. Forecast horizons above the native 1,024-step capacity require explicit autoregressive-unroll opt-in and remain subject to the DIMER request ceiling.

Operational alert or action thresholds are domain-specific and are not supplied by this repository.

###### Approaches to uncertainty and variability

Predictive uncertainty is represented by requested model quantiles from the fixed trained grid. These are **uncalibrated model quantiles** unless a downstream user demonstrates empirical calibration on representative held-out data. Users requiring guaranteed empirical coverage or conformal validity must perform their own calibration/backtesting.

Inference is deterministic on the standard CPU path for fixed inputs and the locked runtime, while floating-point details and latency can still vary across hardware/runtime substrates.

---

#### Ethical considerations and biases

###### Data

Upstream Chronos-2 was pretrained by Amazon on a mixture of synthetic and real-world time-series data described by the upstream project. The full pretraining mixture cannot be independently reconstructed from this repository. Model weights are fetched from Hugging Face and are not committed here. Users remain responsible for lawful and appropriate handling of inference data, including personal, confidential, proprietary, or regulated data.

###### Human Life

The pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, housing, autonomous-vehicle control, or emergency dispatch, and this repository does not certify Chronos-2 for any of them. Validation performed here is limited to the chronological forecasting helpers and the tutorial's CI execution against the pinned weights; no clinical, regulatory, or independent domain validation has been carried out by the developers or by any external body. Where a sensitive-domain deployment is foreseeable — capacity forecasting that gates staffing in a hospital, for example — it would be admissible only with independent domain validation on that operator's own history, a human decision-maker between the forecast and the action, monitoring for drift, and any regulatory clearance the domain requires.

###### Mitigations

Implemented mitigations include:

1. **Supply-chain integrity:** immutable revision pinning, SHA-256 verification of `config.json` and `model.safetensors`, model byte-size validation, refusal of arbitrary/mutable model sources, and rejection of pickle-style weight formats.
2. **Input validation:** explicit checks for timestamps, regularity, gaps, null/non-finite targets, numeric covariates, future-table semantics, and resource ceilings before upstream inference.
3. **Alignment oracles:** verification of one row per `(series, target, step)` in the reviewed upstream layout and an assertion that `prediction == q0.5` when q0.5 is requested.
4. **Reproducibility:** locked dependencies plus structured model/runtime/inference provenance on every forecast.
5. **Evaluation discipline:** chronological holdout helpers and history-only naive baselines; no public random-split helper for tutorial forecasting evaluation.

###### Risks and harms

Model-intrinsic risks: forecasts outside the pretraining distribution — regime changes, structural breaks, series shorter than the model has effectively seen — carry no signal of their own unreliability, so overconfidence is borne by the operator and by whoever the operator's decision affects; the quantile grid is not calibrated in a user's domain, and a reader who treats `q0.1`–`q0.9` as a guaranteed 80 % interval will under-provision at a rate the pipeline does not measure; long-horizon unrolling past the native 1,024 steps compounds error step by step, which is why it is opt-in. Use-context risks: automation bias, where a numerically precise forecast displaces the operator's own judgement; sensor or instrument drift that reaches the model as a level shift in the target and is forecast forward as if real; covariates read causally, so that a knob the operator can turn is assumed to move the target; and undetected leakage where a "known-future" covariate is actually derived from the future target. Likelihood under normal use is highest for the calibration and automation-bias risks because they need no fault to occur; magnitude scales with what the forecast gates, from a misallocated inventory order to a mis-staffed shift.

###### Use cases

Distinct from the capability and decision boundaries above, the developers consider the following uses prohibited even where the model would produce a plausible forecast: surveillance or activity profiling of individuals from telemetry that traces to a person; forecasting a person's behaviour, creditworthiness, employment, or housing outcome as an input to a decision about that person; autonomous control of physical systems where a forecast error can injure someone; deceptive presentation of a model quantile as a certified prediction interval; and any use that violates the Apache-2.0 terms of the upstream `amazon/chronos-2` weights, the `chronos-forecasting` package licence, or the terms of the DIMER deployment. The repository is not authorization for any of these, and organisational controls remain the operator's obligation.

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

The immutable revision SHA is the primary model identity. The loader additionally verifies the configuration and weight digests and expected byte size. Mutable refs such as `main`/`latest`, arbitrary local paths, and unapproved sources are rejected in the standard path.

The pinned Hugging Face repository does not contain a separate `LICENSE` file; Apache-2.0 is declared in repository/model-card metadata, and exported provenance records that basis explicitly.

## Forecast modes

### Mode A — univariate

One target per series. This is the tutorial's main evaluated path.

### Mode B — multiple independent series

Many series IDs can be forecast in one request while preserving series identity in normalized output.

### Mode C — multi-target

A request can contain several target columns. The normalized output preserves `target_name`, and the pipeline asserts the upstream row layout so values cannot silently move to a neighbouring series or target. **The release tutorial executes this primary capability by default.**

### Mode D — covariate-informed

The pipeline distinguishes numeric **past-only covariates** from **known-future covariates** supplied across the horizon. Both sets are recorded separately in provenance. Future target columns are refused as leakage. The release tutorial provides an optional executable Mode D path.

## Covariate policy

v1 accepts numeric covariates only. Numeric strings that coerce without loss are accepted; booleans are normalized to 0/1. Categorical/string and temporal payload columns are refused unless users explicitly derive stable numeric features. Covariates are predictive inputs, not evidence of causal effect.

## Quantile semantics

Chronos-2 was trained on the fixed grid:

```text
0.01 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50
0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 0.99
```

The pipeline rejects levels outside this trained grid rather than allowing an upstream substitution under the requested label. Requested and effective quantiles therefore remain self-describing. The default tutorial requests `[0.1, 0.5, 0.9]`; those outputs must not be interpreted as guaranteed calibrated intervals without downstream evidence.

## Point-forecast semantics

The normalized `prediction` column is the **median / q0.5**, not a statistical mean. For `chronos-forecasting==2.3.1`, the wrapper asserts exact equality between upstream `predictions` and the 0.5 quantile when q0.5 is requested. An upstream semantic change therefore fails loudly.

## Context and horizon

The pinned model exposes an 8,192-timestep context and native 1,024-timestep prediction length. Requests beyond the native horizon are rejected unless `allow_unroll=True`; permitted unrolling remains bounded by DIMER limits, and provenance records requested/effective/model horizon plus whether unrolling occurred.

## Frequency and gaps

DIMER validates observed and declared frequency before `predict_df`. v1 supports fixed-width intervals such as `15min`, `h`, `D`, and `7D`. Calendar-dependent month/quarter/year/business-day offsets remain outside the current contract. Gaps and missing target values are rejected rather than silently interpolated.

## Output contract

Normalized forecasts use:

```text
series_id
timestamp
target_name
prediction
q<level>...
```

The upstream-to-normalized mapping is deterministic, and an output-layout oracle verifies one row per `(series, target, forecast step)`. Provenance records model pins, runtime versions, device/dtype, context/horizon semantics, quantiles, target/covariate counts, past/known-future covariates, batch/cross-learning state, latency metadata, and observed frequency.

## Evaluation

Evaluation is chronological only. Implemented helpers provide chronological tail holdout, MAE, RMSE, quantile pinball loss, empirical outer-interval coverage, last-value baseline, explicit seasonal-naive baseline, and per-series/aggregate outputs. Forecast and truth must cover the same normalized key rows or evaluation fails rather than silently dropping mismatches.

The tutorial runs seasonal-naive comparison only when its 24-step hourly season is structurally valid and sufficient history remains; otherwise it skips that optional comparator. For real data, use representative multi-origin chronological backtesting and domain-relevant baselines.

## Live tutorial

The v1 tutorial is [`tutorials/chronos_2_forecasting_colab.ipynb`](tutorials/chronos_2_forecasting_colab.ipynb), declared as `TASK-INFERENCE` under DIMER Notebook Specification 1.0. It covers locked-runtime bootstrap, deterministic sample/BYOD ingestion, duplicate-header defense, chronological holdout, immutable model verification, univariate forecasting/evaluation, primary Mode C multi-target forecasting, optional Mode D known-future covariates, machine-readable exports, and provenance.

CI executes the notebook code path against real pinned weights on `main`/manual dispatch, including a clean-directory bootstrap path plus BYOD-shaped and Mode D branches. Hosted Colab remains an external substrate; release-grade Colab claims require a recorded clean-Colab execution for the exact release revision.

## Intended use

Appropriate use requires data satisfying the explicit input contract and domain-specific chronological validation before forecasts inform operations. Quantiles and covariates must be interpreted according to the semantics above.

## Limitations

- No training/fine-tuning workflow.
- Fixed-width frequency support only.
- Missing targets and gaps are rejected.
- Categorical/string covariates are not supported by the public path.
- Quantiles are not guaranteed calibrated in a user's domain.
- Long-horizon unrolling requires explicit opt-in and can compound error.
- `cross_learning=true` is not a beginner default and can make outputs depend on request composition.
- The current boundary is tutorial/developer preview; stable production-serving APIs/SLOs/backend packaging are separate follow-on work.

## License

Pipeline code is Apache-2.0 licensed. The pinned Chronos-2 model-repository metadata declares Apache-2.0; wrapper and model provenance are recorded separately.

## References

- Model: `amazon/chronos-2` at `95a9710e2596287d08352589f42634fa5abdf0a7`
- Upstream implementation: `amazon-science/chronos-forecasting`, release `v2.3.1`
- DIMER contract: [`docs/rfc/0001-chronos-2.md`](docs/rfc/0001-chronos-2.md)
- Release-completion tracking: issue #5
