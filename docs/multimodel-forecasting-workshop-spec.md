# DIMER Workshop Specification: Comparing Time-Series Foundation Models

**Status:** Revised proposal  
**Notebook specification:** DIMER `NOTEBOOK_SPEC` **2.1**  
**Notebook profile:** `TASK-INFERENCE`  
**Pedagogical mode:** `WORKSHOP`  
**Proposed filename:** `DIMER_MultiModel_TimeSeries_Forecasting_Workshop.ipynb`  
**Proposed anchor repository:** `kurtvalcorza/chronos-2-forecasting-pipeline`  
**Canonical runtime:** NVIDIA Tesla T4 or equivalent  
**Canonical execution:** standalone, credential-free, top-to-bottom `Run all`

---

## 1. Purpose

This workshop demonstrates how multiple pretrained time-series foundation models solve the **same zero-shot forecasting problem** under a common data and evaluation contract.

The canonical comparison covers:

| Model | DIMER capability | Upstream model | License |
|---|---|---|---|
| **TiRex-2** | Zero-shot probabilistic forecasting | `NX-AI/TiRex-2` @ `05e5b26db52bfb256f1ae1bdf785589850482de3` | Apache-2.0 |
| **Chronos-2** | Zero-shot probabilistic forecasting | `amazon/chronos-2` @ `95a9710e2596287d08352589f42634fa5abdf0a7` | Apache-2.0 |
| **Toto 2.0 2.5B** | Zero-shot multivariate probabilistic forecasting | `Datadog/Toto-2.0-2.5B` @ `51a2812bbe449437c01b79c0e425ed578f335f5b` | Apache-2.0 |

The workshop asks:

> Given the same historical series, forecast horizon, quantiles, baselines, and evaluation periods, how do different time-series foundation models behave?

The notebook MUST NOT present the result as a universal model ranking.

---

## 2. Notebook profile

The notebook SHALL use:

**Profile:** `TASK-INFERENCE`  
**Mode:** `WORKSHOP`

All three models are pretrained zero-shot forecasters.

The canonical workflow MUST NOT introduce artificial training or fine-tuning.

Canonical workflow:

```text
sample acquisition
→ validation
→ chronological split
→ baselines
→ zero-shot validation forecasts
→ compare
→ freeze configuration
→ independent test forecasts
→ evaluate
→ new-data forecast
→ export
```

No deployable adaptation artifact is produced.

---

## 3. Core models

### 3.1 TiRex-2

**Model:** `NX-AI/TiRex-2`  
**Revision:** `05e5b26db52bfb256f1ae1bdf785589850482de3`  
**Runtime:** `tirex-2==0.2.1`  
**Checkpoint:** approximately 381 MB  
**Reference execution:** CPU  
**License:** Apache-2.0

Supported workshop use:

- univariate forecasting;
- multiple target series;
- probabilistic quantiles;
- optional covariates outside the canonical comparison.

---

### 3.2 Chronos-2

**Model:** `amazon/chronos-2`  
**Revision:** `95a9710e2596287d08352589f42634fa5abdf0a7`  
**Runtime:** `chronos-forecasting==2.3.1`  
**Parameters:** approximately 119.5M  
**Checkpoint:** approximately 478 MB  
**License:** Apache-2.0

Supported workshop use:

- zero-shot forecasting;
- multi-series / multi-target forecasting;
- quantile forecasting;
- optional covariates outside the canonical comparison.

---

### 3.3 Toto 2.0

**Model:** `Datadog/Toto-2.0-2.5B`  
**Revision:** `51a2812bbe449437c01b79c0e425ed578f335f5b`  
**Runtime:** `toto-2==2.0.0`  
**Parameters:** approximately 2.5B  
**Checkpoint:** 9,817,176,960 bytes  
**Runtime:** CUDA GPU  
**License:** Apache-2.0

Supported workshop use:

- zero-shot multivariate forecasting;
- probabilistic quantiles.

External covariates are not included in the current DIMER Toto capability.

---

## 4. MOMENT exclusion

`AutonLab/MOMENT-1-base` MUST NOT appear in the canonical forecasting comparison.

The live DIMER MOMENT profile exposes:

- embeddings;
- reconstruction/imputation;
- anomaly scoring; and
- supervised classification adaptation.

Its released base checkpoint does not provide a valid pretrained zero-shot forecasting head.

An optional conceptual section MAY explain:

> Not all time-series foundation models expose the same pretrained task.

MOMENT MUST NOT be included in forecasting metrics or leaderboards.

---

# 5. Canonical workshop dataset

## 5.1 Source

The workshop SHALL reuse the existing sample infrastructure from:

`kurtvalcorza/chronos-2-forecasting-pipeline`

Specifically:

```text
examples/sample-data/
├── DATASET_CARD.md
├── generate_samples.py
├── SHA256SUMS
├── SHA256SUMS.generated
└── chronos_univariate.csv
```

The canonical comparative sample is:

**`chronos_multi_series.csv`**

It is generated deterministically by:

`examples/sample-data/generate_samples.py`

and has canonical SHA-256:

```text
6f12f1475411aaab13a2de860ba50f008bd7185a9266f99147c244db4a5de8b3
```

The generated sample is original synthetic content distributed under the repository's Apache-2.0 license.

No third-party records, personal data, benchmark rows, or proprietary data are included.

---

## 5.2 Dataset structure

`chronos_multi_series.csv` contains:

- **2 series:** `A`, `B`
- **96 observations per series**
- **192 rows total**
- **hourly frequency**
- **one target column**
- deterministic trend and 24-hour seasonality
- different offsets, slopes, amplitudes, and phases between series

Long-form source:

```text
series_id,timestamp,target
A,2026-01-01T00:00:00,...
A,2026-01-01T01:00:00,...
...
B,2026-01-01T00:00:00,...
...
```

The workshop MUST generate this data directly from the pinned formula rather than requiring repository access at runtime.

Generated bytes MUST reproduce the canonical SHA-256 above.

---

## 5.3 Generation formulas

For hourly step `t = 0..95`:

Series A:

```text
target =
120
+ 0.18*t
+ 6*sin(2*pi*t/24)
```

Series B:

```text
target =
155
+ 0.12*t
+ 9*sin(2*pi*(t+5)/24)
```

These series intentionally provide:

- trend;
- daily seasonality;
- different magnitude scales;
- different seasonal amplitudes;
- phase shift.

The dataset is pedagogical and MUST NOT be treated as benchmark evidence.

---

# 6. Common model input

The long-form sample MUST be pivoted into a model-neutral representation:

```text
timestamp,A,B
2026-01-01 00:00,...
2026-01-01 01:00,...
...
```

Internal numerical representation:

```text
shape = (2, 96)
```

where:

- axis 0 = target series;
- axis 1 = chronological timestep.

Every model receives exactly the same float values.

The notebook MUST state:

> The source fixture defines two aligned synthetic series. The comparative workshop presents them as a common two-channel numerical target matrix. This does not imply a causal or physical relationship between the two variables.

---

# 7. Canonical experimental protocol

Use:

| Parameter | Value |
|---|---:|
| Frequency | hourly |
| Number of series | 2 |
| Total observations | 96 |
| Initial context | **48** |
| Validation horizon | **24** |
| Test horizon | **24** |
| Seasonal period | **24** |
| Quantiles | **0.1, 0.5, 0.9** |

Dataset timeline:

```text
0                                                    95
|----------------|----------------|-------------------|
    48 context       24 validation        24 test
```

---

# 8. Validation forecast

For the validation forecast:

```text
context = steps 0–47
truth   = steps 48–71
```

All models receive only steps `0–47`.

Validation target values MUST NOT enter:

- context;
- preprocessing;
- covariates; or
- configuration.

No model selection may use test-period values.

---

# 9. Test forecast

After validation configuration is frozen:

```text
available history = steps 0–71
test truth        = steps 72–95
```

For final testing, the validation observations are now legitimate historical observations.

The test context SHOULD therefore be:

**72 observations**

rather than restricting every forecast origin to the initial 48.

This teaches an important forecasting principle:

> As time advances, observed former "future" values legitimately become history.

The test target values remain hidden until evaluation.

---

# 10. Common-denominator capability contract

The canonical benchmark SHALL use only functionality available across all three models.

### Included

- numerical target histories;
- two aligned series;
- finite values;
- regular sampling;
- common forecast horizon;
- common q0.1/q0.5/q0.9 outputs.

### Excluded

- exogenous covariates;
- missing-value imputation;
- model fine-tuning;
- autoregressive long-horizon unrolling;
- irregular timestamps;
- calendar-frequency special handling;
- model-specific task extensions.

This prevents one model from receiving information unavailable to another.

---

# 11. Common ceilings

For portability across all three models:

### Context

Canonical comparison:

```text
32 <= context_length <= 8192
```

Chronos-2 provides the limiting maximum.

### Horizon

Canonical comparison:

```text
1 <= horizon <= 1024
```

The workshop uses only `24`.

Requests beyond Chronos-2's native 1,024-step prediction length MUST NOT appear in the canonical workflow.

---

# 12. Forecast semantics

The workshop normalizes every model output to:

- `q0.1`
- `q0.5`
- `q0.9`

Point forecast:

```text
prediction = q0.5
```

The notebook MUST assert this equality.

`q0.5` MUST be described as:

> model median

not mean.

The q10–q90 band MUST be described as:

> model quantiles

not a guaranteed 80% confidence interval.

---

# 13. Canonical forecast table

Every model adapter MUST emit:

```text
model
partition
forecast_origin
series_id
timestamp
step
truth
q0.1
q0.5
q0.9
prediction
```

One row corresponds to:

```text
(model, series, forecast timestep)
```

Silent row dropping, target reordering, horizon truncation, or quantile substitution is prohibited.

---

# 14. Data validation

Validation MUST execute before model acquisition.

The notebook MUST check:

- expected source digest;
- `series_id` exists;
- `timestamp` exists;
- `target` exists;
- IDs are non-empty;
- timestamps parse;
- timestamps are unique within each series;
- timestamps increase strictly;
- each series uses identical timestamps;
- frequency is fixed-width hourly;
- every target is numeric;
- all targets are finite;
- no target values are missing;
- all series contain exactly 96 observations;
- context length is valid;
- validation horizon is valid;
- test horizon is valid;
- seasonal period is valid.

The notebook MUST NOT silently:

- interpolate;
- impute;
- resample;
- reorder timestamps;
- drop rows;
- fill NaNs;
- shorten horizons; or
- rename duplicate targets.

---

# 15. Exploratory analysis

Before model inference, the workshop SHOULD show:

1. raw series A and B;
2. normalized overlay;
3. per-series summary table;
4. first differences;
5. 24-hour seasonal pattern;
6. validation and test boundaries.

Participants SHOULD identify:

- trend;
- seasonal period;
- phase difference;
- relative scale.

---

# 16. Baseline A — last value

For each series:

\[
\hat y_{t+h}=y_t
\]

The forecast repeats the most recent observation through the horizon.

This answers:

> Is the foundation model better than assuming the present level continues?

---

# 17. Baseline B — seasonal naive

Season length:

```text
24 hours
```

Forecast:

\[
\hat y_{t+h}=y_{t+h-24}
\]

This is particularly important because the workshop sample was deliberately constructed with daily periodicity.

It answers:

> Does the foundation model improve on repeating yesterday's hourly pattern?

The notebook MUST explain that a sophisticated model can still be unnecessary if a simple seasonal baseline performs similarly or better.

---

# 18. Common point metrics

All forecasts SHALL be scored using one notebook-owned evaluator.

Point prediction:

`q0.5`

Metrics:

### MAE

\[
MAE = \frac{1}{n}\sum |y-\hat y|
\]

### RMSE

\[
RMSE =
\sqrt{
\frac{1}{n}\sum (y-\hat y)^2
}
\]

Report both:

- per series;
- macro-average across series; and
- pooled.

The notebook MUST state that pooled error is scale-dependent.

---

# 19. Baseline-relative skill

Calculate:

\[
Skill_{MAE}
=
1-
\frac{MAE_{model}}
{MAE_{seasonal}}
\]

Interpretation:

| Skill | Meaning |
|---:|---|
| `> 0` | model beats seasonal naive |
| `0` | same error |
| `< 0` | seasonal naive performs better |

Report:

- A skill;
- B skill;
- macro-average skill.

This SHOULD be the principal cross-series comparison metric.

---

# 20. Probabilistic metrics

For q0.1 and q0.9 compute:

### Empirical interval coverage

Fraction of truth values satisfying:

```text
q0.1 <= truth <= q0.9
```

### Mean interval width

\[
mean(q_{0.9}-q_{0.1})
\]

### Pinball loss

Calculate for:

- q0.1;
- q0.5;
- q0.9.

The notebook MUST explain:

> Coverage and interval width must be interpreted together. High coverage alone can be obtained with uninformatively wide intervals.

---

# 21. Runtime isolation

The three models SHOULD run in isolated environments because their dependency stacks differ materially.

Recommended structure:

```text
work/
├── envs/
│   ├── tirex/
│   ├── chronos/
│   └── toto/
├── inputs/
├── validation_runs/
├── test_runs/
└── future_runs/
```

The parent notebook passes:

- JSON configuration;
- CSV/NPY input;
- output directory.

Each environment returns normalized JSON/CSV results.

The notebook MUST NOT:

- clone repositories;
- import repository-local source;
- fetch DIMER source at runtime; or
- call DIMER workers.

---

# 22. TiRex execution

TiRex SHOULD run in its isolated CPU environment.

Reference environment:

```text
tirex-2==0.2.1
torch==2.8.0
numpy==2.3.3
pandas==2.3.3
```

When executed inside a GPU notebook environment, the subprocess SHOULD hide CUDA if required:

```text
CUDA_VISIBLE_DEVICES=""
```

Output MUST normalize:

```text
q0.1
q0.5
q0.9
```

from the model's native nine-quantile output.

---

# 23. Chronos execution

Reference package:

```text
chronos-forecasting==2.3.1
```

The model SHALL receive the same two target series.

Only q0.1/q0.5/q0.9 are retained for common evaluation.

The adapter MUST assert:

```text
prediction == q0.5
```

No covariate table is supplied to the canonical comparison.

---

# 24. Toto execution

Reference package:

```text
toto-2==2.0.0
```

Toto MUST run on CUDA.

Canonical settings SHOULD preserve the currently qualified DIMER decode contract unless later testing justifies a workshop-specific setting.

The two synthetic series are supplied as the two target variates.

Toto MUST be unloaded from GPU memory after each forecasting phase.

---

# 25. Workshop execution tiers

Because Toto requires a 9.8 GB checkpoint, the notebook SHALL expose:

```python
WORKSHOP_TIER = "STANDARD"  # @param ["STANDARD", "FULL"]
```

## STANDARD

Runs:

- TiRex-2
- Chronos-2

This is the canonical `Run all` release path.

## FULL

Runs:

- TiRex-2
- Chronos-2
- Toto 2.0

The FULL path is intended for:

- instructor environments;
- pre-cached sessions;
- extended workshops;
- complete fleet comparison.

Both modes require clean-runtime qualification, but STANDARD is the release-default canonical path.

---

# 26. Validation-stage comparison

Run every selected model on:

```text
history = hours 0–47
forecast = hours 48–71
```

Produce one common result table.

Example:

| Model | A MAE | B MAE | Macro MAE | Seasonal skill | Coverage | Width |
|---|---:|---:|---:|---:|---:|---:|

Also show one chart per series with:

- history;
- validation truth;
- last-value;
- seasonal naive;
- model medians.

A second visualization MAY show model q10–q90 bands.

---

# 27. Freeze-before-test

Before test evaluation, export:

```text
outputs/frozen/frozen_experiment.json
```

It MUST record:

- dataset SHA-256;
- model set;
- model revisions;
- environment pins;
- context policy;
- validation horizon;
- test horizon;
- seasonal period;
- quantiles;
- decode settings;
- baselines;
- metric definitions.

After this file is written, test results MUST NOT alter these choices.

---

# 28. Independent test

Run the frozen configurations using:

```text
history = hours 0–71
truth   = hours 72–95
```

Test horizon:

```text
24
```

The notebook SHOULD emphasize:

> Validation observations are now historical observations and may legitimately be part of the test forecast context.

---

# 29. Final evaluation tables

## 29.1 Aggregate

| Model | MAE | RMSE | Seasonal MAE skill | q10–q90 coverage | Mean interval width | Inference time |
|---|---:|---:|---:|---:|---:|---:|

## 29.2 Per-series

| Model | Series | MAE | RMSE | Seasonal MAE | Skill | Coverage | Width |
|---|---|---:|---:|---:|---:|---:|---:|

Do not present aggregate numbers without the per-series table.

---

# 30. Computational tradeoffs

Record:

- checkpoint size;
- environment installation time;
- model load time;
- forecast time;
- device;
- GPU memory where available;
- number of quantiles produced;
- model-specific capabilities excluded from the common comparison.

Suggested table:

| Model | Weight size | Device | Load time | Forecast time | Notes |
|---|---:|---|---:|---:|---|
| TiRex-2 | ~381 MB | CPU | measured | measured | lightweight |
| Chronos-2 | ~478 MB | CPU/GPU | measured | measured | native 1,024 horizon |
| Toto 2.0 | 9.8 GB | GPU | measured | measured | 2.5B parameters |

This section SHOULD prompt:

> Does any measured improvement justify the additional compute and model footprint for this problem?

---

# 31. New-data forecast

After test evaluation, use the entire 96-step sample as context.

Generate a future:

```text
24-hour
```

forecast with no truth.

Output:

```text
model
series_id
timestamp
step
q0.1
q0.5
q0.9
```

This output MUST be marked:

**not measurable yet**

because no future ground truth exists.

---

# 32. BYOD

The workshop MUST support user-supplied regularly sampled numerical time series.

Recommended controls:

```python
USE_BYOD = False
BYOD_CSV_PATH = ""
BYOD_TIMESTAMP_COLUMN = "timestamp"
BYOD_SERIES_ID_COLUMN = "series_id"
BYOD_TARGET_COLUMN = "target"
BYOD_VALIDATION_HORIZON = 24
BYOD_TEST_HORIZON = 24
BYOD_SEASON_LENGTH = 24
```

If `BYOD_CSV_PATH` is provided, the notebook MUST not open an upload dialog.

---

# 33. BYOD schema

Preferred long format:

```text
series_id,timestamp,target
A,2026-01-01T00:00:00,100
A,2026-01-01T01:00:00,101
...
B,2026-01-01T00:00:00,150
...
```

Requirements:

- unique IDs;
- same timestamp grid for every series;
- fixed-width frequency;
- strictly increasing timestamps;
- finite numerical target;
- no missing observations;
- sufficient history for validation + test + model minimum context.

---

# 34. BYOD workflow

```text
load
→ raw-header validation
→ parse
→ temporal validation
→ reshape
→ EDA
→ chronological split
→ baselines
→ model forecasts
→ validation comparison
→ freeze
→ test
→ new-data forecast
→ export
```

BYOD MUST use the same common evaluator as the built-in fixture.

---

# 35. Privacy guidance

The notebook MUST state:

> User-supplied data remain inside the selected notebook runtime and are not submitted to a DIMER worker or API. A hosted notebook remains an external compute environment. Do not upload confidential, personal, security-sensitive, restricted, regulated, or proprietary operational telemetry unless you are authorized to process it there.

---

# 36. Optional experiment — Chronos generated fixtures

The existing Chronos dataset generator also provides:

- covariate history;
- future-known covariates;
- a second multi-series mode.

An optional section MAY reuse:

```text
chronos_covariates_history.csv
chronos_covariates_future.csv
```

to demonstrate Chronos known-future covariates.

This section MUST be clearly excluded from the common model comparison because Toto does not expose the same input capability.

---

# 37. Optional experiment — TiRex native fixture

The TiRex tutorial currently uses a deterministic 256-step series:

```text
0.02*t
+ sin(t/8)
+ Gaussian noise(seed=7)
```

with a 32-step horizon.

This MAY be included as:

> model-native fixture comparison

to show that individual DIMER tutorial samples are optimized for their own model demonstrations.

It MUST NOT enter the shared leaderboard.

---

# 38. Optional experiment — Toto native fixture

The Toto tutorial currently uses a deterministic 320-step two-variate fixture:

```text
first:
0.01*t + sin(t/9) + noise

second:
0.5*first + cos(t/13) + noise
```

with seed `11`.

This MAY be used to demonstrate Toto's model-native multi-variate path.

Again, it MUST remain separate from the common workshop evaluation.

---

# 39. Workshop exercises

### Exercise A — seasonality

Before running the models:

> Which baseline should perform better on these synthetic series: last value or 24-hour seasonal naive?

Explain why.

---

### Exercise B — model versus baseline

After validation:

> Did every foundation model beat seasonal naive on both series?

If not:

> What does that tell us about using large pretrained models when a simple seasonal pattern dominates the data?

---

### Exercise C — uncertainty

Compare:

- interval coverage;
- interval width.

Ask:

> Would you prefer a narrower interval with lower coverage or a wider interval with higher coverage?

The notebook SHOULD explain that the choice depends on the operational cost of misses and over-conservatism.

---

### Exercise D — compute

Show model size and runtime.

Ask:

> Would the 9.8 GB Toto model be justified for this forecasting workload?

The notebook MUST not prescribe one answer.

---

# 40. Interpretation and limitations

The concluding section MUST state:

### Synthetic data

The dataset was deliberately constructed to be understandable.

Performance on it is **tutorial evidence only**.

### No universal ranking

Two synthetic series and two forecast origins cannot establish general model superiority.

### Baselines are mandatory

A foundation model that cannot consistently improve on seasonal naive may not be worthwhile for the application.

### Quantiles are not guaranteed intervals

q10/q90 require empirical calibration assessment on real data.

### Distribution shift

Real forecasting problems include:

- structural breaks;
- missing data;
- promotions;
- incidents;
- changing sensors;
- changed policies;
- evolving demand;
- interventions;
- nonstationarity.

None is fully represented by this fixture.

---

# 41. Output structure

```text
outputs/
├── data/
│   ├── chronos_multi_series.csv
│   └── dataset_manifest.json
├── validation/
│   ├── baselines.csv
│   ├── tirex.csv
│   ├── chronos.csv
│   ├── toto.csv
│   └── metrics.csv
├── frozen/
│   └── frozen_experiment.json
├── test/
│   ├── forecasts.csv
│   ├── aggregate_metrics.csv
│   └── per_series_metrics.csv
├── future/
│   └── forecasts.csv
├── figures/
├── provenance/
│   ├── models.json
│   └── experiment_manifest.json
└── workshop_summary.json
```

Toto outputs MAY be absent in STANDARD mode.

---

# 42. Provenance

The experiment manifest SHOULD record:

```text
notebook_spec
notebook_profile
notebook_mode
workshop_revision
execution_tier

dataset:
  generator_source
  formula_revision
  sha256
  license
  frequency
  series_ids
  n_timesteps

protocol:
  validation_context
  validation_horizon
  test_context
  test_horizon
  seasonal_period
  quantiles

models:
  id
  revision
  runtime_package
  checkpoint_sha256
  checkpoint_size
  device
  inference_configuration

evaluation:
  point_forecast = q0.5
  baselines
  metrics
  freeze_record
```

---

# 43. Notebook metadata

```json
{
  "dimer": {
    "notebook_spec": "2.1",
    "notebook_profile": "TASK-INFERENCE",
    "notebook_mode": "WORKSHOP",
    "standalone": true,
    "capability": "multi-model-time-series-forecasting",
    "carrier": "self-contained multi-model comparative forecasting workshop",
    "dataset": "chronos_multi_series synthetic fixture",
    "default_tier": "STANDARD",
    "canonical_runtime": "NVIDIA Tesla T4",
    "worker_required": false,
    "credentials_required": false,
    "clean_runtime_evidence": "pending"
  }
}
```

---

# 44. Release acceptance

| Check | STANDARD | FULL |
|---|---:|---:|
| Notebook spec 2.1 | PASS | PASS |
| Fresh `Run all` | PASS | PASS |
| No repository clone | PASS | PASS |
| No DIMER-source runtime fetch | PASS | PASS |
| No DIMER service | PASS | PASS |
| No credentials | PASS | PASS |
| Dataset formula digest | PASS | PASS |
| Chronology validation | PASS | PASS |
| Last-value baseline | PASS | PASS |
| Seasonal-naive baseline | PASS | PASS |
| TiRex-2 | PASS | PASS |
| Chronos-2 | PASS | PASS |
| Toto 2.0 | N/A | PASS |
| Common normalized output | PASS | PASS |
| Freeze-before-test | PASS | PASS |
| Independent test | PASS | PASS |
| Probabilistic metrics | PASS | PASS |
| Future forecast | PASS | PASS |
| BYOD positive case | PASS | PASS |
| BYOD refusal case | PASS | PASS |
| Provenance export | PASS | PASS |

---

# 45. Suggested registry entry

```markdown
| Notebook | Profile | Mode | Capability | Default runtime | Sample | BYOD | Run-all | Release status |
|---|---|---|---|---|---|---|---|---|
| `DIMER_MultiModel_TimeSeries_Forecasting_Workshop.ipynb` | `TASK-INFERENCE` | `WORKSHOP` | zero-shot forecasting with TiRex-2, Chronos-2 and optional Toto 2.0 | T4 | deterministic Chronos two-series hourly fixture | yes | pending | candidate |
```

---

# 46. Implementation principle

This notebook should follow the same design philosophy as the improved tabular FM workshops:

> **one dataset → one experimental protocol → common baselines → multiple foundation models → frozen evaluation → common metrics → interpretable engineering tradeoffs**

The workshop SHOULD reuse the existing Chronos sample-generation logic and digest rather than introduce another dataset solely for this notebook.

The TiRex and Toto native synthetic fixtures remain useful as optional experiments, but the canonical comparison MUST use the shared Chronos two-series fixture so every model receives identical data.