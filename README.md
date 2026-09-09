# Chronos-2 Forecasting — DIMER Pipeline

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/chronos-2-forecasting-pipeline)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/chronos-2-forecasting-pipeline/blob/main/tutorials/chronos_2_forecasting_colab.ipynb)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-amazon%2Fchronos--2-ffcc4d?style=flat)](https://huggingface.co/amazon/chronos-2)
[![Upstream](https://img.shields.io/badge/Upstream-amazon%2Fchronos--forecasting-181717?style=flat&logo=github&logoColor=white)](https://github.com/amazon-science/chronos-forecasting)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

A DIMER pipeline that runs **zero-shot time-series forecasting** with
[Chronos-2](https://huggingface.co/amazon/chronos-2), a pretrained forecasting foundation model
from Amazon. You supply a regular long-format table of observations; the pipeline validates it,
resolves and integrity-checks one pinned model revision, and returns probabilistic forecasts with
complete runtime provenance. Nothing is trained or fine-tuned.

Chronos-2 is a forecasting **specialist**, not a general-purpose time-series model. See
[MODEL_CARD.md](MODEL_CARD.md) for capabilities, limits, supply-chain pins, and licence.

## Status: Phase 3 tutorial / developer-preview candidate

The implementation phases are defined by the original RFC
([`docs/rfc/0001-chronos-2.md`](docs/rfc/0001-chronos-2.md), issue
[#1](https://github.com/kurtvalcorza/chronos-2-forecasting-pipeline/issues/1)). Release-completion
work is tracked in [RFC #5](https://github.com/kurtvalcorza/chronos-2-forecasting-pipeline/issues/5).

| Phase | Scope | State |
| :-- | :-- | :-- |
| 1 | Locked runtime, pinned + verified loader, model card, DIMER validation, univariate/multi-ID inference, real-weight tests | complete |
| 2 | Multi-target, past and known-future covariates, effective context/horizon provenance, row-layout oracle | complete |
| 3 | Deterministic samples, BYOD Colab, visualization, chronological evaluation/baselines, export, executable notebook CI | **implemented; release review** |
| 4 | Stable serving API, resource/latency contract, DIMER backend packaging | not complete |

**What works now:** all four RFC forecasting modes — A (univariate), B (multiple independent
series), C (multi-target), and D (numeric past-only / known-future covariates) — plus
chronological holdout evaluation, MAE/RMSE, pinball loss, interval coverage, last-value and
explicit seasonal-naive baselines, deterministic tutorial samples, and portable CSV/JSON export.

This is a **developer/tutorial preview**, not yet a production-serving stability claim. Phase 4
remains explicit follow-on work.

## Live tutorial

The recommended external-user path is
[`tutorials/chronos_2_forecasting_colab.ipynb`](tutorials/chronos_2_forecasting_colab.ipynb).
It runs the same public Python API as the package and covers:

1. locked-runtime bootstrap;
2. bundled deterministic sample or BYOD CSV;
3. chronological future holdout with no target leakage;
4. verified pinned Chronos-2 acquisition;
5. zero-shot median + quantile forecast;
6. visualization;
7. leakage-safe evaluation against last-value and explicit seasonal-naive baselines;
8. forecast, metrics, and full-provenance export.

The real-weight CI job executes the notebook's actual code cells top-to-bottom and uploads the
resulting `outputs/` directory as workflow evidence. Static notebook JSON validation alone is not
treated as sufficient.

## Quickstart

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). `uv.lock` is the contributor/runtime
resolution contract; CI installs it with `uv sync --locked`.

```bash
uv sync --locked --extra dev
uv run ruff check .
uv run pytest -m "not integration"   # no network or model weights
uv run pytest -m integration         # real pinned weights, CPU
uv run python tools/run_notebook.py tutorials/chronos_2_forecasting_colab.ipynb
```

A minimal forecast:

```python
import pandas as pd
from chronos2_pipeline import ForecastConfig, forecast, load_pinned_model

history = pd.read_csv("my_series.csv")     # series_id, timestamp, target
model = load_pinned_model(device="auto")   # only the pinned revision is accepted
config = ForecastConfig(prediction_length=24, quantile_levels=[0.1, 0.5, 0.9])
result = forecast(history, config, model)

result.forecast      # series_id, timestamp, target_name, prediction, q0.1, q0.5, q0.9
result.provenance    # model pins, runtime versions, context/horizon semantics
```

`prediction` is the **median (q0.5)** point forecast, never a mean — see
[Point-forecast semantics](MODEL_CARD.md#point-forecast-semantics).

Several targets and known-future covariates use the same entry point:

```python
config = ForecastConfig(target=["demand", "price"], prediction_length=24)
result = forecast(history, config, model, future_df=known_future)
```

`history` carries one column per target plus numeric covariates. `known_future` carries the id and
timestamp columns plus covariates known across the horizon, exactly `prediction_length` rows per
series. Future target columns are refused as leakage. Covariates absent from `known_future` are
past-only, and provenance records the two sets separately.

## Chronological evaluation

The public evaluation API deliberately exposes **no random split helper**:

```python
from chronos2_pipeline import (
    chronological_holdout,
    evaluate_forecast,
    last_value_baseline,
)

split = chronological_holdout(history, config)
result = forecast(split.history, config, model)
metrics = evaluate_forecast(result.forecast, split.truth, config)

baseline = last_value_baseline(split.history, split.truth, config)
baseline_metrics = evaluate_forecast(baseline, split.truth, config)
```

The held-out tail is absent from model history. Forecast and truth must cover exactly the same
`(series_id, timestamp, target_name)` rows or evaluation fails rather than silently dropping data.
Quantile evaluation uses pinball loss; empirical interval coverage is reported when at least two
requested quantile columns are present. Seasonal-naive comparison requires an explicit
`season_length`; no seasonal period is inferred heuristically.

## Tutorial samples

[`examples/sample-data/DATASET_CARD.md`](examples/sample-data/DATASET_CARD.md) documents original,
synthetic Apache-2.0 teaching data for univariate, multi-series, and known-future-covariate paths.
The source of truth is `generate_samples.py`, and `SHA256SUMS` pins the canonical generated bytes.
Tests regenerate the artifacts and verify both identity and DIMER validation semantics.

These samples are **not model benchmarks**. Their deterministic, simple structure exists to test
and teach the product contract without third-party redistribution or benchmark-overlap ambiguity.

## Design commitments

These are contract items backed by tests rather than documentation alone:

- **One model source.** `load_pinned_model` accepts only `amazon/chronos-2` at revision
  `95a9710e...`. Mutable refs, other revisions, local paths, `s3://`, and `hf://` are refused.
  The loader verifies the pinned config/weights identity and refuses pickle-format fallback.
- **Validation runs before upstream inference.** The RFC validation rules remain DIMER-owned.
  Supplying `frequency=` cannot bypass regularity, gap, shared-frequency, or minimum-history checks.
- **Fixed-width frequencies only.** Regular `15min`, hourly, daily, and fixed-duration intervals are
  supported; monthly/quarterly/yearly/business-calendar semantics remain out of scope and fail
  explicitly rather than being coerced.
- **Out-of-grid quantiles hard-fail.** The pipeline will not export `q<level>` under a label whose
  value upstream silently substituted from another trained quantile.
- **Rows are checked against the upstream layout.** Multi-target output is asserted to remain one
  row per `(series, target, step)` in the pinned order before it is normalized.
- **Numeric covariates only.** Categorical/string covariates are refused because the pinned
  upstream encoding route changes with target count. Boolean flags are explicitly converted to
  `0.0/1.0`; null and non-finite covariates are refused.
- **Provenance travels with every forecast.** Requested/effective context and horizon, unrolling,
  quantiles, past/known-future covariates, runtime versions, device/dtype, and latency state are
  serialized with the result.
- **Evaluation is chronological.** Tutorial benchmarking never randomly splits a time series or
  exposes held-out future targets to model context/covariates.

## Repository layout

```text
docs/rfc/0001-chronos-2.md          original versioned RFC
examples/sample-data/               deterministic synthetic samples + card + digests
src/chronos2_pipeline/
  config.py                         ForecastConfig
  model.py                          pinned loader + integrity checks
  validation.py                     DIMER-side request validation
  inference.py                      forecast + normalization + provenance
  evaluation.py                     chronological holdout, metrics, naive baselines
  provenance.py                     export metadata
tutorials/chronos_2_forecasting_colab.ipynb
tools/run_notebook.py               executable notebook smoke path
tests/                              unit/contract + real-weight integration suites
```

## Licence

Pipeline code and repository-generated tutorial samples: [Apache-2.0](LICENSE), Copyright 2026
Kurt Valcorza. Chronos-2 model weights carry their own licence, recorded in [MODEL_CARD.md](MODEL_CARD.md).
