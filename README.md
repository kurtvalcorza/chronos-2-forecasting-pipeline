# Chronos-2 Forecasting — DIMER Pipeline

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/chronos-2-forecasting-pipeline)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-amazon%2Fchronos--2-ffcc4d?style=flat)](https://huggingface.co/amazon/chronos-2)
[![Upstream](https://img.shields.io/badge/Upstream-amazon%2Fchronos--forecasting-181717?style=flat&logo=github&logoColor=white)](https://github.com/amazon-science/chronos-forecasting)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

A DIMER pipeline that runs **zero-shot time-series forecasting** with
[Chronos-2](https://huggingface.co/amazon/chronos-2), a pretrained forecasting foundation model
from Amazon. You supply a long-format table of observations — one identifier column, one
timestamp column, one target column — and the pipeline validates the table, resolves and
integrity-checks a single pinned model revision, and returns a quantile forecast together with
complete runtime provenance. Nothing is trained or fine-tuned.

Chronos-2 is a forecasting **specialist**, not a general-purpose time-series model. See
[MODEL_CARD.md](MODEL_CARD.md) for capabilities, limits, supply-chain pins, and licence.

## Status: Phase 2 inference

This repository is being built in the phases defined by the RFC
([`docs/rfc/0001-chronos-2.md`](docs/rfc/0001-chronos-2.md), issue
[#1](https://github.com/kurtvalcorza/chronos-2-forecasting-pipeline/issues/1)):

| Phase | Scope | State |
| :-- | :-- | :-- |
| 1 | RFC in version control, locked environment, pinned + verified loader, model card, DIMER-side validation, univariate inference, raw/normalised output contract tests, real CPU smoke test | complete |
| 2 | Multi-target, past and known-future covariates, effective context/horizon provenance | **this phase** |
| 3 | Sample datasets and cards, BYOD, Colab tutorial, visualisation, evaluation and baselines, export | not started |
| 4 | Stable serving API, resource limits, latency instrumentation, DIMER packaging contract | not started |

**What works today:** all four RFC forecasting modes — A (univariate), B (multiple independent
series), C (multi-target) and D (covariate-informed, past-only and known-future). Covariates must
be numeric; see [Design commitments](#design-commitments). `evaluation.py` raises
`NotImplementedError` until Phase 3.

## Quickstart

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). The lock file is the environment
contract — CI, Colab and local development all install from it.

```bash
uv sync --locked --extra dev      # exact locked stack, plus pytest and ruff
uv run ruff check .
uv run pytest -m "not integration"   # no network, no model weights
uv run pytest -m integration         # downloads ~478 MB of weights on first run, CPU only
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
result.provenance    # model pins, resolved runtime versions, context/horizon semantics
```

`prediction` is the **median (q0.5)** point forecast, never a mean — see
[Point-forecast semantics](MODEL_CARD.md#point-forecast-semantics).

Several targets, and a covariate whose future values you know, are the same call:

```python
config = ForecastConfig(target=["demand", "price"], prediction_length=24)
result = forecast(history, config, model, future_df=known_future)
```

`history` carries one column per target plus any covariate columns; `known_future` carries the id
and timestamp columns plus the covariates you know across the horizon, exactly
`prediction_length` rows per series. Every covariate not in `known_future` is past-only, and
`result.provenance["inference"]` names both sets separately.

## Design commitments

These are contract items from the RFC, each backed by a test rather than by documentation:

- **One model source.** `load_pinned_model` accepts only `amazon/chronos-2` at revision
  `95a9710e...`. `main`, `latest`, any other revision, local paths, `s3://` and `hf://` raise
  `ModelSourceError`. Before anything is downloaded the Hub is asked which commit the pin names,
  and a different answer refuses the load; after download, the `model.safetensors` SHA-256 and
  byte size and the `config.json` SHA-256 are checked; `.bin` weights are refused outright.
- **Validation is ours, and runs first.** All 21 RFC validation rules are enforced before
  `predict_df` is called. Supplying `frequency=` does not bypass the regularity, gap,
  same-frequency or minimum-length checks, *and* it does not reach `predict_df` unless it has been
  confirmed equal to the interval observed in the data — upstream uses `freq` as-is to lay out the
  horizon, so an unchecked value silently moves the forecast onto another time axis.
- **Fixed-width frequencies only.** Regularity is diff equality, so `15min`, `h`, `D`, `7D` are in
  scope and **monthly, quarterly, yearly and business-daily data are rejected**
  (`CALENDAR_FREQUENCY_UNSUPPORTED`). Calendar-frequency support is Phase 2. See
  [Frequency, gaps, and why validation is duplicated](MODEL_CARD.md#frequency-gaps-and-why-validation-is-duplicated).
- **Out-of-grid quantiles hard-fail, with no opt-out.** Upstream silently clamps a request for
  `0.001` to the nearest trained quantile. This pipeline rejects it, because a column labelled
  `q0.001` that actually holds `q0.01` is a false export. Membership is exact float identity, the
  same gate upstream uses.
- **The rename map is deterministic and tested.** Upstream raw columns map to
  `series_id, timestamp, target_name, prediction, q<level>` by an explicit table.
- **Rows are checked against the layout, not trusted.** Upstream ties a forecast value to its
  series and target by row position alone. Every call asserts the frame is one row per
  `(series, target, step)` in that order, so a reordering upstream fails loudly instead of
  exporting each forecast under a neighbouring label.
- **Numeric covariates only.** Upstream accepts a categorical covariate but encodes it by a route
  that differs between a single-target and a multi-target request, so the same column would mean
  different things in two otherwise identical exports. Non-numeric covariates are refused
  (`COVARIATE_NOT_NUMERIC`); `bool` is coerced to `0.0`/`1.0` rather than left on that path.
  Nulls and non-finite covariate values are refused, as they are for targets.
- **Provenance travels with the forecast.** Requested and effective context and horizon, model
  limits, `autoregressive_unrolled`, requested and effective quantiles, resolved library
  versions, device and dtype.

## Repository layout

```text
docs/rfc/0001-chronos-2.md    verbatim RFC; the commit that added it is the mission anchor
src/chronos2_pipeline/
  config.py                   ForecastConfig — user parameters and their validation
  model.py                    pinned loader, integrity checks, ModelIdentity
  validation.py               DIMER-side validation, RFC rules 1-21
  inference.py                forecast(): validate, predict, normalise, record provenance
  provenance.py               export metadata assembly
  evaluation.py               Phase 3 stub
tests/                        contract/unit tests (no network) + integration tests (marked)
```

## Licence

Pipeline code: [Apache-2.0](LICENSE), Copyright 2026 Kurt Valcorza. The Chronos-2 model weights
carry their own licence, recorded in [MODEL_CARD.md](MODEL_CARD.md).
