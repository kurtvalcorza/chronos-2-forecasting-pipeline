# Chronos-2 Forecasting — DIMER Pipeline

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/chronos-2-forecasting-pipeline)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-amazon%2Fchronos--2-ffcc4d?style=flat)](https://huggingface.co/amazon/chronos-2)
[![Upstream](https://img.shields.io/badge/Upstream-amazon%2Fchronos--forecasting-181717?style=flat&logo=github&logoColor=white)](https://github.com/amazon-science/chronos-forecasting)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A DIMER pipeline that runs **zero-shot time-series forecasting** with
[Chronos-2](https://huggingface.co/amazon/chronos-2), a pretrained forecasting foundation model
from Amazon. You supply a long-format table of observations — one identifier column, one
timestamp column, one target column — and the pipeline validates the table, resolves and
integrity-checks a single pinned model revision, and returns a quantile forecast together with
complete runtime provenance. Nothing is trained or fine-tuned.

Chronos-2 is a forecasting **specialist**, not a general-purpose time-series model. See
[MODEL_CARD.md](MODEL_CARD.md) for capabilities, limits, supply-chain pins, and licence.

## Status: Phase 1 foundation

This repository is being built in the phases defined by the RFC
([`docs/rfc/0001-chronos-2.md`](docs/rfc/0001-chronos-2.md), issue
[#1](https://github.com/kurtvalcorza/chronos-2-forecasting-pipeline/issues/1)). Phase 1 is the
foundation layer only:

| Phase | Scope | State |
| :-- | :-- | :-- |
| 1 | RFC in version control, locked environment, pinned + verified loader, model card, DIMER-side validation, univariate inference, raw/normalised output contract tests, real CPU smoke test | **this phase** |
| 2 | Multi-target, past and known-future covariates, effective context/horizon provenance | not started |
| 3 | Sample datasets and cards, BYOD, Colab tutorial, visualisation, evaluation and baselines, export | not started |
| 4 | Stable serving API, resource limits, latency instrumentation, DIMER packaging contract | not started |

**What works today:** univariate, multi-ID forecasting (RFC Modes A and B). Multi-target and
covariates are validated and rejected by the public API in Phase 1 — the upstream multi-target
raw schema is exercised by a contract test, but is not yet a supported entry point.
`evaluation.py` raises `NotImplementedError` until Phase 3.

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

## Design commitments

These are contract items from the RFC, each backed by a test rather than by documentation:

- **One model source.** `load_pinned_model` accepts only `amazon/chronos-2` at revision
  `95a9710e...`. `main`, `latest`, any other revision, local paths, `s3://` and `hf://` raise
  `ModelSourceError`. After download, the resolved commit, the `model.safetensors` SHA-256 and
  byte size, and the `config.json` SHA-256 are all checked; `.bin` weights are refused outright.
- **Validation is ours, and runs first.** All 21 RFC validation rules are enforced before
  `predict_df` is called, and supplying `frequency=` does not bypass the regularity, gap,
  same-frequency or minimum-length checks.
- **Out-of-grid quantiles hard-fail.** Upstream silently clamps a request for `0.001` to the
  nearest trained quantile. This pipeline rejects it, because a column labelled `q0.001` that
  actually holds `q0.01` is a false export.
- **The rename map is deterministic and tested.** Upstream raw columns map to
  `series_id, timestamp, target_name, prediction, q<level>` by an explicit table.
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

Pipeline code: [MIT](LICENSE), Copyright (c) 2026 Kurt Valcorza. The Chronos-2 model weights
carry their own licence, recorded in [MODEL_CARD.md](MODEL_CARD.md).
