---
license: apache-2.0
pipeline_tag: time-series-forecasting
tags:
  - time-series-forecasting
  - time-series-foundation-model
  - zero-shot
base_model: amazon/chronos-2
---

# Chronos-2

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-amazon%2Fchronos--2-ffcc4d?style=flat)](https://huggingface.co/amazon/chronos-2)
[![GitHub](https://img.shields.io/badge/GitHub-amazon--science%2Fchronos--forecasting-181717?style=flat&logo=github&logoColor=white)](https://github.com/amazon-science/chronos-forecasting)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Description

Chronos-2 is a pretrained time-series **forecasting** model from Amazon. It reads a stretch of
historical observations as context and emits a probabilistic forecast for a future horizon
without being trained on the series it is asked about — the in-context-learning paradigm applied
to forecasting rather than to tabular classification.

It is a **forecasting specialist**, not a general-purpose time-series model. It does not do
classification, anomaly detection, imputation, or representation learning, and this pipeline does
not present it as if it did.

Architecture is T5-family (`model_type: t5`, `chronos_pipeline_class: Chronos2Pipeline`) with a
patch-based encoder: input patches of 16 timesteps at stride 16, output patches of 16, up to 64
output patches. Model dimension is 768 and the checkpoint holds roughly 119.5 million parameters
in a single `safetensors` file.

# Model Details

**Model name:** Chronos-2

**Model identifier:** `amazon/chronos-2`

**Code repository:** [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting)

**Hugging Face repository:** [amazon/chronos-2](https://huggingface.co/amazon/chronos-2)

**Developer:** Amazon

**Model family:** Chronos

**Task:** Zero-shot time-series forecasting

**Architecture:** T5-family patch-based encoder (`Chronos2Pipeline`)

**Model / embedding dimension:** 768

**Input patch size / stride:** 16 / 16

**Output patch size:** 16

**Maximum output patches:** 64

**Approximate parameter count:** ~119.5 million

**Weight format:** `safetensors` (single file; no pickle-format checkpoint exists at the pinned revision)

**License:** Apache-2.0 — see [Licence](#licence) for exactly what that claim rests on

# Checkpoint and Artifact Provenance

Every value below was resolved directly from the Hugging Face Hub on 2026-09-08 and is asserted
by a test in this repository. The constants live in
[`src/chronos2_pipeline/model.py`](src/chronos2_pipeline/model.py); the loader refuses to
continue if any of them fails to match.

| Item | Value |
| :-- | :-- |
| Pinned revision | `95a9710e2596287d08352589f42634fa5abdf0a7` |
| Source URL | https://huggingface.co/amazon/chronos-2/tree/95a9710e2596287d08352589f42634fa5abdf0a7 |
| `model.safetensors` SHA-256 | `ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42` |
| `model.safetensors` size | 477,930,472 bytes |
| `config.json` SHA-256 | `ef1143bfdc9c0376d9a056eefca46cb4b1ec3d0ffacd541ff56feb40fb708031` |
| `config.json` size | 1,067 bytes |
| Files at that revision | `.gitattributes`, `README.md` (31 bytes), `config.json`, `model.safetensors` |

**The revision SHA is the primary supply-chain invariant.** It covers repository configuration as
well as weights, so it is checked first. The two digests are secondary, file-level assertions that
additionally catch a corrupted or truncated download.

## What the loader enforces

`load_pinned_model()` refuses, before any network call, anything other than `amazon/chronos-2` at
the pinned commit: mutable refs (`main`, `master`, `latest`, `HEAD`), any other commit SHA, local
filesystem paths, and URI schemes including `s3://`, `hf://`, `https://` and `file://`. After
download it verifies both digests and the weight file's byte size, and refuses outright if any
`.bin`, `.pt`, `.pth`, `.ckpt` or `.pkl` file is present in the snapshot — pickle checkpoints
execute arbitrary code on load, and this pipeline has no fallback to them.

**How the resolved revision is checked.** `snapshot_download` lays a snapshot out under
`snapshots/<requested-revision>/`, and the requested revision has already been forced equal to the
pin, so comparing the directory name against the pin is a tautology that cannot fail — the check
an earlier revision performed. The loader now asks the Hub itself, before downloading anything,
which commit the pin resolves to (`HfApi().model_info(...).sha`) and refuses if that is not the
pinned commit; the snapshot directory must then agree with the Hub's answer as well. Two
consequences worth knowing: loading makes one Hub metadata request, and if the Hub cannot be
reached the load fails closed with `ModelIntegrityError` rather than proceeding on the file
digests alone, because the RFC makes the revision the *primary* anchor.

## Runtime pins

| Component | Pin |
| :-- | :-- |
| `chronos-forecasting` | `==2.3.1` (PyPI; upstream tag `v2.3.1` = commit `7dc4435706a4454feb79df44ca9f33631f3027bf`) |
| `transformers` | `>=4.56,<5` |
| Python | `>=3.12,<3.13` |
| Full resolved stack | [`uv.lock`](uv.lock) and [`requirements.lock.txt`](requirements.lock.txt) (hashed) |

The resolved versions of `chronos-forecasting`, `torch`, `transformers`, `huggingface-hub`,
`numpy`, `pandas`, `accelerate`, `einops` and `safetensors`, along with Python version, platform,
device and dtype, are written into every forecast's provenance block. CI installs with
`uv sync --locked`, so CI and local runs resolve to the same stack.

# Licence

The Hugging Face model card metadata for `amazon/chronos-2` declares `license: apache-2.0`, and
the repository carries the `license:apache-2.0` tag.

**There is no `LICENSE` file in the Hugging Face repository at the pinned revision.** The four
files present are listed in the provenance table above. The licence claim therefore rests on the
model card metadata at that revision and nothing else, and that is exactly how it is recorded in
exported provenance:

> apache-2.0 declared in the Hugging Face model card metadata of amazon/chronos-2 at revision
> 95a9710e2596287d08352589f42634fa5abdf0a7; the repository contains no LICENSE file at that
> revision

Pipeline code in this repository is [MIT](LICENSE), Copyright (c) 2026 Kurt Valcorza. Data you
forecast carries its own licence, which this repository makes no claim about.

# Capabilities and limits

## Quantile grid

Chronos-2 was trained on a fixed 21-level quantile grid, read from `chronos_config.quantiles` in
the pinned `config.json`:

```text
0.01 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50
0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 0.99
```

The pipeline reads this grid off the loaded model at runtime and rejects any requested level that
is not an exact member. There is **no opt-out**: `forecast()` has no `allow_out_of_grid`
parameter, and passing one is a `TypeError`. An earlier revision had that flag; it exported a
column named for the *requested* level while holding the substituted one, and recorded
`effective_quantile_levels == requested_quantile_levels` in provenance, so the substitution was
invisible in the export. Phase 1 has no consumer for the escape hatch, so it was removed rather
than repaired.

Membership is **exact float identity**, matching upstream's own gate
(`set(quantile_levels).issubset(training_quantile_levels)`, `pipeline.py` L797). A tolerance would
accept `0.1 * 7 == 0.7000000000000001`, which upstream then routes to `interpolate_quantiles`
instead of indexing the trained `0.7` — the same silent substitution, one grid step smaller. When a
rejected level is within `1e-6` of a grid member the error names that member.

**Why rejection rather than the upstream default.** Asked for a level outside the trained range,
upstream 2.3.1 substitutes the nearest trained level and emits a warning
(`chronos/chronos2/pipeline.py` L797-815). The returned column is still *named* for the level you
asked for. A request for `0.001` therefore comes back as a column labelled `"0.001"` holding the
`0.01` quantile — an export that says something false about itself. An integration test calls
`predict_df` directly with `quantile_levels=[0.001]` and asserts the values equal the `0.01`
column, which is the evidence behind this refusal.

## Context and horizon

| Limit | Value |
| :-- | :-- |
| Model context length | 8,192 timesteps |
| Model native prediction length | 1,024 timesteps (64 output patches × patch size 16) |

Both are read from the loaded pipeline (`model_context_length`, `model_prediction_length`), not
hardcoded, and asserted against these numbers in the integration suite.

Upstream clamps a longer requested context to 8,192 (`pipeline.py` L610-617) and will satisfy a
horizon beyond 1,024 by unrolling its own output autoregressively — it passes
`limit_prediction_length=False` unconditionally (`pipeline.py` L939), so there is no upstream
guard to lean on. This pipeline rejects `prediction_length > 1024` by default; `allow_unroll=True`
opts in, and provenance then records `autoregressive_unrolled: true`. Requested, effective and
model values are recorded for both context and horizon on every call, whether or not clamping
happened.

Two guards bound the horizon and the order matters. `ResourceLimits.max_prediction_length`
(**4,096** by default) is DIMER's own resource guard and is deliberately set *above* the model's
native 1,024 so that the unroll gate, not the resource guard, is what answers a request beyond the
model's capacity. When the two were equal, `PREDICTION_LENGTH_LIMIT` always won and `allow_unroll`
— and therefore `autoregressive_unrolled: true` — was unreachable on the shipped configuration.
So: 1,025-4,096 needs `allow_unroll=True` and is flagged in provenance; above 4,096 is refused
outright as `PREDICTION_LENGTH_LIMIT`.

## Point-forecast semantics

**`prediction` is the median (q0.5), never a mean.**

Upstream's `predict_df` returns a column called `predictions`, and parts of the upstream
documentation describe it as a mean. It is not. At 2.3.1 it is computed as
`pred[..., training_quantile_levels.index(0.5)]` — the code carries the comment
`# NOTE: the median is returned as the mean here` (`chronos/chronos2/pipeline.py` L816-818).

This pipeline therefore labels the column `prediction` / median point forecast, and asserts on
every call that `raw["predictions"]` equals `raw["0.5"]` exactly whenever `0.5` was requested. A
mismatch raises `UpstreamContractError` rather than relabelling a different statistic. For a
skewed predictive distribution the median and the mean differ, and downstream cost calculations
that assume a mean will be wrong.

## Frequency, gaps, and why validation is duplicated

Upstream's `freq=` argument bypasses frequency inference entirely. Its own docstring says so
(`pipeline.py` L881-885): the supplied value "is used as-is and is not checked against the data,
even when `validate_inputs=True`". A gappy or irregular series passed with `freq="h"` is accepted
and forecast.

This pipeline validates regularity, gaps, shared frequency and minimum length itself, before
`predict_df`. Regularity is established by explicit diff equality per series, not by
`pd.infer_freq`. Series with fewer than three observations are rejected: upstream's own inference
needs three points (`chronos/df_utils.py` L30), and below that regularity is unfalsifiable.

**A declared `frequency` neither suppresses those checks nor reaches upstream unchecked.** The
checks are only half the protection: `freq` also *lays out the forecast horizon*, so a value that
passes the checks but contradicts the data still moves the output onto a different time axis. An
earlier revision compared only fixed-width aliases and let calendar aliases fall through, so
hourly history declared `frequency="ME"` was validated as hourly and came back stamped at month
ends, with no error. Now every declared alias is resolved before `predict_df` and only one that
has been affirmatively confirmed equal to the observed interval is forwarded:

| Declared, on strictly hourly data | Outcome |
| :-- | :-- |
| `"h"`, `"60min"` | accepted, forwarded to `predict_df` |
| `"D"`, `"15min"` | `FREQUENCY_MISMATCH` |
| `"W"`, `"ME"`, `"B"`, `"QS"`, `"YE"` | `FREQUENCY_NOT_FIXED_WIDTH` |
| not declared | accepted; `freq=None`, upstream infers from the validated data |

### Fixed-width frequencies only

**v1 supports fixed-width frequencies only** — those where one period is always the same
`Timedelta`: `15min`, `h`, `D`, `7D`. This follows directly from diff equality, which calendar
frequencies can never satisfy, and it is a real limitation: **monthly, quarterly, yearly and
business-daily data cannot be forecast by this pipeline.** Such a series is rejected with
`CALENDAR_FREQUENCY_UNSUPPORTED`, which names the alias `pd.infer_freq` recognised, rather than
being called `IRREGULAR_FREQUENCY` — regular monthly data is not irregular, and the earlier
message said it was. Widening the rule to calendar offsets is a Phase-2 design decision, not a
Phase-1 patch.

Weekly data *is* in scope: a week is a constant seven days. Declare it as `"7D"`; the pandas alias
`W` is anchored and therefore not fixed-width, so it is refused as a declaration even where the
data itself passes.

v1 rejects gaps and missing target values rather than interpolating them. Silent interpolation of
irregular or gappy input is explicitly out of scope.

## Not supported

- Fine-tuning or pretraining — zero-shot inference only.
- Online or streaming weight updates.
- Causal interpretation of covariates. A covariate that helps the forecast is not thereby a cause.
- Calibration guarantees. Quantiles are the model's, uncalibrated for your data.
- Chronos-Bolt or older Chronos checkpoints.
- `cross_learning=true` as a default or tutorial workflow. When enabled, results depend on batch
  composition, so both the flag and `batch_size` must be exported.

# Evaluation caveat

**Do not treat a tutorial benchmark number from this repository as an unbiased estimate of
Chronos-2's accuracy on your problem.** The datasets used in tutorials and examples may overlap
the corpora Chronos-2 was pretrained or benchmarked on. Neither the pretraining corpus nor the
benchmark suite is enumerated at the pinned revision, so overlap cannot be ruled out by
inspection. Any honest comparison needs your own held-out window, split chronologically, with
baselines scored on the identical horizon.

Evaluation is Phase 3 in this repository; `evaluation.py` raises `NotImplementedError` rather than
offering a placeholder metric that might split at random.

# Upstream references

- Model: https://huggingface.co/amazon/chronos-2 (revision `95a9710e2596287d08352589f42634fa5abdf0a7`)
- Code: https://github.com/amazon-science/chronos-forecasting (release `v2.3.1`, commit `7dc4435706a4454feb79df44ca9f33631f3027bf`)
- Line references above are to the installed `chronos-forecasting==2.3.1` package as inspected on
  2026-09-08, not to upstream `main`.
- This pipeline's contract: [`docs/rfc/0001-chronos-2.md`](docs/rfc/0001-chronos-2.md)
