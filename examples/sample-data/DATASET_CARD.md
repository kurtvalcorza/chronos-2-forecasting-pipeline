# Chronos-2 Tutorial Samples

## Summary

These are **synthetic, deterministic teaching datasets** for the DIMER Chronos-2 pipeline. They exist to exercise the repository's input, forecasting, covariate, evaluation and export contracts without introducing third-party redistribution or benchmark-overlap claims.

The source of truth is [`generate_samples.py`](generate_samples.py). It contains no random draw: every value is produced from fixed trends, sinusoidal components and deterministic calendar flags. Running it recreates the tutorial CSVs and `SHA256SUMS` byte-for-byte under the locked runtime.

`chronos_univariate.csv` is checked in for the default quick-start path. The generator also produces the larger multi-series and covariate examples on demand so their provenance stays inspectable as code rather than as opaque copied data.

## License and redistribution

The samples are original synthetic content generated for this repository and are distributed under the repository's **Apache-2.0** license. They incorporate no third-party records, personal data, copyrighted dataset content or upstream benchmark rows.

## Artifacts

| Artifact | Rows | IDs | Frequency | Targets | Covariates | Purpose |
|---|---:|---:|---|---|---|---|
| `chronos_univariate.csv` | 96 | 1 | hourly | `target` | none | default tutorial, chronological holdout and BYOD-shape reference |
| `chronos_multi_series.csv` | 192 | 2 | hourly | `target` | none | Mode B multiple independent series |
| `chronos_covariates_history.csv` | 192 | 2 | hourly | `demand` | `temperature`, `holiday` | Mode D historical context |
| `chronos_covariates_future.csv` | 48 | 2 | hourly | none | `temperature`, `holiday` | 24-step known-future covariate table |

All timestamps are naive ISO-8601 values. The examples intentionally use fixed-width hourly frequency because calendar-frequency support is outside the current pipeline contract.

## Generation formulas

### Univariate

For hourly step `t = 0..95`:

```text
target = 100 + 0.25*t
       + 7*sin(2*pi*t/24)
       + 1.5*cos(2*pi*t/12)
```

This combines a visible trend with daily and half-daily structure.

### Multiple series

Two independent IDs use different offsets, slopes, amplitudes and phase while preserving the same hourly schema. The sample is designed to show that Mode B is multiple independent series, not multi-target forecasting.

### Covariate-informed

For each series:

```text
temperature = 27 + 4*sin(2*pi*(t + phase)/24)
holiday     = 1 on Saturday/Sunday, otherwise 0
demand      = base + 1.8*temperature + 10*holiday
            + 0.08*t + 3*sin(2*pi*t/12)
```

The future table extends only `temperature` and `holiday` for 24 steps. It deliberately contains **no future target**, so using it cannot leak `demand` into inference.

## Preprocessing

No imputation, interpolation, scaling, resampling or learned preprocessing is applied. Values are emitted at four decimal places. `holiday` is emitted as `0/1`, which enters the pipeline's explicit numeric covariate path.

## Integrity

[`SHA256SUMS`](SHA256SUMS) records the canonical UTF-8 CSV bytes generated with:

- timestamps formatted as `%Y-%m-%dT%H:%M:%S`;
- floats formatted as `%.4f`;
- LF newlines.

Tests regenerate the datasets in a temporary directory and compare both their digests and schemas. A changed formula, row order, float representation or timestamp layout therefore changes the recorded artifact identity.

## Intended use

Use these samples for:

- tutorial and smoke execution;
- validation examples;
- chronological holdout demonstrations;
- naive-baseline comparisons;
- result/provenance export examples.

They are **not benchmark datasets** and should not be used to claim model quality, generalization, calibration or superiority over another forecasting system. Their simple structure is pedagogical and may be easier than real operational data.
