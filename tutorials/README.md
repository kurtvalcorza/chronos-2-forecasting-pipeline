# Tutorials

Notebook specification: **DIMER Notebook Specification 1.0**

| Notebook | Profile | Capability | Default runtime | BYOD | Release status |
|---|---|---|---|---|---|
| [`chronos_2_forecasting_colab.ipynb`](chronos_2_forecasting_colab.ipynb) | `TASK-INFERENCE` | zero-shot probabilistic forecasting | Python 3.12 / CPU | CSV upload/path | release candidate |

The notebook is the authoritative user-facing tutorial for the repository's primary forecasting capability. It uses the repository's public API, immutable upstream model pin, chronological evaluation, naive baselines, machine-readable forecast/evaluation exports, and provenance.

## Verification

Automated release verification covers:

- notebook JSON/source compilation and conformance assertions;
- locked dependency installation and repository bootstrap from a clean working directory;
- real pinned-weight CPU execution of the default tutorial path;
- a BYOD-shaped CSV path through the same notebook ingestion and pipeline calls;
- the optional known-future-covariate path; and
- upload of generated forecast/evaluation/provenance evidence.

A release should record the successful CI run for the exact release commit. Because hosted Colab images can change independently of this repository, maintainers should also perform and record a clean Colab run before describing a new release revision as Colab-verified.

`release candidate` must be changed to `release-grade` only after all applicable DIMER Notebook Specification 1.0 MUST requirements pass and clean-runtime evidence for the release revision is recorded.
