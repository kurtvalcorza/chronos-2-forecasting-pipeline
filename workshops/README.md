# Workshops

Instructor-led comparative notebooks that use this repository's sample fixtures. They are kept apart
from `tutorials/`, which holds the repository's single generated release tutorial
(`tools/validate_release_assets.py` requires exactly one notebook there). Workshop notebooks are not
generated from `src/` and are not covered by the parity checks.

The design contract is [`WORKSHOP_SPEC.md`](WORKSHOP_SPEC.md).

| Notebook | Profile | Mode | Capability | Default runtime | Sample | BYOD | Run-all | Release status |
|---|---|---|---|---|---|---|---|---|
| `DIMER_MultiModel_TimeSeries_Forecasting_Workshop.ipynb` | `TASK-INFERENCE` | `WORKSHOP` | zero-shot forecasting with TiRex-2, Chronos-2 and optional Toto 2.0 | T4 | deterministic Chronos two-series hourly fixture | yes | pending | candidate |

- **Notebook specification:** DIMER Notebook Specification 2.1, standalone: no clone, repository import, DIMER-source fetch, worker or credentials.
- **Dataset:** `chronos_multi_series.csv`, regenerated inline and asserted against SHA-256
  `6f12f1475411aaab13a2de860ba50f008bd7185a9266f99147c244db4a5de8b3`, the digest that
  `examples/sample-data/generate_samples.py` produces (`examples/sample-data/SHA256SUMS.generated`).
- **Tiers:** `STANDARD` (default) runs TiRex-2 and Chronos-2; `FULL` adds Toto 2.0 2.5B (9.8 GB checkpoint, CUDA required).
- **Verification coverage (REL9):** CI does not execute this notebook. Clean-runtime `Run all` evidence for both tiers is pending and must be recorded before the status moves past `candidate`.
