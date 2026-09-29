# Reef numerical core build report

Implemented tools/reef_core.py and tests/test_reef_core.py. No git mutations.

Confirmed: 15 CPU tests pass with exit 0, using --noconftest because the adjacent environment lacks the host pipeline package imported by repository conftest. These checks cover schema/date/negative/nonfinite refusals, missing dates, early validity, inclusive HotSpot >=1, 84-day expiration, seasonal leap days and frozen development fit, future perturbation, split boundaries, empty-event nulls, quantile ordering, missing model outputs and CSV reload equality.

Actual five NOAA source files independently parsed: 115 eligible validation and 184 test origins. Each region has23 in2024; Northern/Central/Eastern have23 each in2025; Western/Southern have0. Calendar-aware DHW reconstruction max absolute error4.285714285856557e-05 on complete valid windows, below0.0001 tolerance.

Core uses no model or project imports and can be embedded verbatim. Main panel index is(region_id,date). Quantile input is28-by-3 forq10/q50/q90; point input must equal raw median. Returned evaluate_forecasts tables are JSON-safe lists: metrics, macro, thresholds, per_origin_errors, paired_differences. Pass expected_arms=['persistence','seasonal','chronos'] at full experiment verification.

No hosted run/model inference asserted. Parent owns runtime, independent subprocess verification, immutable acquisition, notebook and documentation integration. Optional availability-delay experiment remains outside this core. Annual horizon boundary matches the frozen spec's reported23 origins per region/year.
