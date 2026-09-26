# Open-Meteo Hourly Temperature, Philippine Cities (BYOD sample)

## Summary

This is a small **real-data** sample for the "Bring your own data" (BYOD) path of
`tutorials/DIMER_MultiModel_TimeSeries_Forecasting_Workshop.ipynb`. It holds hourly air
temperature at 2 m for Manila, Cebu and Davao over 14 days: 2026-08-26 00:00 to 2026-09-08
23:00 UTC.

It complements the synthetic `chronos_multi_series.csv` fixture. The fixture is deterministic
and noise-free. This sample carries real weather, so a foundation model has no guaranteed
advantage over a seasonal-naive baseline.

This file is third-party data. It is **not** covered by the repository's Apache-2.0 licence and
is not part of the synthetic samples described in `examples/sample-data/DATASET_CARD.md`.

## Source and licence

- **Provider:** [Open-Meteo](https://open-meteo.com/) historical weather API
  (`https://archive-api.open-meteo.com/v1/archive`).
- **Licence:** Open-Meteo publishes its API data under
  [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
  Any use or redistribution must credit Open-Meteo, for example:
  "Weather data by [Open-Meteo.com](https://open-meteo.com/)".
- **Upstream data:** Open-Meteo builds its historical API from weather reanalysis and model
  datasets, including ECMWF ERA5 from the Copernicus Climate Change Service. Check the current
  terms on the [Open-Meteo licence page](https://open-meteo.com/en/licence) before you
  redistribute the file outside this repository.
- **Personal data:** none. The values are gridded weather estimates, not measurements tied to
  any person, station operator or property.

## How it was created

The file is produced by [`fetch_open_meteo.py`](fetch_open_meteo.py). That script uses only the
Python standard library and needs no API key. It was run on 2026-09-26 with this request:

```text
https://archive-api.open-meteo.com/v1/archive?latitude=14.5995,10.3157,7.1907&longitude=120.9842,123.8854,125.4553&start_date=2026-08-26&end_date=2026-09-08&hourly=temperature_2m&timezone=UTC
```

| `series_id` | Requested point (lat, lon) | Grid cell returned (lat, lon) | Grid elevation |
|---|---|---|---:|
| `manila` | 14.5995, 120.9842 | 14.5870, 121.0028 | 9 m |
| `cebu` | 10.3157, 123.8854 | 10.2988, 123.8489 | 39 m |
| `davao` | 7.1907, 125.4553 | 7.2056, 125.4822 | 194 m |

Open-Meteo snaps each requested point to its nearest model grid cell. The values describe that
grid cell, not a single weather station.

The script refuses the response instead of repairing it if any of these hold:

- a location is not in UTC;
- the unit is not °C;
- the timestamps are not the complete hourly grid for the date range;
- any value is missing or non-finite.

It applies no imputation, interpolation, resampling, smoothing or scaling. The only
transformations are:

1. converting the API's column-per-location JSON into the notebook's long CSV format
   (`series_id,timestamp,target`); and
2. adding `:00` seconds to each timestamp, for example `2026-08-26T00:00` becomes
   `2026-08-26T00:00:00`.

Temperatures are written exactly as the API returned them: one decimal place, in °C. The file
uses LF line endings.

To regenerate or verify the file:

```bash
python examples/byod-data/open-meteo-ph-temperature/fetch_open_meteo.py          # rewrite the CSV
python examples/byod-data/open-meteo-ph-temperature/fetch_open_meteo.py --check  # compare only
```

Reanalysis providers sometimes revise recent values, so a later fetch may not match byte for
byte. The committed CSV and [`SHA256SUMS`](SHA256SUMS) are the reference. On 2026-09-26, a
second fetch matched the committed file exactly.

## Contents

| Artifact | Rows | Series | Frequency | Target | Covariates |
|---|---:|---:|---|---|---|
| `openmeteo_ph_hourly_temperature.csv` | 1,008 | 3 | hourly, UTC | `target` (air temperature at 2 m, °C) | none |

| `series_id` | Hours | Min °C | Mean °C | Max °C |
|---|---:|---:|---:|---:|
| `manila` | 336 | 24.5 | 27.3 | 31.7 |
| `cebu` | 336 | 25.7 | 29.5 | 34.4 |
| `davao` | 336 | 22.0 | 26.6 | 31.8 |

SHA-256: `74163ee609cda87869b7c13f4c2aa59343f94b4ba42d2e57331034902fe04f1a`

All three series share one timestamp grid. Timestamps are naive ISO-8601 in UTC, so local
Philippine time (UTC+8) is 8 hours later. The daily temperature peak therefore falls near
04:00–07:00 UTC.

## Use in the workshop notebook

In Colab, upload the CSV and set these controls:

```python
BYOD_CSV_PATH = "/content/openmeteo_ph_hourly_temperature.csv"
VALIDATION_HORIZON = 24
TEST_HORIZON = 24
SEASON_LENGTH = 24
```

With the default horizons, the notebook splits the 14 days chronologically:

- **Validation:** forecast from 2026-08-26 00:00 to 2026-09-06 23:00, scored on 2026-09-07.
- **Test:** forecast from history up to 2026-09-07 23:00, scored on 2026-09-08.
- **Future forecast:** 2026-09-09, which is not measurable against this file.

### Recorded execution

The notebook's `SAMPLE_DATASET = "OPEN_METEO_PH"` option embeds this file and checks its digest.
With that option, the STANDARD tier (TiRex-2 on CPU, Chronos-2 on CUDA) passed in Google Colab
on a Tesla T4 on 2026-09-26, at commit `129142b`, notebook blob `13e92f12ef0b`. See
`docs/release-verification.md`. The same file had earlier passed through the BYOD path in a
local `nbclient` run with CPU-only PyTorch. Both runs gave identical metrics.

Test window, 2026-09-08 (macro-averaged over the three cities):

| Model | MAE (°C) | RMSE (°C) | Skill vs seasonal-naive | q10–q90 coverage | Mean interval width (°C) |
|---|---:|---:|---:|---:|---:|
| Chronos-2 | 0.450 | 0.579 | −0.034 | 0.764 | 1.392 |
| TiRex-2 | 0.463 | 0.593 | −0.061 | 0.819 | 1.601 |
| seasonal-naive | 0.496 | 0.610 | 0 | — | — |
| last-value | 1.804 | 2.243 | −3.796 | — | — |

On the validation day (2026-09-07), both models beat seasonal-naive: Chronos-2 had skill
+0.130 and TiRex-2 had +0.046. On the test day, both fell slightly behind it. The notebook's
Exercise B covers exactly this case: a strong daily cycle can make a simple baseline hard to
beat.

## Pretraining overlap

The observation window starts after the release of every model in the comparison: TiRex-2,
Chronos-2 and Toto 2.0. These specific values therefore cannot appear in their pretraining
data. The models may still have been pretrained on weather data from the same sources for
earlier periods. The file tests generalisation to new dates, not to an unseen domain.

## Intended use and limitations

- Use it for workshop exercises, BYOD demonstrations and smoke runs on real, noisy data.
- It is **not a benchmark.** Three grid cells, one validation day and one test day cannot rank
  forecasting models or establish calibration.
- The values are reanalysis estimates for model grid cells several kilometres across. They can differ from
  station readings, especially near coasts and in terrain, as for Davao.
- Two weeks in the southwest-monsoon season do not show seasonal or year-to-year variation.
