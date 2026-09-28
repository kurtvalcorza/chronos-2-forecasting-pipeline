# Philippine regional reef heat-stress dataset

This capstone uses a frozen snapshot of **NOAA Coral Reef Watch (CRW) v3.1
Regional Virtual Stations**. It contains satellite-derived regional indicators,
not observations of coral bleaching, mortality, fisheries loss, or reef condition.

## Source and rights

NOAA CRW states that its website content is public domain and may be freely
distributed. Retain NOAA CRW attribution; this data status is separate from the
repository's code license. See the [NOAA citation and reuse guidance](https://coralreefwatch.noaa.gov/satellite/docs/recommendations_crw_citation.php).

Citation: NOAA Coral Reef Watch. 2019. NOAA Coral Reef Watch Version 3.1 Daily
Global 5km Satellite Coral Bleaching Heat Stress Monitoring Products: Regional
Virtual Stations, Northern, Central, Western, Eastern and Southern Philippines;
1 January 1985 to 26 September 2026. NOAA Coral Reef Watch, College Park,
Maryland, USA. Accessed 28 September 2026.
[Regional station catalogue](https://coralreefwatch.noaa.gov/product/vs/data.php).

The original acquisition timestamp is **2026-09-28T05:26:03.426497+00:00**.
All five files cover **1985-01-01 through 2026-09-26**.

| Region | Rows | Missing calendar days | Original source |
|---|---:|---:|---|
| Northern | 15,244 | 0 | [TXT](https://coralreefwatch.noaa.gov/product/vs/data/northern_philippines.txt) |
| Central | 15,244 | 0 | [TXT](https://coralreefwatch.noaa.gov/product/vs/data/central_philippines.txt) |
| Western | 15,219 | 25 | [TXT](https://coralreefwatch.noaa.gov/product/vs/data/western_philippines.txt) |
| Eastern | 15,244 | 0 | [TXT](https://coralreefwatch.noaa.gov/product/vs/data/eastern_philippines.txt) |
| Southern | 15,219 | 25 | [TXT](https://coralreefwatch.noaa.gov/product/vs/data/southern_philippines.txt) |

Total: **76,170 rows, 6,324,302 original source bytes**. Western and Southern both
omit 2025-01-05, 2025-01-13, 2025-10-02 through 2025-10-23, and 2026-03-24.
These are properties of this snapshot, not promises about NOAA's growing files.

## Immutable packaging and implementation amendment

The checked-in [archive](../../tutorials/data/reef/noaa_crw_philippines_2026-09-28.zip)
contains exactly five flat `<region>_philippines.txt` members, preserving the
original downloaded bytes. The [manifest](../../tutorials/data/reef/manifest.json)
records every source URL, member size and SHA-256, and archive identity. The
[original audit](../../tutorials/data/reef/audit.json) is preserved byte-for-byte.

- Archive size: **1,040,559 bytes**.
- Archive SHA-256: `320a04998ff0f0ce4a66264fc83b585c0fafd653844d533468d98a84d6d07b43`.
- Dataset identity: `noaa-crw-rvs-v3.1-philippines-2026-09-28`.

**Implementation amendment:** the notebook carries this immutable archive as
base64-encoded data, replacing the specification's proposed remote asset
download. This makes the standalone notebook usable before publication without
inventing an immutable public URL. Archive and member hashes are checked before
parsing; no executable repository source is downloaded. A checksum failure must
stop execution. The notebook must not silently substitute the live NOAA files.

An optional future live refresh is a different dataset version and requires a
new audit, manifest, and experiment identity. The source URLs above describe
provenance; they do not identify immutable data.

[build_snapshot.py](../../tutorials/data/reef/build_snapshot.py) rebuilds the
archive from a directory holding the original five TXT files and audit.json.
It verifies audit hashes, sizes, row counts, and region membership first. ZIP
member order, timestamps, file mode, and compression settings are fixed.
Byte-identical rebuilding was checked in the build environment; deflate output
across different zlib implementations is not promised. The distributed archive
digest remains authoritative.

## Scientific interpretation and validation

Each daily representative SST, SSTA and HotSpot comes from the pixel at the
region's **90th-percentile HotSpot**. That pixel can change daily. These values
are not regional-mean temperatures or fixed sensor measurements. Header
coordinates are regional reference points, not locations of physical sensors.
Do not subtract the regionally averaged maximum monthly mean from representative
SST to reconstruct HotSpot: those quantities need not describe the same pixel.
[Regional methodology](https://coralreefwatch.noaa.gov/product/vs/map.php).

The numeric columns are `SST_MIN`, `SST_MAX`, `SST@90th_HS`, `SSTA@90th_HS`,
`90th_HS>0`, `DHW_from_90th_HS>1`, and `BAA_7day_max`. Dates have separate year,
month and day fields. Header validity dates use **year/day/month** ordering.
Mask DHW before **1985-03-25** and BAA before **1985-03-31**; early numeric zeros
do not establish validity.

Despite the DHW column's `>1` text, reconstruction uses HotSpot **greater than
or equal to 1 degree Celsius**, accumulated over 84 **calendar** days including
the target date and divided by 7. Calendar-aware reconstruction agrees within
0.0001 degree-Celsius-weeks on complete windows. Reindex missing dates explicitly;
never collapse gaps, zero-fill missing observations, or interpolate evaluation
outcomes. [Global product methodology](https://coralreefwatch.noaa.gov/product/5km/methodology.php).

BAA is the legacy regional category scheme. Do not label it as the newer global
alert scheme. DHW thresholds of 4 and 8 are stress indicators, not observed
bleaching labels or full official alert categories.

The archive is reprocessed historical data and does not establish what was
available on each historic forecast date. The experiment is retrospective.
Under the specified 365-day complete-context rule, Western and Southern have
zero eligible 2025 test origins; the notebook must show the common 2024 panel
separately from the three-region 2025 panel.

## Evidence scope

CPU packaging checks verify member hashes against the original audit, all
reported dates and gaps, numeric finiteness, no duplicate dates, complete-window
DHW reconstruction, and deterministic rebuilding in the current environment.
The data build does not establish notebook/model execution or hosted Colab
qualification. See [data build report](../reef-data-build-report.json).
