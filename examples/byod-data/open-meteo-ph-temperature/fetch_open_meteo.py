"""Fetch the Open-Meteo hourly temperature sample used as real-data BYOD for the workshop.

Standard library only, so it runs in any Python 3.10+ without project dependencies.

    python examples/byod-data/open-meteo-ph-temperature/fetch_open_meteo.py
    python examples/byod-data/open-meteo-ph-temperature/fetch_open_meteo.py --check

The default writes ``openmeteo_ph_hourly_temperature.csv`` next to this script and prints its
SHA-256. ``--check`` fetches into memory and compares against the committed file without
writing anything. Open-Meteo serves reanalysis data that its upstream providers can revise, so
a later fetch may not be byte-identical; the committed CSV and ``SHA256SUMS`` are the reference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "openmeteo_ph_hourly_temperature.csv"
ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"

# Requested city-centre coordinates; Open-Meteo snaps each to its nearest grid cell.
LOCATIONS = {
    "manila": (14.5995, 120.9842),
    "cebu": (10.3157, 123.8854),
    "davao": (7.1907, 125.4553),
}
START_DATE = "2026-08-26"
END_DATE = "2026-09-08"
VARIABLE = "temperature_2m"


def request_url() -> str:
    params = {
        "latitude": ",".join(str(lat) for lat, _ in LOCATIONS.values()),
        "longitude": ",".join(str(lon) for _, lon in LOCATIONS.values()),
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": VARIABLE,
        "timezone": "UTC",
    }
    return f"{ENDPOINT}?{urllib.parse.urlencode(params, safe=',')}"


def fetch() -> list[dict]:
    with urllib.request.urlopen(request_url(), timeout=60) as response:
        payload = json.load(response)
    if not isinstance(payload, list) or len(payload) != len(LOCATIONS):
        raise ValueError(f"expected one response per location, got {type(payload).__name__}")
    return payload


def to_rows(payload: list[dict]) -> list[tuple[str, str, float]]:
    start = datetime.fromisoformat(START_DATE)
    n_hours = ((datetime.fromisoformat(END_DATE) - start).days + 1) * 24
    expected_grid = [
        (start + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M") for h in range(n_hours)
    ]

    rows = []
    for series_id, location in zip(LOCATIONS, payload, strict=True):
        if location.get("utc_offset_seconds") != 0:
            raise ValueError(f"{series_id}: response is not in UTC")
        if location["hourly_units"][VARIABLE] != "°C":
            raise ValueError(f"{series_id}: unexpected unit {location['hourly_units'][VARIABLE]!r}")
        times = location["hourly"]["time"]
        values = location["hourly"][VARIABLE]
        # Refuse rather than repair: no gaps, no resampling, no imputation.
        if times != expected_grid:
            raise ValueError(f"{series_id}: timestamps are not the complete hourly UTC grid")
        bad = [t for t, v in zip(times, values, strict=True) if v is None or not math.isfinite(v)]
        if bad:
            raise ValueError(
                f"{series_id}: {len(bad)} missing or non-finite values, first at {bad[0]}"
            )
        rows.extend((series_id, f"{t}:00", float(v)) for t, v in zip(times, values, strict=True))
    return rows


def to_csv_bytes(rows: list[tuple[str, str, float]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["series_id", "timestamp", "target"])
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="compare a fresh fetch with the committed CSV"
    )
    args = parser.parse_args()

    payload = fetch()
    data = to_csv_bytes(to_rows(payload))
    digest = hashlib.sha256(data).hexdigest()
    for series_id, location in zip(LOCATIONS, payload, strict=True):
        print(
            f"{series_id}: grid cell {location['latitude']:.4f}, {location['longitude']:.4f}, "
            f"elevation {location['elevation']} m"
        )

    if args.check:
        committed = OUTPUT.read_bytes()
        if committed == data:
            print(f"OK: fresh fetch matches {OUTPUT.name} ({digest})")
            return 0
        print(
            f"DIFFERS: fresh fetch {digest} != committed {hashlib.sha256(committed).hexdigest()}; "
            "upstream reanalysis values may have been revised",
            file=sys.stderr,
        )
        return 1

    OUTPUT.write_bytes(data)
    print(f"wrote {OUTPUT.name}: {len(data)} bytes, sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
