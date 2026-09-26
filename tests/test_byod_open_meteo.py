"""Offline checks for the Open-Meteo BYOD sample used by the multi-model workshop."""

from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "examples" / "byod-data" / "open-meteo-ph-temperature"
CSV = SAMPLE_DIR / "openmeteo_ph_hourly_temperature.csv"


def load_fetcher():
    spec = importlib.util.spec_from_file_location(
        "fetch_open_meteo", SAMPLE_DIR / "fetch_open_meteo.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_csv_matches_its_manifest() -> None:
    lines = (SAMPLE_DIR / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    digest, name = lines[0].split(maxsplit=1)
    assert name.strip() == CSV.name
    assert hashlib.sha256(CSV.read_bytes()).hexdigest() == digest


def test_csv_satisfies_the_workshop_byod_contract() -> None:
    raw = CSV.read_bytes()
    assert b"\r" not in raw
    frame = pd.read_csv(CSV, parse_dates=["timestamp"])
    assert list(frame.columns) == ["series_id", "timestamp", "target"]
    assert list(frame["series_id"].drop_duplicates()) == ["manila", "cebu", "davao"]
    assert frame["target"].map(math.isfinite).all()

    grids = [block["timestamp"].tolist() for _, block in frame.groupby("series_id", sort=False)]
    assert all(grid == grids[0] for grid in grids)
    expected = pd.date_range("2026-08-26 00:00", "2026-09-08 23:00", freq="h").tolist()
    assert grids[0] == expected


def test_fetcher_transform_reproduces_committed_bytes() -> None:
    """Rebuild the API payload from the CSV and check the script's own transform round-trips."""
    fetcher = load_fetcher()
    frame = pd.read_csv(CSV, dtype={"timestamp": str})
    payload = []
    for series_id in fetcher.LOCATIONS:
        block = frame[frame["series_id"] == series_id]
        payload.append({
            "utc_offset_seconds": 0,
            "hourly_units": {fetcher.VARIABLE: "°C"},
            "hourly": {
                "time": [ts[:-3] for ts in block["timestamp"]],
                fetcher.VARIABLE: block["target"].tolist(),
            },
        })
    assert fetcher.to_csv_bytes(fetcher.to_rows(payload)) == CSV.read_bytes()


def test_dataset_card_records_licence_digest_and_request() -> None:
    card = (SAMPLE_DIR / "DATASET_CARD.md").read_text(encoding="utf-8")
    digest = (SAMPLE_DIR / "SHA256SUMS").read_text(encoding="utf-8").split()[0]
    assert digest in card
    assert "CC BY 4.0" in card
    assert "Open-Meteo.com" in card
    assert load_fetcher().request_url() in card
