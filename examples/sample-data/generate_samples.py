"""Generate the deterministic synthetic datasets used by the Chronos-2 tutorial."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CHECKED_IN = frozenset({"chronos_univariate.csv"})


def build_samples() -> dict[str, pd.DataFrame]:
    """Return all tutorial samples from fixed formulas; no random state is used."""

    n = 96
    timestamps = pd.date_range("2026-01-01", periods=n, freq="h")
    step = np.arange(n, dtype=float)

    univariate = pd.DataFrame(
        {
            "series_id": "A",
            "timestamp": timestamps,
            "target": (
                100.0
                + 0.25 * step
                + 7.0 * np.sin(2.0 * np.pi * step / 24.0)
                + 1.5 * np.cos(2.0 * np.pi * step / 12.0)
            ),
        }
    )

    multi_parts = []
    for series_id, offset, slope, amplitude, phase in (
        ("A", 0.0, 0.18, 6.0, 0.0),
        ("B", 35.0, 0.12, 9.0, 5.0),
    ):
        target = (
            120.0
            + offset
            + slope * step
            + amplitude * np.sin(2.0 * np.pi * (step + phase) / 24.0)
        )
        multi_parts.append(
            pd.DataFrame(
                {
                    "series_id": series_id,
                    "timestamp": timestamps,
                    "target": target,
                }
            )
        )
    multi_series = pd.concat(multi_parts, ignore_index=True)

    history_parts = []
    future_parts = []
    horizon = 24
    for series_id, base, phase in (("A", 80.0, 0.0), ("B", 110.0, 4.0)):
        temperature = 27.0 + 4.0 * np.sin(2.0 * np.pi * (step + phase) / 24.0)
        holiday = (pd.Series(timestamps).dt.dayofweek >= 5).astype(int).to_numpy()
        demand = (
            base
            + 1.8 * temperature
            + 10.0 * holiday
            + 0.08 * step
            + 3.0 * np.sin(2.0 * np.pi * step / 12.0)
        )
        history_parts.append(
            pd.DataFrame(
                {
                    "series_id": series_id,
                    "timestamp": timestamps,
                    "demand": demand,
                    "temperature": temperature,
                    "holiday": holiday,
                }
            )
        )

        future_step = np.arange(n, n + horizon, dtype=float)
        future_timestamps = pd.date_range(
            timestamps[-1] + pd.Timedelta(hours=1),
            periods=horizon,
            freq="h",
        )
        future_temperature = 27.0 + 4.0 * np.sin(
            2.0 * np.pi * (future_step + phase) / 24.0
        )
        future_holiday = (
            pd.Series(future_timestamps).dt.dayofweek >= 5
        ).astype(int).to_numpy()
        future_parts.append(
            pd.DataFrame(
                {
                    "series_id": series_id,
                    "timestamp": future_timestamps,
                    "temperature": future_temperature,
                    "holiday": future_holiday,
                }
            )
        )

    return {
        "chronos_univariate.csv": univariate,
        "chronos_multi_series.csv": multi_series,
        "chronos_covariates_history.csv": pd.concat(history_parts, ignore_index=True),
        "chronos_covariates_future.csv": pd.concat(future_parts, ignore_index=True),
    }


def csv_bytes(frame: pd.DataFrame) -> bytes:
    """Canonical checked-in byte representation."""

    text = frame.to_csv(
        index=False,
        date_format="%Y-%m-%dT%H:%M:%S",
        float_format="%.4f",
        lineterminator="\n",
    )
    return text.encode("utf-8")


def _manifest(digests: dict[str, str], names: set[str] | frozenset[str]) -> str:
    return "".join(f"{digests[name]}  {name}\n" for name in sorted(names))


def generate(root: Path = ROOT) -> dict[str, str]:
    """Write samples plus separate checked-in and generated-artifact manifests."""

    root.mkdir(parents=True, exist_ok=True)
    digests: dict[str, str] = {}
    samples = build_samples()
    for name, frame in samples.items():
        payload = csv_bytes(frame)
        (root / name).write_bytes(payload)
        digests[name] = hashlib.sha256(payload).hexdigest()

    generated_only = set(samples) - set(CHECKED_IN)
    (root / "SHA256SUMS").write_text(
        _manifest(digests, CHECKED_IN),
        encoding="utf-8",
        newline="\n",
    )
    (root / "SHA256SUMS.generated").write_text(
        _manifest(digests, generated_only),
        encoding="utf-8",
        newline="\n",
    )
    return digests


if __name__ == "__main__":
    generate()
