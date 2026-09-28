"""Scientific contracts for the exploratory published-source-block benchmark.

Indices refer to row order, never verified dates. Lag zero is the latest
available observation; lag one is the immediately preceding source block.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

CASE_LAGS = (0, 1, 2, 3, 4, 8, 12, 52)
WEATHER_LAGS = (0, 1, 2, 3, 4)


def validate_rows(rows: list[dict[str, Any]]) -> None:
    """Reject incomplete, unordered source cycles and missing selected values."""
    if len(rows) < 260 or len(rows) % 52:
        raise ValueError("Need at least five complete 52-block source years")
    first = rows[0]["year"]
    if isinstance(first, bool) or int(first) != first:
        raise ValueError("Year must be integral")
    for i, row in enumerate(rows):
        year, block = int(first) + i // 52, 1 + i % 52
        if row.get("year") != year or row.get("block") != block:
            raise ValueError(f"Missing, repeated or unordered source block at row {i}")
        for name in ("cases", "rain", "temp"):
            value = row.get(name)
            if value is None or isinstance(value, bool):
                raise ValueError(f"Missing/invalid {name} at row {i}; missing is not zero")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid {name} at row {i}") from exc
            if not np.isfinite(numeric) or (name != "temp" and numeric < 0):
                raise ValueError(f"Invalid {name} at row {i}")


def origins(rows: list[dict[str, Any]], years: list[int]) -> list[int]:
    """Return issuance row indices before target blocks 1, 5, ..., 49."""
    validate_rows(rows)
    selected = set(years)
    if not selected.issubset({r["year"] for r in rows}):
        raise ValueError("Requested source year is absent")
    result = [
        i - 1 for i, r in enumerate(rows) if r["year"] in selected and (r["block"] - 1) % 4 == 0
    ]
    if any(i < 52 for i in result):
        raise ValueError("Requested origin has insufficient case-lag history")
    return result


def target_key(rows: list[dict[str, Any]], index: int) -> tuple[int, int]:
    """Map row index, including future blocks, without inspecting target values."""
    if index < 0:
        raise ValueError("Negative source-block index")
    return int(rows[0]["year"]) + index // 52, index % 52 + 1


def feature_names(weather: bool = False) -> list[str]:
    names = [f"cases_lag_{lag}" for lag in CASE_LAGS]
    names += [f"cases_mean_{n}" for n in (4, 8, 12)]
    names += ["target_block_sin", "target_block_cos"]
    if weather:
        for variable in ("rain", "temp"):
            names += [f"{variable}_lag_{lag}" for lag in WEATHER_LAGS]
            names += [f"{variable}_mean_{n}" for n in (4, 8)]
    return names


def feature_row(
    rows: list[dict[str, Any]], origin: int, horizon: int, delay: int = 0, weather: bool = False
) -> np.ndarray:
    """Read only observations through origin-delay; target calendar is known."""
    if horizon not in (1, 2, 3, 4) or delay < 0 or int(delay) != delay:
        raise ValueError("Horizon must be 1..4 and delay a nonnegative integer")
    cutoff = origin - delay
    if cutoff < 52 or origin >= len(rows):
        raise ValueError("Insufficient history or unknown issuance block")
    values = [float(rows[cutoff - lag]["cases"]) for lag in CASE_LAGS]
    values += [
        float(np.mean([rows[i]["cases"] for i in range(cutoff - n + 1, cutoff + 1)]))
        for n in (4, 8, 12)
    ]
    _, block = target_key(rows, origin + horizon)
    angle = 2 * np.pi * (block - 1) / 52
    values += [float(np.sin(angle)), float(np.cos(angle))]
    if weather:
        for variable in ("rain", "temp"):
            values += [float(rows[cutoff - lag][variable]) for lag in WEATHER_LAGS]
            values += [
                float(np.mean([rows[i][variable] for i in range(cutoff - n + 1, cutoff + 1)]))
                for n in (4, 8)
            ]
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Selected features contain nonfinite values")
    return result


def training_task(
    rows: list[dict[str, Any]],
    origin: int,
    horizon: int,
    window: int = 104,
    delay: int = 0,
    weather: bool = False,
) -> dict[str, Any]:
    """Return mature direct-forecast examples from trailing issuance blocks.

    y is log1p(cases); query is one unscaled feature vector. Transforms must be
    fit on X alone. The first of 104 issuance blocks is origin-103.
    """
    query = feature_row(rows, origin, horizon, delay, weather)
    if window < 1 or int(window) != window:
        raise ValueError("Window must be a positive integer")
    cutoff = origin - delay
    indices = np.asarray(
        [
            s
            for s in range(max(52 + delay, origin - window + 1), origin + 1)
            if s + horizon <= cutoff
        ],
        dtype=np.int64,
    )
    if not len(indices):
        raise ValueError("No mature supervised examples")
    targets = indices + horizon
    y = np.asarray([rows[int(i)]["cases"] for i in targets], dtype=float)
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Missing/invalid mature target; never replace with zero")
    return {
        "X": np.stack([feature_row(rows, int(s), horizon, delay, weather) for s in indices]),
        "y": np.log1p(y),
        "query": query,
        "issuance_indices": indices,
        "target_indices": targets,
        "feature_names": feature_names(weather),
        "cutoff": cutoff,
    }


def _paired(ref: Any, pred: Any) -> tuple[np.ndarray, np.ndarray]:
    a, b = np.asarray(ref, dtype=float), np.asarray(pred, dtype=float)
    if not a.size or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Expected nonempty paired finite arrays of identical shape")
    return a, b


def metrics(ref: Any, pred: Any) -> dict[str, float | int]:
    a, b = _paired(ref, pred)
    error = b - a
    return {
        "n": int(a.size),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "bias": float(error.mean()),
    }


def skill(mae: float, baseline_mae: float) -> float | None:
    if not np.isfinite([mae, baseline_mae]).all() or min(mae, baseline_mae) < 0:
        raise ValueError("MAE inputs must be finite and nonnegative")
    return None if baseline_mae == 0 else 1 - float(mae) / float(baseline_mae)


def inverse_counts(log_predictions: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(log_predictions, dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        raw = np.expm1(values)
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite inverse-transformed prediction")
    return raw, np.maximum(raw, 0), raw < 0


def interval_metrics(ref: Any, quantiles: Any) -> dict[str, float | int]:
    y, q = np.asarray(ref, dtype=float), np.asarray(quantiles, dtype=float)
    if (
        q.shape != y.shape + (3,)
        or not y.size
        or not np.isfinite(q).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("Expected finite q0.1/q0.5/q0.9 for each observation")
    if (np.diff(q, axis=-1) < 0).any():
        raise ValueError("Crossing quantiles refused")
    error = y[..., None] - q
    levels = np.asarray([0.1, 0.5, 0.9])
    return {
        "n": int(y.size),
        "median_mae": float(np.abs(y - q[..., 1]).mean()),
        "coverage_80": float(((y >= q[..., 0]) & (y <= q[..., 2])).mean()),
        "mean_width": float((q[..., 2] - q[..., 0]).mean()),
        "pinball_loss": float(np.maximum(levels * error, (levels - 1) * error).mean()),
    }


def paired_bootstrap(
    ref: Any, a: Any, b: Any, n_boot: int = 2000, seed: int = 42, group_size: int = 4
) -> dict[str, Any]:
    """Resample fixed contiguous four-origin groups, retaining all horizons.

    Arrays are chronological [origins, 4]. Last shorter group is retained.
    Difference is B minus A: negative favours B. No significance claim.
    """
    y, pa = _paired(ref, a)
    _, pb = _paired(ref, b)
    if y.ndim != 2 or y.shape[1] != 4 or n_boot < 1 or group_size < 1:
        raise ValueError("Require [origins,4], positive replicates and group size")
    groups = [np.arange(i, min(i + group_size, len(y))) for i in range(0, len(y), group_size)]
    if len(groups) < 2:
        raise ValueError("At least two contiguous groups required")
    delta = np.abs(pb - y) - np.abs(pa - y)
    rng = np.random.default_rng(seed)
    replicates = np.asarray(
        [
            delta[
                np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
            ].mean()
            for _ in range(n_boot)
        ]
    )
    return {
        "difference_b_minus_a": float(delta.mean()),
        "ci95": np.quantile(replicates, [0.025, 0.975]).tolist(),
        "n_origins": len(y),
        "n_pairs": int(y.size),
        "n_groups": len(groups),
        "group_sizes": [len(g) for g in groups],
        "replicates": n_boot,
        "seed": seed,
    }


def digest_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
