"""Execute the notebook's lightweight BYOD and optional learning paths.

These checks use real notebook cells, but do not load foundation models and do
not claim hosted-runtime or full REL12 evidence.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials/DIMER_MultiModel_TimeSeries_Forecasting_Workshop.ipynb"


def source(title):
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    return next(
        "".join(cell["source"])
        for cell in cells
        if "".join(cell["source"]).startswith(f"# @title {title}\n")
    )


def setup_byod(tmp_path, monkeypatch, payload):
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "input.csv"
    input_path.write_bytes(payload)
    namespace = {"display": lambda *_: None}
    exec(source("1.1 Notebook controls"), namespace)
    # With a location set, even USE_BYOD=True must bypass google.colab upload.
    namespace.update(BYOD_CSV_PATH=str(input_path), USE_BYOD=True)
    exec(source("3.1 Generate or load the sample"), namespace)
    return namespace


def sample_bytes():
    # A user-shaped fixture independent of the notebook's embedded sample.
    hours = np.arange(96)
    return pd.concat([
        pd.DataFrame({
            "series_id": sid,
            "timestamp": pd.date_range("2026-01-01", periods=96, freq="h"),
            "target": offset + hours * 0.1 + np.sin(hours * 2 * np.pi / 24),
        })
        for sid, offset in [("A", 10), ("B", 30)]
    ]).to_csv(index=False).encode()


def test_byod_path_reaches_validation_splits_and_real_baseline_scoring(tmp_path, monkeypatch):
    ns = setup_byod(tmp_path, monkeypatch, sample_bytes())
    exec(source("3.2 Check the data before any model download"), ns)
    exec(source("5.1 Split the data, build the baselines and define the metrics"), ns)
    assert ns["sample_kind"] == "BYOD"
    assert ns["panel"]["n_series"] == 2
    assert len(ns["validation_truth"]) == 48
    scored, _, aggregate = ns["evaluate_forecasts"](
        ns["validation_seasonal"], ns["validation_truth"], ns["validation_seasonal"]
    )
    assert len(scored) == 48
    assert np.isfinite(aggregate["macro_mae"]).all()
    assert (tmp_path / "outputs/data/dataset_manifest.json").exists()


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("missing_target", "Missing required columns"),
        ("duplicate_time", "Duplicate .* observations"),
        ("gap", "irregular or gappy"),
        ("non_finite", "missing or non-finite"),
    ],
)
def test_byod_rejects_incompatible_input_before_models(tmp_path, monkeypatch, corruption, message):
    frame = pd.read_csv(io.BytesIO(sample_bytes()))
    if corruption == "missing_target":
        frame = frame.drop(columns="target")
    elif corruption == "duplicate_time":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif corruption == "gap":
        frame = frame.drop(index=1)
    else:
        frame.loc[0, "target"] = np.inf
    ns = setup_byod(tmp_path, monkeypatch, frame.to_csv(index=False).encode())
    with pytest.raises(ValueError, match=message):
        exec(source("3.2 Check the data before any model download"), ns)
    assert not (tmp_path / "outputs/data/dataset_manifest.json").exists()


def test_activity_is_optional_and_preserves_the_frozen_comparison(tmp_path, monkeypatch):
    ns = setup_byod(tmp_path, monkeypatch, sample_bytes())
    exec(source("3.2 Check the data before any model download"), ns)
    exec(source("5.1 Split the data, build the baselines and define the metrics"), ns)
    baseline_before = ns["validation_seasonal"].copy(deep=True)
    history_before = ns["validation_history"].copy(deep=True)
    truth_before = ns["validation_truth"].copy(deep=True)
    canonical_files = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    # Sentinel test values must not affect a validation-only experiment.
    ns["test_truth"] = object()
    ns["test_history"] = object()
    activity = source("Optional validation-only seasonal-period comparison")
    exec(activity, ns)
    assert "seasonal_activity" not in ns
    assert not (tmp_path / "outputs/activities").exists()
    exec(activity.replace("RUN_SEASONAL_ACTIVITY = False", "RUN_SEASONAL_ACTIVITY = True"), ns)
    result = ns["seasonal_activity"]
    assert len(result) == 4
    for row in result.itertuples(index=False):
        history = history_before[history_before.series_id == row.series_id].target.to_numpy()
        truth = truth_before[truth_before.series_id == row.series_id].target.to_numpy()
        expected = np.mean(np.abs(truth - np.resize(history[-row.season_length:], len(truth))))
        assert row.mae == pytest.approx(expected)
    pd.testing.assert_frame_equal(ns["validation_history"], history_before)
    pd.testing.assert_frame_equal(ns["validation_truth"], truth_before)
    pd.testing.assert_frame_equal(ns["validation_seasonal"], baseline_before)
    assert ns["SEASON_LENGTH"] == 24
    assert all(path.read_bytes() == contents for path, contents in canonical_files.items())
    assert (tmp_path / "outputs/activities/seasonal_period_comparison.csv").exists()
