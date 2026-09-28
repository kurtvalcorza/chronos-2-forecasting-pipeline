"""CPU contract checks for calendar, leakage and heat accumulation."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "reef_core", Path(__file__).parents[1] / "tools/reef_core.py"
)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def panel():
    dates = pd.date_range("2017-01-01", "2026-01-31")
    hs = np.where(dates.month >= 6, 1.5, 0.5)
    frame = pd.DataFrame(dict(region_id="region", date=dates, hotspot_c=hs))
    frame["dhw_c_weeks"] = core.reconstruct_dhw(pd.Series(hs, index=dates)).to_numpy()
    return core.daily_panel([frame])


def noaa():
    return (
        "Name:\nRegion\nFirst Valid DHW Date:\n1985 25 03\nFirst Valid BAA Date:\n1985 31 03\n"
        + " ".join(core.REEF_COLUMNS)
        + "\n1985 01 01 25 29 27 0 1 0 0\n1985 01 03 25 29 27 0 1 0 0\n"
    )


def test_parse_and_missing_calendar():
    f = core.parse_noaa(noaa(), "region")
    assert f.dhw_c_weeks.isna().all()
    assert f.raw_dhw_c_weeks.eq(0).all()
    p = core.daily_panel([f])
    assert len(p) == 3 and p.missing_day.sum() == 1
    assert np.isnan(p.loc[("region", "1985-01-02"), "hotspot_c"])


@pytest.mark.parametrize(
    "old,new",
    [
        ("SST_MIN", "bad"),
        ("1985 25 03", "1985 32 03"),
        ("1985 01 03", "1985 01 01"),
        ("1985 01 03", "1985 02 30"),
        ("27 0 1 0 0", "27 0 nan 0 0"),
        ("27 0 1 0 0", "27 0 -1 0 0"),
    ],
)
def test_noaa_refusals(old, new):
    with pytest.raises((ValueError, TypeError)):
        core.parse_noaa(noaa().replace(old, new), "region")


def test_dhw_threshold_boundary_and_gap():
    hs = pd.Series(np.zeros(170), index=pd.date_range("2020-01-01", periods=170))
    hs.iloc[:3] = [0.99, 1.0, 1.01]
    out = core.reconstruct_dhw(hs)
    assert out.iloc[:83].isna().all()
    assert out.iloc[83] == pytest.approx(2.01 / 7)
    assert out.iloc[85] == pytest.approx(1.01 / 7)
    assert out.iloc[86] == 0
    out = core.reconstruct_dhw(hs.drop(hs.index[90]))
    assert out.iloc[90:].isna().all()


def test_composed_known_and_new():
    history = np.ones(365)
    forecast = np.full(28, 1.01)
    result = core.compose_dhw(history, forecast)
    assert result["known"][0] == pytest.approx(83 / 7)
    assert result["known"][-1] == pytest.approx(56 / 7)
    assert result["predicted"][-1] == pytest.approx(28 * 1.01 / 7)
    assert np.allclose(result["dhw"], result["known"] + result["predicted"])


def test_origin_boundaries_and_gap_exclusions():
    p = panel()
    manifest = core.origin_manifest(p)
    assert manifest.groupby("year").eligible.sum().to_dict() == {2023: 23, 2024: 23, 2025: 23}
    p.loc[("region", pd.Timestamp("2025-01-05")), "hotspot_c"] = np.nan
    m = core.origin_manifest(p)
    assert m[m.year == 2025].eligible.sum() == 0


def test_seasonal_leap_and_future_isolation():
    p = panel()
    seasonal = core.fit_seasonal(p)
    assert len(seasonal) == 366
    feb = seasonal[(seasonal.month == 2) & (seasonal.day == 29)].hotspot_c.item()
    assert feb == 0.5
    changed = p.copy()
    changed.loc[("region", slice(pd.Timestamp("2023-01-02"), None)), "hotspot_c"] = 99
    pd.testing.assert_frame_equal(seasonal, core.fit_seasonal(changed))
    pd.testing.assert_series_equal(
        core.context_at(p, "region", "2023-01-01"), core.context_at(changed, "region", "2023-01-01")
    )
    dates = pd.date_range("2023-01-02", periods=28)
    a = core.baseline_forecasts(
        core.context_at(p, "region", "2023-01-01"), dates, seasonal, "region"
    )
    b = core.baseline_forecasts(
        core.context_at(changed, "region", "2023-01-01"), dates, seasonal, "region"
    )
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_byod_refusals_and_valid_fixture():
    frame = pd.DataFrame(
        dict(
            region_id="r",
            date=pd.date_range("2020-01-01", periods=365).strftime("%Y-%m-%d"),
            hotspot_c=0.0,
        )
    )
    assert len(core.validate_byod(frame, semantics_confirmed=True)) == 365
    for bad in [
        frame.iloc[:-1],
        pd.concat([frame, frame.iloc[:1]]),
        frame.assign(hotspot_c=-1),
        frame.assign(date="2020-01-01T00:00:00"),
    ]:
        with pytest.raises(ValueError):
            core.validate_byod(bad, semantics_confirmed=True)
    with pytest.raises(ValueError):
        core.validate_byod(frame)


def test_metrics_null_support_order_and_parity(tmp_path):
    p = panel()
    raw = np.zeros(28)
    f = core.forecast_frame(p, "region", "2024-03-01", "chronos", raw, "test", np.zeros((28, 3)))
    result = core.evaluate_forecasts(f)
    assert result["thresholds"][0]["precision"] is None
    assert result["thresholds"][0]["precision_reason"] == "no_predicted_positives"
    json.dumps(result, allow_nan=False)
    f.to_csv(tmp_path / "forecasts.csv", index=False)
    loaded = pd.read_csv(tmp_path / "forecasts.csv", parse_dates=["origin", "target_date"])
    assert core.evaluate_forecasts(loaded) == result
    with pytest.raises(ValueError, match="Quantiles"):
        core.forecast_frame(
            p, "region", "2024-03-01", "chronos", raw, "test", np.tile([2, 0, 1], (28, 1))
        )
    with pytest.raises(ValueError, match="Missing model outputs"):
        core.evaluate_forecasts(f.iloc[:-1])


def test_negative_predictions_preserve_raw():
    f = core.forecast_frame(panel(), "region", "2024-03-01", "model", np.full(28, -1), "test")
    assert f.raw_hotspot_c.eq(-1).all() and f.hotspot_c.eq(0).all()


def test_future_targets_change_only_scoring():
    p = panel()
    raw = np.ones(28)
    f = core.forecast_frame(p, "region", "2024-03-01", "model", raw, "test")
    changed = p.copy()
    changed.loc[
        ("region", slice(pd.Timestamp("2024-03-02"), pd.Timestamp("2024-03-29"))), "hotspot_c"
    ] = 2
    g = core.forecast_frame(changed, "region", "2024-03-01", "model", raw, "test")
    assert np.array_equal(f.dhw_c_weeks, g.dhw_c_weeks)
    assert (
        core.evaluate_forecasts(f)["metrics"][0]["hotspot_mae"]
        != core.evaluate_forecasts(g)["metrics"][0]["hotspot_mae"]
    )
