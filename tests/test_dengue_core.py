"""Known-answer and leakage contracts, without model or network dependencies."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import dengue_core as dc  # noqa: E402


@pytest.fixture
def rows():
    return [
        {
            "year": 2010 + i // 52,
            "block": i % 52 + 1,
            "cases": float(i),
            "rain": float(i) / 2,
            "temp": 20 + i / 1000,
        }
        for i in range(16 * 52)
    ]


def test_origins_and_december_boundary(rows):
    result = dc.origins(rows, [2024, 2025])
    assert len(result) == 26
    assert rows[result[0]]["year"] == 2023
    assert rows[result[0]]["block"] == 52
    assert dc.target_key(rows, result[0] + 1) == (2024, 1)
    assert dc.target_key(rows, result[-1] + 4) == (2025, 52)
    assert dc.target_key(rows, len(rows) + 3) == (2026, 4)


@pytest.mark.parametrize(
    "field,value",
    [
        ("cases", None),
        ("cases", float("nan")),
        ("rain", -1),
        ("temp", float("inf")),
        ("cases", False),
    ],
)
def test_missing_invalid_not_zero(rows, field, value):
    rows[14][field] = value
    with pytest.raises(ValueError):
        dc.validate_rows(rows)


def test_incomplete_or_duplicate_cycle_refused(rows):
    with pytest.raises(ValueError):
        dc.validate_rows(rows[:-1])
    rows[15]["block"] = 15
    with pytest.raises(ValueError):
        dc.validate_rows(rows)


def test_latest_and_lags_and_trailing_mean(rows):
    x = dc.feature_row(rows, 200, 4, delay=2, weather=True)
    names = dc.feature_names(True)
    assert len(x) == 27
    assert x[names.index("cases_lag_0")] == 198
    assert x[names.index("cases_lag_52")] == 146
    assert x[names.index("cases_mean_4")] == 196.5
    assert x[names.index("rain_lag_4")] == 97


@pytest.mark.parametrize("horizon", [1, 2, 3, 4])
@pytest.mark.parametrize("delay", [0, 2])
def test_mature_targets_and_identical_weather_cohorts(rows, horizon, delay):
    plain = dc.training_task(rows, 400, horizon, delay=delay)
    weather = dc.training_task(rows, 400, horizon, delay=delay, weather=True)
    np.testing.assert_array_equal(plain["issuance_indices"], weather["issuance_indices"])
    np.testing.assert_array_equal(plain["y"], weather["y"])
    assert plain["issuance_indices"][0] == 297
    assert plain["target_indices"][-1] == 400 - delay
    assert len(plain["y"]) == 104 - horizon - delay
    np.testing.assert_allclose(np.expm1(plain["y"]), plain["target_indices"])
    # Each historical input obeys its own cutoff, not the current cutoff.
    np.testing.assert_array_equal(plain["X"][:, 0], plain["issuance_indices"] - delay)


def test_poison_future_never_changes_prior_task(rows):
    before = dc.training_task(rows, 400, 4, delay=2, weather=True)
    changed = copy.deepcopy(rows)
    for row in changed[399:]:
        row.update(cases=float("nan"), rain=float("nan"), temp=float("nan"))
    after = dc.training_task(changed, 400, 4, delay=2, weather=True)
    for key in ("X", "y", "query", "issuance_indices"):
        np.testing.assert_array_equal(before[key], after[key])


def test_insufficient_feature_history_refused(rows):
    with pytest.raises(ValueError):
        dc.feature_row(rows, 53, 1, delay=2)


def test_known_metrics_and_skill():
    result = dc.metrics([0, 2, 4], [1, 0, 4])
    assert result["mae"] == 1
    assert result["rmse"] == pytest.approx(np.sqrt(5 / 3))
    assert result["bias"] == pytest.approx(-1 / 3)
    assert dc.skill(1, 2) == 0.5
    assert dc.skill(1, 0) is None
    with pytest.raises(ValueError):
        dc.metrics([1], [np.nan])


def test_inverse_counts_keeps_fractional_and_clipping_evidence():
    raw, effective, clipped = dc.inverse_counts([np.log(1.5), -1])
    assert raw[0] == pytest.approx(0.5)
    assert effective[0] == pytest.approx(0.5)
    assert effective[1] == 0
    assert clipped.tolist() == [False, True]
    with pytest.raises(ValueError):
        dc.inverse_counts([1000])


def test_interval_known_answer_and_crossing():
    result = dc.interval_metrics([1, 3], [[0, 1, 2], [2, 3, 4]])
    assert result["coverage_80"] == 1
    assert result["median_mae"] == 0
    assert result["mean_width"] == 2
    assert result["pinball_loss"] == pytest.approx(1 / 15)
    with pytest.raises(ValueError):
        dc.interval_metrics([1], [[2, 1, 3]])


def test_bootstrap_preserves_pairing_and_tail():
    ref = np.zeros((26, 4))
    a, b = np.ones((26, 4)) * 3, np.ones((26, 4))
    result = dc.paired_bootstrap(ref, a, b)
    assert result["difference_b_minus_a"] == -2
    assert result["ci95"] == [-2, -2]
    assert result["group_sizes"] == [4, 4, 4, 4, 4, 4, 2]
    assert result == dc.paired_bootstrap(ref, a, b)


def test_bootstrap_resamples_whole_groups_all_horizons():
    ref = np.zeros((8, 4))
    a = np.zeros_like(ref)
    b = np.repeat(np.array([1] * 4 + [9] * 4)[:, None], 4, axis=1)
    result = dc.paired_bootstrap(ref, a, b)
    assert result["difference_b_minus_a"] == 5
    assert result["ci95"] == [1, 9]


def test_json_digest_canonical_and_refuses_nan():
    assert dc.digest_json({"a": 1, "b": 2}) == dc.digest_json({"b": 2, "a": 1})
    with pytest.raises(ValueError):
        dc.digest_json({"value": float("nan")})
