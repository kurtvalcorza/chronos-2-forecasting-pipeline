from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chronos2_pipeline import (
    ForecastConfig,
    ValidationError,
    chronological_holdout,
    evaluate_forecast,
    last_value_baseline,
    seasonal_naive_baseline,
)


def make_frame(n: int = 12, *, multi: bool = False) -> pd.DataFrame:
    stamps = pd.date_range("2026-01-01", periods=n, freq="h")
    frames = []
    for offset, series_id in enumerate(("A", "B")):
        target = np.arange(n, dtype=float) + offset * 100.0
        data: dict[str, object] = {
            "series_id": series_id,
            "timestamp": stamps,
            "target": target,
        }
        if multi:
            data["target2"] = target * 2.0
        frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True)


def normalised_truth_forecast(truth: pd.DataFrame, config: ForecastConfig) -> pd.DataFrame:
    rows = []
    for _, row in truth.iterrows():
        for target in config.target_names:
            value = float(row[target])
            rows.append(
                {
                    "series_id": row["series_id"],
                    "timestamp": row["timestamp"],
                    "target_name": target,
                    "prediction": value,
                    "q0.1": value - 1.0,
                    "q0.5": value,
                    "q0.9": value + 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_chronological_holdout_removes_the_future_from_each_series() -> None:
    config = ForecastConfig(prediction_length=3)
    split = chronological_holdout(make_frame(), config)

    assert split.horizon == 3
    assert len(split.history) == 2 * 9
    assert len(split.truth) == 2 * 3
    for series_id in ("A", "B"):
        history = split.history[split.history["series_id"] == series_id]
        truth = split.truth[split.truth["series_id"] == series_id]
        assert history["timestamp"].max() < truth["timestamp"].min()
        assert not set(history["timestamp"]) & set(truth["timestamp"])


def test_holdout_refuses_a_series_too_short_for_context_plus_horizon() -> None:
    config = ForecastConfig(prediction_length=3)
    with pytest.raises(ValidationError) as exc:
        chronological_holdout(make_frame(n=5), config, minimum_context=3)
    assert exc.value.code == "EVALUATION_SERIES_TOO_SHORT"


def test_exact_point_forecast_scores_zero_and_interval_covers_everything() -> None:
    config = ForecastConfig(prediction_length=3)
    split = chronological_holdout(make_frame(), config)
    forecast = normalised_truth_forecast(split.truth, config)

    result = evaluate_forecast(forecast, split.truth, config)

    assert result.aggregate["n"] == 6
    assert result.aggregate["mae"] == pytest.approx(0.0)
    assert result.aggregate["rmse"] == pytest.approx(0.0)
    assert result.aggregate["interval_coverage"] == pytest.approx(1.0)
    assert len(result.per_series) == 2
    assert set(result.quantiles["quantile"]) == {0.1, 0.5, 0.9}
    median_loss = result.quantiles.loc[
        result.quantiles["quantile"] == 0.5, "pinball_loss"
    ].item()
    assert median_loss == pytest.approx(0.0)


def test_evaluation_refuses_missing_forecast_rows_instead_of_silently_dropping_them() -> None:
    config = ForecastConfig(prediction_length=3)
    split = chronological_holdout(make_frame(), config)
    forecast = normalised_truth_forecast(split.truth, config).iloc[:-1].copy()

    with pytest.raises(ValidationError) as exc:
        evaluate_forecast(forecast, split.truth, config)
    assert exc.value.code == "EVALUATION_COVERAGE_MISMATCH"


def test_last_value_baseline_uses_only_the_last_history_value() -> None:
    config = ForecastConfig(prediction_length=3)
    split = chronological_holdout(make_frame(), config)
    baseline = last_value_baseline(split.history, split.truth, config)

    for series_id, expected in (("A", 8.0), ("B", 108.0)):
        values = baseline.loc[
            baseline["series_id"] == series_id, "prediction"
        ].to_numpy()
        assert np.array_equal(values, np.full(3, expected))


def test_seasonal_naive_repeats_only_the_explicit_final_season() -> None:
    config = ForecastConfig(prediction_length=3)
    split = chronological_holdout(make_frame(), config)
    baseline = seasonal_naive_baseline(
        split.history,
        split.truth,
        config,
        season_length=3,
    )

    assert np.array_equal(
        baseline.loc[baseline["series_id"] == "A", "prediction"].to_numpy(),
        np.array([6.0, 7.0, 8.0]),
    )
    assert np.array_equal(
        baseline.loc[baseline["series_id"] == "B", "prediction"].to_numpy(),
        np.array([106.0, 107.0, 108.0]),
    )


def test_multi_target_truth_is_melted_without_crossing_target_labels() -> None:
    config = ForecastConfig(target=["target", "target2"], prediction_length=2)
    split = chronological_holdout(make_frame(multi=True), config)
    forecast = normalised_truth_forecast(split.truth, config)

    result = evaluate_forecast(forecast, split.truth, config)

    assert result.aggregate["n"] == 2 * 2 * 2
    assert set(result.per_series["target_name"]) == {"target", "target2"}
    assert result.aggregate["mae"] == pytest.approx(0.0)


def test_pinball_loss_uses_requested_quantile_semantics() -> None:
    config = ForecastConfig(prediction_length=1)
    truth = pd.DataFrame(
        {
            "series_id": ["A"],
            "timestamp": [pd.Timestamp("2026-01-01T01:00:00")],
            "target": [10.0],
        }
    )
    forecast = pd.DataFrame(
        {
            "series_id": ["A"],
            "timestamp": [pd.Timestamp("2026-01-01T01:00:00")],
            "target_name": ["target"],
            "prediction": [10.0],
            "q0.1": [8.0],
            "q0.5": [10.0],
            "q0.9": [12.0],
        }
    )

    result = evaluate_forecast(forecast, truth, config)

    losses = dict(
        zip(
            result.quantiles["quantile"],
            result.quantiles["pinball_loss"],
            strict=True,
        )
    )
    assert losses[0.1] == pytest.approx(0.2)
    assert losses[0.5] == pytest.approx(0.0)
    assert losses[0.9] == pytest.approx(0.2)
