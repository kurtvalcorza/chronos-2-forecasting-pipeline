"""Output normalisation, the median oracle, and the pre-predict rejection path.

These tests drive a fake pipeline that reproduces the pinned upstream raw schema
exactly, so the rename map and the oracles are exercised without weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chronos2_pipeline import (
    ForecastConfig,
    UpstreamContractError,
    ValidationError,
    build_rename_map,
    forecast,
    normalized_columns,
)
from chronos2_pipeline.inference import quantile_column_name
from conftest import FakeLoadedModel, SentinelPipeline, make_identity, make_series


def raw_frame(
    series_ids: list[str],
    horizon: int,
    quantiles: list[float],
    targets: list[str] = ["target"],  # noqa: B006 - fixture convenience
    *,
    id_column: str = "series_id",
    timestamp_column: str = "timestamp",
    point_equals_median: bool = True,
    start: str = "2026-01-03T16:00:00",
) -> pd.DataFrame:
    """A frame shaped exactly like upstream ``predict_df`` output.

    Column order and row order follow ``chronos/chronos2/pipeline.py`` L950-960
    in 2.3.1: rows are ordered (item, target, step); ``predictions`` is the
    median; quantile columns are named ``str(level)``.
    """
    rows = len(series_ids) * len(targets) * horizon
    frame = pd.DataFrame(
        {
            id_column: np.repeat(series_ids, len(targets) * horizon),
            timestamp_column: np.tile(
                pd.date_range(start, periods=horizon, freq="h"), len(series_ids) * len(targets)
            ),
            "target_name": np.tile(np.repeat(targets, horizon), len(series_ids)),
        }
    )
    base = np.arange(rows, dtype=float)
    for i, q in enumerate(quantiles):
        frame[str(q)] = base + i
    median_col = str(0.5) if 0.5 in quantiles else str(quantiles[0])
    frame["predictions"] = (
        frame[median_col].to_numpy() if point_equals_median else frame[median_col].to_numpy() + 1.0
    )
    # Upstream appends `predictions` before the quantile columns; reorder to match.
    ordered = [
        id_column,
        timestamp_column,
        "target_name",
        "predictions",
        *(str(q) for q in quantiles),
    ]
    return frame[ordered]


class FakePipeline:
    """Returns a canned raw frame and records the kwargs it was called with."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.calls: list[dict] = []

    def predict_df(self, df, **kwargs):
        self.calls.append({"df": df, **kwargs})
        return self.frame.copy()


def model_returning(frame: pd.DataFrame, **identity_overrides) -> FakeLoadedModel:
    return FakeLoadedModel(
        pipeline=FakePipeline(frame), identity=make_identity(**identity_overrides)
    )


# --------------------------------------------------------------------------
# The rename map
# --------------------------------------------------------------------------


def test_rename_map_is_the_documented_table():
    config = ForecastConfig()
    assert build_rename_map(config) == {
        "series_id": "series_id",
        "timestamp": "timestamp",
        "predictions": "prediction",
        "0.1": "q0.1",
        "0.5": "q0.5",
        "0.9": "q0.9",
    }


def test_rename_map_follows_custom_column_names():
    config = ForecastConfig(id_column="item_id", timestamp_column="ts", quantile_levels=[0.5])
    assert build_rename_map(config) == {
        "item_id": "series_id",
        "ts": "timestamp",
        "predictions": "prediction",
        "0.5": "q0.5",
    }


def test_rename_map_never_maps_target_name():
    assert "target_name" not in build_rename_map(ForecastConfig())


def test_rename_map_is_deterministic_across_calls():
    config = ForecastConfig(quantile_levels=[0.05, 0.5, 0.95])
    assert build_rename_map(config) == build_rename_map(config)


def test_normalized_column_order_is_fixed():
    assert normalized_columns(ForecastConfig()) == [
        "series_id",
        "timestamp",
        "target_name",
        "prediction",
        "q0.1",
        "q0.5",
        "q0.9",
    ]


@pytest.mark.parametrize(("level", "name"), [(0.1, "q0.1"), (0.5, "q0.5"), (0.99, "q0.99")])
def test_quantile_column_names(level, name):
    assert quantile_column_name(level) == name


# --------------------------------------------------------------------------
# End-to-end normalisation against a faithful fake
# --------------------------------------------------------------------------


def test_forecast_normalises_the_raw_frame():
    quantiles = [0.1, 0.5, 0.9]
    frame = raw_frame(["A", "B"], horizon=8, quantiles=quantiles)
    model = model_returning(frame)
    history = pd.concat([make_series("A"), make_series("B", seed=2)], ignore_index=True)
    config = ForecastConfig(prediction_length=8, quantile_levels=quantiles)

    result = forecast(history, config, model)

    assert list(result.forecast.columns) == normalized_columns(config)
    assert len(result.forecast) == 2 * 8
    assert sorted(result.forecast["series_id"].unique()) == ["A", "B"]
    # Values are carried, not recomputed.
    np.testing.assert_array_equal(
        result.forecast["prediction"].to_numpy(), frame["predictions"].to_numpy()
    )
    np.testing.assert_array_equal(
        result.forecast["q0.9"].to_numpy(), frame["0.9"].to_numpy()
    )
    assert result.raw.equals(frame)


def test_normalisation_drops_nothing_and_adds_nothing():
    frame = raw_frame(["A"], horizon=4, quantiles=[0.1, 0.5, 0.9])
    model = model_returning(frame)
    result = forecast(make_series(), ForecastConfig(prediction_length=4), model)
    assert set(result.forecast.columns) == set(normalized_columns(ForecastConfig()))
    assert len(result.forecast) == len(frame)


def test_forecast_passes_the_configured_arguments_through_to_predict_df():
    frame = raw_frame(["A"], horizon=4, quantiles=[0.1, 0.5, 0.9])
    model = model_returning(frame)
    config = ForecastConfig(prediction_length=4, batch_size=32, context_length=48, frequency="h")
    forecast(make_series(), config, model)
    call = model.pipeline.calls[0]
    assert call["prediction_length"] == 4
    assert call["batch_size"] == 32
    assert call["context_length"] == 48
    assert call["freq"] == "h"
    assert call["validate_inputs"] is True
    assert call["cross_learning"] is False
    assert call["target"] == "target"
    assert call["quantile_levels"] == [0.1, 0.5, 0.9]


def test_predict_df_receives_the_sorted_normalised_history():
    frame = raw_frame(["A", "B"], horizon=4, quantiles=[0.5])
    model = model_returning(frame)
    history = pd.concat([make_series("B", seed=2), make_series("A")], ignore_index=True)
    forecast(history, ForecastConfig(prediction_length=4, quantile_levels=[0.5]), model)
    passed = model.pipeline.calls[0]["df"]
    assert list(passed["series_id"].unique()) == ["A", "B"]
    assert passed["timestamp"].is_monotonic_increasing is False  # two series, resets at B
    assert passed.groupby("series_id")["timestamp"].apply(lambda s: s.is_monotonic_increasing).all()


# --------------------------------------------------------------------------
# The median oracle (RFC C-2)
# --------------------------------------------------------------------------


def test_median_oracle_passes_when_predictions_equals_q05():
    frame = raw_frame(["A"], horizon=4, quantiles=[0.1, 0.5, 0.9], point_equals_median=True)
    result = forecast(make_series(), ForecastConfig(prediction_length=4), model_returning(frame))
    np.testing.assert_array_equal(
        result.forecast["prediction"].to_numpy(), result.forecast["q0.5"].to_numpy()
    )


def test_median_oracle_fails_when_predictions_diverges_from_q05():
    frame = raw_frame(["A"], horizon=4, quantiles=[0.1, 0.5, 0.9], point_equals_median=False)
    with pytest.raises(UpstreamContractError, match="no longer exactly the 0.5 quantile"):
        forecast(make_series(), ForecastConfig(prediction_length=4), model_returning(frame))


def test_median_oracle_detects_a_single_differing_row():
    frame = raw_frame(["A"], horizon=8, quantiles=[0.1, 0.5, 0.9])
    frame.loc[3, "predictions"] = frame.loc[3, "predictions"] + 1e-9
    with pytest.raises(UpstreamContractError) as exc:
        forecast(make_series(), ForecastConfig(prediction_length=8), model_returning(frame))
    assert "1 differing row" in str(exc.value)


def test_median_oracle_is_skipped_when_05_was_not_requested():
    quantiles = [0.1, 0.9]
    frame = raw_frame(["A"], horizon=4, quantiles=quantiles, point_equals_median=False)
    result = forecast(
        make_series(),
        ForecastConfig(prediction_length=4, quantile_levels=quantiles),
        model_returning(frame),
    )
    assert list(result.forecast.columns) == [
        "series_id",
        "timestamp",
        "target_name",
        "prediction",
        "q0.1",
        "q0.9",
    ]


# --------------------------------------------------------------------------
# The raw-schema tripwire
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dropped", ["target_name", "predictions", "0.5", "series_id"])
def test_missing_raw_column_raises_upstream_contract_error(dropped):
    frame = raw_frame(["A"], horizon=4, quantiles=[0.1, 0.5, 0.9]).drop(columns=[dropped])
    with pytest.raises(UpstreamContractError, match="missing expected column"):
        forecast(make_series(), ForecastConfig(prediction_length=4), model_returning(frame))


def test_raw_multi_target_frame_keeps_target_name():
    """Guardrail 5: the multi-target raw schema is a Phase 1 contract even though
    the public API is univariate. The shape is asserted here; the real upstream
    frame is asserted in the integration suite."""
    frame = raw_frame(["A", "B"], horizon=4, quantiles=[0.5], targets=["y1", "y2"])
    assert "target_name" in frame.columns
    assert len(frame) == 2 * 2 * 4
    assert sorted(frame["target_name"].unique()) == ["y1", "y2"]


# --------------------------------------------------------------------------
# Validation runs before predict_df (RFC C-3, C-5)
# --------------------------------------------------------------------------


def test_irregular_series_with_explicit_frequency_never_reaches_predict_df(sentinel_model):
    history = make_series(n=48)
    history.loc[20, "timestamp"] += pd.Timedelta(minutes=17)
    config = ForecastConfig(prediction_length=8, frequency="h")

    with pytest.raises(ValidationError) as exc:
        forecast(history, config, sentinel_model)

    assert exc.value.code == "IRREGULAR_FREQUENCY"
    assert sentinel_model.pipeline.called is False


def test_gappy_series_with_explicit_frequency_never_reaches_predict_df(sentinel_model):
    history = make_series(n=48).drop(index=[30]).reset_index(drop=True)
    with pytest.raises(ValidationError) as exc:
        forecast(history, ForecastConfig(prediction_length=8, frequency="h"), sentinel_model)
    assert exc.value.code == "SERIES_GAP"
    assert sentinel_model.pipeline.called is False


def test_out_of_grid_quantile_never_reaches_predict_df(sentinel_model):
    config = ForecastConfig(prediction_length=8, quantile_levels=[0.001, 0.5])
    with pytest.raises(ValidationError) as exc:
        forecast(make_series(), config, sentinel_model)
    assert exc.value.code == "QUANTILE_NOT_IN_GRID"
    assert sentinel_model.pipeline.called is False


def test_the_sentinel_pipeline_would_have_flagged_a_leak():
    """Proves the sentinel is a real oracle: calling it fails loudly."""
    sentinel = SentinelPipeline()
    with pytest.raises(AssertionError, match="predict_df was called"):
        sentinel.predict_df(pd.DataFrame())


# --------------------------------------------------------------------------
# Phase 1 boundaries
# --------------------------------------------------------------------------


def test_multi_target_public_api_is_refused_in_phase_1(sentinel_model):
    history = make_series()
    history["target2"] = 1.0
    config = ForecastConfig(target=["target", "target2"], prediction_length=8)
    with pytest.raises(ValidationError) as exc:
        forecast(history, config, sentinel_model)
    assert exc.value.code == "MULTI_TARGET_NOT_SUPPORTED"
    assert sentinel_model.pipeline.called is False


def test_covariate_columns_are_refused_in_phase_1(sentinel_model):
    history = make_series()
    history["temperature"] = 20.0
    with pytest.raises(ValidationError) as exc:
        forecast(history, ForecastConfig(prediction_length=8), sentinel_model)
    assert exc.value.code == "COVARIATES_NOT_SUPPORTED"
    assert sentinel_model.pipeline.called is False


def test_future_df_is_refused_in_phase_1(sentinel_model):
    with pytest.raises(ValidationError) as exc:
        forecast(make_series(), ForecastConfig(prediction_length=8), sentinel_model, pd.DataFrame())
    assert exc.value.code == "COVARIATES_NOT_SUPPORTED"
    assert sentinel_model.pipeline.called is False


def test_evaluation_is_an_explicit_phase_3_stub():
    from chronos2_pipeline import evaluation

    stubs = (
        evaluation.evaluate_forecast,
        evaluation.seasonal_naive_baseline,
        evaluation.last_value_baseline,
    )
    for fn in stubs:
        with pytest.raises(NotImplementedError, match="Phase 3"):
            fn()
