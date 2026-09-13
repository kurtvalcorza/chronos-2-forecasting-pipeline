"""Offline tests for the public validation and evaluation stage helpers (DAT24 / EVAL21).

No weights and no network: the model is the conftest stand-in and forecasts are frames.
"""

# ruff: noqa: E501  -- offline fixtures and assertions are kept on single lines for readability
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chronos2_pipeline import (
    INPUT_SCHEMA,
    MODEL_ID,
    MODEL_REVISION,
    ForecastConfig,
    ValidationError,
    chronological_holdout,
    evaluation_report,
    forecast,
    validate_inputs,
)
from chronos2_pipeline.inference import ForecastResult
from chronos2_pipeline.validation import DEFAULT_LIMITS, MIN_OBSERVATIONS
from conftest import FakeLoadedModel, SentinelPipeline, make_identity, make_series


def sentinel() -> FakeLoadedModel:
    return FakeLoadedModel(pipeline=SentinelPipeline(), identity=make_identity())


def truth_shaped_forecast(truth: pd.DataFrame, config: ForecastConfig, *, offset: float = 0.0) -> pd.DataFrame:
    rows = []
    for _, row in truth.iterrows():
        for target in config.target_names:
            value = float(row[target]) + offset
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


def as_result(frame: pd.DataFrame) -> ForecastResult:
    return ForecastResult(
        forecast=frame,
        raw=frame,
        provenance={
            "model": {"name": MODEL_ID, "revision": MODEL_REVISION},
            "inference": {"effective_context_length": 84},
        },
    )


# --------------------------------------------------------------------------
# validate_inputs
# --------------------------------------------------------------------------


def test_validate_inputs_returns_manifest_with_schema_and_identity(two_id_history: pd.DataFrame) -> None:
    config = ForecastConfig(prediction_length=6, quantile_levels=[0.1, 0.5, 0.9])
    manifest = validate_inputs(two_id_history, config, sentinel(), names=["first", "second"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["min_observations_per_series"] == MIN_OBSERVATIONS
    assert manifest["schema"]["max_ids"] == DEFAULT_LIMITS.max_ids
    assert manifest["schema"]["max_prediction_length"] == DEFAULT_LIMITS.max_prediction_length
    assert [entry["id"] for entry in manifest["inputs"]] == ["first", "second"]
    assert [entry["series_id"] for entry in manifest["inputs"]] == ["A", "B"]
    assert all(entry["n_observations"] == 64 for entry in manifest["inputs"])
    assert manifest["prediction_length"] == 6
    assert manifest["requested_quantile_levels"] == [0.1, 0.5, 0.9]
    assert manifest["observed_frequency"] == "0 days 01:00:00"
    assert manifest["n_rows"] == 128
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_default_ids_are_the_series_ids(univariate_history: pd.DataFrame) -> None:
    manifest = validate_inputs(univariate_history, ForecastConfig(prediction_length=3), sentinel())
    assert [entry["id"] for entry in manifest["inputs"]] == ["A"]
    assert manifest["past_covariate_names"] == []
    assert manifest["known_future_covariate_names"] == []


def test_validate_inputs_rejects_exactly_like_forecast(univariate_history: pd.DataFrame) -> None:
    config = ForecastConfig(prediction_length=3)
    model = sentinel()
    broken = univariate_history.copy()
    broken.loc[5, "target"] = np.nan
    with pytest.raises(ValidationError) as via_helper:
        validate_inputs(broken, config, model)
    with pytest.raises(ValidationError) as via_forecast:
        forecast(broken, config, model)
    assert via_helper.value.code == via_forecast.value.code
    assert not model.pipeline.called

    gappy = univariate_history.drop(index=[10]).reset_index(drop=True)
    with pytest.raises(ValidationError) as via_helper:
        validate_inputs(gappy, config, model)
    with pytest.raises(ValidationError) as via_forecast:
        forecast(gappy, config, model)
    assert via_helper.value.code == via_forecast.value.code

    leak = univariate_history.tail(3).copy()  # a future table that carries the target column
    with pytest.raises(ValidationError) as via_helper:
        validate_inputs(univariate_history, config, model, future_df=leak)
    with pytest.raises(ValidationError) as via_forecast:
        forecast(univariate_history, config, model, leak)
    assert via_helper.value.code == via_forecast.value.code
    assert not model.pipeline.called


def test_validate_inputs_names_must_match_the_series(two_id_history: pd.DataFrame) -> None:
    with pytest.raises(ValidationError) as exc:
        validate_inputs(two_id_history, ForecastConfig(prediction_length=3), sentinel(), names=["only-one"])
    assert exc.value.code == "NAMES_LENGTH_MISMATCH"


# --------------------------------------------------------------------------
# evaluation_report
# --------------------------------------------------------------------------


def test_evaluation_report_not_measurable_without_truth() -> None:
    config = ForecastConfig(prediction_length=3, quantile_levels=[0.1, 0.5, 0.9])
    split = chronological_holdout(make_series(n=24), config)
    report = evaluation_report(as_result(truth_shaped_forecast(split.truth, config)), config=config)
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["baselines"] == []
    assert "chronological_holdout" in report["needs"]
    assert report["horizon"] == 3
    assert report["n_series"] == 1
    assert report["targets"] == ["target"]
    assert report["effective_context_length"] == 84
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_evaluation_report_sample_sanity_with_truth_and_baselines() -> None:
    config = ForecastConfig(prediction_length=3, quantile_levels=[0.1, 0.5, 0.9])
    history = make_series(n=30)
    split = chronological_holdout(history, config)
    result = as_result(truth_shaped_forecast(split.truth, config, offset=0.5))
    report = evaluation_report(
        result, split.truth, config=config, history_df=split.history, season_length=24, sample_kind="BYOD"
    )
    assert report["verdict"] == "sample-sanity"
    assert report["sample_kind"] == "BYOD"
    assert {m["id"] for m in report["metrics"]} == {"evaluate_forecast"}
    by_metric = {(m["metric"], m.get("quantile")): m["value"] for m in report["metrics"]}
    assert by_metric[("mae", None)] == pytest.approx(0.5)
    assert by_metric[("rmse", None)] == pytest.approx(0.5)
    assert by_metric[("interval_coverage", None)] == pytest.approx(1.0)
    assert {q for (name, q) in by_metric if name == "pinball_loss"} == {0.1, 0.5, 0.9}
    assert [b["id"] for b in report["baselines"]] == ["last_value_baseline", "seasonal_naive_baseline"]
    assert report["baselines"][1]["season_length"] == 24
    assert all(m["estimation"] for m in report["metrics"])
    assert "3 held-out" in report["reason"]


def test_evaluation_report_accepts_a_bare_forecast_frame_and_skips_baselines_without_history() -> None:
    config = ForecastConfig(prediction_length=2, quantile_levels=[0.5])
    split = chronological_holdout(make_series(n=12), config)
    frame = truth_shaped_forecast(split.truth, config)[["series_id", "timestamp", "target_name", "prediction", "q0.5"]]
    report = evaluation_report(frame, split.truth, config=config)
    assert report["verdict"] == "sample-sanity"
    assert report["baselines"] == []
    assert report["model_id"] is None
    assert [m["metric"] for m in report["metrics"]] == ["mae", "rmse", "pinball_loss"]


def test_evaluation_report_raises_what_evaluate_forecast_raises() -> None:
    config = ForecastConfig(prediction_length=3, quantile_levels=[0.5])
    split = chronological_holdout(make_series(n=12), config)
    short = truth_shaped_forecast(split.truth, config).iloc[:-1]
    with pytest.raises(ValidationError) as exc:
        evaluation_report(as_result(short), split.truth, config=config)
    assert exc.value.code == "EVALUATION_COVERAGE_MISMATCH"
