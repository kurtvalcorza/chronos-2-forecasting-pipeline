"""Provenance export: requested vs effective context and horizon, runtime block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chronos2_pipeline import ForecastConfig, build_provenance, forecast, runtime_versions
from chronos2_pipeline.provenance import PROVENANCE_SCHEMA_VERSION
from conftest import MODEL_PREDICTION_LENGTH, FakeLoadedModel, make_identity, make_series
from test_inference_contract import raw_frame


class FakePipeline:
    def __init__(self, frame):
        self.frame = frame
        self.calls = []

    def predict_df(self, df, **kwargs):
        self.calls.append(kwargs)
        return self.frame.copy()


def model_for(horizon: int, quantiles=(0.1, 0.5, 0.9), **overrides) -> FakeLoadedModel:
    frame = raw_frame(["A"], horizon=horizon, quantiles=list(quantiles))
    return FakeLoadedModel(pipeline=FakePipeline(frame), identity=make_identity(**overrides))


def test_runtime_versions_reports_the_installed_stack():
    versions = runtime_versions()
    for package in ("python", "chronos-forecasting", "torch", "transformers", "pandas", "numpy"):
        assert versions[package], f"{package} version missing from provenance"
    assert versions["chronos-forecasting"].startswith("2.3")


def test_runtime_versions_reports_none_for_an_absent_package():
    absent = "definitely-not-installed-xyz"
    assert runtime_versions((absent,))[absent] is None


def test_provenance_has_the_three_rfc_blocks():
    model = model_for(4)
    result = forecast(make_series(), ForecastConfig(prediction_length=4), model)
    assert set(result.provenance) == {"schema_version", "model", "runtime", "inference"}
    assert result.provenance["schema_version"] == PROVENANCE_SCHEMA_VERSION


def test_model_block_carries_all_four_supply_chain_pins():
    from chronos2_pipeline import model as model_mod

    result = forecast(make_series(), ForecastConfig(prediction_length=4), model_for(4))
    block = result.provenance["model"]
    assert block["revision"] == model_mod.PINNED_REVISION
    assert block["weights_sha256"] == model_mod.PINNED_WEIGHTS_SHA256
    assert block["weights_bytes"] == model_mod.PINNED_WEIGHTS_BYTES
    assert block["config_sha256"] == model_mod.PINNED_CONFIG_SHA256
    assert block["license"] == "apache-2.0"
    assert "no LICENSE file" in block["license_source"]
    assert len(block["trained_quantiles"]) == 21


def test_runtime_block_records_the_resolved_device_and_dtype():
    result = forecast(make_series(), ForecastConfig(prediction_length=4), model_for(4))
    runtime = result.provenance["runtime"]
    assert runtime["device"] == "cpu"
    assert runtime["dtype"] == "torch.float32"
    assert runtime["torch"]


def test_context_semantics_requested_effective_and_model():
    config = ForecastConfig(prediction_length=4, context_length=48)
    result = forecast(make_series(), config, model_for(4))
    inf = result.provenance["inference"]
    assert inf["requested_context_length"] == 48
    assert inf["effective_context_length"] == 48
    assert inf["model_context_length"] == 8192


def test_context_length_none_is_bounded_by_the_history_not_the_model_limit():
    """With no request, the data is what binds -- not the model's ceiling.

    This used to export ``effective_context_length: 8192`` for a 64-observation
    series: a number larger than the history that existed. RFC C-6 asks for the
    *effective* context, and no series can contribute more than it holds.
    """
    result = forecast(make_series(n=64), ForecastConfig(prediction_length=4), model_for(4))
    inf = result.provenance["inference"]
    assert inf["requested_context_length"] is None
    assert inf["effective_context_length"] == 64
    assert inf["model_context_length"] == 8192
    assert inf["longest_series_length"] == 64


def test_effective_context_never_exceeds_the_available_history():
    """Whichever of request, model limit and data binds first, it is never the data."""
    for requested in (None, 32, 4096, 8192):
        config = ForecastConfig(prediction_length=4, context_length=requested)
        inf = forecast(make_series(n=64), config, model_for(4)).provenance["inference"]
        assert inf["effective_context_length"] <= inf["longest_series_length"]
        assert inf["effective_context_length"] <= inf["model_context_length"]
        if requested is not None:
            assert inf["effective_context_length"] <= requested


def test_mixed_length_series_report_both_ends():
    """``effective_context_length`` is a ceiling across the request, so a shorter
    series contributed less. Both observed lengths ship so that is legible."""
    history = pd.concat(
        [make_series("A", n=64), make_series("B", n=40)], ignore_index=True
    )
    model = FakeLoadedModel(
        pipeline=FakePipeline(raw_frame(["A", "B"], horizon=4, quantiles=[0.1, 0.5, 0.9])),
        identity=make_identity(),
    )
    inf = forecast(history, ForecastConfig(prediction_length=4), model).provenance["inference"]
    assert inf["longest_series_length"] == 64
    assert inf["shortest_series_length"] == 40
    assert inf["effective_context_length"] == 64


def test_requested_context_above_the_model_limit_is_recorded_as_clamped():
    """Upstream resets context to the model limit (pipeline.py L610-617 in 2.3.1);
    provenance must show the clamp rather than echoing the request.

    Both series here are longer than the model limit, so the model ceiling is what
    binds and the clamp is what is under test -- not the history bound above.
    """
    config = ForecastConfig(prediction_length=4, context_length=8192)
    result = forecast(make_series(n=9000), config, model_for(4))
    inf = result.provenance["inference"]
    assert inf["requested_context_length"] == 8192
    assert inf["effective_context_length"] == 8192

    small_model = model_for(4, model_context_length=512)
    config = ForecastConfig(prediction_length=4, context_length=4096)
    result = forecast(make_series(n=600), config, small_model)
    inf = result.provenance["inference"]
    assert inf["requested_context_length"] == 4096
    assert inf["effective_context_length"] == 512
    assert inf["model_context_length"] == 512


def test_autoregressive_unrolled_is_false_within_the_native_horizon():
    result = forecast(make_series(), ForecastConfig(prediction_length=4), model_for(4))
    assert result.provenance["inference"]["autoregressive_unrolled"] is False


def test_autoregressive_unrolled_is_true_beyond_the_native_horizon():
    small_model = model_for(8, model_prediction_length=4)
    result = forecast(
        make_series(n=64), ForecastConfig(prediction_length=8), small_model, allow_unroll=True
    )
    inf = result.provenance["inference"]
    assert inf["requested_prediction_length"] == 8
    assert inf["effective_prediction_length"] == 8
    assert inf["model_prediction_length"] == 4
    assert inf["autoregressive_unrolled"] is True


def test_native_horizon_boundary_is_not_flagged_as_unrolled():
    model = model_for(4, model_prediction_length=4)
    result = forecast(make_series(), ForecastConfig(prediction_length=4), model)
    assert result.provenance["inference"]["autoregressive_unrolled"] is False
    assert MODEL_PREDICTION_LENGTH == 1024  # the real pin, asserted in the integration suite


def test_quantile_and_batch_semantics_are_recorded():
    config = ForecastConfig(prediction_length=4, quantile_levels=[0.1, 0.5, 0.9], batch_size=64)
    result = forecast(make_series(), config, model_for(4))
    inf = result.provenance["inference"]
    assert inf["requested_quantile_levels"] == [0.1, 0.5, 0.9]
    assert inf["effective_quantile_levels"] == [0.1, 0.5, 0.9]
    assert inf["batch_size"] == 64
    assert inf["cross_learning"] is False


def test_shape_and_frequency_are_recorded():
    frame = raw_frame(["A", "B"], horizon=4, quantiles=[0.1, 0.5, 0.9])
    model = FakeLoadedModel(pipeline=FakePipeline(frame), identity=make_identity())
    history = pd.concat([make_series("A"), make_series("B", seed=3)], ignore_index=True)
    result = forecast(history, ForecastConfig(prediction_length=4), model)
    inf = result.provenance["inference"]
    assert inf["n_ids"] == 2
    assert inf["n_targets"] == 1
    assert inf["n_covariates"] == 0
    assert inf["observed_frequency"] == "0 days 01:00:00"
    assert inf["declared_frequency"] == ""


def test_latency_is_recorded_and_warm_up_is_off_by_default():
    model = model_for(4)
    result = forecast(make_series(), ForecastConfig(prediction_length=4), model)
    inf = result.provenance["inference"]
    assert inf["latency_seconds"] >= 0.0
    assert inf["warm_up_performed"] is False
    assert len(model.pipeline.calls) == 1


def test_warm_up_runs_one_discarded_call_before_the_scored_one():
    model = model_for(4)
    result = forecast(
        make_series(), ForecastConfig(prediction_length=4), model, measure_latency=True
    )
    assert result.provenance["inference"]["warm_up_performed"] is True
    assert len(model.pipeline.calls) == 2


def test_build_provenance_is_json_serialisable():
    import json

    payload = build_provenance(
        make_identity(),
        device="cpu",
        dtype="torch.float32",
        n_ids=1,
        n_targets=1,
        n_covariates=0,
        requested_context_length=None,
        effective_context_length=64,
        longest_series_length=64,
        shortest_series_length=64,
        requested_prediction_length=24,
        effective_prediction_length=24,
        autoregressive_unrolled=False,
        requested_quantiles=[0.1, 0.5, 0.9],
        effective_quantiles=[0.1, 0.5, 0.9],
        batch_size=256,
        cross_learning=False,
        latency_seconds=1.25,
        warm_up_performed=True,
        frequency="h",
        observed_frequency="0 days 01:00:00",
    )
    assert json.loads(json.dumps(payload))["inference"]["latency_seconds"] == 1.25


def test_forecast_values_are_finite():
    result = forecast(make_series(), ForecastConfig(prediction_length=4), model_for(4))
    assert np.isfinite(result.forecast["prediction"].to_numpy()).all()


def test_result_accessors():
    result = forecast(make_series(), ForecastConfig(prediction_length=4), model_for(4))
    assert result.model is result.provenance["model"]
    assert result.inference is result.provenance["inference"]


@pytest.mark.parametrize("field", ["n_ids", "n_targets", "n_covariates", "latency_seconds"])
def test_required_inference_fields_present(field):
    result = forecast(make_series(), ForecastConfig(prediction_length=4), model_for(4))
    assert field in result.provenance["inference"]
