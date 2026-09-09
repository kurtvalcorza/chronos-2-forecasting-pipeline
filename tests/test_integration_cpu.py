"""Integration: the real pinned weights, on CPU.

Marked ``integration`` and excluded from the default run. Downloads ~478 MB the
first time. Nothing here requires CUDA.

These are the tests that make the rest of the suite honest: everything else
asserts against a fake shaped like the recorded upstream contract, and only this
file checks that the recorded contract is still what upstream actually does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chronos2_pipeline import ForecastConfig, ValidationError, forecast, load_pinned_model
from chronos2_pipeline import model as model_mod
from chronos2_pipeline.inference import normalized_columns
from conftest import make_series

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def loaded():
    """Load the pinned checkpoint once for the whole module."""
    return load_pinned_model(device="cpu")


# --------------------------------------------------------------------------
# Supply chain, against the real download
# --------------------------------------------------------------------------


def test_resolved_revision_and_digests_match_the_pins(loaded):
    identity = loaded.identity
    assert identity.name == model_mod.PINNED_MODEL_ID
    assert identity.revision == model_mod.PINNED_REVISION
    assert identity.weights_sha256 == model_mod.PINNED_WEIGHTS_SHA256
    assert identity.weights_bytes == model_mod.PINNED_WEIGHTS_BYTES
    assert identity.config_sha256 == model_mod.PINNED_CONFIG_SHA256


def test_snapshot_contains_no_pickle_weights(loaded):
    from pathlib import Path

    files = [p for p in Path(loaded.identity.snapshot_path).rglob("*") if p.is_file()]
    suffixes = {p.suffix.lower() for p in files}
    assert not suffixes & set(model_mod.REFUSED_WEIGHT_SUFFIXES)
    assert ".safetensors" in suffixes


def test_trained_quantile_grid_read_from_the_model_is_the_21_level_grid(loaded):
    grid = loaded.trained_quantiles
    assert len(grid) == 21
    np.testing.assert_allclose(grid, model_mod.EXPECTED_TRAINED_QUANTILES, rtol=0, atol=1e-12)


def test_model_limits_read_from_the_pipeline(loaded):
    """8192 context; 64 output patches x patch size 16 = 1024 native horizon."""
    assert loaded.model_context_length == 8192
    assert loaded.model_prediction_length == 1024


def test_the_config_json_on_disk_hashes_to_the_pin(loaded):
    from pathlib import Path

    config_path = Path(loaded.identity.snapshot_path) / model_mod.CONFIG_FILENAME
    assert model_mod.sha256_file(config_path) == model_mod.PINNED_CONFIG_SHA256


def test_loading_reports_the_actual_device_and_dtype(loaded):
    assert loaded.device.startswith("cpu")
    assert loaded.dtype.startswith("torch.")


# --------------------------------------------------------------------------
# CPU smoke: univariate, two IDs
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def smoke_result(loaded):
    history = pd.concat(
        [make_series("A", n=64, seed=1), make_series("B", n=64, seed=2)], ignore_index=True
    )
    config = ForecastConfig(prediction_length=8, quantile_levels=[0.1, 0.5, 0.9])
    return config, forecast(history, config, loaded)


def test_cpu_smoke_shape_and_columns(smoke_result):
    config, result = smoke_result
    assert list(result.forecast.columns) == normalized_columns(config)
    assert len(result.forecast) == 2 * 8
    assert sorted(result.forecast["series_id"].unique()) == ["A", "B"]
    assert result.forecast["target_name"].unique().tolist() == ["target"]


def test_cpu_smoke_outputs_are_finite(smoke_result):
    _, result = smoke_result
    for column in ("prediction", "q0.1", "q0.5", "q0.9"):
        values = result.forecast[column].to_numpy(dtype=float)
        assert np.isfinite(values).all(), f"{column} contains non-finite values"


def test_cpu_smoke_point_forecast_is_exactly_the_median(smoke_result):
    _, result = smoke_result
    np.testing.assert_array_equal(
        result.forecast["prediction"].to_numpy(), result.forecast["q0.5"].to_numpy()
    )
    np.testing.assert_array_equal(
        result.raw["predictions"].to_numpy(), result.raw["0.5"].to_numpy()
    )


def test_cpu_smoke_quantiles_are_ordered(smoke_result):
    _, result = smoke_result
    assert (result.forecast["q0.1"] <= result.forecast["q0.5"] + 1e-6).all()
    assert (result.forecast["q0.5"] <= result.forecast["q0.9"] + 1e-6).all()


def test_cpu_smoke_forecast_timestamps_continue_the_history(smoke_result):
    _, result = smoke_result
    for series_id in ("A", "B"):
        stamps = pd.DatetimeIndex(
            result.forecast.loc[result.forecast["series_id"] == series_id, "timestamp"]
        )
        assert stamps.is_monotonic_increasing
        assert (stamps.to_series().diff().dropna() == pd.Timedelta(hours=1)).all()


def test_cpu_smoke_provenance_is_complete(smoke_result):
    _, result = smoke_result
    inference = result.provenance["inference"]
    assert inference["n_ids"] == 2
    assert inference["requested_prediction_length"] == 8
    assert inference["effective_prediction_length"] == 8
    assert inference["model_prediction_length"] == 1024
    assert inference["autoregressive_unrolled"] is False
    assert inference["latency_seconds"] > 0
    assert result.provenance["model"]["revision"] == model_mod.PINNED_REVISION
    assert result.provenance["runtime"]["device"].startswith("cpu")


def test_cpu_smoke_predictions_are_in_the_neighbourhood_of_the_history(smoke_result):
    """A crude sanity oracle: the series oscillates around 100, so a forecast
    ten times that magnitude means the context was not actually used."""
    _, result = smoke_result
    values = result.forecast["prediction"].to_numpy(dtype=float)
    assert values.min() > 50.0
    assert values.max() < 200.0


# --------------------------------------------------------------------------
# Upstream oracles that need the real pipeline
# --------------------------------------------------------------------------


def test_raw_multi_target_output_preserves_target_name(loaded):
    """Guardrail 5 / RFC C-1, asserted against ``predict_df`` directly.

    Phase 2 reaches the same path through :func:`forecast`; this stays on the
    raw call so the recorded upstream schema is checked without the normalising
    layer in between."""
    history = pd.concat(
        [make_series("A", n=64, seed=1), make_series("B", n=64, seed=2)], ignore_index=True
    )
    history["target2"] = history["target"] * 1.5
    horizon = 8

    raw = loaded.pipeline.predict_df(
        history,
        id_column="series_id",
        timestamp_column="timestamp",
        target=["target", "target2"],
        prediction_length=horizon,
        quantile_levels=[0.1, 0.5, 0.9],
        validate_inputs=True,
    )

    assert "target_name" in raw.columns
    assert sorted(raw["target_name"].unique()) == ["target", "target2"]
    assert len(raw) == 2 * horizon * 2  # n_ids * horizon * n_targets
    for name in ("0.1", "0.5", "0.9"):
        assert name in raw.columns
    assert "predictions" in raw.columns
    np.testing.assert_array_equal(raw["predictions"].to_numpy(), raw["0.5"].to_numpy())
    # target_name actually disambiguates: the two targets differ.
    by_target = raw.groupby("target_name")["predictions"].mean()
    assert by_target["target2"] > by_target["target"]


def test_upstream_clamps_an_out_of_grid_quantile_and_dimer_refuses_it_first(loaded):
    """Documents WHY rule 14 hard-fails.

    Called directly, upstream answers a request for 0.001 with the edge quantile
    it was trained on (0.01) under a column labelled "0.001". DIMER rejects the
    same request before ``predict_df`` is reached.
    """
    history = make_series("A", n=64, seed=1)

    clamped = loaded.pipeline.predict_df(
        history,
        id_column="series_id",
        timestamp_column="timestamp",
        target="target",
        prediction_length=8,
        quantile_levels=[0.001, 0.5],
        validate_inputs=True,
    )
    edge = loaded.pipeline.predict_df(
        history,
        id_column="series_id",
        timestamp_column="timestamp",
        target="target",
        prediction_length=8,
        quantile_levels=[0.01, 0.5],
        validate_inputs=True,
    )
    # The column labelled 0.001 holds the trained 0.01 quantile, not an
    # extrapolation: upstream substituted the edge level.
    np.testing.assert_allclose(
        clamped["0.001"].to_numpy(), edge["0.01"].to_numpy(), rtol=0, atol=1e-6
    )

    with pytest.raises(ValidationError) as exc:
        forecast(history, ForecastConfig(prediction_length=8, quantile_levels=[0.001, 0.5]), loaded)
    assert exc.value.code == "QUANTILE_NOT_IN_GRID"


def test_upstream_ignores_a_declared_freq_that_contradicts_the_data(loaded):
    """Documents WHY rule 7-10 validation is ours and runs first.

    Upstream's own docstring says ``freq`` is used as-is and is never checked
    against the data. Here a gappy series is accepted by upstream with
    ``validate_inputs=True``, while DIMER refuses it.
    """
    gappy = make_series("A", n=64, seed=1).drop(index=[30]).reset_index(drop=True)

    accepted = loaded.pipeline.predict_df(
        gappy,
        id_column="series_id",
        timestamp_column="timestamp",
        target="target",
        prediction_length=4,
        quantile_levels=[0.5],
        validate_inputs=True,
        freq="h",
    )
    assert len(accepted) == 4  # upstream produced a forecast from a gappy series

    with pytest.raises(ValidationError) as exc:
        forecast(
            gappy,
            ForecastConfig(prediction_length=4, frequency="h", quantile_levels=[0.5]),
            loaded,
        )
    assert exc.value.code == "SERIES_GAP"


def test_loader_still_refuses_a_mutable_revision_with_the_cache_warm():
    from chronos2_pipeline import ModelSourceError

    with pytest.raises(ModelSourceError):
        load_pinned_model(revision="main", device="cpu")


# --------------------------------------------------------------------------
# Phase 2 — Modes C and D against the real weights
# --------------------------------------------------------------------------


def covariate_history(n: int = 64) -> pd.DataFrame:
    """Two series, one target, one covariate that genuinely drives the target."""
    frames = []
    for i, sid in enumerate(("A", "B")):
        block = make_series(sid, n=n, seed=i + 1)
        temperature = 20.0 + 5.0 * np.sin(np.arange(n) / 6.0)
        block["temperature"] = temperature
        block["target"] = block["target"] + 2.0 * temperature
        frames.append(block)
    return pd.concat(frames, ignore_index=True)


def future_table(history: pd.DataFrame, horizon: int, column: str) -> pd.DataFrame:
    """Known-future values for one covariate, aligned to the forecast origin."""
    blocks = []
    for sid, block in history.groupby("series_id", sort=True):
        stamps = pd.date_range(
            block["timestamp"].max() + pd.Timedelta("1h"), periods=horizon, freq="h"
        )
        blocks.append(
            pd.DataFrame({"series_id": sid, "timestamp": stamps, column: 20.0})
        )
    return pd.concat(blocks, ignore_index=True)


def test_multi_target_through_the_public_api(loaded):
    """Mode C end to end: two targets, two series, real weights."""
    history = pd.concat(
        [make_series("A", n=64, seed=1), make_series("B", n=64, seed=2)], ignore_index=True
    )
    history["target2"] = history["target"] * 1.5
    horizon = 8
    config = ForecastConfig(target=["target", "target2"], prediction_length=horizon)

    result = forecast(history, config, loaded)

    assert list(result.forecast.columns) == normalized_columns(config)
    assert len(result.forecast) == 2 * 2 * horizon
    #: The layout oracle already ran inside forecast(); this states the
    #: consequence it protects — each block is the target it is labelled with.
    assert list(result.forecast["target_name"][:horizon].unique()) == ["target"]
    assert list(result.forecast["target_name"][horizon : 2 * horizon].unique()) == ["target2"]
    by_target = result.forecast.groupby("target_name")["prediction"].mean()
    assert by_target["target2"] > by_target["target"]
    assert result.inference["n_targets"] == 2
    assert result.inference["n_covariates"] == 0


def test_a_past_only_covariate_runs_and_is_recorded_as_past_only(loaded):
    """Mode D, past-only: no future table, so the covariate stops at the origin."""
    history = covariate_history()
    config = ForecastConfig(prediction_length=8)

    result = forecast(history, config, loaded)

    assert len(result.forecast) == 2 * 8
    assert np.isfinite(result.forecast["prediction"]).all()
    assert result.inference["past_covariate_names"] == ["temperature"]
    assert result.inference["known_future_covariate_names"] == []


def test_a_known_future_covariate_runs_and_is_recorded_as_known_future(loaded):
    """Mode D, known-future: the same column, now supplied across the horizon."""
    history = covariate_history()
    horizon = 8
    future = future_table(history, horizon, "temperature")
    config = ForecastConfig(prediction_length=horizon)

    result = forecast(history, config, loaded, future)

    assert len(result.forecast) == 2 * horizon
    assert np.isfinite(result.forecast["prediction"]).all()
    assert result.inference["known_future_covariate_names"] == ["temperature"]
    assert result.inference["past_covariate_names"] == []


def test_the_known_future_covariate_actually_changes_the_forecast(loaded):
    """Otherwise the whole Mode D path could be a no-op and every test above
    would still pass. The covariate drives the target at ``+2.0`` per unit, so
    two future tables differing by 10 degrees must not produce one forecast."""
    history = covariate_history()
    horizon = 8
    config = ForecastConfig(prediction_length=horizon)

    cold = future_table(history, horizon, "temperature")
    hot = cold.copy()
    hot["temperature"] = 30.0

    cold_result = forecast(history, config, loaded, cold)
    hot_result = forecast(history, config, loaded, hot)

    assert not np.allclose(
        cold_result.forecast["prediction"].to_numpy(),
        hot_result.forecast["prediction"].to_numpy(),
    )


def test_upstream_accepts_a_categorical_covariate_that_dimer_refuses(loaded):
    """Documents WHY the covariate contract is numeric-only.

    Upstream takes a string covariate without complaint and encodes it — by
    target encoding for a single target and ordinal encoding for several
    (``preprocess.py`` L415). DIMER refuses it before ``predict_df`` rather than
    export a column whose meaning depends on how many targets were requested.
    """
    history = covariate_history()
    history["weather"] = np.where(history["temperature"] > 20.0, "warm", "cool")
    config = ForecastConfig(prediction_length=8)

    with pytest.raises(ValidationError) as exc:
        forecast(history, config, loaded)
    assert exc.value.code == "COVARIATE_NOT_NUMERIC"

    #: Upstream, called directly with the same frame, does not object.
    raw = loaded.pipeline.predict_df(
        history,
        id_column="series_id",
        timestamp_column="timestamp",
        target="target",
        prediction_length=8,
        quantile_levels=[0.5],
        validate_inputs=True,
    )
    assert len(raw) == 2 * 8
