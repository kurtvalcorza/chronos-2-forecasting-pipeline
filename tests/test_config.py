"""ForecastConfig scalar validation."""

from __future__ import annotations

import pytest

from chronos2_pipeline import ForecastConfig, ValidationError
from chronos2_pipeline.config import DEFAULT_QUANTILE_LEVELS


def test_defaults_match_the_rfc_user_parameters_block():
    config = ForecastConfig()
    assert config.id_column == "series_id"
    assert config.timestamp_column == "timestamp"
    assert config.target == "target"
    assert config.prediction_length == 24
    assert config.quantile_levels == [0.1, 0.5, 0.9]
    assert config.batch_size == 256
    assert config.context_length is None
    assert config.frequency is None
    assert config.device == "auto"
    assert config.cross_learning is False
    assert DEFAULT_QUANTILE_LEVELS == (0.1, 0.5, 0.9)


def test_target_normalises_to_a_list_preserving_order():
    assert ForecastConfig(target="y").target_names == ["y"]
    assert ForecastConfig(target=["b", "a"]).target_names == ["b", "a"]
    assert ForecastConfig(target=["b", "a"]).n_targets == 2


def test_reserved_columns_are_id_timestamp_and_every_target():
    config = ForecastConfig(id_column="i", timestamp_column="t", target=["y1", "y2"])
    assert config.reserved_columns == ["i", "t", "y1", "y2"]


@pytest.mark.parametrize("bad", [0, -1, -24])
def test_prediction_length_must_be_positive(bad):
    with pytest.raises(ValidationError) as exc:
        ForecastConfig(prediction_length=bad)
    assert exc.value.code == "CONFIG_INVALID_PREDICTION_LENGTH"


def test_prediction_length_must_be_an_int_not_a_bool():
    with pytest.raises(ValidationError) as exc:
        ForecastConfig(prediction_length=True)
    assert exc.value.code == "CONFIG_INVALID_PREDICTION_LENGTH"


@pytest.mark.parametrize(
    ("levels", "reason"),
    [
        ([0.1, 0.1, 0.9], "duplicate"),
        ([0.9, 0.5, 0.1], "unsorted"),
        ([0.0, 0.5], "zero is not strictly inside (0, 1)"),
        ([0.5, 1.0], "one is not strictly inside (0, 1)"),
        ([-0.1, 0.5], "negative"),
        ([], "empty"),
    ],
)
def test_bad_quantile_levels_are_rejected(levels, reason):
    with pytest.raises(ValidationError) as exc:
        ForecastConfig(quantile_levels=levels)
    assert exc.value.code == "CONFIG_INVALID_QUANTILES", reason


def test_sorted_unique_in_range_quantiles_are_accepted():
    config = ForecastConfig(quantile_levels=[0.01, 0.5, 0.99])
    assert config.quantile_levels == [0.01, 0.5, 0.99]


@pytest.mark.parametrize("bad", [0, -1])
def test_batch_size_must_be_at_least_one(bad):
    with pytest.raises(ValidationError) as exc:
        ForecastConfig(batch_size=bad)
    assert exc.value.code == "CONFIG_INVALID_BATCH_SIZE"


def test_batch_size_of_one_is_allowed():
    assert ForecastConfig(batch_size=1).batch_size == 1


@pytest.mark.parametrize("bad", [0, -5])
def test_context_length_must_be_positive_when_set(bad):
    with pytest.raises(ValidationError) as exc:
        ForecastConfig(context_length=bad)
    assert exc.value.code == "CONFIG_INVALID_CONTEXT_LENGTH"


def test_context_length_none_means_model_default():
    assert ForecastConfig(context_length=None).context_length is None


def test_unknown_device_is_rejected():
    with pytest.raises(ValidationError) as exc:
        ForecastConfig(device="tpu")
    assert exc.value.code == "CONFIG_INVALID_DEVICE"


@pytest.mark.parametrize("device", ["auto", "cpu", "cuda"])
def test_supported_devices_are_accepted(device):
    assert ForecastConfig(device=device).device == device


def test_empty_target_list_is_rejected():
    with pytest.raises(ValidationError) as exc:
        ForecastConfig(target=[])
    assert exc.value.code == "CONFIG_INVALID_TARGET"


def test_duplicate_target_names_are_rejected():
    with pytest.raises(ValidationError) as exc:
        ForecastConfig(target=["y", "y"])
    assert exc.value.code == "CONFIG_INVALID_TARGET"


def test_target_may_not_collide_with_the_id_or_timestamp_column():
    with pytest.raises(ValidationError) as exc:
        ForecastConfig(id_column="series_id", target="series_id")
    assert exc.value.code == "CONFIG_INVALID_TARGET"


def test_empty_id_column_is_rejected():
    with pytest.raises(ValidationError) as exc:
        ForecastConfig(id_column="")
    assert exc.value.code == "CONFIG_INVALID_COLUMN"


def test_validation_error_carries_code_message_and_details():
    with pytest.raises(ValidationError) as exc:
        ForecastConfig(prediction_length=0)
    err = exc.value
    assert err.code == "CONFIG_INVALID_PREDICTION_LENGTH"
    assert err.details["prediction_length"] == 0
    assert err.code in str(err)
    assert isinstance(err, ValueError)
