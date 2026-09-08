"""DIMER-side validation, RFC rules 1-21.

Every rule gets a passing fixture and a failing fixture. The passing fixture is
what makes each failing assertion discriminating: it proves the rule can be
satisfied, so a rejection is attributable to the mutation under test rather than
to the fixture being malformed in some other way.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chronos2_pipeline import ForecastConfig, ValidationError
from chronos2_pipeline.model import EXPECTED_TRAINED_QUANTILES
from chronos2_pipeline.validation import (
    MIN_OBSERVATIONS,
    ResourceLimits,
    quantiles_in_grid,
    validate_forecast_request,
)
from conftest import MODEL_CONTEXT_LENGTH, MODEL_PREDICTION_LENGTH, make_series


def run(history, config=None, **kwargs):
    """Validate with the pinned model's real limits and quantile grid."""
    kwargs.setdefault("trained_quantiles", EXPECTED_TRAINED_QUANTILES)
    kwargs.setdefault("model_context_length", MODEL_CONTEXT_LENGTH)
    kwargs.setdefault("model_prediction_length", MODEL_PREDICTION_LENGTH)
    return validate_forecast_request(history, config or ForecastConfig(), **kwargs)


def code_of(excinfo) -> str:
    return excinfo.value.code


# --------------------------------------------------------------------------
# The passing baseline every failing case is a mutation of
# --------------------------------------------------------------------------


def test_valid_univariate_request_passes(univariate_history):
    result = run(univariate_history)
    assert result.n_ids == 1
    assert result.n_targets == 1
    assert result.n_covariates == 0
    assert result.frequency == pd.Timedelta(hours=1)
    assert result.n_rows == 64
    assert result.min_series_length == result.max_series_length == 64


def test_valid_multi_id_request_passes(two_id_history):
    result = run(two_id_history)
    assert result.n_ids == 2
    assert sorted(str(i) for i in result.series_ids) == ["A", "B"]
    assert result.frequency == pd.Timedelta(hours=1)


def test_valid_multi_target_request_passes():
    history = make_series()
    history["target2"] = history["target"] * 2.0
    config = ForecastConfig(target=["target", "target2"])
    result = run(history, config)
    assert result.target_names == ["target", "target2"]
    assert result.n_covariates == 0


def test_normalisation_sorts_deterministically_and_resets_the_index(two_id_history):
    shuffled = two_id_history.sample(frac=1.0, random_state=7).reset_index(drop=True)
    result = run(shuffled)
    expected = two_id_history.sort_values(["series_id", "timestamp"], kind="mergesort")
    assert list(result.history["series_id"]) == list(expected["series_id"])
    assert list(result.history["timestamp"]) == list(expected["timestamp"])
    assert list(result.history.index) == list(range(len(result.history)))
    # Sorting twice is a fixed point.
    assert run(result.history).history.equals(result.history)


# --------------------------------------------------------------------------
# Rule 1 — required columns
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dropped", ["series_id", "timestamp", "target"])
def test_rule01_missing_required_column_rejected(univariate_history, dropped):
    with pytest.raises(ValidationError) as exc:
        run(univariate_history.drop(columns=[dropped]))
    assert code_of(exc) == "MISSING_COLUMNS"
    assert dropped in exc.value.details["missing"]


def test_rule01_custom_column_names_are_honoured():
    history = make_series().rename(
        columns={"series_id": "item", "timestamp": "ts", "target": "y"}
    )
    result = run(history, ForecastConfig(id_column="item", timestamp_column="ts", target="y"))
    assert result.n_ids == 1


# --------------------------------------------------------------------------
# Rule 2 — timestamps parse
# --------------------------------------------------------------------------


def test_rule02_unparseable_timestamp_rejected(univariate_history):
    bad = univariate_history.copy()
    bad["timestamp"] = bad["timestamp"].astype(str)
    bad.loc[3, "timestamp"] = "not-a-date"
    with pytest.raises(ValidationError) as exc:
        run(bad)
    assert code_of(exc) == "TIMESTAMP_UNPARSEABLE"


def test_rule02_iso_strings_parse(univariate_history):
    as_strings = univariate_history.copy()
    as_strings["timestamp"] = as_strings["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    result = run(as_strings)
    assert pd.api.types.is_datetime64_any_dtype(result.history["timestamp"])


def test_rule02_null_timestamp_rejected(univariate_history):
    bad = univariate_history.copy()
    bad.loc[5, "timestamp"] = pd.NaT
    with pytest.raises(ValidationError) as exc:
        run(bad)
    assert code_of(exc) == "TIMESTAMP_UNPARSEABLE"


# --------------------------------------------------------------------------
# Rule 3 — non-null ids
# --------------------------------------------------------------------------


def test_rule03_null_id_rejected(univariate_history):
    bad = univariate_history.copy()
    bad.loc[2, "series_id"] = None
    with pytest.raises(ValidationError) as exc:
        run(bad)
    assert code_of(exc) == "NULL_IDS"


# --------------------------------------------------------------------------
# Rules 4 and 19 — numeric targets, explicit missing-value policy
# --------------------------------------------------------------------------


def test_rule04_non_coercible_target_rejected_under_strict_policy(univariate_history):
    bad = univariate_history.copy()
    bad["target"] = bad["target"].astype(object)
    bad.loc[10, "target"] = "n/a"
    with pytest.raises(ValidationError) as exc:
        run(bad)
    assert code_of(exc) == "TARGET_NOT_NUMERIC"
    assert exc.value.details["n_offending"] == 1


def test_rule04_numeric_strings_are_coerced(univariate_history):
    coercible = univariate_history.copy()
    coercible["target"] = coercible["target"].map(lambda v: f"{v:.6f}")
    result = run(coercible)
    assert result.history["target"].dtype == np.float64
    np.testing.assert_allclose(
        result.history["target"].to_numpy(),
        univariate_history["target"].to_numpy(),
        rtol=0,
        atol=1e-5,
    )


def test_rule19_null_target_rejected_not_imputed(univariate_history):
    bad = univariate_history.copy()
    bad.loc[7, "target"] = np.nan
    with pytest.raises(ValidationError) as exc:
        run(bad)
    assert code_of(exc) == "TARGET_MISSING_VALUES"


def test_rule19_infinite_target_rejected(univariate_history):
    bad = univariate_history.copy()
    bad.loc[7, "target"] = np.inf
    with pytest.raises(ValidationError) as exc:
        run(bad)
    assert code_of(exc) == "TARGET_NOT_FINITE"


def test_unknown_target_policy_rejected(univariate_history):
    with pytest.raises(ValidationError) as exc:
        run(univariate_history, target_policy="coerce")
    assert code_of(exc) == "TARGET_POLICY_UNSUPPORTED"


# --------------------------------------------------------------------------
# Rule 5 — duplicate observations
# --------------------------------------------------------------------------


def test_rule05_duplicate_id_timestamp_rejected(univariate_history):
    bad = pd.concat([univariate_history, univariate_history.iloc[[4]]], ignore_index=True)
    with pytest.raises(ValidationError) as exc:
        run(bad)
    assert code_of(exc) == "DUPLICATE_OBSERVATIONS"
    assert exc.value.details["n_duplicate_rows"] == 2


def test_rule05_same_timestamp_in_two_series_is_not_a_duplicate(two_id_history):
    assert run(two_id_history).n_ids == 2


# --------------------------------------------------------------------------
# Rules 7, 9 — regularity and gaps, and rule 8 — shared frequency
# --------------------------------------------------------------------------


def test_rule07_irregular_series_rejected(univariate_history):
    bad = univariate_history.copy()
    bad.loc[20, "timestamp"] = bad.loc[20, "timestamp"] + pd.Timedelta(minutes=17)
    with pytest.raises(ValidationError) as exc:
        run(bad)
    assert code_of(exc) == "IRREGULAR_FREQUENCY"


def test_rule09_gap_rejected_and_reported_as_a_gap(univariate_history):
    gapped = univariate_history.drop(index=[30, 31]).reset_index(drop=True)
    with pytest.raises(ValidationError) as exc:
        run(gapped)
    assert code_of(exc) == "SERIES_GAP"
    assert exc.value.details["base_interval"] == "0 days 01:00:00"


def test_rule08_mixed_frequency_across_series_rejected():
    hourly = make_series("A", n=48, freq="h", seed=1)
    daily = make_series("B", n=48, freq="D", seed=2)
    with pytest.raises(ValidationError) as exc:
        run(pd.concat([hourly, daily], ignore_index=True))
    assert code_of(exc) == "MIXED_FREQUENCY"
    assert len(exc.value.details["distinct"]) == 2


def test_rule08_same_frequency_across_series_passes(two_id_history):
    assert run(two_id_history).frequency == pd.Timedelta(hours=1)


@pytest.mark.parametrize("freq", ["h", "15min", "D"])
def test_regular_series_at_several_frequencies_pass(freq):
    result = run(make_series(n=20, freq=freq))
    assert result.frequency == pd.Timedelta(pd.tseries.frequencies.to_offset(freq).nanos, unit="ns")


# --------------------------------------------------------------------------
# Rule 10 — minimum observations
# --------------------------------------------------------------------------


def test_rule10_two_point_series_rejected():
    with pytest.raises(ValidationError) as exc:
        run(make_series(n=2))
    assert code_of(exc) == "SERIES_TOO_SHORT"
    assert exc.value.details["minimum"] == MIN_OBSERVATIONS


def test_rule10_three_point_series_is_the_boundary_and_passes():
    assert run(make_series(n=3)).min_series_length == 3


def test_rule10_short_series_rejected_even_when_another_series_is_long():
    mixed = pd.concat([make_series("A", n=48), make_series("B", n=2)], ignore_index=True)
    with pytest.raises(ValidationError) as exc:
        run(mixed)
    assert code_of(exc) == "SERIES_TOO_SHORT"
    assert exc.value.details["series_id"] == "B"


# --------------------------------------------------------------------------
# Rule 11 — non-empty history / minimum context
# --------------------------------------------------------------------------


def test_rule11_empty_history_rejected(univariate_history):
    with pytest.raises(ValidationError) as exc:
        run(univariate_history.iloc[0:0])
    assert code_of(exc) == "EMPTY_HISTORY"


def test_rule11_context_length_below_minimum_rejected(univariate_history):
    with pytest.raises(ValidationError) as exc:
        run(univariate_history, ForecastConfig(context_length=2))
    assert code_of(exc) == "CONTEXT_TOO_SHORT"


def test_rule11_reasonable_context_length_passes(univariate_history):
    assert run(univariate_history, ForecastConfig(context_length=48)).n_rows == 64


# --------------------------------------------------------------------------
# Rule 12 — horizon limits
# --------------------------------------------------------------------------


def test_rule12_prediction_length_over_the_dimer_guard_rejected(univariate_history):
    limits = ResourceLimits(max_prediction_length=12)
    with pytest.raises(ValidationError) as exc:
        run(univariate_history, ForecastConfig(prediction_length=13), limits=limits)
    assert code_of(exc) == "PREDICTION_LENGTH_LIMIT"


def test_rule12_prediction_length_at_the_guard_passes(univariate_history):
    limits = ResourceLimits(max_prediction_length=12)
    assert run(univariate_history, ForecastConfig(prediction_length=12), limits=limits).n_ids == 1


def test_horizon_beyond_the_model_native_length_rejected_by_default(univariate_history):
    config = ForecastConfig(prediction_length=MODEL_PREDICTION_LENGTH + 1)
    with pytest.raises(ValidationError) as exc:
        run(univariate_history, config, limits=ResourceLimits(max_prediction_length=99999))
    assert code_of(exc) == "PREDICTION_LENGTH_EXCEEDS_MODEL"


def test_horizon_beyond_the_model_native_length_allowed_when_unroll_opted_in(univariate_history):
    config = ForecastConfig(prediction_length=MODEL_PREDICTION_LENGTH + 1)
    result = run(
        univariate_history,
        config,
        limits=ResourceLimits(max_prediction_length=99999),
        allow_unroll=True,
    )
    assert result.n_ids == 1


# --------------------------------------------------------------------------
# Rules 13 and 14 — trained quantile grid
# --------------------------------------------------------------------------


def test_quantiles_in_grid_helper_reports_only_the_missing_levels():
    assert quantiles_in_grid([0.1, 0.5, 0.9], EXPECTED_TRAINED_QUANTILES) == []
    assert quantiles_in_grid([0.001, 0.5], EXPECTED_TRAINED_QUANTILES) == [0.001]
    # 0.02 sits between the trained 0.01 and 0.05 and is not a member.
    assert quantiles_in_grid([0.02], EXPECTED_TRAINED_QUANTILES) == [0.02]


@pytest.mark.parametrize("level", [0.001, 0.02, 0.125, 0.999])
def test_rule14_out_of_grid_quantile_hard_fails_by_default(univariate_history, level):
    config = ForecastConfig(quantile_levels=sorted({level, 0.5}))
    with pytest.raises(ValidationError) as exc:
        run(univariate_history, config)
    assert code_of(exc) == "QUANTILE_NOT_IN_GRID"
    assert exc.value.details["out_of_grid"] == [level]


@pytest.mark.parametrize("level", [0.01, 0.15, 0.5, 0.95, 0.99])
def test_rule13_in_grid_quantiles_pass(univariate_history, level):
    config = ForecastConfig(quantile_levels=[level])
    assert run(univariate_history, config).requested_quantiles == [level]


def test_rule14_out_of_grid_can_be_opted_into_explicitly(univariate_history):
    config = ForecastConfig(quantile_levels=[0.001, 0.5])
    result = run(univariate_history, config, allow_out_of_grid=True)
    assert result.requested_quantiles == [0.001, 0.5]


def test_grid_membership_uses_the_grid_it_is_given_not_a_constant(univariate_history):
    """A model advertising a coarser grid must reject 0.1, proving the grid is data."""
    with pytest.raises(ValidationError) as exc:
        run(univariate_history, ForecastConfig(quantile_levels=[0.1]), trained_quantiles=(0.5,))
    assert code_of(exc) == "QUANTILE_NOT_IN_GRID"
    assert exc.value.details["trained_quantiles"] == [0.5]


# --------------------------------------------------------------------------
# Rules 15-18 — the future-covariate table
# --------------------------------------------------------------------------


def future_table(history: pd.DataFrame, horizon: int, freq: str = "h") -> pd.DataFrame:
    rows = []
    for series_id, block in history.groupby("series_id"):
        origin = block["timestamp"].max()
        step = pd.tseries.frequencies.to_offset(freq)
        stamps = pd.date_range(origin + step, periods=horizon, freq=freq)
        rows.append(
            pd.DataFrame(
                {
                    "series_id": series_id,
                    "timestamp": stamps,
                    "temperature": np.arange(horizon, dtype=float),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def covariate_history(two_id_history):
    history = two_id_history.copy()
    history["temperature"] = np.linspace(20.0, 25.0, len(history))
    return history


def run_with_future(history, future, horizon=8, **kwargs):
    config = ForecastConfig(prediction_length=horizon)
    return run(history, config, future_df=future, **kwargs)


def test_valid_future_table_passes(covariate_history):
    future = future_table(covariate_history, horizon=8)
    result = run_with_future(covariate_history, future)
    assert result.future is not None
    assert len(result.future) == 16
    assert result.covariate_names == ["temperature"]


def test_rule16_wrong_number_of_future_rows_rejected(covariate_history):
    future = future_table(covariate_history, horizon=8).drop(index=[0]).reset_index(drop=True)
    with pytest.raises(ValidationError) as exc:
        run_with_future(covariate_history, future)
    assert code_of(exc) == "FUTURE_LENGTH_MISMATCH"


def test_rule17_future_column_absent_from_history_rejected(covariate_history):
    future = future_table(covariate_history, horizon=8).rename(columns={"temperature": "humidity"})
    with pytest.raises(ValidationError) as exc:
        run_with_future(covariate_history, future)
    assert code_of(exc) == "FUTURE_COLUMN_NOT_IN_HISTORY"
    assert exc.value.details["extra_columns"] == ["humidity"]


def test_rule18_future_target_leakage_rejected(covariate_history):
    future = future_table(covariate_history, horizon=8)
    future["target"] = 1.0
    with pytest.raises(ValidationError) as exc:
        run_with_future(covariate_history, future)
    assert code_of(exc) == "FUTURE_TARGET_LEAKAGE"
    assert exc.value.details["leaked_columns"] == ["target"]


def test_future_ids_must_match_historical_ids(covariate_history):
    future = future_table(covariate_history, horizon=8)
    future.loc[future["series_id"] == "B", "series_id"] = "C"
    with pytest.raises(ValidationError) as exc:
        run_with_future(covariate_history, future)
    assert code_of(exc) == "FUTURE_ID_MISMATCH"


def test_rule15_future_timestamps_must_start_at_the_forecast_origin(covariate_history):
    future = future_table(covariate_history, horizon=8)
    future["timestamp"] = future["timestamp"] + pd.Timedelta(hours=1)
    with pytest.raises(ValidationError) as exc:
        run_with_future(covariate_history, future)
    assert code_of(exc) == "FUTURE_TIMESTAMP_MISALIGNED"


def test_rule15_future_timestamps_must_follow_the_validated_frequency(covariate_history):
    future = future_table(covariate_history, horizon=8)
    future.loc[4, "timestamp"] = future.loc[4, "timestamp"] + pd.Timedelta(minutes=30)
    with pytest.raises(ValidationError) as exc:
        run_with_future(covariate_history, future)
    assert code_of(exc) == "FUTURE_TIMESTAMP_MISALIGNED"


def test_future_table_missing_id_column_rejected(covariate_history):
    future = future_table(covariate_history, horizon=8).drop(columns=["series_id"])
    with pytest.raises(ValidationError) as exc:
        run_with_future(covariate_history, future)
    assert code_of(exc) == "FUTURE_MISSING_COLUMNS"


# --------------------------------------------------------------------------
# Rule 20 — resource guards
# --------------------------------------------------------------------------


def test_rule20_too_many_ids_rejected():
    history = pd.concat(
        [make_series(f"S{i}", n=4, seed=i) for i in range(5)], ignore_index=True
    )
    with pytest.raises(ValidationError) as exc:
        run(history, limits=ResourceLimits(max_ids=4))
    assert code_of(exc) == "RESOURCE_LIMIT"
    assert exc.value.details["guard"] == "max_ids"


def test_rule20_ids_at_the_guard_pass():
    history = pd.concat(
        [make_series(f"S{i}", n=4, seed=i) for i in range(4)], ignore_index=True
    )
    assert run(history, limits=ResourceLimits(max_ids=4)).n_ids == 4


def test_rule20_too_many_targets_rejected():
    history = make_series()
    for i in range(3):
        history[f"t{i}"] = 1.0
    config = ForecastConfig(target=["target", "t0", "t1", "t2"])
    with pytest.raises(ValidationError) as exc:
        run(history, config, limits=ResourceLimits(max_targets=3))
    assert exc.value.details["guard"] == "max_targets"


def test_rule20_too_many_covariates_rejected(univariate_history):
    history = univariate_history.copy()
    for i in range(3):
        history[f"c{i}"] = 1.0
    with pytest.raises(ValidationError) as exc:
        run(history, limits=ResourceLimits(max_covariates=2))
    assert exc.value.details["guard"] == "max_covariates"


def test_rule20_context_length_over_the_guard_rejected(univariate_history):
    with pytest.raises(ValidationError) as exc:
        run(univariate_history, ForecastConfig(context_length=9000))
    assert code_of(exc) == "RESOURCE_LIMIT"
    assert exc.value.details["guard"] == "max_context_length"


def test_rule20_row_count_guard(univariate_history):
    with pytest.raises(ValidationError) as exc:
        run(univariate_history, limits=ResourceLimits(max_rows=10))
    assert exc.value.details["guard"] == "max_rows"


# --------------------------------------------------------------------------
# Rule 21 — batch size accounts for targets plus covariates
# --------------------------------------------------------------------------


def test_rule21_batch_size_must_cover_targets_plus_covariates(univariate_history):
    history = univariate_history.copy()
    history["temperature"] = 1.0
    history["holiday"] = 0.0
    # 1 target + 2 covariates = 3
    with pytest.raises(ValidationError) as exc:
        run(history, ForecastConfig(batch_size=2))
    assert code_of(exc) == "BATCH_SIZE_TOO_SMALL"
    assert exc.value.details["required"] == 3


def test_rule21_batch_size_exactly_at_the_requirement_passes(univariate_history):
    history = univariate_history.copy()
    history["temperature"] = 1.0
    history["holiday"] = 0.0
    assert run(history, ForecastConfig(batch_size=3)).n_covariates == 2


def test_rule21_univariate_no_covariates_needs_only_batch_size_one(univariate_history):
    assert run(univariate_history, ForecastConfig(batch_size=1)).n_targets == 1


# --------------------------------------------------------------------------
# RFC C-3 — an explicit `frequency` does not bypass rules 7-10
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    [
        ("irregular", "IRREGULAR_FREQUENCY"),
        ("gap", "SERIES_GAP"),
        ("short", "SERIES_TOO_SHORT"),
    ],
)
def test_explicit_frequency_does_not_bypass_regularity_rules(mutate, expected_code):
    history = make_series(n=48)
    if mutate == "irregular":
        history.loc[20, "timestamp"] += pd.Timedelta(minutes=17)
    elif mutate == "gap":
        history = history.drop(index=[30]).reset_index(drop=True)
    else:
        history = history.iloc[:2].reset_index(drop=True)

    with pytest.raises(ValidationError) as exc:
        run(history, ForecastConfig(frequency="h"))
    assert code_of(exc) == expected_code


def test_declared_frequency_that_contradicts_the_data_is_rejected(univariate_history):
    with pytest.raises(ValidationError) as exc:
        run(univariate_history, ForecastConfig(frequency="D"))
    assert code_of(exc) == "FREQUENCY_MISMATCH"
    assert exc.value.details["observed_interval"] == "0 days 01:00:00"


def test_declared_frequency_matching_the_data_passes(univariate_history):
    assert run(univariate_history, ForecastConfig(frequency="h")).frequency == pd.Timedelta(hours=1)


def test_unparseable_declared_frequency_is_rejected(univariate_history):
    with pytest.raises(ValidationError) as exc:
        run(univariate_history, ForecastConfig(frequency="every-other-tuesday"))
    assert code_of(exc) == "FREQUENCY_UNPARSEABLE"


def test_non_dataframe_history_rejected():
    with pytest.raises(ValidationError) as exc:
        run([{"series_id": "A"}])
    assert code_of(exc) == "HISTORY_NOT_A_DATAFRAME"
