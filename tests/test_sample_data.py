from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pandas as pd

from chronos2_pipeline import (
    EXPECTED_TRAINED_QUANTILES,
    ForecastConfig,
    validate_forecast_request,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "examples" / "sample-data"
GENERATOR = SAMPLE_DIR / "generate_samples.py"

EXPECTED = {
    "chronos_covariates_future.csv": (
        "fc9383639d3e66c3178f1e0aa93af84b26d92266afbadb441f0bf9e7ecb6d076"
    ),
    "chronos_covariates_history.csv": (
        "f12e248646d9fb21502fb63b96136f4dc34ee8f1be066973ff40dc99af31ece0"
    ),
    "chronos_multi_series.csv": (
        "6f12f1475411aaab13a2de860ba50f008bd7185a9266f99147c244db4a5de8b3"
    ),
    "chronos_univariate.csv": (
        "eff96b1a9bec5a81aa4021b4d308c29e18fc89be0cf8eb54e755c2efe535c7f7"
    ),
}


def load_generator():
    spec = importlib.util.spec_from_file_location("chronos_sample_generator", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_generator_reproduces_every_recorded_artifact(tmp_path: Path) -> None:
    generator = load_generator()
    digests = generator.generate(tmp_path)

    assert digests == EXPECTED
    assert {name: sha256(tmp_path / name) for name in EXPECTED} == EXPECTED
    assert (tmp_path / "SHA256SUMS").read_text(encoding="utf-8") == (
        SAMPLE_DIR / "SHA256SUMS"
    ).read_text(encoding="utf-8")
    assert (tmp_path / "SHA256SUMS.generated").read_text(encoding="utf-8") == (
        SAMPLE_DIR / "SHA256SUMS.generated"
    ).read_text(encoding="utf-8")


def test_checked_in_manifest_is_verifiable_from_a_clean_checkout() -> None:
    lines = (SAMPLE_DIR / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert lines
    for line in lines:
        digest, name = line.split(maxsplit=1)
        path = SAMPLE_DIR / name.strip()
        assert path.is_file()
        assert sha256(path) == digest


def test_generated_manifest_records_only_generated_examples() -> None:
    names = {
        line.split(maxsplit=1)[1].strip()
        for line in (SAMPLE_DIR / "SHA256SUMS.generated")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    }
    assert names == {
        "chronos_multi_series.csv",
        "chronos_covariates_history.csv",
        "chronos_covariates_future.csv",
    }


def test_checked_in_quickstart_sample_is_the_generated_canonical_bytes(tmp_path: Path) -> None:
    generator = load_generator()
    generator.generate(tmp_path)

    assert (tmp_path / "chronos_univariate.csv").read_bytes() == (
        SAMPLE_DIR / "chronos_univariate.csv"
    ).read_bytes()


def test_generated_samples_cover_modes_a_b_and_d_and_pass_dimer_validation(
    tmp_path: Path,
) -> None:
    generator = load_generator()
    generator.generate(tmp_path)

    univariate = pd.read_csv(tmp_path / "chronos_univariate.csv")
    multi = pd.read_csv(tmp_path / "chronos_multi_series.csv")
    history = pd.read_csv(tmp_path / "chronos_covariates_history.csv")
    future = pd.read_csv(tmp_path / "chronos_covariates_future.csv")

    assert len(univariate) == 96
    assert univariate["series_id"].nunique() == 1
    assert len(multi) == 192
    assert multi["series_id"].nunique() == 2
    assert len(history) == 192
    assert len(future) == 48
    assert set(future.columns) == {"series_id", "timestamp", "temperature", "holiday"}
    assert "demand" not in future.columns

    validate_forecast_request(
        univariate,
        ForecastConfig(prediction_length=12),
        trained_quantiles=EXPECTED_TRAINED_QUANTILES,
        model_context_length=8192,
        model_prediction_length=1024,
    )
    multi_result = validate_forecast_request(
        multi,
        ForecastConfig(prediction_length=12),
        trained_quantiles=EXPECTED_TRAINED_QUANTILES,
        model_context_length=8192,
        model_prediction_length=1024,
    )
    assert multi_result.n_ids == 2

    covariate_result = validate_forecast_request(
        history,
        ForecastConfig(target="demand", prediction_length=24),
        trained_quantiles=EXPECTED_TRAINED_QUANTILES,
        model_context_length=8192,
        model_prediction_length=1024,
        future_df=future,
    )
    assert covariate_result.past_covariate_names == []
    assert covariate_result.known_future_covariate_names == ["temperature", "holiday"]
