"""Shared fixtures. Nothing here touches the network or the model weights."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pytest

from chronos2_pipeline.model import EXPECTED_TRAINED_QUANTILES, ModelIdentity

MODEL_CONTEXT_LENGTH = 8192
MODEL_PREDICTION_LENGTH = 1024


def make_series(
    series_id: str = "A",
    n: int = 64,
    start: str = "2026-01-01",
    freq: str = "h",
    seed: int = 0,
) -> pd.DataFrame:
    """One regular hourly series with a mild seasonal shape."""
    rng = np.random.default_rng(seed)
    stamps = pd.date_range(start=start, periods=n, freq=freq)
    t = np.arange(n)
    values = 100.0 + 10.0 * np.sin(2 * np.pi * t / 24.0) + rng.normal(0, 0.5, n)
    return pd.DataFrame(
        {"series_id": series_id, "timestamp": stamps, "target": values}
    )


@pytest.fixture
def univariate_history() -> pd.DataFrame:
    return make_series()


@pytest.fixture
def two_id_history() -> pd.DataFrame:
    return pd.concat(
        [make_series("A", seed=1), make_series("B", seed=2)], ignore_index=True
    )


class SentinelPipeline:
    """Stand-in for ``Chronos2Pipeline`` that records whether it was called.

    Used by the tests that must prove DIMER rejected a request *before* it could
    reach ``predict_df``. ``predict_df`` raises rather than returning, so a test
    that reached it fails on the call, not only on a flag.
    """

    def __init__(self, frame: pd.DataFrame | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._frame = frame

    @property
    def called(self) -> bool:
        return bool(self.calls)

    def predict_df(self, df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        self.calls.append({"df": df, **kwargs})
        if self._frame is None:
            raise AssertionError(
                "predict_df was called; DIMER validation should have rejected this "
                "request before any upstream call."
            )
        return self._frame


@dataclass(frozen=True)
class FakeLoadedModel:
    """Minimal stand-in for ``LoadedModel`` — same attribute surface, no weights."""

    pipeline: Any
    identity: ModelIdentity
    device: str = "cpu"
    dtype: str = "torch.float32"

    @property
    def trained_quantiles(self) -> tuple[float, ...]:
        return self.identity.trained_quantiles

    @property
    def model_context_length(self) -> int:
        return self.identity.model_context_length

    @property
    def model_prediction_length(self) -> int:
        return self.identity.model_prediction_length


def make_identity(**overrides: Any) -> ModelIdentity:
    from chronos2_pipeline import model as model_mod

    defaults: dict[str, Any] = {
        "name": model_mod.PINNED_MODEL_ID,
        "revision": model_mod.PINNED_REVISION,
        "config_sha256": model_mod.PINNED_CONFIG_SHA256,
        "weights_sha256": model_mod.PINNED_WEIGHTS_SHA256,
        "weights_bytes": model_mod.PINNED_WEIGHTS_BYTES,
        "license": model_mod.PINNED_LICENSE,
        "license_source": model_mod.PINNED_LICENSE_SOURCE,
        "source_url": model_mod.PINNED_MODEL_URL,
        "trained_quantiles": EXPECTED_TRAINED_QUANTILES,
        "model_context_length": MODEL_CONTEXT_LENGTH,
        "model_prediction_length": MODEL_PREDICTION_LENGTH,
    }
    defaults.update(overrides)
    return ModelIdentity(**defaults)


@pytest.fixture
def sentinel_model() -> FakeLoadedModel:
    """A model whose ``predict_df`` fails the test if it is ever reached."""
    return FakeLoadedModel(pipeline=SentinelPipeline(), identity=make_identity())


@pytest.fixture
def trained_quantiles() -> tuple[float, ...]:
    return EXPECTED_TRAINED_QUANTILES
