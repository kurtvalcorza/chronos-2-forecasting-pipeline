"""Export metadata: what model, what runtime, what request.

A forecast without this block is not reproducible. Everything recorded here is
read from the live objects — resolved library versions, the device the weights
actually landed on, the context and horizon the model actually used — never from
a constant or from what the caller asked for.
"""

from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from .model import ModelIdentity

__all__ = ["PROVENANCE_SCHEMA_VERSION", "TRACKED_PACKAGES", "runtime_versions", "build_provenance"]

#: Bump when the shape of the exported dict changes in a way consumers must notice.
PROVENANCE_SCHEMA_VERSION = "1.0"

#: RFC "Runtime/dependency reproducibility": these versions travel with results.
TRACKED_PACKAGES: tuple[str, ...] = (
    "chronos-forecasting",
    "torch",
    "transformers",
    "huggingface-hub",
    "numpy",
    "pandas",
    "accelerate",
    "einops",
    "safetensors",
)


def runtime_versions(packages: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str | None]:
    """Resolved distribution versions, ``None`` for anything not installed."""
    resolved: dict[str, str | None] = {"python": platform.python_version()}
    for name in packages:
        try:
            resolved[name] = _pkg_version(name)
        except PackageNotFoundError:
            resolved[name] = None
    return resolved


def build_provenance(
    identity: ModelIdentity,
    *,
    device: str,
    dtype: str,
    n_ids: int,
    n_targets: int,
    n_covariates: int,
    requested_context_length: int | None,
    effective_context_length: int,
    longest_series_length: int,
    shortest_series_length: int,
    requested_prediction_length: int,
    effective_prediction_length: int,
    autoregressive_unrolled: bool,
    requested_quantiles: list[float],
    effective_quantiles: list[float],
    batch_size: int,
    cross_learning: bool,
    latency_seconds: float,
    warm_up_performed: bool,
    frequency: str,
    observed_frequency: str,
) -> dict[str, Any]:
    """Assemble the three-block export metadata dict.

    Notes
    -----
    ``effective_quantiles`` is the list actually present in the output. In v1 it
    always equals ``requested_quantiles``, because out-of-grid levels are refused
    rather than clamped — but both are exported so a future release that permits
    clamping cannot do it invisibly.

    ``effective_context_length`` is the context bound actually in force: the
    request, the model's limit and the longest series in the request, whichever
    binds first. It is not a promise that every series contributed that much —
    a series shorter than it contributed its whole history and no more, which is
    what ``shortest_series_length`` is for.

    ``latency_seconds`` times the scored ``predict_df`` call only.
    ``warm_up_performed`` says whether a discarded warm-up call preceded it; a
    cold first call includes lazy CUDA/kernel initialisation and is not a
    comparable latency figure.
    """
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "model": identity.as_dict(),
        "runtime": {
            **runtime_versions(),
            "platform": platform.platform(),
            "device": device,
            "dtype": dtype,
        },
        "inference": {
            "n_ids": n_ids,
            "n_targets": n_targets,
            "n_covariates": n_covariates,
            "requested_context_length": requested_context_length,
            "effective_context_length": effective_context_length,
            "model_context_length": identity.model_context_length,
            "longest_series_length": longest_series_length,
            "shortest_series_length": shortest_series_length,
            "requested_prediction_length": requested_prediction_length,
            "effective_prediction_length": effective_prediction_length,
            "model_prediction_length": identity.model_prediction_length,
            "autoregressive_unrolled": autoregressive_unrolled,
            "requested_quantile_levels": list(requested_quantiles),
            "effective_quantile_levels": list(effective_quantiles),
            "batch_size": batch_size,
            "cross_learning": cross_learning,
            "declared_frequency": frequency,
            "observed_frequency": observed_frequency,
            "latency_seconds": latency_seconds,
            "warm_up_performed": warm_up_performed,
        },
    }
