"""Zero-shot time-series forecasting with a pinned Amazon Chronos-2 checkpoint.

Phase 1 (foundation): pinned verified loader, DIMER-side validation, univariate
multi-ID inference, deterministic output normalisation, provenance export.

Phase 2 (inference): multi-target forecasting, past-only and known-future
covariates, and the row-layout oracle that keeps a forecast attached to the
series and target it was made for.

Phase 3 (tutorial/evaluation): chronological holdout evaluation, naive
baselines, deterministic sample assets, and the live tutorial path.
"""

from __future__ import annotations

from .config import DEFAULT_QUANTILE_LEVELS, ForecastConfig
from .errors import (
    Chronos2PipelineError,
    HubUnavailableError,
    ModelIntegrityError,
    ModelSourceError,
    UpstreamContractError,
    ValidationError,
)
from .evaluation import (
    EvaluationResult,
    HoldoutSplit,
    chronological_holdout,
    evaluate_forecast,
    evaluation_report,
    last_value_baseline,
    seasonal_naive_baseline,
)
from .inference import ForecastResult, build_rename_map, forecast, normalized_columns
from .model import (
    DEFAULT_WEIGHTS_DIR,
    EXPECTED_TRAINED_QUANTILES,
    MANIFEST_NAME,
    MODEL_ID,
    MODEL_KEY,
    MODEL_LICENSE,
    MODEL_REVISION,
    PINNED_CONFIG_SHA256,
    PINNED_LICENSE,
    PINNED_MODEL_ID,
    PINNED_REVISION,
    PINNED_WEIGHTS_BYTES,
    PINNED_WEIGHTS_SHA256,
    LoadedModel,
    ModelIdentity,
    load_pinned_model,
    resolve_hub_revision,
    stage_missing_files,
    verify_snapshot,
)
from .provenance import build_provenance, runtime_versions
from .validation import (
    INPUT_SCHEMA,
    ResourceLimits,
    ValidationResult,
    validate_forecast_request,
    validate_inputs,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # config
    "ForecastConfig",
    "DEFAULT_QUANTILE_LEVELS",
    # errors
    "Chronos2PipelineError",
    "ModelSourceError",
    "ModelIntegrityError",
    "HubUnavailableError",
    "ValidationError",
    "UpstreamContractError",
    # model
    "load_pinned_model",
    "verify_snapshot",
    "stage_missing_files",
    "DEFAULT_WEIGHTS_DIR",
    "MANIFEST_NAME",
    "MODEL_ID",
    "MODEL_REVISION",
    "MODEL_LICENSE",
    "MODEL_KEY",
    "resolve_hub_revision",
    "LoadedModel",
    "ModelIdentity",
    "PINNED_MODEL_ID",
    "PINNED_REVISION",
    "PINNED_CONFIG_SHA256",
    "PINNED_WEIGHTS_SHA256",
    "PINNED_WEIGHTS_BYTES",
    "PINNED_LICENSE",
    "EXPECTED_TRAINED_QUANTILES",
    # validation
    "validate_forecast_request",
    "validate_inputs",
    "INPUT_SCHEMA",
    "ValidationResult",
    "ResourceLimits",
    # inference
    "forecast",
    "ForecastResult",
    "build_rename_map",
    "normalized_columns",
    # evaluation
    "chronological_holdout",
    "evaluate_forecast",
    "evaluation_report",
    "last_value_baseline",
    "seasonal_naive_baseline",
    "HoldoutSplit",
    "EvaluationResult",
    # provenance
    "build_provenance",
    "runtime_versions",
]
