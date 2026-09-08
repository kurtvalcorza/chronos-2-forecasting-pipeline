"""User-facing forecast configuration.

Mirrors the RFC's "User parameters" block one field at a time. Validation here
covers only the scalar fields — anything that needs the data or the model
(quantile-grid membership, frequency, resource limits) belongs to
:mod:`chronos2_pipeline.validation`, which runs later and has both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .errors import ValidationError

__all__ = ["ForecastConfig", "DEFAULT_QUANTILE_LEVELS", "SUPPORTED_DEVICES"]

#: RFC default. Kept as a module constant so tests and docs cite one source.
DEFAULT_QUANTILE_LEVELS: tuple[float, ...] = (0.1, 0.5, 0.9)

SUPPORTED_DEVICES: tuple[str, ...] = ("auto", "cpu", "cuda")

DeviceName = Literal["auto", "cpu", "cuda"]


@dataclass(frozen=True)
class ForecastConfig:
    """Configuration for one forecast request.

    Field defaults are exactly the RFC "User parameters" YAML block.

    Notes
    -----
    ``cross_learning`` stays ``False`` in v1 and is not a beginner knob. When it
    is ever flipped, both its value and ``batch_size`` must be exported, because
    cross-learning makes results depend on batch composition.
    """

    id_column: str = "series_id"
    timestamp_column: str = "timestamp"
    target: str | list[str] = "target"
    prediction_length: int = 24
    quantile_levels: list[float] = field(default_factory=lambda: list(DEFAULT_QUANTILE_LEVELS))
    batch_size: int = 256
    context_length: int | None = None
    frequency: str | None = None
    device: DeviceName = "auto"
    cross_learning: bool = False

    def __post_init__(self) -> None:
        self._validate_columns()
        self._validate_prediction_length()
        self._validate_quantiles()
        self._validate_batch_size()
        self._validate_context_length()
        self._validate_device()

    # -- accessors ---------------------------------------------------------

    @property
    def target_names(self) -> list[str]:
        """``target`` normalised to a list, preserving order."""
        if isinstance(self.target, str):
            return [self.target]
        return list(self.target)

    @property
    def n_targets(self) -> int:
        return len(self.target_names)

    @property
    def reserved_columns(self) -> list[str]:
        """Columns that are structural rather than covariates."""
        return [self.id_column, self.timestamp_column, *self.target_names]

    # -- scalar validation -------------------------------------------------

    def _validate_columns(self) -> None:
        names = {
            "id_column": self.id_column,
            "timestamp_column": self.timestamp_column,
        }
        for label, value in names.items():
            if not isinstance(value, str) or not value:
                raise ValidationError(
                    "CONFIG_INVALID_COLUMN",
                    f"{label} must be a non-empty string, got {value!r}.",
                    {"field": label, "value": value},
                )

        targets = self.target
        if isinstance(targets, str):
            targets = [targets]
        if not isinstance(targets, list) or not targets:
            raise ValidationError(
                "CONFIG_INVALID_TARGET",
                "target must be a non-empty string or a non-empty list of strings.",
                {"target": self.target},
            )
        if any(not isinstance(t, str) or not t for t in targets):
            raise ValidationError(
                "CONFIG_INVALID_TARGET",
                "every target name must be a non-empty string.",
                {"target": self.target},
            )
        if len(set(targets)) != len(targets):
            raise ValidationError(
                "CONFIG_INVALID_TARGET",
                "target names must be unique.",
                {"target": self.target},
            )
        overlap = set(targets) & {self.id_column, self.timestamp_column}
        if overlap:
            raise ValidationError(
                "CONFIG_INVALID_TARGET",
                f"target names collide with the id/timestamp columns: {sorted(overlap)}.",
                {"collisions": sorted(overlap)},
            )

    def _validate_prediction_length(self) -> None:
        value = self.prediction_length
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValidationError(
                "CONFIG_INVALID_PREDICTION_LENGTH",
                f"prediction_length must be an int, got {type(value).__name__}.",
                {"prediction_length": value},
            )
        if value <= 0:
            raise ValidationError(
                "CONFIG_INVALID_PREDICTION_LENGTH",
                f"prediction_length must be > 0, got {value}.",
                {"prediction_length": value},
            )

    def _validate_quantiles(self) -> None:
        levels = self.quantile_levels
        if not isinstance(levels, list) or not levels:
            raise ValidationError(
                "CONFIG_INVALID_QUANTILES",
                "quantile_levels must be a non-empty list of floats.",
                {"quantile_levels": levels},
            )
        if any(isinstance(q, bool) or not isinstance(q, (int, float)) for q in levels):
            raise ValidationError(
                "CONFIG_INVALID_QUANTILES",
                "quantile_levels must contain only numbers.",
                {"quantile_levels": levels},
            )
        if len(set(levels)) != len(levels):
            raise ValidationError(
                "CONFIG_INVALID_QUANTILES",
                f"quantile_levels must be unique, got {levels}.",
                {"quantile_levels": levels},
            )
        if list(levels) != sorted(levels):
            raise ValidationError(
                "CONFIG_INVALID_QUANTILES",
                f"quantile_levels must be sorted ascending, got {levels}.",
                {"quantile_levels": levels},
            )
        out_of_range = [q for q in levels if not (0.0 < float(q) < 1.0)]
        if out_of_range:
            raise ValidationError(
                "CONFIG_INVALID_QUANTILES",
                f"quantile_levels must lie strictly inside (0, 1); offenders: {out_of_range}.",
                {"out_of_range": out_of_range},
            )

    def _validate_batch_size(self) -> None:
        value = self.batch_size
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValidationError(
                "CONFIG_INVALID_BATCH_SIZE",
                f"batch_size must be an int, got {type(value).__name__}.",
                {"batch_size": value},
            )
        if value < 1:
            raise ValidationError(
                "CONFIG_INVALID_BATCH_SIZE",
                f"batch_size must be >= 1, got {value}.",
                {"batch_size": value},
            )

    def _validate_context_length(self) -> None:
        value = self.context_length
        if value is None:
            return
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValidationError(
                "CONFIG_INVALID_CONTEXT_LENGTH",
                f"context_length must be an int or None, got {type(value).__name__}.",
                {"context_length": value},
            )
        if value <= 0:
            raise ValidationError(
                "CONFIG_INVALID_CONTEXT_LENGTH",
                f"context_length must be > 0 when set, got {value}.",
                {"context_length": value},
            )

    def _validate_device(self) -> None:
        if self.device not in SUPPORTED_DEVICES:
            raise ValidationError(
                "CONFIG_INVALID_DEVICE",
                f"device must be one of {list(SUPPORTED_DEVICES)}, got {self.device!r}.",
                {"device": self.device},
            )
        if not isinstance(self.cross_learning, bool):
            raise ValidationError(
                "CONFIG_INVALID_CROSS_LEARNING",
                f"cross_learning must be a bool, got {type(self.cross_learning).__name__}.",
                {"cross_learning": self.cross_learning},
            )
