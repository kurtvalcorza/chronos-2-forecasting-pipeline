"""Chronological, leakage-safe evaluation — Phase 3.

Deliberately unimplemented. The RFC's evaluation contract (chronological splits
only, no future targets as covariates, statistics derived from historical
context only, baselines on the identical window) is a design surface that has
not been reviewed yet, and a placeholder metric that silently splits at random
would be worse than nothing.
"""

from __future__ import annotations

from typing import Any, NoReturn

__all__ = ["evaluate_forecast", "seasonal_naive_baseline", "last_value_baseline"]

_PHASE = "Phase 3"


def _not_yet(name: str) -> NoReturn:
    raise NotImplementedError(
        f"{name} is {_PHASE}. See the evaluation contract in docs/rfc/0001-chronos-2.md."
    )


def evaluate_forecast(*args: Any, **kwargs: Any) -> NoReturn:
    """Score a forecast against held-out future truth. Not implemented."""
    _not_yet("evaluate_forecast")


def seasonal_naive_baseline(*args: Any, **kwargs: Any) -> NoReturn:
    """Seasonal-naive baseline on the identical horizon/window. Not implemented."""
    _not_yet("seasonal_naive_baseline")


def last_value_baseline(*args: Any, **kwargs: Any) -> NoReturn:
    """Last-value baseline on the identical horizon/window. Not implemented."""
    _not_yet("last_value_baseline")
