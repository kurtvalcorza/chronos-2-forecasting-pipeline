"""Exception hierarchy for the pipeline.

Every failure mode the pipeline is contractually required to detect raises a
distinct, catchable type. Nothing here inherits from ``Warning`` and nothing is
downgraded to a log line: the RFC's position is that a silently-degraded
forecast is worse than no forecast.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "Chronos2PipelineError",
    "ModelSourceError",
    "ModelIntegrityError",
    "ValidationError",
    "UpstreamContractError",
]


class Chronos2PipelineError(Exception):
    """Base class for every error raised by this package."""


class ModelSourceError(Chronos2PipelineError, ValueError):
    """A model source other than the single pinned revision was requested.

    Also inherits :class:`ValueError` so that callers written against the RFC's
    "``ValueError`` subclass" wording keep working.
    """


class ModelIntegrityError(Chronos2PipelineError, ValueError):
    """The downloaded snapshot does not match the pinned revision or digests."""


class ValidationError(Chronos2PipelineError, ValueError):
    """A DIMER-side validation rule rejected the request.

    Parameters
    ----------
    code
        Stable machine-readable identifier for the rule that failed, e.g.
        ``"IRREGULAR_FREQUENCY"``. Callers and tests match on this, never on the
        prose message.
    message
        Human-readable explanation.
    details
        Structured context — offending ids, observed values, limits.
    """

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        self.code = code
        self.message = message
        self.details: dict[str, Any] = dict(details or {})
        super().__init__(f"[{code}] {message}")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ValidationError(code={self.code!r}, message={self.message!r}, "
            f"details={self.details!r})"
        )


class UpstreamContractError(Chronos2PipelineError, RuntimeError):
    """The pinned upstream release behaved differently from the recorded contract.

    Raised, for example, when ``predict_df`` stops returning the median in its
    ``predictions`` column, or drops a column the normalisation map depends on.
    This is deliberately fatal: it is the tripwire that must fail CI when a
    future upstream version changes semantics underneath us.
    """
