"""Provider error taxonomy.

Adapters translate SDK exceptions into `ProviderError` so the retry layer can
decide without importing provider-specific exception types. `retryable` is the
only field the retry loop reads; `status_code` exists for logging and tests.
"""

from __future__ import annotations

# Status codes worth retrying: rate limits, overload, and transient 5xx.
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})


class ProviderError(Exception):
    """A failed provider call.

    Args:
        message: Human-readable cause.
        status_code: HTTP status when the SDK reported one.
        retryable: Override the status-derived decision. When None it is
            derived from `status_code`; an absent status counts as
            non-retryable, so unknown failures fail fast instead of burning
            budget on repeats.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable if retryable is not None else status_code in RETRYABLE_STATUS


class BudgetExhaustedError(Exception):
    """A call was refused because the task's token budget is spent."""
