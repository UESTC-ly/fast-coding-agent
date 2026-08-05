"""Billing wrapper: retry layer + ledger tracking.

The唯一计费入口. Every provider call — routing classifier, FastPath attempts,
retries, Full Agent, sub-agents — goes through `BilledClient.invoke`, which
records usage into the shared `CostLedger` before returning. Nothing bypasses it.

Retry logic wraps the adapter: transient 429/5xx errors are retried with
exponential backoff; non-retryable failures propagate immediately. Each retry's
tokens still count — a request that succeeds on the third attempt bills for all
three, not just the successful one.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from qqcode.models.errors import BudgetExhaustedError, ProviderError
from qqcode.models.protocol import (
    Budget,
    Completion,
    CostLedger,
    ModelClient,
    ModelTier,
    Msg,
    OutputSpec,
    Phase,
    ToolSpec,
)


@dataclass
class RetryPolicy:
    """Exponential backoff parameters."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    # Override for testing; production default sleeps wall-clock.
    sleep: Callable[[float], None] = time.sleep

    def delays(self) -> list[float]:
        """Compute backoff delays for attempts 2..max_attempts."""
        out = []
        delay = self.base_delay
        for _ in range(self.max_attempts - 1):
            out.append(min(delay, self.max_delay))
            delay *= 2
        return out


DEFAULT_RETRY = RetryPolicy()


class BilledClient:
    """ModelClient + automatic retry + ledger accounting.

    Every call increments the ledger, including retries. A call that succeeds on
    the third attempt bills for all three.
    """

    def __init__(
        self,
        adapter: ModelClient,
        *,
        ledger: CostLedger,
        retry_policy: RetryPolicy = DEFAULT_RETRY,
    ):
        """
        Args:
            adapter: An `AnthropicAdapter` or `OpenAIAdapter` instance (or any
                object implementing `ModelClient.invoke`).
            ledger: Where usage accumulates across the session.
            retry_policy: Backoff parameters.
        """
        self._adapter = adapter
        self._ledger = ledger
        self._retry = retry_policy

    def invoke(
        self,
        messages: list[Msg],
        tools: list[ToolSpec] | None = None,
        output_spec: OutputSpec | None = None,
        budget: Budget | None = None,
        temperature: float = 0.0,
        *,
        phase: Phase,
        tier: ModelTier = ModelTier.BALANCED,
        **kwargs: Any,
    ) -> Completion:
        """Invoke with automatic retry and ledger recording.

        Args:
            phase: Billing phase this call belongs to (routing / fastpath /
                fullagent / subagent).
            tier: Model tier. The adapter maps this to a concrete model id.
            All other args match `ModelClient.invoke`.

        Returns:
            The completion from the first successful attempt.

        Raises:
            BudgetExhaustedError: The task's token budget is spent.
            ProviderError: All retries exhausted or a non-retryable failure.
        """
        budget = budget or Budget()
        if budget.total_limit is not None:
            spent = self._ledger.automatic_total
            if spent >= budget.total_limit:
                raise BudgetExhaustedError(
                    f"Token budget exhausted: {spent} >= {budget.total_limit}"
                )

        last_exc: ProviderError | None = None
        delays = self._retry.delays()

        for attempt in range(self._retry.max_attempts):
            try:
                completion = self._adapter.invoke(
                    messages,
                    tools=tools,
                    output_spec=output_spec,
                    budget=budget,
                    temperature=temperature,
                    tier=tier,
                    **kwargs,
                )
                self._ledger.add(completion.usage, phase, retried=attempt > 0)
                return completion

            except ProviderError as exc:
                last_exc = exc
                self._ledger.calls += 1
                if attempt > 0:
                    self._ledger.retried_calls += 1

                if not exc.retryable or attempt + 1 >= self._retry.max_attempts:
                    raise

                self._retry.sleep(delays[attempt])

        assert last_exc is not None  # Loop exited without success; last_exc was set.
        raise last_exc
