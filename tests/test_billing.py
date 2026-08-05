"""Tests for BilledClient: the single billing entry point.

The invariants here are the ones the cost target depends on. If a retry's tokens
vanish from the ledger, or a budget check happens after the request leaves, the
`automatic_total` stops being trustworthy and the "≤ 50% of full Full Agent"
claim becomes unmeasurable.

`RetryPolicy.sleep` is injected everywhere so no test spends wall-clock time.
"""

from __future__ import annotations

from typing import Any

import pytest

from qqcode.models.billing import DEFAULT_RETRY, BilledClient, RetryPolicy
from qqcode.models.errors import BudgetExhaustedError, ProviderError
from qqcode.models.protocol import (
    Budget,
    Completion,
    CostLedger,
    ModelTier,
    Msg,
    Role,
    TextContent,
    Usage,
)

MESSAGES = [Msg(role=Role.USER, content=[TextContent(text="go")])]


class ScriptedAdapter:
    """Adapter that replays a script of completions and exceptions.

    Each element is either a `Completion` to return or an exception to raise.
    Records every invocation so tests can assert on attempt counts and params.
    """

    def __init__(self, script: list[Any]):
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    def invoke(self, messages: list[Msg], **kwargs: Any) -> Completion:
        self.calls.append({"messages": messages, **kwargs})
        if not self._script:
            raise AssertionError("ScriptedAdapter exhausted: more calls than scripted")
        item = self._script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    @property
    def attempts(self) -> int:
        return len(self.calls)


def completion(*, input_tokens: int = 100, output_tokens: int = 20) -> Completion:
    return Completion(
        content=[TextContent(text="ok")],
        stop_reason="end_turn",
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        raw={},
    )


def no_sleep_policy(max_attempts: int = 3) -> RetryPolicy:
    """A retry policy that records delays instead of sleeping."""
    slept: list[float] = []
    policy = RetryPolicy(max_attempts=max_attempts, base_delay=1.0, sleep=slept.append)
    policy.slept = slept  # type: ignore[attr-defined]
    return policy


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------


def test_successful_call_records_usage_under_its_phase() -> None:
    ledger = CostLedger()
    client = BilledClient(ScriptedAdapter([completion()]), ledger=ledger)

    client.invoke(MESSAGES, phase="fastpath")

    assert ledger.calls == 1
    assert ledger.retried_calls == 0
    assert ledger.fastpath_tokens == 120
    assert ledger.automatic_total == 120


@pytest.mark.parametrize(
    ("phase", "attr"),
    [
        ("routing", "routing_tokens"),
        ("fastpath", "fastpath_tokens"),
        ("fullagent", "fullagent_tokens"),
        ("subagent", "subagent_tokens"),
    ],
)
def test_each_phase_bills_to_its_own_line_item(phase: str, attr: str) -> None:
    ledger = CostLedger()
    client = BilledClient(ScriptedAdapter([completion()]), ledger=ledger)

    client.invoke(MESSAGES, phase=phase)  # type: ignore[arg-type]

    assert getattr(ledger, attr) == 120
    assert ledger.automatic_total == 120


def test_tier_and_params_reach_the_adapter() -> None:
    adapter = ScriptedAdapter([completion()])
    client = BilledClient(adapter, ledger=CostLedger())

    client.invoke(MESSAGES, temperature=0.7, phase="fullagent", tier=ModelTier.DEEP)

    call = adapter.calls[0]
    assert call["tier"] is ModelTier.DEEP
    assert call["temperature"] == 0.7


def test_extra_kwargs_pass_through() -> None:
    adapter = ScriptedAdapter([completion()])
    client = BilledClient(adapter, ledger=CostLedger())

    client.invoke(MESSAGES, phase="fullagent", model="forced-model")

    assert adapter.calls[0]["model"] == "forced-model"


def test_a_shared_ledger_accumulates_across_phases() -> None:
    """One task's routing, FastPath, and Full Agent spend land in one total."""
    ledger = CostLedger()
    adapter = ScriptedAdapter(
        [
            completion(input_tokens=300, output_tokens=80),
            completion(input_tokens=1500, output_tokens=300),
            completion(input_tokens=6000, output_tokens=1200),
        ]
    )
    client = BilledClient(adapter, ledger=ledger)

    client.invoke(MESSAGES, phase="routing")
    client.invoke(MESSAGES, phase="fastpath")
    client.invoke(MESSAGES, phase="fullagent")

    assert ledger.routing_tokens == 380
    assert ledger.fastpath_tokens == 1800
    assert ledger.fullagent_tokens == 7200
    assert ledger.automatic_total == 9380


# --------------------------------------------------------------------------
# Retry accounting — the load-bearing invariant
# --------------------------------------------------------------------------


def test_retry_succeeds_and_is_marked_as_retried() -> None:
    ledger = CostLedger()
    policy = no_sleep_policy()
    adapter = ScriptedAdapter(
        [ProviderError("overloaded", status_code=529), completion()]
    )
    client = BilledClient(adapter, ledger=ledger, retry_policy=policy)

    client.invoke(MESSAGES, phase="fastpath")

    assert adapter.attempts == 2
    # One failed attempt + one successful attempt.
    assert ledger.calls == 2
    assert ledger.retried_calls == 1
    assert policy.slept == [1.0]  # type: ignore[attr-defined]


def test_every_attempt_counts_toward_the_ledger() -> None:
    """A call that succeeds on the third attempt bills for all three."""
    ledger = CostLedger()
    adapter = ScriptedAdapter(
        [
            ProviderError("boom", status_code=500),
            ProviderError("boom", status_code=500),
            completion(),
        ]
    )
    client = BilledClient(adapter, ledger=ledger, retry_policy=no_sleep_policy())

    client.invoke(MESSAGES, phase="fullagent")

    assert adapter.attempts == 3
    assert ledger.calls == 3
    assert ledger.retried_calls == 2


def test_exhausted_retries_raise_and_still_bill_every_attempt() -> None:
    ledger = CostLedger()
    adapter = ScriptedAdapter([ProviderError("boom", status_code=503)] * 3)
    client = BilledClient(adapter, ledger=ledger, retry_policy=no_sleep_policy())

    with pytest.raises(ProviderError):
        client.invoke(MESSAGES, phase="fastpath")

    assert adapter.attempts == 3
    assert ledger.calls == 3
    assert ledger.retried_calls == 2


def test_non_retryable_failure_stops_after_one_attempt() -> None:
    ledger = CostLedger()
    policy = no_sleep_policy()
    adapter = ScriptedAdapter([ProviderError("bad request", status_code=400)])
    client = BilledClient(adapter, ledger=ledger, retry_policy=policy)

    with pytest.raises(ProviderError):
        client.invoke(MESSAGES, phase="fastpath")

    assert adapter.attempts == 1
    assert ledger.calls == 1
    assert ledger.retried_calls == 0
    assert policy.slept == []  # type: ignore[attr-defined]


def test_failed_fastpath_cost_survives_escalation_to_full_agent() -> None:
    """The headline cost rule: a failed FastPath still counts in the total."""
    ledger = CostLedger()
    adapter = ScriptedAdapter(
        [
            completion(input_tokens=300, output_tokens=80),  # routing
            ProviderError("overloaded", status_code=529),  # fastpath attempt 1
            completion(input_tokens=1500, output_tokens=300),  # fastpath retry
            completion(input_tokens=6000, output_tokens=1200),  # fullagent
        ]
    )
    client = BilledClient(adapter, ledger=ledger, retry_policy=no_sleep_policy())

    client.invoke(MESSAGES, phase="routing")
    client.invoke(MESSAGES, phase="fastpath")
    client.invoke(MESSAGES, phase="fullagent")

    assert ledger.retried_calls == 1
    assert ledger.fastpath_tokens == 1800
    assert ledger.automatic_total == 9380
    assert ledger.summary()["by_phase"]["fastpath"] == 1800


def test_backoff_delays_grow_exponentially_and_cap() -> None:
    policy = RetryPolicy(max_attempts=5, base_delay=1.0, max_delay=4.0, sleep=lambda _: None)
    assert policy.delays() == [1.0, 2.0, 4.0, 4.0]


def test_single_attempt_policy_never_sleeps() -> None:
    policy = RetryPolicy(max_attempts=1, sleep=lambda _: None)
    assert policy.delays() == []

    ledger = CostLedger()
    adapter = ScriptedAdapter([ProviderError("boom", status_code=503)])
    client = BilledClient(adapter, ledger=ledger, retry_policy=policy)

    with pytest.raises(ProviderError):
        client.invoke(MESSAGES, phase="fastpath")
    assert adapter.attempts == 1


def test_default_retry_policy_is_three_attempts() -> None:
    assert DEFAULT_RETRY.max_attempts == 3


# --------------------------------------------------------------------------
# Budget enforcement
# --------------------------------------------------------------------------


def test_budget_check_fires_before_the_request_leaves() -> None:
    """An exhausted budget must not waste a network round trip."""
    ledger = CostLedger()
    ledger.add(Usage(input_tokens=900, output_tokens=200), "fullagent")
    adapter = ScriptedAdapter([completion()])
    client = BilledClient(adapter, ledger=ledger)

    with pytest.raises(BudgetExhaustedError):
        client.invoke(MESSAGES, budget=Budget(total_limit=1000), phase="fullagent")

    assert adapter.attempts == 0
    assert ledger.calls == 1  # unchanged: only the pre-existing recorded call


def test_call_proceeds_while_budget_remains() -> None:
    ledger = CostLedger()
    ledger.add(Usage(input_tokens=100, output_tokens=50), "fullagent")
    adapter = ScriptedAdapter([completion()])
    client = BilledClient(adapter, ledger=ledger)

    client.invoke(MESSAGES, budget=Budget(total_limit=10_000), phase="fullagent")

    assert adapter.attempts == 1


def test_budget_boundary_is_inclusive() -> None:
    """Spent exactly equal to the limit counts as exhausted."""
    ledger = CostLedger()
    ledger.add(Usage(input_tokens=800, output_tokens=200), "fullagent")
    client = BilledClient(ScriptedAdapter([completion()]), ledger=ledger)

    with pytest.raises(BudgetExhaustedError):
        client.invoke(MESSAGES, budget=Budget(total_limit=1000), phase="fullagent")


def test_absent_total_limit_disables_the_budget_gate() -> None:
    ledger = CostLedger()
    ledger.add(Usage(input_tokens=10**6, output_tokens=10**6), "fullagent")
    adapter = ScriptedAdapter([completion()])
    client = BilledClient(adapter, ledger=ledger)

    client.invoke(MESSAGES, budget=Budget(), phase="fullagent")

    assert adapter.attempts == 1


def test_budget_is_forwarded_to_the_adapter() -> None:
    adapter = ScriptedAdapter([completion()])
    client = BilledClient(adapter, ledger=CostLedger())
    budget = Budget(max_tokens=777)

    client.invoke(MESSAGES, budget=budget, phase="fullagent")

    assert adapter.calls[0]["budget"] is budget


def test_omitted_budget_becomes_a_default_budget() -> None:
    """The adapter must always receive a Budget, never None."""
    adapter = ScriptedAdapter([completion()])
    client = BilledClient(adapter, ledger=CostLedger())

    client.invoke(MESSAGES, phase="fullagent")

    assert isinstance(adapter.calls[0]["budget"], Budget)


# --------------------------------------------------------------------------
# Nothing bypasses the ledger
# --------------------------------------------------------------------------


def test_value_error_from_the_adapter_is_not_swallowed_or_billed() -> None:
    """Structured-output/tool misuse is a programming error, not a retryable one."""
    ledger = CostLedger()
    adapter = ScriptedAdapter([ValueError("structured output cannot be combined with tools")])
    client = BilledClient(adapter, ledger=ledger, retry_policy=no_sleep_policy())

    with pytest.raises(ValueError):
        client.invoke(MESSAGES, phase="fastpath")

    assert adapter.attempts == 1
    assert ledger.calls == 0
    assert ledger.automatic_total == 0


def test_cache_tokens_are_tracked_without_inflating_the_billed_total() -> None:
    ledger = CostLedger()
    adapter = ScriptedAdapter(
        [
            Completion(
                content=[TextContent(text="ok")],
                stop_reason="end_turn",
                usage=Usage(
                    input_tokens=200,
                    output_tokens=50,
                    cache_creation_tokens=4000,
                    cache_read_tokens=12000,
                ),
                raw={},
            )
        ]
    )
    client = BilledClient(adapter, ledger=ledger)

    client.invoke(MESSAGES, phase="fullagent")

    assert ledger.cache_creation == 4000
    assert ledger.cache_read == 12000
    assert ledger.automatic_total == 250
