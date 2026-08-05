"""Tests for CostLedger accounting."""

from __future__ import annotations

from qqcode.models.protocol import CostLedger, Usage


def test_usage_total_sums_input_and_output() -> None:
    assert Usage(input_tokens=100, output_tokens=50).total == 150


def test_ledger_starts_empty() -> None:
    ledger = CostLedger()
    assert ledger.automatic_total == 0
    assert ledger.calls == 0


def test_ledger_accumulates_across_phases() -> None:
    ledger = CostLedger()
    ledger.add(Usage(input_tokens=500, output_tokens=100), "routing")
    ledger.add(Usage(input_tokens=2000, output_tokens=400), "fastpath")
    ledger.add(Usage(input_tokens=8000, output_tokens=1500), "fullagent")

    assert ledger.calls == 3
    assert ledger.routing_tokens == 600
    assert ledger.fastpath_tokens == 2400
    assert ledger.fullagent_tokens == 9500
    assert ledger.automatic_total == 12500


def test_failed_fastpath_cost_survives_escalation() -> None:
    """A failed FastPath attempt must still count toward the automatic total."""
    ledger = CostLedger()
    ledger.add(Usage(input_tokens=300, output_tokens=80), "routing")
    ledger.add(Usage(input_tokens=1500, output_tokens=300), "fastpath")  # failed
    ledger.add(Usage(input_tokens=1500, output_tokens=250), "fastpath")  # provider retry
    ledger.add(Usage(input_tokens=6000, output_tokens=1200), "fullagent")

    assert ledger.fastpath_tokens == 3550
    assert ledger.automatic_total == 11130
    assert ledger.summary()["by_phase"]["fastpath"] == 3550


def test_ledger_tracks_cache_tokens_separately() -> None:
    ledger = CostLedger()
    ledger.add(
        Usage(
            input_tokens=200,
            output_tokens=50,
            cache_creation_tokens=4000,
            cache_read_tokens=12000,
        ),
        "fullagent",
    )

    assert ledger.cache_creation == 4000
    assert ledger.cache_read == 12000
    # Cache tokens are reported separately and do not inflate the billed total.
    assert ledger.automatic_total == 250


def test_summary_shape() -> None:
    ledger = CostLedger()
    ledger.add(Usage(input_tokens=10, output_tokens=5), "routing")
    summary = ledger.summary()

    assert set(summary) == {
        "total_input",
        "total_output",
        "cache_creation",
        "cache_read",
        "calls",
        "retried_calls",
        "automatic_total",
        "by_phase",
    }
    assert set(summary["by_phase"]) == {"routing", "fastpath", "fullagent", "subagent"}


def test_subagent_cost_is_tracked_separately_but_counted() -> None:
    """Sub-agent spend needs its own line item yet must reach automatic_total."""
    ledger = CostLedger()
    ledger.add(Usage(input_tokens=4000, output_tokens=600), "fullagent")
    ledger.add(Usage(input_tokens=3000, output_tokens=500), "subagent")
    ledger.add(Usage(input_tokens=2000, output_tokens=400), "subagent")

    assert ledger.subagent_tokens == 5900
    assert ledger.fullagent_tokens == 4600
    assert ledger.automatic_total == 10500
