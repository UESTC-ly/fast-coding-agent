"""Tests for the benchmark evaluator's failure attribution.

The evaluator's whole output is a behavioral rate, so it is only as trustworthy
as its ability to tell "the agent's fix was wrong" apart from "the measurement
apparatus broke". `RunRecord.incident_type` exists for exactly that split and
documents itself as "set when failure is NOT due to agent capability".

`_run_acceptance` defeated it: a `git apply` failure of the hidden test patch
was caught and returned as `passed=False`, indistinguishable from tests that
ran and failed. That happens whenever the agent edits a test file the hidden
patch also touches — which FastPath's prompt explicitly forbids, so the case
that most needs to be visible was the one being silently scored as incapacity.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_BENCH = Path(__file__).resolve().parent.parent / "benchmarks" / "qqcode_benchmark.py"


def _load_benchmark_module() -> Any:
    """Import the benchmark script, which lives outside the package."""
    spec = importlib.util.spec_from_file_location("qqcode_benchmark", _BENCH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["qqcode_benchmark"] = module
    spec.loader.exec_module(module)
    return module


bench = _load_benchmark_module()


DIFF_AGAINST_TEST_FILE = """\
diff --git a/tests/test_thing.py b/tests/test_thing.py
--- a/tests/test_thing.py
+++ b/tests/test_thing.py
@@ -1,2 +1,3 @@
 def test_one():
-    assert compute() == 1
+    assert compute() == 2
+    assert compute() != 3
"""


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A workspace whose test file does NOT match the hidden patch's context."""
    root = tmp_path / "ws"
    (root / "tests").mkdir(parents=True)
    # The agent rewrote this file, so the hidden patch's context no longer applies.
    (root / "tests" / "test_thing.py").write_text(
        "def test_one():\n    assert compute() == 99  # agent edited this\n"
    )
    (root / "thing.py").write_text("def compute():\n    return 1\n")
    return root


def _task() -> dict[str, Any]:
    return {
        "id": "probe",
        "acceptance_command": ["{python}", "-m", "pytest", "-q"],
        "fixture": {},
    }


def _record() -> Any:
    """A RunRecord for an agent run that succeeded on its own terms."""
    return bench.RunRecord(
        task_id="probe", category="simple", mode="automatic", order="AB", cycle=1,
        agent_success=True, mode_used="fastpath", finish_reason="fastpath_ok",
        fastpath_attempted=True, fastpath_success=True, turns_used=0,
        tokens_total=800, tokens_routing=0, tokens_fastpath=800, tokens_fullagent=0,
        duration_ms=1.0, behavioral_complete=False, acceptance_output="",
    )


class TestTestPatchConflictErrorIsAnIncident:
    def test_conflict_is_reported_not_scored_as_failure(self, workspace: Path) -> None:
        """A hidden patch that will not apply must be attributable, not a verdict.

        The agent's fix may have been perfect; we cannot tell, because the
        measurement never ran. Folding this into `passed=False` inflates the
        apparent failure rate with apparatus faults.
        """
        result = bench._run_acceptance(
            _task(), workspace, DIFF_AGAINST_TEST_FILE, Path(sys.executable)
        )

        assert result.conflict, (
            "a test_patch that cannot be applied must be flagged as a conflict, "
            f"got {result!r}"
        )
        assert not result.passed

    def test_incident_type_is_set_for_a_conflict(self, workspace: Path) -> None:
        """The conflict must reach `RunRecord.incident_type`.

        Detecting it inside `_run_acceptance` is worthless if the caller still
        records the run as a clean behavioral failure — the rate is computed
        from `incident_type`, not from the returned value.
        """
        result = bench._run_acceptance(
            _task(), workspace, DIFF_AGAINST_TEST_FILE, Path(sys.executable)
        )
        record = _record()
        bench._apply_acceptance_result(record, result)

        assert record.incident_type == bench.INCIDENT_TEST_CONFLICT
        assert record.behavioral_complete is False

    def test_clean_failure_stays_a_verdict(self, tmp_path: Path) -> None:
        """A patch that applies and then fails tests is a real behavioral failure.

        This is the control: the fix must not reclassify genuine failures as
        incidents, which would hide real incapacity instead of real faults.
        """
        root = tmp_path / "ws"
        (root / "tests").mkdir(parents=True)
        (root / "tests" / "test_thing.py").write_text(
            "def test_one():\n    assert compute() == 1\n"
        )
        (root / "thing.py").write_text("def compute():\n    return 1\n")

        result = bench._run_acceptance(
            _task(), root, DIFF_AGAINST_TEST_FILE, Path(sys.executable)
        )
        assert not result.conflict, "an applicable patch must not be called a conflict"

        record = _record()
        bench._apply_acceptance_result(record, result)
        assert record.incident_type is None, (
            "a genuine test failure must remain an agent verdict"
        )

    def test_conflict_is_excluded_from_the_rate(self) -> None:
        """A conflicted run must not count as a denominator entry.

        `incident_type` already drives that exclusion, so this pins the
        end-to-end consequence rather than the mechanism.
        """
        rows = [
            {"behavioral_complete": True, "incident_type": None},
            {"behavioral_complete": False, "incident_type": None},
            {"behavioral_complete": False, "incident_type": bench.INCIDENT_TEST_CONFLICT},
        ]
        clean = [r for r in rows if r["incident_type"] is None]
        assert len(clean) == 2, "the conflicted row must drop out of the denominator"


# ---------------------------------------------------------------------------
# Derivability: a fixture whose hidden assertion the statement never implies
# cannot measure capability, so it must not sit in the behavioral denominator.
# ---------------------------------------------------------------------------


class TestDerivabilityAudit:
    """The audit file must be loaded, and it must actually change the report.

    Shipping the verdicts as data while the report keeps averaging over all 15
    tasks would be the "parsed but not wired" failure this repo has hit twice.
    """

    def test_audit_file_covers_every_task(self) -> None:
        """Every task needs a verdict, or the denominator is silently partial."""
        audit = bench.load_derivability()
        task_ids = {t["id"] for t in bench.load_tasks()}

        assert task_ids, "no tasks loaded"
        assert task_ids <= set(audit), f"unaudited tasks: {sorted(task_ids - set(audit))}"

    def test_verdicts_are_from_the_known_set(self) -> None:
        audit = bench.load_derivability()
        unknown = {v for v in audit.values() if v not in bench.DERIVABILITY_VERDICTS}
        assert not unknown, f"unknown verdicts: {unknown}"

    def test_only_derivable_counts_toward_the_rate(self) -> None:
        """The report's headline rate must exclude non-derivable fixtures.

        Two runs: one on a derivable task (failed), one on a not_derivable task
        (also failed). The measurable rate must be 0/1, not 0/2 — the second
        failure says nothing about capability.
        """
        audit = bench.load_derivability()
        derivable = next(k for k, v in audit.items() if v == "derivable")
        not_derivable = next(k for k, v in audit.items() if v == "not_derivable")

        records = []
        for task_id in (derivable, not_derivable):
            rec = _record()
            rec.task_id = task_id
            rec.behavioral_complete = False
            records.append(rec)

        report = bench._build_report(records, {})
        measurable = report["derivability"]

        assert measurable["measurable_tasks"] == 1, (
            "only the derivable task may sit in the denominator, got "
            f"{measurable['measurable_tasks']}"
        )
        assert measurable["excluded"][not_derivable] == "not_derivable"
        assert measurable["behavioral_rate_measurable"] == 0.0

    def test_excluded_fixtures_are_reported_not_hidden(self) -> None:
        """Exclusion must be visible, otherwise it looks like cherry-picking."""
        audit = bench.load_derivability()
        rec = _record()
        rec.task_id = next(k for k, v in audit.items() if v == "not_derivable")
        report = bench._build_report([rec], {})

        excluded = report["derivability"]["excluded"]
        assert rec.task_id in excluded
        assert report["derivability"]["measurable_tasks"] == 0
        assert "note" in report["derivability"]
