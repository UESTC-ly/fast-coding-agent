"""Tests for M5 memory module: trace store and calibration replay."""

from __future__ import annotations

from pathlib import Path

import pytest

from qqcode.memory.replay import (
    FASTPATH,
    FULLAGENT,
    INDETERMINATE,
    ReplayEngine,
    route_offline,
)
from qqcode.memory.trace import TraceRecord, TraceStore
from qqcode.routing.router import RoutingThresholds


# --- helpers ------------------------------------------------------------------

def _fp_trace(
    *,
    task_length: int = 50,
    l0_triggered: bool = False,
    l0_reason: str = "",
    l1_decision: str = "fastpath",
    l1_confidence: float = 0.9,
    files_hint_count: int = 1,
    fastpath_attempted: bool = True,
    fastpath_success: bool = True,
    skills: list[str] | None = None,
    tokens: int = 500,
) -> TraceRecord:
    r = TraceRecord()
    r.task_snippet = "add a docstring"
    r.task_length = task_length
    r.route_layer = "l0" if l0_triggered else "l1_l2"
    r.route_decision = "fastpath"
    r.l0_triggered = l0_triggered
    r.l0_reason = l0_reason
    r.l1_decision = l1_decision
    r.l1_confidence = l1_confidence
    r.files_hint_count = files_hint_count
    r.fastpath_attempted = fastpath_attempted
    r.fastpath_success = fastpath_success
    r.final_success = fastpath_success
    r.mode_used = "fastpath"
    r.finish_reason = "fastpath_ok" if fastpath_success else "escalated"
    r.tokens_total = tokens
    r.skills_used = skills or []
    return r


def _fa_trace(
    *,
    task_length: int = 600,
    l0_triggered: bool = True,
    l0_reason: str = "L0: Task length 600 exceeds FastPath budget",
) -> TraceRecord:
    r = TraceRecord()
    r.task_snippet = "update the auth module"  # no FULLMUST_KEYWORDS
    r.task_length = task_length
    r.route_layer = "l0"
    r.route_decision = "fullagent"
    r.l0_triggered = l0_triggered
    r.l0_reason = l0_reason
    r.final_success = True
    r.mode_used = "fullagent"
    r.finish_reason = "explicit"
    r.tokens_total = 3000
    return r


# --- TraceStore ---------------------------------------------------------------


class TestTraceStore:
    def test_write_and_read_round_trip(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / ".qqcode/trace.db")
        r = TraceRecord.from_task("add a type hint")
        r.route_decision = "fastpath"
        r.final_success = True
        r.mode_used = "fastpath"
        r.finish_reason = "fastpath_ok"
        r.tokens_total = 800

        store.write(r)
        records = store.all()
        store.close()

        assert len(records) == 1
        got = records[0]
        assert got.id == r.id
        assert got.task_snippet == "add a type hint"
        assert got.final_success is True
        assert got.tokens_total == 800

    def test_count_empty(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / ".qqcode/trace.db")
        assert store.count() == 0
        store.close()

    def test_count_after_writes(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / ".qqcode/trace.db")
        store.write(TraceRecord.from_task("t1"))
        store.write(TraceRecord.from_task("t2"))
        assert store.count() == 2
        store.close()

    def test_duplicate_id_is_idempotent(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / ".qqcode/trace.db")
        r = TraceRecord.from_task("task")
        store.write(r)
        store.write(r)  # should not raise
        assert store.count() == 1
        store.close()

    def test_ordered_by_timestamp(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / ".qqcode/trace.db")
        r1 = TraceRecord.from_task("first")
        r1.timestamp = "2026-01-01T00:00:00+00:00"
        r2 = TraceRecord.from_task("second")
        r2.timestamp = "2026-01-02T00:00:00+00:00"
        store.write(r2)
        store.write(r1)
        records = store.all()
        store.close()
        assert records[0].task_snippet == "first"
        assert records[1].task_snippet == "second"

    def test_skills_used_round_trip(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / ".qqcode/trace.db")
        r = TraceRecord.from_task("x")
        r.skills_used = ["python-patterns", "testing"]
        store.write(r)
        got = store.all()[0]
        store.close()
        assert got.skills_used == ["python-patterns", "testing"]

    def test_context_manager(self, tmp_path: Path) -> None:
        with TraceStore(tmp_path / ".qqcode/trace.db") as store:
            store.write(TraceRecord.from_task("x"))
            assert store.count() == 1

    def test_for_repo_creates_subdirectory(self, tmp_path: Path) -> None:
        store = TraceStore.for_repo(tmp_path)
        store.write(TraceRecord.from_task("x"))
        store.close()
        assert (tmp_path / ".qqcode" / "trace.db").exists()

    def test_bool_fields_round_trip(self, tmp_path: Path) -> None:
        store = TraceStore(tmp_path / ".qqcode/trace.db")
        r = TraceRecord.from_task("t")
        r.l0_triggered = True
        r.l2_override = True
        r.fastpath_attempted = True
        r.fastpath_success = False
        r.final_success = True
        store.write(r)
        got = store.all()[0]
        store.close()
        assert got.l0_triggered is True
        assert got.l2_override is True
        assert got.fastpath_attempted is True
        assert got.fastpath_success is False
        assert got.final_success is True


# --- route_offline ------------------------------------------------------------


class TestRouteOffline:
    defaults = RoutingThresholds()

    def test_l1_fastpath_high_confidence(self) -> None:
        t = _fp_trace(l1_confidence=0.9, files_hint_count=1)
        assert route_offline(t, self.defaults) == FASTPATH

    def test_l1_low_confidence_escalates(self) -> None:
        t = _fp_trace(l1_confidence=0.5)
        assert route_offline(t, self.defaults) == FULLAGENT

    def test_l1_too_many_files_escalates(self) -> None:
        t = _fp_trace(l1_confidence=0.95, files_hint_count=4)
        assert route_offline(t, self.defaults) == FULLAGENT

    def test_l0_keyword_always_fullagent(self) -> None:
        t = _fa_trace(l0_triggered=True, l0_reason="L0: Complex keyword detected")
        t.task_snippet = "refactor the auth module"
        assert route_offline(t, self.defaults) == FULLAGENT

    def test_l0_length_still_triggers_at_same_threshold(self) -> None:
        t = _fa_trace(task_length=600, l0_reason="L0: Task length 600 exceeds FastPath budget")
        assert route_offline(t, self.defaults) == FULLAGENT

    def test_l0_length_becomes_indeterminate_when_threshold_raised(self) -> None:
        t = _fa_trace(task_length=400, l0_reason="L0: Task length 400 exceeds FastPath budget")
        high = RoutingThresholds(max_task_length=600)
        assert route_offline(t, high) == INDETERMINATE

    def test_raising_tau_blocks_borderline_trace(self) -> None:
        t = _fp_trace(l1_confidence=0.72)  # passes default τ=0.70
        strict = RoutingThresholds(confidence=0.80)
        assert route_offline(t, strict) == FULLAGENT

    def test_lowering_tau_passes_borderline_trace(self) -> None:
        t = _fp_trace(l1_confidence=0.62)
        t.route_decision = "fullagent"  # was blocked at default τ
        loose = RoutingThresholds(confidence=0.60)
        assert route_offline(t, loose) == FASTPATH

    def test_raising_max_files_passes_wide_hint(self) -> None:
        t = _fp_trace(l1_confidence=0.95, files_hint_count=4)  # blocked at K=3
        wider = RoutingThresholds(max_files=5)
        assert route_offline(t, wider) == FASTPATH

    def test_l1_fullagent_decision_stays_fullagent(self) -> None:
        t = _fp_trace(l1_decision="fullagent", l1_confidence=0.9)
        t.route_decision = "fullagent"
        assert route_offline(t, self.defaults) == FULLAGENT


# --- ReplayEngine -------------------------------------------------------------


class TestReplayEngine:
    def _make_traces(self) -> list[TraceRecord]:
        return [
            _fp_trace(l1_confidence=0.85, fastpath_success=True, tokens=400),
            _fp_trace(l1_confidence=0.75, fastpath_success=True, tokens=450),
            _fp_trace(l1_confidence=0.65, fastpath_success=False, tokens=600),
            _fa_trace(task_length=600),
            _fp_trace(l1_confidence=0.90, files_hint_count=4, tokens=800),
        ]

    def test_calibrate_tau_row_count(self) -> None:
        engine = ReplayEngine(self._make_traces())
        rows = engine.calibrate_tau(tau_range=[0.60, 0.70, 0.80])
        assert len(rows) == 3

    def test_baseline_tau_has_zero_delta(self) -> None:
        engine = ReplayEngine(self._make_traces())
        rows = engine.calibrate_tau(tau_range=[0.70])
        assert rows[0].delta_vs_baseline == pytest.approx(0.0, abs=0.001)

    def test_lower_tau_routes_more_to_fastpath(self) -> None:
        engine = ReplayEngine(self._make_traces())
        rows = engine.calibrate_tau(tau_range=[0.60, 0.80])
        assert rows[0].fp_routed >= rows[1].fp_routed

    def test_fp_precision_from_attempted_traces(self) -> None:
        # At τ=0.70: conf=0.65 trace blocked → only 2 FP attempts, both succeed
        engine = ReplayEngine(self._make_traces())
        rows = engine.calibrate_tau(tau_range=[0.70])
        assert rows[0].fp_precision == pytest.approx(1.0)

    def test_calibrate_task_length_row_count(self) -> None:
        engine = ReplayEngine(self._make_traces())
        rows = engine.calibrate_task_length(length_range=[300, 500, 700])
        assert len(rows) == 3

    def test_calibrate_max_files_wider_admits_more(self) -> None:
        engine = ReplayEngine(self._make_traces())
        rows = engine.calibrate_max_files(files_range=[2, 3, 5])
        # K=5 should admit the files_hint_count=4 trace that K=3 blocks
        assert rows[2].fp_routed >= rows[0].fp_routed

    def test_empty_traces(self) -> None:
        engine = ReplayEngine([])
        rows = engine.calibrate_tau(tau_range=[0.70])
        assert rows[0].fp_routed == 0
        assert rows[0].fp_precision is None

    def test_indeterminate_traces_counted(self) -> None:
        t = _fa_trace(task_length=400, l0_reason="L0: Task length 400 exceeds FastPath budget")
        engine = ReplayEngine([t], baseline=RoutingThresholds(max_task_length=600))
        rows = engine.calibrate_task_length(length_range=[600])
        assert rows[0].indeterminate == 1

    def test_skill_impact_groups_by_skill(self) -> None:
        traces = [
            _fp_trace(skills=["py-patterns"], fastpath_success=True),
            _fp_trace(skills=["py-patterns"], fastpath_success=True),
            _fp_trace(skills=[], fastpath_success=False),
        ]
        engine = ReplayEngine(traces)
        rows = engine.skill_impact()
        names = [r.skill_name for r in rows]
        assert "py-patterns" in names
        assert "(no skill)" in names

    def test_skill_impact_sorted_descending(self) -> None:
        traces = [
            _fp_trace(skills=["good-skill"], fastpath_success=True),
            _fp_trace(skills=["bad-skill"], fastpath_success=False),
        ]
        engine = ReplayEngine(traces)
        rows = engine.skill_impact()
        assert rows[0].fp_hit_rate >= rows[-1].fp_hit_rate

    def test_skill_impact_empty(self) -> None:
        engine = ReplayEngine([])
        assert engine.skill_impact() == []
