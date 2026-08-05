"""Offline calibration: replay routing decisions to tune τ, L, K.

Never calls a model. Uses recorded L1 output from past traces to compute what
each trace's routing decision would have been under different threshold settings.

The sweep produces a calibration table where each row shows:
- How many tasks would route to FastPath / FullAgent / indeterminate
- The observed FastPath precision (% that actually succeeded)
- Estimated token cost per task under that configuration
- Cost delta versus the current baseline

A route replay can be INDETERMINATE when L0 fired on task length in the
original run but the new threshold would NOT fire — meaning the task would
reach L1, but no L1 data was recorded. Those traces are counted but excluded
from the precision calculation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qqcode.memory.trace import TraceRecord
from qqcode.routing.router import DEFAULT_THRESHOLDS, FULLMUST_KEYWORDS, RoutingThresholds

INDETERMINATE = "indeterminate"
FASTPATH = "fastpath"
FULLAGENT = "fullagent"


# --------------------------------------------------------------------------
# Offline route replay
# --------------------------------------------------------------------------


def route_offline(trace: TraceRecord, thresholds: RoutingThresholds) -> str:
    """Compute the routing decision for a trace under different thresholds.

    Returns 'fastpath', 'fullagent', or INDETERMINATE.

    INDETERMINATE means the replay requires an L1 model call that never happened
    in the original run — e.g., L0 fired on length originally but a new higher
    limit would let the task through to L1.
    """
    # --- L0 replay ---
    # Keyword triggers are threshold-independent
    task_lower = trace.task_snippet.lower()
    if any(kw in task_lower for kw in FULLMUST_KEYWORDS):
        return FULLAGENT

    # Length threshold may have changed
    if trace.task_length > thresholds.max_task_length:
        return FULLAGENT

    # L0 no longer fires under new thresholds.
    # If it fired originally, determine why.
    if trace.l0_triggered:
        original_was_length = "Task length" in trace.l0_reason
        original_was_keyword = "keyword" in trace.l0_reason.lower()
        if original_was_length and not original_was_keyword:
            # Length no longer triggers, no L1 data recorded → indeterminate
            return INDETERMINATE
        # Skill hint or other L0 trigger — assume still fires
        return trace.route_decision

    # --- L1 data available (L0 did not fire originally) ---
    if not trace.l1_decision:
        # Fallback route (no client) — treat as deterministic
        return trace.route_decision

    # --- L2 replay with new thresholds ---
    if trace.l1_decision == FASTPATH:
        if trace.files_hint_count > thresholds.max_files:
            return FULLAGENT
        if trace.l1_confidence < thresholds.confidence:
            return FULLAGENT
        return FASTPATH

    return FULLAGENT


# --------------------------------------------------------------------------
# Calibration report structures
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationRow:
    """Routing outcome and estimated cost for one threshold configuration."""

    thresholds: RoutingThresholds
    # Counts (determinate traces only)
    fp_routed: int
    fa_routed: int
    indeterminate: int
    # Precision: fraction of FP-attempted traces that succeeded.
    # None when no FP-routed trace has an observed FastPath outcome.
    fp_precision: float | None
    # Mean total tokens per task under this configuration
    mean_tokens_per_task: float
    # Fractional cost change vs baseline (+ = more expensive, − = cheaper)
    delta_vs_baseline: float


@dataclass(frozen=True)
class SkillImpactRow:
    """FastPath performance split by skill presence."""

    skill_name: str   # "" means "no skill matched"
    trace_count: int
    fp_routed_count: int
    fp_success_count: int
    # fp_success_count / trace_count (0.0 when trace_count == 0)
    fp_hit_rate: float


# --------------------------------------------------------------------------
# Replay engine
# --------------------------------------------------------------------------


@dataclass
class ReplayEngine:
    """Compute calibration curves from a list of trace records.

    Usage::

        engine = ReplayEngine(traces)
        rows = engine.calibrate_tau()         # sweep τ (confidence)
        rows = engine.calibrate_task_length() # sweep L (max task chars)
        rows = engine.calibrate_max_files()   # sweep K (max files hint)
        skill_rows = engine.skill_impact()
    """

    traces: list[TraceRecord]
    baseline: RoutingThresholds = field(default_factory=lambda: DEFAULT_THRESHOLDS)

    def calibrate_tau(self, tau_range: list[float] | None = None) -> list[CalibrationRow]:
        """Sweep the confidence threshold τ, fixing L and K at baseline values."""
        if tau_range is None:
            tau_range = [round(0.50 + i * 0.05, 2) for i in range(10)]
        configs = [
            RoutingThresholds(
                confidence=tau,
                max_task_length=self.baseline.max_task_length,
                max_files=self.baseline.max_files,
            )
            for tau in tau_range
        ]
        return self._sweep(configs)

    def calibrate_task_length(
        self, length_range: list[int] | None = None
    ) -> list[CalibrationRow]:
        """Sweep the maximum task length L, fixing τ and K at baseline values."""
        if length_range is None:
            length_range = list(range(200, 1100, 100))
        configs = [
            RoutingThresholds(
                confidence=self.baseline.confidence,
                max_task_length=length,
                max_files=self.baseline.max_files,
            )
            for length in length_range
        ]
        return self._sweep(configs)

    def calibrate_max_files(
        self, files_range: list[int] | None = None
    ) -> list[CalibrationRow]:
        """Sweep the maximum file count K, fixing τ and L at baseline values."""
        if files_range is None:
            files_range = list(range(1, 7))
        configs = [
            RoutingThresholds(
                confidence=self.baseline.confidence,
                max_task_length=self.baseline.max_task_length,
                max_files=k,
            )
            for k in files_range
        ]
        return self._sweep(configs)

    def _sweep(self, configs: list[RoutingThresholds]) -> list[CalibrationRow]:
        baseline_cost = self._evaluate(self.baseline).mean_tokens_per_task
        rows: list[CalibrationRow] = []
        for cfg in configs:
            row = self._evaluate(cfg)
            delta = (
                (row.mean_tokens_per_task - baseline_cost) / baseline_cost
                if baseline_cost > 0
                else 0.0
            )
            rows.append(CalibrationRow(
                thresholds=row.thresholds,
                fp_routed=row.fp_routed,
                fa_routed=row.fa_routed,
                indeterminate=row.indeterminate,
                fp_precision=row.fp_precision,
                mean_tokens_per_task=row.mean_tokens_per_task,
                delta_vs_baseline=delta,
            ))
        return rows

    def _evaluate(self, thresholds: RoutingThresholds) -> CalibrationRow:
        fp_routed = 0
        fa_routed = 0
        indet = 0
        fp_with_outcome = 0
        fp_successes = 0
        total_tokens = 0

        for t in self.traces:
            decision = route_offline(t, thresholds)
            tokens = t.tokens_total or (
                t.tokens_routing + t.tokens_fastpath + t.tokens_fullagent
            )
            total_tokens += tokens

            if decision == INDETERMINATE:
                indet += 1
            elif decision == FASTPATH:
                fp_routed += 1
                if t.fastpath_attempted:
                    fp_with_outcome += 1
                    if t.fastpath_success:
                        fp_successes += 1
            else:
                fa_routed += 1

        n = fp_routed + fa_routed + indet
        mean_tokens = total_tokens / n if n > 0 else 0.0
        fp_precision = fp_successes / fp_with_outcome if fp_with_outcome > 0 else None

        return CalibrationRow(
            thresholds=thresholds,
            fp_routed=fp_routed,
            fa_routed=fa_routed,
            indeterminate=indet,
            fp_precision=fp_precision,
            mean_tokens_per_task=mean_tokens,
            delta_vs_baseline=0.0,
        )

    def skill_impact(self) -> list[SkillImpactRow]:
        """FastPath performance split by which skills were active.

        Returns one row per distinct skill name (plus a row for no-skill tasks),
        sorted by fp_hit_rate descending.
        """
        stats: dict[str, dict[str, int]] = {}

        def _ensure(key: str) -> None:
            if key not in stats:
                stats[key] = {"count": 0, "fp_routed": 0, "fp_success": 0}

        for t in self.traces:
            keys = t.skills_used if t.skills_used else ["(no skill)"]
            for key in keys:
                _ensure(key)
                stats[key]["count"] += 1
                if t.fastpath_attempted:
                    stats[key]["fp_routed"] += 1
                    if t.fastpath_success:
                        stats[key]["fp_success"] += 1

        rows = [
            SkillImpactRow(
                skill_name=name,
                trace_count=s["count"],
                fp_routed_count=s["fp_routed"],
                fp_success_count=s["fp_success"],
                fp_hit_rate=s["fp_success"] / s["count"] if s["count"] > 0 else 0.0,
            )
            for name, s in stats.items()
        ]
        rows.sort(key=lambda r: r.fp_hit_rate, reverse=True)
        return rows
