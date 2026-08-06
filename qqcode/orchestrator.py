"""Orchestrator: route → FastPath → (escalate) Full Agent → finalize.

Single entry point for running a task. Handles all three modes (auto/fast/full)
and wires the three-condition gate for Full Agent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from qqcode.acceptance import AcceptanceHarness, all_passed, filter_acceptance_paths, first_failure
from qqcode.agents.full_agent import FullAgentInput, execute_full_agent
from qqcode.config import Config
from qqcode.events import EventCallback
from qqcode.memory.trace import TraceRecord, TraceStore
from qqcode.models.billing import BilledClient
from qqcode.models.factory import build_client, uniform_tiers
from qqcode.models.protocol import CostLedger, ModelTier
from qqcode.review import ConfirmCallback, build_review
from qqcode.routing import RoutingDecision, execute_fastpath, route_task
from qqcode.routing.fastpath import FastPathInput, build_escalation_context
from qqcode.skills.index import SkillIndex
from qqcode.tools.builtins import default_registry
from qqcode.workspace.worktree import Seed, WorktreeWorkspace

Mode = Literal["auto", "fast", "full"]


@dataclass
class RunResult:
    """Outcome of one task run."""

    success: bool
    mode_used: str        # "fastpath" | "fullagent"
    finish_reason: str    # "fastpath_ok" | "explicit" | "max_turns" | "budget" | "stuck" | "error"
    changed_files: frozenset[str]
    reasoning: str
    ledger: CostLedger
    dry_run: bool = False
    error: str | None = None
    turns_used: int = 0
    # True when the agent produced a valid change that the user declined.
    # Distinct from failure: the work was sound, the person did not want it.
    rejected: bool = False


def run_task(
    task: str,
    repo: Path,
    config: Config,
    *,
    mode: Mode = "auto",
    dry_run: bool = False,
    harness: AcceptanceHarness | None = None,
    provider: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
    max_turns: int = 30,
    trace_store: TraceStore | None = None,
    confirm: ConfirmCallback | None = None,
    seed: Seed = "head",
    history: str = "",
    on_event: EventCallback | None = None,
) -> RunResult:
    """Run a task against a repository.

    Args:
        task: Natural-language task description.
        repo: Repository root path.
        config: Loaded from .env via Config.from_env().
        mode: "auto" = intelligent routing; "fast" = FastPath only;
              "full" = Full Agent directly.
        dry_run: Apply changes in shadow workspace but skip finalize.
        harness: Hidden acceptance tests (task author supplies these).
        provider: Override default provider ("anthropic" | "openai").
        model: Pin every tier to this model id (useful for benchmarking).
        reasoning_effort: Pin reasoning effort ("low" | "medium" | "high").
            OpenAI path only; ignored for Anthropic.
        max_turns: Full Agent turn cap.
        trace_store: If provided, one TraceRecord is written per run.
        confirm: Human verdict callback. `None` (the default) means the objective
            conditions decide alone — identical to batch behavior. When supplied,
            a change that passes every objective condition is still only
            finalized if this returns True. Used by the conversational layer,
            where no hidden acceptance test exists.
        seed: Whether each shadow starts from the last commit ("head", default)
            or the live working tree ("worktree"). Conversation needs
            "worktree" so turn N builds on turn N-1's uncommitted output.
    """
    t0 = time.monotonic()
    # An explicit --model beats the configured default: the flag is the more
    # specific signal, and benchmarking depends on being able to override.
    effective_model = model or config.default_model
    tier_models = uniform_tiers(effective_model) if effective_model else None
    client, ledger = build_client(
        config,
        provider=provider,
        tier_models=tier_models,
        reasoning_effort=reasoning_effort,
    )
    skill_index = SkillIndex.discover(repo)

    # Start trace record (filled in as the run progresses)
    record = TraceRecord.from_task(task) if trace_store is not None else None

    # Collect active skill names for the trace
    if record is not None:
        matched = skill_index.match(task=task, paths=())
        record.skills_used = [s.name for s in matched]

    if mode == "full":
        if record is not None:
            record.route_layer = "mode_forced"
            record.route_decision = "fullagent"
        result = _run_fullagent(
            task, repo, client, ledger, skill_index, repo,
            harness=harness, dry_run=dry_run, max_turns=max_turns,
            escalation_context="", confirm=confirm, seed=seed,
            history=history, on_event=on_event,
        )
        _finalise_trace(record, result, ledger, time.monotonic() - t0, trace_store)
        return result

    if mode == "fast":
        if record is not None:
            record.route_layer = "mode_forced"
            record.route_decision = "fastpath"
        fp_result, esc = _try_fastpath(
            task, repo, client, skill_index, (), harness, dry_run, record,
            confirm=confirm, seed=seed, history=history,
        )
        if fp_result is not None:
            _finalise_trace(record, fp_result, ledger, time.monotonic() - t0, trace_store)
            return fp_result
        blocked = RunResult(
            success=False, mode_used="fastpath", finish_reason="escalation_blocked",
            changed_files=frozenset(), reasoning="", ledger=ledger,
            error=f"FastPath failed and mode=fast prevents escalation. {esc[:300]}",
        )
        _finalise_trace(record, blocked, ledger, time.monotonic() - t0, trace_store)
        return blocked

    # auto: route → FastPath → (escalate) Full Agent
    routing = route_task(task, skill_index, client)

    if record is not None:
        record.route_layer = routing.layer
        record.route_decision = routing.decision
        record.l0_triggered = routing.l0_triggered
        record.l0_reason = routing.reasoning if routing.l0_triggered else ""
        record.l1_decision = routing.l1_decision
        record.l1_confidence = routing.l1_confidence
        record.l2_override = routing.l2_override
        record.l2_reason = routing.reasoning if routing.l2_override else ""
        record.files_hint_count = len(routing.files_hint)

    if routing.decision == RoutingDecision.FASTPATH:
        fp_result, esc = _try_fastpath(
            task, repo, client, skill_index, routing.files_hint, harness, dry_run, record,
            confirm=confirm, seed=seed, history=history,
        )
        if fp_result is not None:
            _finalise_trace(record, fp_result, ledger, time.monotonic() - t0, trace_store)
            return fp_result
    else:
        esc = ""

    result = _run_fullagent(
        task, repo, client, ledger, skill_index, repo,
        harness=harness, dry_run=dry_run, max_turns=max_turns,
        escalation_context=esc, confirm=confirm, seed=seed,
        history=history, on_event=on_event,
    )
    _finalise_trace(record, result, ledger, time.monotonic() - t0, trace_store)
    return result


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------


def _accepted_by_user(
    confirm: ConfirmCallback | None,
    *,
    task: str,
    mode_used: str,
    reasoning: str,
    changed_files: frozenset[str],
    source: Path,
    shadow: Path,
) -> bool:
    """Whether the human verdict permits finalizing.

    Returns True when no callback was supplied: batch runs have no one to ask,
    and the objective conditions have already decided. This is what keeps
    `confirm=None` byte-for-byte identical to the pre-conversation behavior.
    """
    if confirm is None:
        return True
    review = build_review(
        task=task,
        mode_used=mode_used,
        reasoning=reasoning,
        changed_files=changed_files,
        source=source,
        shadow=shadow,
    )
    return confirm(review)


def _try_fastpath(
    task: str,
    repo: Path,
    client: BilledClient,
    skill_index: SkillIndex,
    files_hint: tuple[str, ...],
    harness: AcceptanceHarness | None,
    dry_run: bool,
    record: TraceRecord | None = None,
    *,
    confirm: ConfirmCallback | None = None,
    seed: Seed = "head",
    history: str = "",
) -> tuple[RunResult | None, str]:
    """Attempt FastPath in a fresh shadow workspace.

    Returns (RunResult, "") on success, (None, escalation_context) on failure.
    """
    ledger = client._ledger  # noqa: SLF001
    if record is not None:
        record.fastpath_attempted = True

    with WorktreeWorkspace(repo, use_git=True, seed=seed) as workspace:
        baseline = workspace.snapshot()
        inp = FastPathInput(
            task=task,
            baseline=baseline,
            skill_index=skill_index,
            tool_registry=default_registry(),
            files_hint=files_hint,
            history=history,
        )
        fp = execute_fastpath(inp, workspace, client, harness=harness)

        if record is not None:
            record.fastpath_success = fp.success
            record.fastpath_reason = "ok" if fp.success else fp.escalation_reason

        if not fp.success:
            return None, build_escalation_context(fp, task)

        if not _accepted_by_user(
            confirm,
            task=task,
            mode_used="fastpath",
            reasoning=fp.reasoning,
            changed_files=fp.changed_files,
            source=repo,
            shadow=workspace.root,
        ):
            return RunResult(
                success=False, mode_used="fastpath", finish_reason="rejected",
                changed_files=fp.changed_files, reasoning=fp.reasoning,
                ledger=ledger, dry_run=dry_run, rejected=True,
            ), ""

        if not dry_run:
            workspace.finalize(repo)
        return RunResult(
            success=True, mode_used="fastpath", finish_reason="fastpath_ok",
            changed_files=fp.changed_files, reasoning=fp.reasoning,
            ledger=ledger, dry_run=dry_run,
        ), ""


def _finalise_trace(
    record: TraceRecord | None,
    result: RunResult,
    ledger: CostLedger,
    elapsed: float,
    store: TraceStore | None,
) -> None:
    """Fill final fields into the trace record and persist it."""
    if record is None or store is None:
        return
    record.final_success = result.success
    record.mode_used = result.mode_used
    record.finish_reason = result.finish_reason
    record.turns_used = result.turns_used
    # Error text takes precedence; on success this holds the agent's closing summary.
    record.finish_summary = result.error or result.reasoning
    record.duration_ms = int(elapsed * 1000)
    s = ledger.summary()
    by_phase = s.get("by_phase", {})
    record.tokens_routing = by_phase.get("routing", 0)
    record.tokens_fastpath = by_phase.get("fastpath", 0)
    record.tokens_fullagent = by_phase.get("fullagent", 0)
    record.tokens_total = s.get("automatic_total", 0)
    store.write(record)


def _run_fullagent(
    task: str,
    repo: Path,
    client: BilledClient,
    ledger: CostLedger,
    skill_index: SkillIndex,
    target: Path,
    *,
    harness: AcceptanceHarness | None,
    dry_run: bool,
    max_turns: int,
    escalation_context: str,
    confirm: ConfirmCallback | None = None,
    seed: Seed = "head",
    history: str = "",
    on_event: EventCallback | None = None,
) -> RunResult:
    with WorktreeWorkspace(repo, use_git=True, seed=seed) as workspace:
        baseline = workspace.snapshot()
        inp = FullAgentInput(
            task=task,
            baseline=baseline,
            skill_index=skill_index,
            tool_registry=default_registry(),
            escalation_context=escalation_context,
            max_turns=max_turns,
            model_tier=ModelTier.BALANCED,
            history=history,
            on_event=on_event,
        )
        fa = execute_full_agent(inp, workspace, client)

        if not fa.success:
            return RunResult(
                success=False, mode_used="fullagent", finish_reason=fa.finish_reason,
                changed_files=fa.changed_files, reasoning=fa.reasoning,
                ledger=ledger, error=fa.error, turns_used=fa.turns_used,
            )

        changed = filter_acceptance_paths(fa.changed_files)

        if harness is not None and len(harness) > 0:
            acc = harness.run(workspace.root)
            if not all_passed(acc):
                fail = first_failure(acc)
                return RunResult(
                    success=False, mode_used="fullagent", finish_reason="acceptance_failed",
                    changed_files=changed, reasoning=fa.reasoning, ledger=ledger,
                    error=str(fail.diagnostic()) if fail else "acceptance failed",
                    turns_used=fa.turns_used,
                )

        if not _accepted_by_user(
            confirm,
            task=task,
            mode_used="fullagent",
            reasoning=fa.reasoning,
            changed_files=changed,
            source=repo,
            shadow=workspace.root,
        ):
            return RunResult(
                success=False, mode_used="fullagent", finish_reason="rejected",
                changed_files=changed, reasoning=fa.reasoning, ledger=ledger,
                dry_run=dry_run, turns_used=fa.turns_used, rejected=True,
            )

        if not dry_run:
            workspace.finalize(target)

        return RunResult(
            success=True, mode_used="fullagent", finish_reason=fa.finish_reason,
            changed_files=changed, reasoning=fa.reasoning,
            ledger=ledger, dry_run=dry_run, turns_used=fa.turns_used,
        )
