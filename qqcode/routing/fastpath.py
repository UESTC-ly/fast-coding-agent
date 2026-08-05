"""FastPath execution: single-shot patch generation, then the three-condition gate.

FastPath spends one model call on a complete patch, applies it to the shadow
workspace, and then decides whether the result may be finalized. It never
iterates: a task that needs a second look is a task that belongs to Full Agent,
and the cheapest way to learn that is to fail fast and escalate.

The gate is all three of these, in this order:

1. **Agent finish state valid** — checked first because it is free. A truncated
   response (`max_tokens`) or a missing patch means there is nothing to verify.
2. **Diff ⊆ expected file set** — checked before acceptance because it needs no
   subprocess. A patch that touched the wrong files is already disqualified, so
   running its tests would only cost wall-clock.
3. **Hidden acceptance passes** — last, because it is the expensive one.

Every failure returns a structured diagnostic rather than raising. The diagnostic
is what makes escalation worth more than a cold retry: Full Agent starts from the
clean baseline but knows what FastPath tried and how it failed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qqcode.acceptance import (
    ACCEPTANCE_DIR,
    AcceptanceHarness,
    all_passed,
    filter_acceptance_paths,
    first_failure,
    is_acceptance_path,
)
from qqcode.models.billing import BilledClient
from qqcode.models.protocol import (
    Budget,
    ContentBlock,
    ModelTier,
    Msg,
    OutputSpec,
    Role,
    TextContent,
    ToolUseContent,
)
from qqcode.skills.index import SkillIndex
from qqcode.tools.registry import ToolRegistry
from qqcode.workspace.protocol import Workspace, WorkspaceSnapshot

# Truncation is the only stop reason that invalidates a patch: the tail is
# missing, so writing it would corrupt files rather than fail cleanly. Both
# providers report truncation as `max_tokens`.
#
# Do NOT invert this into a whitelist of "good" stop reasons. Structured output
# is a tool call on Anthropic's wire (`tool_use`) but plain message content on
# OpenAI's (`stop` -> `end_turn`), so demanding one provider's success token
# rejects every valid OpenAI patch.
TRUNCATED_STOP_REASON = "max_tokens"

PATCH_TOOL_NAME = "submit_patch"

# Budget for file contents inlined into the prompt. Reading is local, so this
# costs no extra model call — but the tokens still land in the request, and
# FastPath's whole value is being cheap. A task whose files exceed this belongs
# to Full Agent, which can read them selectively instead of all at once.
MAX_PREFETCH_TOTAL_CHARS = 20_000
MAX_PREFETCH_FILE_CHARS = 8_000


class EscalationReason:
    """Why FastPath handed the task to Full Agent.

    A closed set so callers can branch on it and the trace store can aggregate
    it. Grouped by which of the three conditions failed.
    """

    # Condition 2 — no valid finish state
    MODEL_ERROR = "model_error"
    NO_PATCH = "no_patch"
    TRUNCATED = "truncated"
    DECLINED = "declined"
    # Applying the patch never got far enough to check anything
    WRITE_ERROR = "write_error"
    # Condition 3 — diff exceeded the expected set
    UNEXPECTED_MODIFICATIONS = "unexpected_modifications"
    ACCEPTANCE_TAMPERING = "acceptance_tampering"
    # Condition 1 — hidden behavioral verification
    ACCEPTANCE_FAILED = "acceptance_failed"
    HARNESS_ERROR = "harness_error"


@dataclass(frozen=True)
class FastPathInput:
    """Task and context for one FastPath attempt."""

    task: str
    baseline: WorkspaceSnapshot
    skill_index: SkillIndex
    tool_registry: ToolRegistry
    # Expected touched files, from L0/L1. Empty means the diff check cannot be
    # enforced — see `execute_fastpath` for why that is not a silent pass.
    files_hint: tuple[str, ...] = ()


PATCH_OUTPUT_SPEC = OutputSpec(
    tool_name=PATCH_TOOL_NAME,
    schema={
        "type": "object",
        "required": ["reasoning", "files"],
        "properties": {
            "reasoning": {
                "type": "string",
                "description": (
                    "Why this approach solves the task. If the task cannot be done in one "
                    "pass, explain what is missing here and return an empty files array."
                ),
            },
            "files": {
                "type": "array",
                "description": (
                    "Complete replacement content for each file to write. Paths are "
                    "relative to the repository root. Empty means the task needs more "
                    "than one pass."
                ),
                "items": {
                    "type": "object",
                    "required": ["path", "content"],
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                },
            },
        },
    },
)


@dataclass(frozen=True)
class FastPathResult:
    """FastPath outcome: either finalizable, or an escalation with diagnostics."""

    success: bool
    final_snapshot: WorkspaceSnapshot | None = None
    changed_files: frozenset[str] = field(default_factory=frozenset)
    reasoning: str = ""
    # Populated only when success is False.
    escalation_reason: str = ""
    diagnostic: dict[str, str] = field(default_factory=dict)

    @property
    def escalated(self) -> bool:
        return not self.success


SYSTEM_PROMPT = f"""You are a FastPath coding agent. You get exactly one attempt.

Produce a complete patch by calling {PATCH_TOOL_NAME}. Rules:
- Write full replacement content for each file, not a diff.
- Do not explore, ask questions, or plan for a second pass.
- Touch only files the task requires. Do not refactor or reformat code the task
  did not ask about.
- Never add or edit test files unless the task explicitly asks for tests. This
  work is verified by tests you cannot see; editing tests cannot make the task
  pass and may make a correct patch score as a failure.
- Preserve everything the task did not ask you to change: imports, unrelated
  functions, docstrings, and formatting. You are writing whole files, so an
  omission silently deletes working code.
- Handle the edge cases the task names explicitly (empty input, None, zero,
  boundary values). If the task states an error message or return value, use it
  verbatim.
- If the task is ambiguous, needs investigation, or cannot be finished in one
  pass, return an empty files array and say why in reasoning. That is a useful
  answer, not a failure — it routes the task to a more capable agent.

Current file contents (if any exist) are provided below the task description.
Base your patch on them rather than refusing to work. Missing files mean you are
creating them from scratch.
"""


def _build_messages(task: str, skill_bodies: list[str], file_contents: dict[str, str]) -> list[Msg]:
    """Assemble the prompt.

    Skill bodies go after the cache breakpoint. Putting per-task skill text
    before it would give every task a different cache prefix, so the prefix
    would never hit and the recompute would cost more than the skill saved.

    File contents come before the task description, after skills. They are
    local context the model needs, not the instruction itself.
    """
    messages = [
        Msg(role=Role.SYSTEM, content=[TextContent(text=SYSTEM_PROMPT)], cache_breakpoint=True)
    ]

    tail = list(skill_bodies)

    # Inline file contents if any were prefetched
    if file_contents:
        tail.append("## Current file contents\n")
        for path, content in sorted(file_contents.items()):
            tail.append(f"### {path}\n```\n{content}\n```")

    tail.append(f"## Task\n{task}")
    messages.append(Msg(role=Role.USER, content=[TextContent(text=t) for t in tail]))
    return messages


def execute_fastpath(
    inp: FastPathInput,
    workspace: Workspace,
    client: BilledClient,
    *,
    harness: AcceptanceHarness | None = None,
) -> FastPathResult:
    """Generate a patch, apply it to the shadow, and run the three-condition gate.

    Args:
        inp: Task and context.
        workspace: Shadow workspace. Nothing here reaches the real repository —
            that only happens if the caller calls `workspace.finalize` after a
            successful result.
        client: Billed client. Every call lands in the shared ledger, including
            this one when it fails and the task escalates.
        harness: Hidden acceptance tests. `None` or an empty harness means no
            behavioral criteria were supplied, so condition 1 imposes nothing;
            the other two conditions still apply.

    Returns:
        A result that is either finalizable or carries an escalation diagnostic.
        This function does not raise for task-level failures — an escalation is a
        normal outcome, not an error.
    """
    _, skills = inp.skill_index.select("fastpath", task=inp.task, paths=inp.files_hint)
    file_contents = _prefetch_files(inp.files_hint, workspace)
    messages = _build_messages(inp.task, [s.body for s in skills], file_contents)

    try:
        completion = client.invoke(
            messages=messages,
            output_spec=PATCH_OUTPUT_SPEC,
            budget=Budget(max_tokens=8192),
            tier=ModelTier.FAST,
            phase="fastpath",
        )
    except Exception as exc:
        return _escalate(
            EscalationReason.MODEL_ERROR,
            {"error": f"{type(exc).__name__}: {exc}"},
        )

    # ---- Condition 2: the agent reached a valid finish state ----
    patch = _extract_patch(completion.content)
    if patch is None:
        return _escalate(
            EscalationReason.NO_PATCH,
            {
                "stop_reason": str(completion.stop_reason),
                "content_types": ",".join(type(b).__name__ for b in completion.content),
            },
        )

    reasoning = str(patch.get("reasoning", ""))
    files = _extract_files(patch)

    if completion.stop_reason == TRUNCATED_STOP_REASON:
        # A truncated patch may look well-formed while missing its tail; writing
        # it would corrupt files rather than fail cleanly.
        return _escalate(
            EscalationReason.TRUNCATED,
            {"stop_reason": str(completion.stop_reason), "reasoning": reasoning},
        )

    if not files:
        # The model took the documented exit: one pass is not enough.
        return _escalate(EscalationReason.DECLINED, {"reasoning": reasoning})

    # ---- Tampering check, before any write ----
    # `PathGuard` has no opinion on the harness directory, so a patch could drop
    # a `conftest.py` there and run arbitrary code during test collection. The
    # harness would overwrite and clean up its own files, but collection happens
    # in between. Refuse the patch outright.
    intruding = sorted(p for p in files if is_acceptance_path(p))
    if intruding:
        return _escalate(
            EscalationReason.ACCEPTANCE_TAMPERING,
            {
                "paths": ",".join(intruding),
                "acceptance_dir": ACCEPTANCE_DIR,
                "reasoning": reasoning,
            },
        )

    # ---- Apply to the shadow ----
    try:
        for path, content in files.items():
            workspace.write_file(path, content)
    except Exception as exc:
        return _escalate(
            EscalationReason.WRITE_ERROR,
            {
                "error": f"{type(exc).__name__}: {exc}",
                "attempted": ",".join(sorted(files)),
                "reasoning": reasoning,
            },
        )

    # ---- Condition 3: diff ⊆ expected ----
    # Runs before acceptance: it needs no subprocess, and a patch that touched
    # the wrong files is disqualified regardless of whether its tests pass.
    changed = filter_acceptance_paths(inp.baseline.changed_files(workspace.snapshot()))

    if inp.files_hint:
        unexpected = changed - set(inp.files_hint)
        if unexpected:
            return _escalate(
                EscalationReason.UNEXPECTED_MODIFICATIONS,
                {
                    "expected": ",".join(sorted(inp.files_hint)),
                    "unexpected": ",".join(sorted(unexpected)),
                    "reasoning": reasoning,
                },
            )
    # With no hint there is nothing to compare against, so this condition is
    # unenforceable rather than satisfied. The caller sees the full changed set
    # in the result and remains responsible for reviewing it before finalizing.

    # ---- Condition 1: hidden acceptance ----
    if harness is not None and len(harness) > 0:
        try:
            results = harness.run(_workspace_root(workspace))
        except Exception as exc:
            # A broken harness is our bug, not the agent's. Escalating on it
            # would silently convert an infrastructure fault into a task verdict.
            return _escalate(
                EscalationReason.HARNESS_ERROR,
                {"error": f"{type(exc).__name__}: {exc}", "reasoning": reasoning},
            )

        if not all_passed(results):
            failure = first_failure(results)
            diagnostic = {"reasoning": reasoning}
            if failure is not None:
                diagnostic.update(failure.diagnostic())
            return _escalate(EscalationReason.ACCEPTANCE_FAILED, diagnostic)

        # The harness cleans up after itself, but re-snapshot rather than trust
        # that: the returned changed set must describe the workspace as it now
        # stands, not as it stood before the tests ran.
        changed = filter_acceptance_paths(inp.baseline.changed_files(workspace.snapshot()))

    return FastPathResult(
        success=True,
        final_snapshot=workspace.snapshot(),
        changed_files=frozenset(changed),
        reasoning=reasoning,
    )


def _escalate(reason: str, diagnostic: dict[str, str]) -> FastPathResult:
    """Build a failure result carrying context for Full Agent."""
    return FastPathResult(success=False, escalation_reason=reason, diagnostic=diagnostic)


def _workspace_root(workspace: Workspace) -> Path:
    """The workspace root as a Path, for the harness's subprocess cwd."""
    return Path(workspace.root)


def _extract_patch(content: list[ContentBlock]) -> dict[str, Any] | None:
    """The structured patch payload, or None when the model did not submit one."""
    for block in content:
        if isinstance(block, ToolUseContent) and block.name == PATCH_TOOL_NAME:
            return block.input
    return None


def _prefetch_files(paths: tuple[str, ...], workspace: Workspace) -> dict[str, str]:
    """Read files the model is expected to modify and cap to budget.

    Files that don't exist or fail to read are silently omitted. The model sees
    their absence and knows to treat them as new files rather than edits.
    """
    contents: dict[str, str] = {}
    total_chars = 0

    for path in paths:
        if total_chars >= MAX_PREFETCH_TOTAL_CHARS:
            break
        try:
            text = workspace.read_file(path)
            if len(text) > MAX_PREFETCH_FILE_CHARS:
                text = text[:MAX_PREFETCH_FILE_CHARS] + "\n... (truncated)"
            contents[path] = text
            total_chars += len(text)
        except Exception:
            pass  # missing or unreadable; model will create it from scratch

    return contents


def _extract_files(patch: dict[str, Any]) -> dict[str, str]:
    """Normalize the patch's file list into `{path: content}`.

    Malformed entries are dropped rather than raised on: a partially valid patch
    still fails the gate below, and dropping keeps the failure attributable to
    the gate instead of to a parse error.
    """
    out: dict[str, str] = {}
    raw = patch.get("files")
    if not isinstance(raw, list):
        return out
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        path = entry.get("path")
        content = entry.get("content")
        if isinstance(path, str) and path and isinstance(content, str):
            out[path] = content
    return out


def build_escalation_context(result: FastPathResult, task: str) -> str:
    """Render a failed attempt as prompt text for Full Agent.

    Full Agent restarts from the clean baseline — the shadow is discarded — so
    this text is the only thing that survives. It says what was tried and how it
    failed, and deliberately does not include the patch itself: a wrong patch is
    an anchor, and the point of escalating is to get an independent attempt.
    """
    if result.success:
        raise ValueError("build_escalation_context is for failed attempts only")

    lines = [
        f"A previous FastPath attempt at this task failed: {task}",
        f"Failure mode: {result.escalation_reason}",
    ]
    if result.diagnostic:
        lines.append("Diagnostics:")
        lines.extend(f"  {k}: {v}" for k, v in sorted(result.diagnostic.items()))
    lines.append(
        "The workspace has been reset to its original state. Treat the above as "
        "evidence about the task, not as a starting point to repair."
    )
    return "\n".join(lines)
