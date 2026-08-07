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

import re
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

# Cap on paths recovered from the task text when no hint was supplied. The
# total-chars budget already bounds the prompt; this bounds the tree scan and
# keeps a task that lists a dozen filenames from crowding out the task itself.
MAX_PREFETCH_RESOLVED_FILES = 3

# Depth ceiling for resolving a bare basename ("config.py") to a real path.
# Bounds an otherwise O(repo) search on the code path whose whole justification
# is being cheap: a repository vendoring a virtualenv can hold >100k files.
MAX_PREFETCH_SCAN_DEPTH = 4

# Directories never worth searching for task context: build artefacts, vendored
# dependencies, and caches. Skipping them also removes the main source of
# ambiguous basename collisions (every venv has its own `config.py`).
_EXCLUDED_DIRS = frozenset({
    ".git",
    ".hg",
    ".svn",
    ".tox",
    ".nox",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "site-packages",
    "dist",
    "build",
    ".eggs",
})

# A path-looking token: a name ending in a known source extension, optionally
# with directory components. Requiring a known extension is what keeps ordinary
# prose ("Fix the bug.Then add a test.") from parsing as a filename — a bare
# `\S+\.\S+` matches sentences far more often than it matches paths.
_PATH_TOKEN = re.compile(
    r"\b[\w][\w./-]*\.(?:py|pyi|js|jsx|ts|tsx|go|rs|rb|java|c|h|cpp|hpp|cs|php|swift|kt"
    r"|sh|sql|css|scss|html|json|toml|yaml|yml|ini|cfg|md|rst|txt)\b"
)


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
    # Files to inline when `files_hint` is empty. Advisory only: read for prompt
    # context, never compared against the diff. Kept separate from `files_hint`
    # precisely so a guess cannot become condition 3's contract.
    prefetch_hint: tuple[str, ...] = ()
    # Digest of earlier turns. Empty in batch mode. FastPath is a single call,
    # so this is its only chance to learn what "that change" refers to.
    history: str = ""


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

About the code you can see:
- Only the files listed under "Current file contents" have been read for you.
  Base your edits on that text and preserve everything in it you were not asked
  to change.
- A file NOT listed there has not been read. That says nothing about whether it
  exists. Do not reconstruct such a file from memory of what it probably
  contains: you write whole files, so a guess silently destroys the real one.
  Return an empty files array instead and name the file you need in reasoning.
- Creating a genuinely new file is still fine when the task calls for one.
"""


def _build_messages(
    task: str,
    skill_bodies: list[str],
    file_contents: dict[str, str],
    history: str = "",
) -> list[Msg]:
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

    # After the breakpoint with the skills, and before the task: history is
    # per-conversation, so putting it in the cached prefix would poison the
    # prefix for every other task.
    if history:
        tail.append(history)

    tail.append(f"## Task\n{task}")
    messages.append(Msg(role=Role.USER, content=[TextContent(text=t) for t in tail]))
    return messages


def names_a_path(task: str) -> bool:
    """Whether the task text offers at least one filename to prefetch.

    Exposed for the router, which must decide whether recovering names from text
    is even possible before it spends a call asking a model for them. Shares
    `_PATH_TOKEN` with `resolve_prefetch_paths` so the two cannot disagree about
    what counts as a filename.
    """
    return bool(_PATH_TOKEN.search(task))


def resolve_prefetch_paths(
    task: str,
    files_hint: tuple[str, ...],
    workspace: Workspace,
    prefetch_hint: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Decide which files to inline into the prompt.

    `files_hint` serves two masters. For prefetch it is advisory — a guess that
    costs tokens when wrong. For condition 3 it is a contract: the diff must be
    a subset of it. Only L1 produces one, so L0 and the fallback route reach
    FastPath with `()`, and the prefetch then reads nothing while the prompt
    still claims file contents were provided. The model cannot see the code it
    was asked to change, so it takes the documented exit and the call is wasted.

    This closes that gap on the prefetch side only. When a hint exists it is
    returned untouched, including paths that do not exist yet — a hint naming a
    new file means "create it", and filtering it here would silently narrow the
    enforcement set. Otherwise filenames are recovered from the task text and
    kept only when they resolve to exactly one real file.

    The result deliberately does not flow back into `files_hint`. Feeding it
    there would turn a guess into a contract and reject correct patches that
    touch a file the task never named — trading a decline for a wrong rejection.

    `prefetch_hint` is the same idea from the other direction: a guess the router
    obtained (L1 names files even when L0 made the decision) that is safe to read
    but must never be enforced. Text extraction is tried first because the task
    naming a file is stronger evidence than a classifier's guess about it; the
    hint is the fallback for statements that name no file at all, which is what
    real issue reports look like.
    """
    if files_hint:
        return files_hint

    candidates = [raw.lstrip("./") for raw in _PATH_TOKEN.findall(task)]
    if not candidates:
        return _resolve_advisory(prefetch_hint, workspace)

    resolved: list[str] = []
    unresolved: list[str] = []

    # An explicit relative path needs no search. This is the common case and it
    # costs one stat per candidate, so try it before walking anything.
    for candidate in candidates:
        if _is_file(workspace, candidate):
            _append_unique(resolved, candidate)
        else:
            unresolved.append(candidate)

    # Only a bare basename ("config.py") justifies a directory walk. Search a
    # bounded, shallow neighbourhood rather than the whole tree: `list_files()`
    # is O(repo) and this runs on the path whose entire purpose is being fast —
    # a repository vendoring a virtualenv can hold >100k files, where a full
    # rglob costs seconds. Prefer no context over a slow prompt.
    if unresolved and len(resolved) < MAX_PREFETCH_RESOLVED_FILES:
        for candidate in unresolved:
            match = _find_unique_shallow(workspace, candidate)
            if match is not None:
                _append_unique(resolved, match)

    if resolved:
        return tuple(sorted(resolved)[:MAX_PREFETCH_RESOLVED_FILES])

    # The task named files but none of them exist. The advisory hint is the only
    # remaining source, and sending nothing is the outcome being fixed here.
    return _resolve_advisory(prefetch_hint, workspace)


def _resolve_advisory(prefetch_hint: tuple[str, ...], workspace: Workspace) -> tuple[str, ...]:
    """Keep the advisory paths that name a readable file in this workspace.

    Verification is not optional. These names come from a classifier that never
    saw the repository, so they are plausible-looking guesses: a wrong one either
    fails the read or, worse, inlines an unrelated file as authoritative context.
    Containment goes through the workspace guard, like every other prefetch path.

    Unlike `files_hint`, a non-existent path is dropped rather than kept. A hint
    means "create this"; an advisory guess about an existing bug does not.
    """
    verified = [p for p in prefetch_hint if _is_file(workspace, p)]
    return tuple(sorted(verified)[:MAX_PREFETCH_RESOLVED_FILES])


def _find_unique_shallow(workspace: Workspace, basename: str) -> str | None:
    """The one file named `basename` within a shallow search, or None.

    Returns None when the name is missing *or* ambiguous. Inlining an arbitrary
    `config.py` would spend tokens on misleading context, which is worse than
    sending none.

    Depth is capped because the cost is unbounded otherwise, and because a file
    the task refers to by bare name is rarely buried deep. A deeper file is not
    resolved — it simply gets no prefetch, which is the pre-fix behaviour.
    """
    root = Path(workspace.root)
    matches: list[str] = []

    for depth in range(MAX_PREFETCH_SCAN_DEPTH + 1):
        pattern = "/".join(["*"] * depth + [basename]) if depth else basename
        try:
            found = [p for p in root.glob(pattern) if p.is_file()]
        except OSError:
            return None
        for p in found:
            rel = p.relative_to(root).as_posix()
            if _EXCLUDED_DIRS.isdisjoint(p.relative_to(root).parts[:-1]):
                _append_unique(matches, rel)
        if len(matches) > 1:
            return None  # ambiguous; stop early

    return matches[0] if len(matches) == 1 else None


def _is_file(workspace: Workspace, path: str) -> bool:
    """Whether `path` names a readable file, without trusting it to be safe.

    Goes through `read_file` rather than the filesystem so the workspace's own
    path guard decides what is reachable; a task naming `../../etc/passwd`
    must not become prompt context.
    """
    try:
        workspace.read_file(path)
    except Exception:
        return False
    return True


def _append_unique(paths: list[str], path: str) -> None:
    """Add `path` if absent — a task may name the same file twice."""
    if path not in paths:
        paths.append(path)


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
    prefetch_paths = resolve_prefetch_paths(
        inp.task, inp.files_hint, workspace, inp.prefetch_hint
    )
    file_contents = _prefetch_files(prefetch_paths, workspace)
    messages = _build_messages(inp.task, [s.body for s in skills], file_contents, inp.history)

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

    Files that don't exist or fail to read are silently omitted, and the omission
    is not self-describing: absence means "not read", which is a different claim
    from "does not exist". Do not infer the latter — a path dropped here because
    it exceeded the budget or failed a guard check names a file that is still
    very much on disk. `SYSTEM_PROMPT` tells the model the same thing, because a
    whole-file write based on the wrong inference destroys the real file.
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
            pass  # missing, over budget, or guard-refused — indistinguishable here

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
