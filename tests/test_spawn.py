"""Tests for sub-agent spawn callback (make_spawn_callback).

Verifies that spawning sub-agents via the LangGraph nested-graph mechanism:
- Returns an error string for unknown presets (no crash)
- Runs a sub-agent to completion and surfaces finish_summary
- Respects preset max_turns independently of the parent
- Charges tokens to the subagent phase, not fullagent
- Injects pinned skill bodies into the sub-agent system prompt
- Does not allow sub-agents to spawn further (no recursion)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from qqcode.agents.graph import make_spawn_callback
from qqcode.agents.subagent import EXPLORER, register_preset
from qqcode.models.billing import BilledClient, RetryPolicy
from qqcode.models.protocol import (
    Completion,
    CostLedger,
    Msg,
    Role,
    TextContent,
    ToolUseContent,
    Usage,
)
from qqcode.skills.index import SkillIndex
from qqcode.skills.skill import RoutingHint, Skill
from qqcode.tools.builtins import default_registry
from qqcode.workspace.worktree import WorktreeWorkspace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def finish_completion(summary: str = "done") -> Completion:
    return Completion(
        content=[ToolUseContent(id="fin", name="finish", input={"summary": summary})],
        stop_reason="tool_use",
        usage=Usage(input_tokens=100, output_tokens=50),
        raw={},
    )


def text_completion(text: str = "thinking...") -> Completion:
    return Completion(
        content=[TextContent(text=text)],
        stop_reason="end_turn",
        usage=Usage(input_tokens=50, output_tokens=20),
        raw={},
    )


def tool_completion(name: str, **kwargs: Any) -> Completion:
    return Completion(
        content=[ToolUseContent(id=f"c_{name}", name=name, input=kwargs)],
        stop_reason="tool_use",
        usage=Usage(input_tokens=80, output_tokens=30),
        raw={},
    )


class RecordingAdapter:
    """Replays a script and records every invoke() call."""

    def __init__(self, script: list[Completion]) -> None:
        self._script = list(script)
        self.calls: list[list[Msg]] = []

    def invoke(self, messages: list[Msg], **kwargs: Any) -> Completion:
        self.calls.append(list(messages))
        if not self._script:
            raise AssertionError("Script exhausted")
        return self._script.pop(0)


def make_client(
    script: list[Completion],
    *,
    adapter: RecordingAdapter | None = None,
) -> tuple[BilledClient, RecordingAdapter]:
    rec = adapter or RecordingAdapter(script)
    client = BilledClient(
        rec,
        ledger=CostLedger(),
        retry_policy=RetryPolicy(max_attempts=1, sleep=lambda _: None),
    )
    return client, rec


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSpawnCallback:
    def setup_method(self) -> None:
        self.tmp = tempfile.mkdtemp()
        root = Path(self.tmp)
        (root / "src").mkdir()
        (root / "src" / "main.py").write_text("def main(): pass\n")
        self.workspace = WorktreeWorkspace(root, use_git=False)
        self.skill_index = SkillIndex()
        self.registry = default_registry()

    def teardown_method(self) -> None:
        self.workspace.cleanup()

    def _spawn_fn(
        self,
        script: list[Completion],
        *,
        adapter: RecordingAdapter | None = None,
    ) -> tuple[Any, BilledClient, RecordingAdapter]:
        client, rec = make_client(script, adapter=adapter)
        fn = make_spawn_callback(self.workspace, client, self.skill_index, self.registry)
        return fn, client, rec

    # --- Unknown preset -------------------------------------------------------

    def test_unknown_preset_returns_error_string(self) -> None:
        spawn, _, _ = self._spawn_fn([])
        result = spawn("does-not-exist", "task")
        assert "Unknown" in result
        assert "does-not-exist" in result

    def test_unknown_preset_does_not_raise(self) -> None:
        spawn, _, _ = self._spawn_fn([])
        result = spawn("__totally_fake__", "task")
        assert isinstance(result, str)

    # --- Successful execution -------------------------------------------------

    def test_returns_finish_summary(self) -> None:
        spawn, _, _ = self._spawn_fn([finish_completion("found 3 entry points")])
        result = spawn("explorer", "map the codebase")
        assert result == "found 3 entry points"

    def test_stuck_returns_fallback_string(self) -> None:
        # Two consecutive text turns → stuck
        spawn, _, _ = self._spawn_fn([text_completion(), text_completion()])
        result = spawn("explorer", "map")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_max_turns_terminates_early(self) -> None:
        # Register a 1-turn preset so we can verify it stops at 1 model call
        tight = EXPLORER.derive(name="tight-explorer", max_turns=1)
        register_preset(tight, overwrite=True)

        # Provide 5 completions but only 1 turn is allowed
        script = [tool_completion("read_file", path="src/main.py")] * 5
        spawn, _, rec = self._spawn_fn(script)
        spawn("tight-explorer", "read everything")
        # Exactly 1 model call (turn 1 hits max_turns, then graph ends)
        assert len(rec.calls) == 1

    # --- Token phase accounting -----------------------------------------------

    def test_tokens_charged_to_subagent_phase(self) -> None:
        spawn, client, _ = self._spawn_fn([finish_completion("done")])
        spawn("explorer", "task")
        ledger = client._ledger  # noqa: SLF001
        assert ledger.subagent_tokens > 0

    def test_subagent_tokens_not_in_fullagent_phase(self) -> None:
        spawn, client, _ = self._spawn_fn([finish_completion("x"), finish_completion("y")])
        spawn("explorer", "t1")
        spawn("reviewer", "t2")
        ledger = client._ledger  # noqa: SLF001
        assert ledger.fullagent_tokens == 0

    # --- Tool surface isolation -----------------------------------------------

    def test_read_only_preset_write_attempt_returns_error_result(self) -> None:
        # explorer is READ_ONLY; write_file is not in allowed_tools →
        # executor returns an error result, sub-agent can still finish
        script = [
            tool_completion("write_file", path="x.txt", content="y"),
            finish_completion("done despite error"),
        ]
        spawn, _, _ = self._spawn_fn(script)
        result = spawn("explorer", "write something")
        assert result == "done despite error"

    # --- No recursive spawning ------------------------------------------------

    def test_subagent_cannot_spawn_further(self) -> None:
        # Sub-agent calls spawn_subagent → ToolExecutor returns "not available"
        # error because spawn_callback=None; sub-agent sees the error and finishes
        script = [
            tool_completion("spawn_subagent", preset="reviewer", task="review"),
            finish_completion("completed despite spawn error"),
        ]
        spawn, _, _ = self._spawn_fn(script)
        result = spawn("planner", "plan something")
        assert result == "completed despite spawn error"

    # --- Pinned skills --------------------------------------------------------

    def test_pinned_skills_injected_in_system_prompt(self) -> None:
        skill = Skill(
            name="test-guide",
            description="testing conventions",
            body="Always use pytest. Run with pytest -q.",
            routing_hint=RoutingHint.NONE,
        )
        self.skill_index._skills["test-guide"] = skill  # noqa: SLF001

        skilled = EXPLORER.derive(
            name="skilled-explorer",
            pinned_skills=("test-guide",),
        )
        register_preset(skilled, overwrite=True)

        rec = RecordingAdapter([finish_completion("done")])
        spawn, _, _ = self._spawn_fn([], adapter=rec)
        spawn("skilled-explorer", "explore tests")

        assert rec.calls, "No invoke calls made"
        system_text = " ".join(
            b.text
            for m in rec.calls[0]
            if m.role == Role.SYSTEM
            for b in m.content
            if isinstance(b, TextContent)
        )
        assert "Always use pytest" in system_text

    def test_unknown_pinned_skill_silently_skipped(self) -> None:
        spec = EXPLORER.derive(
            name="ghost-skill-explorer",
            pinned_skills=("nonexistent-skill",),
        )
        register_preset(spec, overwrite=True)

        spawn, _, _ = self._spawn_fn([finish_completion("done")])
        result = spawn("ghost-skill-explorer", "task")
        assert result == "done"
