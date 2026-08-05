"""Tests for M4: ToolExecutor and Full Agent loop termination conditions."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest

from qqcode.agents.full_agent import FullAgentInput, execute_full_agent
from qqcode.models.billing import BilledClient, RetryPolicy
from qqcode.models.errors import BudgetExhaustedError, ProviderError
from qqcode.models.protocol import (
    Completion,
    CostLedger,
    Msg,
    Role,
    TextContent,
    ToolResultContent,
    ToolUseContent,
    Usage,
)
from qqcode.skills import SkillIndex
from qqcode.tools.artifacts import InMemoryArtifactStore
from qqcode.tools.builtins import default_registry
from qqcode.tools.executor import ToolExecutor
from qqcode.workspace.worktree import WorktreeWorkspace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_call(name: str, **kwargs: Any) -> ToolUseContent:
    return ToolUseContent(id=f"call_{name}", name=name, input=kwargs)


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


class FakeAdapter:
    """Replays a script of completions."""

    def __init__(self, script: list[Completion]) -> None:
        self._script = list(script)

    def invoke(self, messages: list[Msg], **kwargs: Any) -> Completion:
        if not self._script:
            raise AssertionError("Script exhausted")
        return self._script.pop(0)


def make_client(script: list[Completion]) -> BilledClient:
    return BilledClient(
        FakeAdapter(script),
        ledger=CostLedger(),
        retry_policy=RetryPolicy(max_attempts=1, sleep=lambda _: None),
    )


# ---------------------------------------------------------------------------
# ToolExecutor tests
# ---------------------------------------------------------------------------


class TestToolExecutor:
    def setup_method(self) -> None:
        self.tmp = tempfile.mkdtemp()
        root = Path(self.tmp)
        (root / "hello.txt").write_text("Hello, world!\n")
        (root / "src").mkdir()
        (root / "src" / "main.py").write_text("def main(): pass\n")
        self.workspace = WorktreeWorkspace(root, use_git=False)
        self.store = InMemoryArtifactStore()
        self.executor = ToolExecutor(self.workspace, self.store, SkillIndex())

    def teardown_method(self) -> None:
        self.workspace.cleanup()

    def test_read_file_returns_content(self) -> None:
        result = self.executor.execute(make_call("read_file", path="hello.txt"))
        assert not result.is_error
        assert "Hello, world!" in result.content

    def test_read_file_missing(self) -> None:
        result = self.executor.execute(make_call("read_file", path="nope.txt"))
        assert result.is_error

    def test_read_file_path_traversal_blocked(self) -> None:
        result = self.executor.execute(make_call("read_file", path="../../etc/passwd"))
        assert result.is_error

    def test_list_files_returns_paths(self) -> None:
        result = self.executor.execute(make_call("list_files"))
        assert not result.is_error
        assert "hello.txt" in result.content

    def test_grep_finds_matches(self) -> None:
        result = self.executor.execute(make_call("grep", pattern="Hello"))
        assert not result.is_error
        assert "hello.txt:1:" in result.content

    def test_grep_no_matches(self) -> None:
        result = self.executor.execute(make_call("grep", pattern="xyzzy_nonexistent"))
        assert not result.is_error
        assert "(no matches)" in result.content

    def test_grep_invalid_regex(self) -> None:
        result = self.executor.execute(make_call("grep", pattern="[invalid"))
        assert result.is_error
        assert "Invalid regex" in result.content

    def test_write_file_creates_file(self) -> None:
        result = self.executor.execute(
            make_call("write_file", path="new.txt", content="new content\n")
        )
        assert not result.is_error
        assert "new.txt" in self.executor.files_touched
        assert self.workspace.read_file("new.txt") == "new content\n"

    def test_write_file_dotenv_blocked(self) -> None:
        result = self.executor.execute(
            make_call("write_file", path=".env", content="SECRET=1")
        )
        assert result.is_error

    def test_edit_file_replaces_unique_string(self) -> None:
        result = self.executor.execute(
            make_call("edit_file", path="hello.txt", old_string="Hello", new_string="Goodbye")
        )
        assert not result.is_error
        assert "Goodbye" in self.workspace.read_file("hello.txt")

    def test_edit_file_missing_old_string(self) -> None:
        result = self.executor.execute(
            make_call("edit_file", path="hello.txt", old_string="XYZ", new_string="abc")
        )
        assert result.is_error
        assert "not found" in result.content

    def test_edit_file_non_unique_string(self) -> None:
        self.workspace.write_file("dup.txt", "aa\naa\n")
        result = self.executor.execute(
            make_call("edit_file", path="dup.txt", old_string="aa", new_string="bb")
        )
        assert result.is_error
        assert "2 times" in result.content

    def test_run_command_allowed(self) -> None:
        result = self.executor.execute(make_call("run_command", cmd=["echo", "hi"]))
        assert not result.is_error
        assert "hi" in result.content

    def test_run_command_denied_binary(self) -> None:
        result = self.executor.execute(make_call("run_command", cmd=["curl", "http://x.com"]))
        assert result.is_error

    def test_run_command_string_rejected(self) -> None:
        result = self.executor.execute(make_call("run_command", cmd="echo hi"))
        assert result.is_error
        assert "array" in result.content

    def test_artifact_roundtrip(self) -> None:
        artifact_id = self.store.put("big content" * 100)
        result = self.executor.execute(make_call("read_artifact", artifact_id=artifact_id))
        assert not result.is_error
        assert "big content" in result.content

    def test_artifact_unknown_id(self) -> None:
        result = self.executor.execute(make_call("read_artifact", artifact_id="art_9999"))
        assert result.is_error

    def test_unknown_tool_returns_error(self) -> None:
        result = self.executor.execute(make_call("nonexistent_tool"))
        assert result.is_error
        assert "Unknown tool" in result.content

    def test_oversized_output_is_stored_as_artifact(self) -> None:
        big = "x" * 10_000
        self.workspace.write_file("big.txt", big)
        result = self.executor.execute(make_call("read_file", path="big.txt"))
        assert not result.is_error
        assert "omitted" in result.content
        assert len(self.store) == 1

    def test_spawn_without_callback_returns_error(self) -> None:
        result = self.executor.execute(
            make_call("spawn_subagent", preset="explorer", task="map the code")
        )
        assert result.is_error
        assert "not available" in result.content

    def test_spawn_with_callback(self) -> None:
        executor = ToolExecutor(
            self.workspace,
            self.store,
            SkillIndex(),
            spawn_callback=lambda preset, task: f"Result from {preset}: done",
        )
        result = executor.execute(
            make_call("spawn_subagent", preset="explorer", task="map the code")
        )
        assert not result.is_error
        assert "explorer" in result.content

    def test_files_touched_tracks_writes(self) -> None:
        self.executor.execute(make_call("write_file", path="a.txt", content="a"))
        self.executor.execute(make_call("write_file", path="b.txt", content="b"))
        self.executor.execute(make_call("edit_file", path="hello.txt", old_string="Hello", new_string="Hi"))
        assert self.executor.files_touched == {"a.txt", "b.txt", "hello.txt"}


# ---------------------------------------------------------------------------
# Full Agent loop termination tests (offline, fake client)
# ---------------------------------------------------------------------------


class TestFullAgentLoop:
    def setup_method(self) -> None:
        self.tmp = tempfile.mkdtemp()
        root = Path(self.tmp)
        (root / "hello.txt").write_text("Hello\n")
        self.workspace = WorktreeWorkspace(root, use_git=False)
        self.baseline = self.workspace.snapshot()

    def teardown_method(self) -> None:
        self.workspace.cleanup()

    def _inp(self, **kw: Any) -> FullAgentInput:
        return FullAgentInput(
            task="test task",
            baseline=self.baseline,
            skill_index=SkillIndex(),
            tool_registry=default_registry(),
            **kw,
        )

    def test_explicit_finish_terminates(self) -> None:
        client = make_client([
            tool_completion("read_file", path="hello.txt"),
            finish_completion("all done"),
        ])
        result = execute_full_agent(self._inp(), self.workspace, client)
        assert result.success
        assert result.finish_reason == "explicit"
        assert result.reasoning == "all done"

    def test_max_turns_terminates(self) -> None:
        client = make_client([
            tool_completion("read_file", path="hello.txt"),
            tool_completion("read_file", path="hello.txt"),
            tool_completion("read_file", path="hello.txt"),
        ])
        result = execute_full_agent(self._inp(max_turns=3), self.workspace, client)
        assert result.finish_reason == "max_turns"
        assert result.turns_used == 3

    def test_stuck_two_text_turns(self) -> None:
        client = make_client([text_completion("thinking"), text_completion("still thinking")])
        result = execute_full_agent(self._inp(), self.workspace, client)
        assert result.finish_reason == "stuck"
        assert not result.success

    def test_stuck_repeated_same_error(self) -> None:
        # .env write fails with same error 3 times
        client = make_client([
            tool_completion("write_file", path=".env", content="bad"),
            tool_completion("write_file", path=".env", content="bad"),
            tool_completion("write_file", path=".env", content="bad"),
        ])
        result = execute_full_agent(self._inp(), self.workspace, client)
        assert result.finish_reason == "stuck"

    def test_budget_exhausted_terminates(self) -> None:
        class ExhaustOnFirst:
            def invoke(self, messages: list[Msg], **kwargs: Any) -> Completion:
                raise BudgetExhaustedError("spent")

        client = BilledClient(
            ExhaustOnFirst(),
            ledger=CostLedger(),
            retry_policy=RetryPolicy(max_attempts=1, sleep=lambda _: None),
        )
        result = execute_full_agent(self._inp(), self.workspace, client)
        assert result.finish_reason == "budget"

    def test_provider_error_terminates(self) -> None:
        class AlwaysFails:
            def invoke(self, messages: list[Msg], **kwargs: Any) -> Completion:
                raise ProviderError("broken", status_code=400, retryable=False)

        client = BilledClient(
            AlwaysFails(),
            ledger=CostLedger(),
            retry_policy=RetryPolicy(max_attempts=1, sleep=lambda _: None),
        )
        result = execute_full_agent(self._inp(), self.workspace, client)
        assert result.finish_reason == "error"
        assert result.error is not None
        assert not result.success

    def test_write_then_finish_changes_captured(self) -> None:
        client = make_client([
            tool_completion("write_file", path="output.txt", content="result\n"),
            finish_completion("wrote output"),
        ])
        result = execute_full_agent(self._inp(), self.workspace, client)
        assert result.success
        assert "output.txt" in result.changed_files

    def test_tool_results_are_carried_on_the_tool_role(self) -> None:
        """Tool results must be Role.TOOL so the OpenAI adapter can encode them.

        Anthropic folds Role.TOOL and Role.USER to the same wire role, so a
        mislabelled result is invisible there but is dropped entirely by the
        OpenAI adapter.
        """
        client = make_client([
            tool_completion("read_file", path="hello.txt"),
            finish_completion("done"),
        ])
        result = execute_full_agent(self._inp(), self.workspace, client)
        carriers = [
            m.role for m in result.messages
            if any(isinstance(b, ToolResultContent) for b in m.content)
        ]
        assert carriers, "no tool result was recorded in the transcript"
        assert set(carriers) == {Role.TOOL}

    def test_escalation_context_in_system_prompt(self) -> None:
        client = make_client([finish_completion("done")])
        inp = self._inp(escalation_context="FastPath failed: acceptance_failed")
        result = execute_full_agent(inp, self.workspace, client)
        assert result.success
        system_msgs = [m for m in result.messages if m.role == Role.SYSTEM]
        system_text = " ".join(
            b.text for m in system_msgs for b in m.content if isinstance(b, TextContent)
        )
        assert "FastPath failed" in system_text

    def test_tool_registry_none_uses_defaults(self) -> None:
        client = make_client([finish_completion("done")])
        inp = FullAgentInput(
            task="test",
            baseline=self.baseline,
            skill_index=SkillIndex(),
            tool_registry=None,  # should use default_registry()
        )
        result = execute_full_agent(inp, self.workspace, client)
        assert result.success
