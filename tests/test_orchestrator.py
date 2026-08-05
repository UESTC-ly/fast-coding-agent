"""End-to-end orchestrator tests — all paths, no real API calls.

Tests every path through run_task:
  1. FastPath success
  2. FastPath declined → escalation_blocked (mode=fast)
  3. Full Agent success
  4. FastPath fails → Full Agent recovers  (critical fallback)
  5. Auto router picks Full Agent directly
  6. Trace written on every run
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from qqcode.memory.trace import TraceRecord, TraceStore
from qqcode.models.billing import BilledClient, RetryPolicy
from qqcode.models.protocol import (
    Completion,
    CostLedger,
    Msg,
    ToolUseContent,
    Usage,
)
from qqcode.skills.index import SkillIndex
from qqcode.workspace.worktree import WorktreeWorkspace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeAdapter:
    def __init__(self, script: list[Completion]) -> None:
        self._script = list(script)
        self.calls = 0

    def invoke(self, messages: list[Msg], **kwargs: Any) -> Completion:
        self.calls += 1
        if not self._script:
            raise AssertionError("FakeAdapter script exhausted")
        return self._script.pop(0)


def make_client(script: list[Completion]) -> tuple[BilledClient, FakeAdapter]:
    adapter = FakeAdapter(script)
    client = BilledClient(
        adapter,
        ledger=CostLedger(),
        retry_policy=RetryPolicy(max_attempts=1, sleep=lambda _: None),
    )
    return client, adapter


def patch_ok(path: str = "main.py") -> Completion:
    """FastPath submit_patch — writes one file."""
    return Completion(
        content=[ToolUseContent(
            id="fp1", name="submit_patch",
            input={"reasoning": "done",
                   "files": [{"path": path, "content": '"""Module."""\ndef main(): pass\n'}]},
        )],
        stop_reason="tool_use",
        usage=Usage(input_tokens=500, output_tokens=200), raw={},
    )


def patch_declined() -> Completion:
    """FastPath submit_patch — empty files → DECLINED."""
    return Completion(
        content=[ToolUseContent(
            id="fp_dec", name="submit_patch",
            input={"reasoning": "needs more context", "files": []},
        )],
        stop_reason="tool_use",
        usage=Usage(input_tokens=400, output_tokens=100), raw={},
    )


def fa_finish(summary: str = "done") -> Completion:
    return Completion(
        content=[ToolUseContent(id="fin", name="finish", input={"summary": summary})],
        stop_reason="tool_use",
        usage=Usage(input_tokens=100, output_tokens=50), raw={},
    )


def fa_write_file(path: str = "out.txt") -> Completion:
    return Completion(
        content=[ToolUseContent(id="w1", name="write_file",
                                input={"path": path, "content": "result\n"})],
        stop_reason="tool_use",
        usage=Usage(input_tokens=80, output_tokens=30), raw={},
    )


def l1_classify(decision: str = "fastpath", conf: float = 0.9) -> Completion:
    return Completion(
        content=[ToolUseContent(
            id="r1", name="classify_task",
            input={"decision": decision, "confidence": conf,
                   "files": ["main.py"], "reasoning": "test"},
        )],
        stop_reason="tool_use",
        usage=Usage(input_tokens=100, output_tokens=50), raw={},
    )


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "main.py").write_text("def main(): pass\n")
    return tmp_path


def _run(repo: Path, script: list[Completion], mode: str = "auto",
         trace_store: TraceStore | None = None,
         task: str = "test task") -> tuple[Any, FakeAdapter]:
    from qqcode.config import Config, ProviderConfig
    from qqcode.orchestrator import run_task

    client, adapter = make_client(script)
    config = Config(anthropic=ProviderConfig(api_key="fake", base_url=None),
                    openai=None, default_provider="anthropic")
    with patch("qqcode.orchestrator.build_client",
               return_value=(client, client._ledger)):  # noqa: SLF001
        result = run_task(task=task, repo=repo, config=config,
                          mode=mode, dry_run=True, trace_store=trace_store)
    return result, adapter


# ---------------------------------------------------------------------------
# _try_fastpath direct tests
# ---------------------------------------------------------------------------

class TestTryFastpath:
    def setup_method(self) -> None:
        self.tmp = tempfile.mkdtemp()
        root = Path(self.tmp)
        (root / "main.py").write_text("def main(): pass\n")
        self.workspace = WorktreeWorkspace(root, use_git=False)

    def teardown_method(self) -> None:
        self.workspace.cleanup()

    def test_success_returns_result_and_changed_files(self) -> None:
        from qqcode.orchestrator import _try_fastpath
        client, _ = make_client([patch_ok()])
        result, esc = _try_fastpath(
            task="add docstring", repo=Path(self.tmp), client=client,
            skill_index=SkillIndex(), files_hint=("main.py",),
            harness=None, dry_run=True,
        )
        assert result is not None and result.success
        assert result.mode_used == "fastpath"
        assert "main.py" in result.changed_files
        assert esc == ""

    def test_declined_returns_none_with_context(self) -> None:
        from qqcode.orchestrator import _try_fastpath
        client, _ = make_client([patch_declined()])
        result, esc = _try_fastpath(
            task="complex task", repo=Path(self.tmp), client=client,
            skill_index=SkillIndex(), files_hint=(),
            harness=None, dry_run=True,
        )
        assert result is None
        assert len(esc) > 0

    def test_record_populated_on_success(self) -> None:
        from qqcode.orchestrator import _try_fastpath
        client, _ = make_client([patch_ok()])
        record = TraceRecord()
        _try_fastpath(
            task="t", repo=Path(self.tmp), client=client,
            skill_index=SkillIndex(), files_hint=("main.py",),
            harness=None, dry_run=True, record=record,
        )
        assert record.fastpath_attempted is True
        assert record.fastpath_success is True
        assert record.fastpath_reason == "ok"

    def test_record_populated_on_failure(self) -> None:
        from qqcode.orchestrator import _try_fastpath
        client, _ = make_client([patch_declined()])
        record = TraceRecord()
        _try_fastpath(
            task="t", repo=Path(self.tmp), client=client,
            skill_index=SkillIndex(), files_hint=(),
            harness=None, dry_run=True, record=record,
        )
        assert record.fastpath_attempted is True
        assert record.fastpath_success is False


# ---------------------------------------------------------------------------
# run_task integration: all three modes + escalation
# ---------------------------------------------------------------------------

class TestRunTaskIntegration:
    def test_mode_full_succeeds(self, repo: Path) -> None:
        result, _ = _run(repo, [fa_finish("done")], mode="full")
        assert result.success
        assert result.mode_used == "fullagent"

    def test_mode_full_writes_file(self, repo: Path) -> None:
        result, _ = _run(repo, [fa_write_file(), fa_finish("wrote")], mode="full")
        assert result.success
        assert "out.txt" in result.changed_files

    def test_mode_fast_fastpath_succeeds(self, repo: Path) -> None:
        result, _ = _run(repo, [patch_ok()], mode="fast")
        assert result.success
        assert result.mode_used == "fastpath"

    def test_mode_fast_blocked_when_fastpath_declines(self, repo: Path) -> None:
        result, _ = _run(repo, [patch_declined()], mode="fast")
        assert not result.success
        assert result.finish_reason == "escalation_blocked"

    def test_mode_auto_fastpath_succeeds(self, repo: Path) -> None:
        # Use a task that avoids L0 skill triggers (no "test", "import", etc.)
        result, adapter = _run(repo, [l1_classify("fastpath", 0.92), patch_ok()],
                               mode="auto", task="update the readme file")
        assert result.success
        assert result.mode_used == "fastpath"
        assert adapter.calls == 2  # routing + fastpath

    def test_mode_auto_fastpath_fails_fullagent_recovers(self, repo: Path) -> None:
        """THE CRITICAL PATH: FP declines → FA takes over and completes the task."""
        result, adapter = _run(repo, [
            l1_classify("fastpath", 0.85),   # router says fastpath
            patch_declined(),                 # fastpath declines
            fa_finish("fixed via full agent"),  # full agent succeeds
        ])
        assert result.success
        assert result.mode_used == "fullagent"
        assert result.finish_reason == "explicit"
        assert adapter.calls == 3

    def test_mode_auto_routes_to_fullagent_directly(self, repo: Path) -> None:
        result, adapter = _run(repo, [l1_classify("fullagent", 0.95), fa_finish()])
        assert result.success
        assert result.mode_used == "fullagent"
        assert adapter.calls == 2  # routing + FA

    def test_trace_written_after_run(self, repo: Path, tmp_path: Path) -> None:
        store = TraceStore.for_repo(tmp_path)
        result, _ = _run(repo, [fa_finish("done")], mode="full", trace_store=store)
        records = store.all()
        store.close()
        assert len(records) == 1
        assert records[0].mode_used == "fullagent"
        assert records[0].final_success is True

    def test_trace_records_turns_used(self, repo: Path, tmp_path: Path) -> None:
        store = TraceStore.for_repo(tmp_path)
        _run(repo, [fa_write_file(), fa_finish()], mode="full", trace_store=store)
        records = store.all()
        store.close()
        assert records[0].turns_used >= 1

    def test_trace_records_error_cause(self, repo: Path, tmp_path: Path) -> None:
        """A failed run must persist *why* it failed, not just finish_reason=error.

        Without this the trace DB records an unexplained error and the only way
        to recover the cause is to monkey-patch the model adapter.
        """
        store = TraceStore.for_repo(tmp_path)
        # Empty script → FakeAdapter raises on the first call → graph errors out.
        result, _ = _run(repo, [], mode="full", trace_store=store)
        records = store.all()
        store.close()
        assert result.success is False
        assert records[0].finish_reason == "error"
        assert "script exhausted" in records[0].finish_summary

    def test_trace_records_summary_on_success(self, repo: Path, tmp_path: Path) -> None:
        store = TraceStore.for_repo(tmp_path)
        _run(repo, [fa_finish("patched the parser")], mode="full", trace_store=store)
        records = store.all()
        store.close()
        assert records[0].finish_summary == "patched the parser"
