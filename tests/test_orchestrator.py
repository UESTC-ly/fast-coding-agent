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
        """L1 says fullagent, so FastPath is never attempted.

        The task must not match any built-in skill: "test task" matches
        `pytest-patterns`, whose FAST hint fires L0 and never reaches L1, so this
        test used to pass while exercising a different path than its name claims.
        """
        result, adapter = _run(
            repo, [l1_classify("fullagent", 0.95), fa_finish()],
            task="adjust the greeting wording",
        )
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

    def test_trace_records_the_recovered_prefetch_count(
        self, repo: Path, tmp_path: Path
    ) -> None:
        """The L0 → L1 recovery must be observable in the trace.

        "test task" matches the built-in `pytest-patterns` skill, so L0 decides
        FASTPATH with no hint, and the text names no file — the exact branch that
        declined 4/4 times in the measured baseline. Without this assertion the
        count can be computed and never recorded, which is the defect shape this
        repo keeps hitting.
        """
        store = TraceStore.for_repo(tmp_path)
        _run(repo, [l1_classify("fastpath", 0.9), patch_declined()],
             mode="auto", trace_store=store)
        records = store.all()
        store.close()

        assert records[0].prefetch_hint_count == 1, (
            "L1 named main.py, so the recovered count must reach the trace"
        )
        assert records[0].files_hint_count == 0, (
            "the recovery must not become condition 3's contract"
        )

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


# ---------------------------------------------------------------------------
# confirm= : the human verdict source
# ---------------------------------------------------------------------------

def _run_confirm(
    repo: Path,
    script: list[Completion],
    confirm: Any,
    *,
    mode: str = "fast",
    task: str = "test task",
) -> Any:
    """Run with a real finalize (dry_run=False) so acceptance is observable."""
    from qqcode.config import Config, ProviderConfig
    from qqcode.orchestrator import run_task

    client, _ = make_client(script)
    config = Config(anthropic=ProviderConfig(api_key="fake", base_url=None),
                    openai=None, default_provider="anthropic")
    with patch("qqcode.orchestrator.build_client",
               return_value=(client, client._ledger)):  # noqa: SLF001
        return run_task(task=task, repo=repo, config=config, mode=mode,
                        dry_run=False, confirm=confirm)


class TestConfirmCallback:
    """The third gate condition gains a human variant without losing the others."""

    def test_none_confirm_finalizes_as_before(self, repo: Path) -> None:
        """confirm=None must behave exactly like the batch path."""
        result = _run_confirm(repo, [patch_ok()], None)
        assert result.success
        assert not result.rejected
        assert '"""Module."""' in (repo / "main.py").read_text()

    def test_approval_finalizes(self, repo: Path) -> None:
        seen: list[Any] = []

        def approve(review: Any) -> bool:
            seen.append(review)
            return True

        result = _run_confirm(repo, [patch_ok()], approve)
        assert result.success
        assert not result.rejected
        assert '"""Module."""' in (repo / "main.py").read_text()
        assert len(seen) == 1
        assert seen[0].changed_files == ("main.py",)

    def test_rejection_does_not_touch_the_repo(self, repo: Path) -> None:
        before = (repo / "main.py").read_text()
        result = _run_confirm(repo, [patch_ok()], lambda _r: False)

        assert not result.success
        assert result.rejected
        assert result.finish_reason == "rejected"
        assert (repo / "main.py").read_text() == before

    def test_review_carries_a_readable_diff(self, repo: Path) -> None:
        captured: list[Any] = []
        _run_confirm(repo, [patch_ok()], lambda r: captured.append(r) or False)

        review = captured[0]
        assert review.task == "test task"
        assert review.mode_used == "fastpath"
        assert not review.is_empty()
        diff = review.diffs[0]
        assert diff.status == "modified"
        assert "+++ b/main.py" in diff.diff_text
        assert '+"""Module."""' in diff.diff_text

    def test_rejection_on_fullagent_path(self, repo: Path) -> None:
        before = (repo / "main.py").read_text()
        result = _run_confirm(
            repo,
            [fa_write_file("out.txt"), fa_finish("wrote it")],
            lambda _r: False,
            mode="full",
        )
        assert result.rejected
        assert result.finish_reason == "rejected"
        assert not (repo / "out.txt").exists()
        assert (repo / "main.py").read_text() == before

    def test_approval_on_fullagent_path(self, repo: Path) -> None:
        result = _run_confirm(
            repo,
            [fa_write_file("out.txt"), fa_finish("wrote it")],
            lambda _r: True,
            mode="full",
        )
        assert result.success
        assert (repo / "out.txt").read_text() == "result\n"

    def test_objective_failure_never_reaches_the_user(self, repo: Path) -> None:
        """A change that fails an objective condition is not offered for review."""
        asked: list[Any] = []
        result = _run_confirm(
            repo, [patch_declined()], lambda r: asked.append(r) or True
        )
        assert not result.success
        assert not result.rejected          # declined, not rejected
        assert asked == []                  # the human was never consulted


class TestSeedAcrossTurns:
    """Two consecutive turns, the real multi-turn invariant.

    Turn 2 touches a different file than turn 1 and never commits. With
    seed="worktree" turn 1's output survives; with seed="head" the second
    finalize would clobber it, which the dirty guard now refuses outright.
    """

    def _git_repo(self, tmp_path: Path) -> Path:
        import subprocess
        root = tmp_path / "r"
        root.mkdir()
        (root / "app.py").write_text("def greet(): pass\n")
        env = {
            "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
            "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
        }
        for cmd in (["git", "init", "-q"], ["git", "add", "."], ["git", "commit", "-qm", "i"]):
            subprocess.run(cmd, cwd=root, check=True, capture_output=True, env=env)
        return root

    def _turn(self, repo: Path, path: str, content: str, seed: str) -> Any:
        from qqcode.config import Config, ProviderConfig
        from qqcode.orchestrator import run_task

        completion = Completion(
            content=[ToolUseContent(
                id="p", name="submit_patch",
                input={"reasoning": "x", "files": [{"path": path, "content": content}]},
            )],
            stop_reason="tool_use",
            usage=Usage(input_tokens=10, output_tokens=5), raw={},
        )
        client, _ = make_client([completion])
        config = Config(anthropic=ProviderConfig(api_key="fake", base_url=None),
                        openai=None, default_provider="anthropic")
        with patch("qqcode.orchestrator.build_client",
                   return_value=(client, client._ledger)):  # noqa: SLF001
            return run_task(task=f"edit {path}", repo=repo, config=config, mode="fast",
                            dry_run=False, seed=seed)  # type: ignore[arg-type]

    def test_worktree_seed_preserves_the_previous_turn(self, tmp_path: Path) -> None:
        repo = self._git_repo(tmp_path)
        assert self._turn(repo, "app.py", 'def greet(): return "hi"\n', "worktree").success
        assert self._turn(repo, "util.py", "def helper(): return 42\n", "worktree").success

        # Turn 1's edit survived turn 2, which never touched app.py.
        assert (repo / "app.py").read_text() == 'def greet(): return "hi"\n'
        assert (repo / "util.py").read_text() == "def helper(): return 42\n"

    def test_head_seed_refuses_the_second_turn(self, tmp_path: Path) -> None:
        """The dirty guard catches what would otherwise be a silent revert."""
        from qqcode.workspace.worktree import DirtyWorktreeError

        repo = self._git_repo(tmp_path)
        assert self._turn(repo, "app.py", 'def greet(): return "hi"\n', "head").success

        with pytest.raises(DirtyWorktreeError):
            self._turn(repo, "util.py", "def helper(): return 42\n", "head")

        # Turn 1's work is intact precisely because the guard fired.
        assert (repo / "app.py").read_text() == 'def greet(): return "hi"\n'


class TestDefaultModelWiring:
    """config.default_model must actually reach the client, not just parse.

    Config-level tests only prove the value was read from the env file. Without
    these, dropping the orchestrator's fallback is invisible to the suite.
    """

    def _config(self, default_model: str | None) -> Any:
        from qqcode.config import Config, ProviderConfig
        return Config(
            anthropic=ProviderConfig(api_key="fake", base_url=None),
            openai=None,
            default_provider="anthropic",
            default_model=default_model,
        )

    def _captured_tiers(self, config: Any, model: str = "") -> Any:
        """Run one task, returning the tier_models build_client was given."""
        from qqcode.orchestrator import run_task

        client, _ = make_client([patch_ok()])
        seen: dict[str, Any] = {}

        def fake_build(cfg: Any, **kw: Any) -> Any:
            seen["tier_models"] = kw.get("tier_models")
            return client, client._ledger  # noqa: SLF001

        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "main.py").write_text("def main(): pass\n")
            with patch("qqcode.orchestrator.build_client", side_effect=fake_build):
                run_task(task="t", repo=repo, config=config, mode="fast",
                         dry_run=True, model=model or None)
        return seen["tier_models"]

    def test_default_model_pins_every_tier(self) -> None:
        tiers = self._captured_tiers(self._config("proxy-model-id"))
        assert tiers is not None
        assert set(tiers.values()) == {"proxy-model-id"}

    def test_explicit_model_overrides_the_default(self) -> None:
        """--model is the more specific signal; benchmarking depends on it."""
        tiers = self._captured_tiers(self._config("proxy-model-id"), model="pinned-id")
        assert set(tiers.values()) == {"pinned-id"}

    def test_no_default_leaves_adapter_defaults_alone(self) -> None:
        assert self._captured_tiers(self._config(None)) is None
