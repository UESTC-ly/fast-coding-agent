"""REPL tests: conversational semantics, driven through a scripted console.

No real terminal and no real model calls. `run_task` is patched at the point the
REPL imports it, so these tests exercise the loop's own behavior — turn
accounting, interruption, rejection, and the working-tree seeding that makes
turn N build on turn N-1.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from rich.console import Console

from qqcode.config import Config, ProviderConfig
from qqcode.models.protocol import CostLedger
from qqcode.orchestrator import RunResult
from qqcode.repl import Session, make_confirm, render_review, run_repl
from qqcode.review import ChangeReview, FileDiff


def _console(inputs: list[str]) -> Console:
    """A console whose `input()` replays a script and whose output is captured."""
    console = Console(file=io.StringIO(), force_terminal=False, width=100)
    pending = list(inputs)

    def fake_input(prompt: str = "", **kwargs: Any) -> str:
        if not pending:
            raise EOFError
        return pending.pop(0)

    console.input = fake_input  # type: ignore[method-assign]
    return console


def _config() -> Config:
    return Config(
        anthropic=ProviderConfig(api_key="fake", base_url=None),
        openai=None,
        default_provider="anthropic",
    )


def _result(
    *, success: bool = True, rejected: bool = False, changed: tuple[str, ...] = ("a.py",)
) -> RunResult:
    return RunResult(
        success=success,
        mode_used="fastpath",
        finish_reason="rejected" if rejected else "fastpath_ok",
        changed_files=frozenset(changed),
        reasoning="did the thing",
        ledger=CostLedger(),
        rejected=rejected,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "a.py").write_text("x = 1\n")
    return tmp_path


class TestLoopControl:
    def test_exit_command_ends_session(self, repo: Path) -> None:
        console = _console(["/exit"])
        session = run_repl(repo, _config(), console=console)
        assert session.turns == []

    def test_eof_ends_session(self, repo: Path) -> None:
        session = run_repl(repo, _config(), console=_console([]))
        assert session.turns == []

    def test_blank_input_is_skipped(self, repo: Path) -> None:
        console = _console(["", "   ", "/exit"])
        with patch("qqcode.repl.run_task") as rt:
            session = run_repl(repo, _config(), console=console)
        rt.assert_not_called()
        assert session.turns == []

    def test_multiple_turns_are_recorded(self, repo: Path) -> None:
        console = _console(["first task", "second task", "/exit"])
        with patch("qqcode.repl.run_task", return_value=_result()):
            session = run_repl(repo, _config(), console=console)

        assert [t.task for t in session.turns] == ["first task", "second task"]
        assert session.accepted_count == 2


class TestTurnOutcomes:
    def test_rejected_turn_is_not_counted_as_accepted(self, repo: Path) -> None:
        console = _console(["do it", "/exit"])
        with patch("qqcode.repl.run_task", return_value=_result(success=False, rejected=True)):
            session = run_repl(repo, _config(), console=console)

        assert session.turns[0].outcome == "rejected"
        assert session.accepted_count == 0

    def test_failed_turn_is_recorded(self, repo: Path) -> None:
        console = _console(["do it", "/exit"])
        with patch("qqcode.repl.run_task", return_value=_result(success=False)):
            session = run_repl(repo, _config(), console=console)

        assert session.turns[0].outcome == "failed"

    def test_interrupt_cancels_turn_but_not_session(self, repo: Path) -> None:
        """Ctrl-C must end the turn and return to the prompt."""
        console = _console(["long task", "next task", "/exit"])
        calls: list[str] = []

        def flaky(task: str, *a: Any, **kw: Any) -> RunResult:
            calls.append(task)
            if task == "long task":
                raise KeyboardInterrupt
            return _result()

        with patch("qqcode.repl.run_task", side_effect=flaky):
            session = run_repl(repo, _config(), console=console)

        assert calls == ["long task", "next task"]
        assert session.turns[0].outcome == "interrupted"
        assert session.turns[1].outcome == "accepted"

    def test_exception_is_surfaced_not_swallowed(self, repo: Path) -> None:
        console = _console(["boom", "/exit"])
        with patch("qqcode.repl.run_task", side_effect=RuntimeError("upstream 400")):
            session = run_repl(repo, _config(), console=console)

        assert session.turns[0].outcome == "failed"
        output = console.file.getvalue()  # type: ignore[attr-defined]
        assert "RuntimeError" in output
        assert "upstream 400" in output


class TestConversationalWiring:
    def test_turns_are_seeded_from_the_working_tree(self, repo: Path) -> None:
        """The invariant that makes turn N build on turn N-1."""
        console = _console(["task", "/exit"])
        with patch("qqcode.repl.run_task", return_value=_result()) as rt:
            run_repl(repo, _config(), console=console)

        assert rt.call_args.kwargs["seed"] == "worktree"

    def test_changes_are_applied_not_dry_run(self, repo: Path) -> None:
        console = _console(["task", "/exit"])
        with patch("qqcode.repl.run_task", return_value=_result()) as rt:
            run_repl(repo, _config(), console=console)

        assert rt.call_args.kwargs["dry_run"] is False

    def test_a_confirm_callback_is_supplied(self, repo: Path) -> None:
        """Without a harness, the human is the verdict source."""
        console = _console(["task", "/exit"])
        with patch("qqcode.repl.run_task", return_value=_result()) as rt:
            run_repl(repo, _config(), console=console)

        assert callable(rt.call_args.kwargs["confirm"])


class TestConfirmPrompt:
    def _review(self, diffs: tuple[FileDiff, ...] = ()) -> ChangeReview:
        return ChangeReview(
            task="t", mode_used="fastpath", reasoning="because", diffs=diffs
        )

    def _one_diff(self) -> tuple[FileDiff, ...]:
        return (FileDiff(path="a.py", status="modified", diff_text="--- a/a.py\n+++ b/a.py\n+x\n"),)

    @pytest.mark.parametrize("answer", ["y", "Y", "yes", "  yes  "])
    def test_affirmative_answers_accept(self, answer: str) -> None:
        confirm = make_confirm(_console([answer]))
        assert confirm(self._review(self._one_diff())) is True

    @pytest.mark.parametrize("answer", ["n", "no", "", "maybe", "q"])
    def test_everything_else_rejects(self, answer: str) -> None:
        """Anything but an explicit yes discards. Silence is not consent."""
        confirm = make_confirm(_console([answer]))
        assert confirm(self._review(self._one_diff())) is False

    def test_empty_change_is_rejected_without_asking(self) -> None:
        console = _console([])  # any input attempt would raise EOFError
        confirm = make_confirm(console)
        assert confirm(self._review()) is False


class TestRenderReview:
    def test_renders_paths_and_diff(self) -> None:
        console = _console([])
        review = ChangeReview(
            task="t",
            mode_used="fastpath",
            reasoning="tidy up",
            diffs=(
                FileDiff(
                    path="a.py", status="modified",
                    diff_text="--- a/a.py\n+++ b/a.py\n+x = 2\n",
                ),
                FileDiff(path="new.py", status="added", diff_text="+++ b/new.py\n+y = 3\n"),
            ),
        )
        render_review(review, console)
        out = console.file.getvalue()  # type: ignore[attr-defined]

        assert "tidy up" in out
        assert "a.py" in out
        assert "new.py" in out
        assert "2 file(s) changed" in out

    def test_flags_a_success_with_no_changes(self) -> None:
        console = _console([])
        render_review(ChangeReview(task="t", mode_used="fastpath", reasoning=""), console)
        assert "changed no files" in console.file.getvalue()  # type: ignore[attr-defined]


class TestSessionAccounting:
    def test_counts_and_totals(self) -> None:
        from qqcode.repl import TurnRecord

        session = Session(repo=Path("."))
        session.turns = [
            TurnRecord(task="a", outcome="accepted", tokens=100),
            TurnRecord(task="b", outcome="rejected", tokens=50),
            TurnRecord(task="c", outcome="accepted", tokens=25),
        ]
        assert session.accepted_count == 2
        assert session.total_tokens == 175

    def test_provider_and_model_reach_run_task(self, repo: Path) -> None:
        """--provider/--model must not be silently dropped in chat mode."""
        console = _console(["task", "/exit"])
        with patch("qqcode.repl.run_task", return_value=_result()) as rt:
            run_repl(repo, _config(), console=console,
                     provider="openai", model="gpt-4o-mini")

        assert rt.call_args.kwargs["provider"] == "openai"
        assert rt.call_args.kwargs["model"] == "gpt-4o-mini"

    def test_model_defaults_to_none(self, repo: Path) -> None:
        console = _console(["task", "/exit"])
        with patch("qqcode.repl.run_task", return_value=_result()) as rt:
            run_repl(repo, _config(), console=console)

        assert rt.call_args.kwargs["model"] is None
        assert rt.call_args.kwargs["provider"] is None
