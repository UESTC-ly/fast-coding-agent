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
from qqcode.memory.session import SessionRecord
from qqcode.models.protocol import CostLedger
from qqcode.orchestrator import RunResult
from qqcode.repl import make_confirm, render_review, run_repl
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
        from qqcode.memory.session import TurnRecord

        session = SessionRecord(repo=".")
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


class TestPersistence:
    """Turns are saved as they complete, not at exit."""

    def _store(self, tmp_path: Path) -> Any:
        from qqcode.memory.session import SessionStore
        return SessionStore(tmp_path / "s.db")

    def test_turns_are_persisted(self, repo: Path, tmp_path: Path) -> None:
        store = self._store(tmp_path)
        console = _console(["one", "two", "/exit"])
        with patch("qqcode.repl.run_task", return_value=_result()):
            session = run_repl(repo, _config(), console=console, session_store=store)

        reloaded = store.load(session.id)
        assert reloaded is not None
        assert [t.task for t in reloaded.turns] == ["one", "two"]
        store.close()

    def test_a_turn_survives_a_crash_in_a_later_turn(self, repo: Path, tmp_path: Path) -> None:
        """Saving per turn is what makes this true; saving at exit would not."""
        store = self._store(tmp_path)
        console = _console(["good", "bad", "/exit"])

        def flaky(task: str, *a: Any, **kw: Any) -> RunResult:
            if task == "bad":
                raise RuntimeError("upstream exploded")
            return _result()

        with patch("qqcode.repl.run_task", side_effect=flaky):
            session = run_repl(repo, _config(), console=console, session_store=store)

        reloaded = store.load(session.id)
        assert reloaded is not None
        assert [(t.task, t.outcome) for t in reloaded.turns] == [
            ("good", "accepted"), ("bad", "failed"),
        ]
        store.close()

    def test_interrupted_turns_are_persisted(self, repo: Path, tmp_path: Path) -> None:
        store = self._store(tmp_path)
        console = _console(["slow", "/exit"])
        with patch("qqcode.repl.run_task", side_effect=KeyboardInterrupt):
            session = run_repl(repo, _config(), console=console, session_store=store)

        reloaded = store.load(session.id)
        assert reloaded is not None
        assert reloaded.turns[0].outcome == "interrupted"
        store.close()

    def test_no_store_still_works(self, repo: Path) -> None:
        """session_store=None keeps the session in memory rather than failing."""
        console = _console(["task", "/exit"])
        with patch("qqcode.repl.run_task", return_value=_result()):
            session = run_repl(repo, _config(), console=console)
        assert len(session.turns) == 1


class TestResume:
    def test_resumed_session_appends_rather_than_restarting(
        self, repo: Path, tmp_path: Path
    ) -> None:
        from qqcode.memory.session import SessionStore

        store = SessionStore(tmp_path / "s.db")
        with patch("qqcode.repl.run_task", return_value=_result()):
            first = run_repl(repo, _config(), console=_console(["one", "/exit"]),
                             session_store=store)
            second = run_repl(repo, _config(), console=_console(["two", "/exit"]),
                              session_store=store, resume=first)

        assert second.id == first.id
        assert [t.task for t in second.turns] == ["one", "two"]
        assert store.count() == 1          # continued, not duplicated
        store.close()

    def test_resume_keeps_the_original_base_commit(self, repo: Path, tmp_path: Path) -> None:
        """The undo anchor must still point at where the conversation started."""
        from qqcode.memory.session import SessionRecord, SessionStore

        store = SessionStore(tmp_path / "s.db")
        prior = SessionRecord(repo=str(repo.resolve()), base_commit="f" * 40)
        store.save(prior)

        with patch("qqcode.repl.run_task", return_value=_result()):
            resumed = run_repl(repo, _config(), console=_console(["task", "/exit"]),
                               session_store=store, resume=prior)

        assert resumed.base_commit == "f" * 40
        store.close()

    def test_resume_header_shows_prior_turns(self, repo: Path, tmp_path: Path) -> None:
        from qqcode.memory.session import TurnRecord

        prior = SessionRecord(repo=str(repo.resolve()))
        prior.turns.append(TurnRecord(task="earlier work", outcome="accepted"))
        console = _console(["/exit"])
        run_repl(repo, _config(), console=console, resume=prior)

        out = console.file.getvalue()  # type: ignore[attr-defined]
        assert "resumed session" in out
        assert "earlier work" in out

    def test_fresh_session_records_the_current_commit(self, tmp_path: Path) -> None:
        import subprocess

        root = tmp_path / "g"
        root.mkdir()
        (root / "f.py").write_text("x = 1\n")
        env = {
            "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
            "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
        }
        for cmd in (["git", "init", "-q"], ["git", "add", "."], ["git", "commit", "-qm", "i"]):
            subprocess.run(cmd, cwd=root, check=True, capture_output=True, env=env)

        session = run_repl(root, _config(), console=_console(["/exit"]))
        assert len(session.base_commit) == 40

    def test_non_git_repo_yields_empty_base_commit(self, repo: Path) -> None:
        session = run_repl(repo, _config(), console=_console(["/exit"]))
        assert session.base_commit == ""


class TestUndoCommand:
    """/undo reverses the last applied turn, through the real REPL loop."""

    def _applied(self, repo: Path, path: str = "a.py") -> Any:
        """A run_task stand-in that writes to the repo like finalize would."""
        def fake(task: str, *a: Any, **kw: Any) -> RunResult:
            confirm = kw["confirm"]
            before = (repo / path).read_text() if (repo / path).is_file() else None
            after = f"# {task}\n"
            review = ChangeReview(
                task=task, mode_used="fastpath", reasoning="r",
                diffs=(FileDiff(
                    path=path, status="modified", diff_text=f"+{after}",
                    before_exists=before is not None, before_text=before,
                    after_exists=True, after_text=after,
                ),),
            )
            if not confirm(review):
                return _result(success=False, rejected=True, changed=(path,))
            (repo / path).write_text(after)      # stands in for finalize
            return _result(changed=(path,))
        return fake

    def test_undo_reverts_the_last_turn(self, repo: Path) -> None:
        console = _console(["do it", "y", "/undo", "/exit"])
        with patch("qqcode.repl.run_task", side_effect=self._applied(repo)):
            run_repl(repo, _config(), console=console)

        assert (repo / "a.py").read_text() == "x = 1\n"      # back to the fixture
        assert "Reverted" in console.file.getvalue()  # type: ignore[attr-defined]

    def test_undo_with_no_history_says_so(self, repo: Path) -> None:
        console = _console(["/undo", "/exit"])
        run_repl(repo, _config(), console=console)
        assert "Nothing to undo" in console.file.getvalue()  # type: ignore[attr-defined]

    def test_undo_is_recorded_as_a_turn(self, repo: Path) -> None:
        console = _console(["do it", "y", "/undo", "/exit"])
        with patch("qqcode.repl.run_task", side_effect=self._applied(repo)):
            session = run_repl(repo, _config(), console=console)

        assert [t.outcome for t in session.turns] == ["accepted", "undone"]

    def test_only_applied_turns_are_undoable(self, repo: Path) -> None:
        """A rejected turn never reached the repo, so there is nothing to undo."""
        console = _console(["do it", "n", "/undo", "/exit"])
        with patch("qqcode.repl.run_task", side_effect=self._applied(repo)):
            run_repl(repo, _config(), console=console)

        assert "Nothing to undo" in console.file.getvalue()  # type: ignore[attr-defined]
        assert (repo / "a.py").read_text() == "x = 1\n"

    def test_undo_only_reaches_back_one_turn(self, repo: Path) -> None:
        console = _console(["first", "y", "second", "y", "/undo", "/exit"])
        with patch("qqcode.repl.run_task", side_effect=self._applied(repo)):
            run_repl(repo, _config(), console=console)

        # Reverses "second", leaving "first" in place.
        assert (repo / "a.py").read_text() == "# first\n"

    def test_repeated_undo_walks_back_the_stack(self, repo: Path) -> None:
        console = _console(["first", "y", "second", "y", "/undo", "/undo", "/exit"])
        with patch("qqcode.repl.run_task", side_effect=self._applied(repo)):
            run_repl(repo, _config(), console=console)

        assert (repo / "a.py").read_text() == "x = 1\n"

    def test_undo_asks_before_discarding_a_later_edit(self, repo: Path) -> None:
        """A hand edit after the turn must not vanish silently."""
        console = _console(["do it", "y", "/undo", "n", "/exit"])
        applied = self._applied(repo)

        def then_edit(task: str, *a: Any, **kw: Any) -> RunResult:
            out = applied(task, *a, **kw)
            (repo / "a.py").write_text("hand-edited afterwards\n")
            return out

        with patch("qqcode.repl.run_task", side_effect=then_edit):
            run_repl(repo, _config(), console=console)

        out = console.file.getvalue()  # type: ignore[attr-defined]
        assert "Changed since that turn" in out
        assert (repo / "a.py").read_text() == "hand-edited afterwards\n"

    def test_confirmed_conflict_undo_proceeds(self, repo: Path) -> None:
        console = _console(["do it", "y", "/undo", "y", "/exit"])
        applied = self._applied(repo)

        def then_edit(task: str, *a: Any, **kw: Any) -> RunResult:
            out = applied(task, *a, **kw)
            (repo / "a.py").write_text("hand-edited afterwards\n")
            return out

        with patch("qqcode.repl.run_task", side_effect=then_edit):
            run_repl(repo, _config(), console=console)

        assert (repo / "a.py").read_text() == "x = 1\n"
