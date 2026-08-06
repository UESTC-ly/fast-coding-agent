"""Conversation context tests: what a later turn is told about earlier ones."""

from __future__ import annotations

from qqcode.conversation import (
    MAX_FILES_LISTED,
    MAX_SUMMARY_CHARS,
    build_context,
)
from qqcode.memory.session import TurnRecord


def _turn(
    task: str = "do a thing",
    outcome: str = "accepted",
    summary: str = "",
    changed_files: tuple[str, ...] = (),
) -> TurnRecord:
    return TurnRecord(
        task=task, outcome=outcome, summary=summary, changed_files=changed_files
    )


class TestEmptyCases:
    def test_no_turns_yields_no_text(self) -> None:
        """The first turn of a session must look exactly like the old behavior."""
        ctx = build_context([])
        assert ctx.is_empty()
        assert ctx.text == ""
        assert ctx.turns_included == 0

    def test_only_interrupted_turns_yields_no_text(self) -> None:
        """Cancelled work was never endorsed; presenting it invites resuming it."""
        ctx = build_context([_turn(outcome="interrupted")])
        assert ctx.is_empty()


class TestContent:
    def test_request_and_outcome_appear(self) -> None:
        ctx = build_context([_turn(task="add a docstring to divide")])
        assert "add a docstring to divide" in ctx.text
        assert "applied" in ctx.text

    def test_summary_appears(self) -> None:
        ctx = build_context([_turn(summary="Added a Raises section")])
        assert "Added a Raises section" in ctx.text

    def test_changed_files_appear(self) -> None:
        ctx = build_context([_turn(changed_files=("calc.py", "util.py"))])
        assert "calc.py" in ctx.text
        assert "util.py" in ctx.text

    def test_rejection_is_recorded_distinctly(self) -> None:
        """'That approach was wrong' needs the wrong approach on record."""
        ctx = build_context([_turn(outcome="rejected", task="rewrite everything")])
        assert "rewrite everything" in ctx.text
        assert "declined" in ctx.text
        assert "not in the files" in ctx.text

    def test_failure_is_recorded(self) -> None:
        ctx = build_context([_turn(outcome="failed")])
        assert "failed" in ctx.text

    def test_accepted_says_the_change_is_present(self) -> None:
        """The agent must not re-apply a change that is already on disk."""
        ctx = build_context([_turn(outcome="accepted")])
        assert "present in the files" in ctx.text


class TestOrdering:
    def test_turns_are_chronological(self) -> None:
        ctx = build_context([_turn(task="first"), _turn(task="second")])
        assert ctx.text.index("first") < ctx.text.index("second")

    def test_interrupted_turns_are_skipped_but_others_kept(self) -> None:
        ctx = build_context(
            [_turn(task="kept one"), _turn(task="cancelled", outcome="interrupted")]
        )
        assert "kept one" in ctx.text
        assert "cancelled" not in ctx.text
        assert ctx.turns_included == 1


class TestBudgets:
    def test_turn_count_is_capped(self) -> None:
        turns = [_turn(task=f"task {i}") for i in range(20)]
        ctx = build_context(turns, max_turns=3)
        assert ctx.turns_included == 3
        assert "task 19" in ctx.text      # newest kept
        assert "task 0" not in ctx.text   # oldest dropped

    def test_oldest_turns_drop_first_under_a_char_budget(self) -> None:
        """Recent turns are what pronouns refer to, so they survive truncation."""
        turns = [_turn(task=f"task {i}", summary="x" * 200) for i in range(10)]
        ctx = build_context(turns, max_chars=600)
        assert "task 9" in ctx.text
        assert "task 0" not in ctx.text

    def test_char_budget_is_respected(self) -> None:
        turns = [_turn(task=f"task {i}", summary="x" * 100) for i in range(10)]
        ctx = build_context(turns, max_chars=500)
        assert len(ctx) <= 500 + len("## Earlier in this conversation\n\n")

    def test_long_summary_is_truncated(self) -> None:
        ctx = build_context([_turn(summary="y" * 5_000)])
        assert len(ctx) < 5_000
        assert "…" in ctx.text

    def test_many_files_collapse_to_a_count(self) -> None:
        files = tuple(f"f{i}.py" for i in range(MAX_FILES_LISTED + 7))
        ctx = build_context([_turn(changed_files=files)])
        assert "and 7 more" in ctx.text

    def test_a_single_huge_turn_cannot_blow_the_budget(self) -> None:
        """One turn touching hundreds of files must not swamp the prompt."""
        files = tuple(f"path/to/module_{i}.py" for i in range(500))
        ctx = build_context([_turn(changed_files=files, summary="z" * MAX_SUMMARY_CHARS)])
        assert len(ctx) < 1_500
