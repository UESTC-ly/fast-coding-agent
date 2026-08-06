"""Undo tests: reversing one applied turn, and refusing when that is unsafe."""

from __future__ import annotations

from pathlib import Path

import pytest

from qqcode.review import FileDiff, build_review
from qqcode.undo import (
    UndoConflictError,
    UndoSnapshot,
    apply_undo,
    plan_undo,
)


def _diff(
    path: str,
    *,
    before: str | None,
    after: str | None,
    truncated: bool = False,
) -> FileDiff:
    """A FileDiff carrying the two sides undo actually reads."""
    status = "modified"
    if before is None:
        status = "added"
    elif after is None:
        status = "deleted"
    return FileDiff(
        path=path,
        status=status,
        diff_text="(irrelevant to undo)",
        truncated=truncated,
        before_exists=before is not None,
        before_text=before,
        after_exists=after is not None,
        after_text=after,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return tmp_path


class TestRestore:
    def test_modified_file_is_restored(self, repo: Path) -> None:
        (repo / "a.py").write_text("new\n")
        snap = UndoSnapshot(task="t", files=(_diff("a.py", before="old\n", after="new\n"),))

        apply_undo(snap, repo)
        assert (repo / "a.py").read_text() == "old\n"

    def test_created_file_is_deleted(self, repo: Path) -> None:
        (repo / "new.py").write_text("fresh\n")
        snap = UndoSnapshot(task="t", files=(_diff("new.py", before=None, after="fresh\n"),))

        apply_undo(snap, repo)
        assert not (repo / "new.py").exists()

    def test_deleted_file_is_recreated(self, repo: Path) -> None:
        snap = UndoSnapshot(task="t", files=(_diff("gone.py", before="was here\n", after=None),))

        apply_undo(snap, repo)
        assert (repo / "gone.py").read_text() == "was here\n"

    def test_nested_paths_are_recreated_with_parents(self, repo: Path) -> None:
        snap = UndoSnapshot(
            task="t", files=(_diff("pkg/sub/m.py", before="x = 1\n", after=None),)
        )
        apply_undo(snap, repo)
        assert (repo / "pkg" / "sub" / "m.py").read_text() == "x = 1\n"

    def test_several_files_in_one_turn(self, repo: Path) -> None:
        (repo / "a.py").write_text("new a\n")
        (repo / "b.py").write_text("created\n")
        snap = UndoSnapshot(
            task="t",
            files=(
                _diff("a.py", before="old a\n", after="new a\n"),
                _diff("b.py", before=None, after="created\n"),
            ),
        )

        plan = apply_undo(snap, repo)
        assert (repo / "a.py").read_text() == "old a\n"
        assert not (repo / "b.py").exists()
        assert plan.to_restore == ("a.py",)
        assert plan.to_delete == ("b.py",)


class TestPlanning:
    def test_plan_touches_nothing(self, repo: Path) -> None:
        """plan_undo is a dry run; it must not write."""
        (repo / "a.py").write_text("new\n")
        snap = UndoSnapshot(task="t", files=(_diff("a.py", before="old\n", after="new\n"),))

        plan_undo(snap, repo)
        assert (repo / "a.py").read_text() == "new\n"

    def test_clean_plan_when_untouched(self, repo: Path) -> None:
        (repo / "a.py").write_text("new\n")
        snap = UndoSnapshot(task="t", files=(_diff("a.py", before="old\n", after="new\n"),))

        assert plan_undo(snap, repo).is_clean

    def test_binary_file_is_reported_unrestorable(self, repo: Path) -> None:
        """A file whose prior bytes were never captured must not be 'restored'."""
        (repo / "img.png").write_bytes(b"\x89PNG\x00\x01")
        binary = FileDiff(
            path="img.png", status="modified", diff_text="",
            before_exists=True, before_text=None,      # unreadable as text
            after_exists=True, after_text=None,
        )
        snap = UndoSnapshot(task="t", files=(binary,))

        plan = plan_undo(snap, repo)
        assert plan.unrestorable == ("img.png",)
        assert not plan.is_clean
        assert not snap.is_restorable

    def test_unrestorable_file_is_left_alone(self, repo: Path) -> None:
        (repo / "img.png").write_bytes(b"\x89PNG\x00\x01")
        binary = FileDiff(
            path="img.png", status="modified", diff_text="",
            before_exists=True, before_text=None, after_exists=True, after_text=None,
        )
        apply_undo(UndoSnapshot(task="t", files=(binary,)), repo)
        assert (repo / "img.png").read_bytes() == b"\x89PNG\x00\x01"


class TestConflicts:
    """Undo does not merge. A later edit must not be silently discarded."""

    def test_edited_since_raises(self, repo: Path) -> None:
        (repo / "a.py").write_text("edited by hand afterwards\n")
        snap = UndoSnapshot(task="t", files=(_diff("a.py", before="old\n", after="new\n"),))

        with pytest.raises(UndoConflictError, match="a.py"):
            apply_undo(snap, repo)

    def test_refused_undo_writes_nothing(self, repo: Path) -> None:
        """A conflict aborts before any file is touched, not halfway through."""
        (repo / "clean.py").write_text("turn output\n")
        (repo / "dirty.py").write_text("edited afterwards\n")
        snap = UndoSnapshot(
            task="t",
            files=(
                _diff("clean.py", before="orig\n", after="turn output\n"),
                _diff("dirty.py", before="orig\n", after="turn output\n"),
            ),
        )

        with pytest.raises(UndoConflictError):
            apply_undo(snap, repo)

        assert (repo / "clean.py").read_text() == "turn output\n"
        assert (repo / "dirty.py").read_text() == "edited afterwards\n"

    def test_force_overrides_the_conflict(self, repo: Path) -> None:
        (repo / "a.py").write_text("edited afterwards\n")
        snap = UndoSnapshot(task="t", files=(_diff("a.py", before="old\n", after="new\n"),))

        apply_undo(snap, repo, force=True)
        assert (repo / "a.py").read_text() == "old\n"

    def test_missing_file_is_a_conflict(self, repo: Path) -> None:
        """The turn wrote it and it is gone now — something else happened."""
        snap = UndoSnapshot(task="t", files=(_diff("a.py", before="old\n", after="new\n"),))
        assert plan_undo(snap, repo).conflicts == ("a.py",)

    def test_recreated_deletion_is_a_conflict(self, repo: Path) -> None:
        """The turn deleted it; finding a file means someone put it back."""
        (repo / "gone.py").write_text("resurrected\n")
        snap = UndoSnapshot(task="t", files=(_diff("gone.py", before="was\n", after=None),))
        assert plan_undo(snap, repo).conflicts == ("gone.py",)

    def test_truncated_diff_does_not_block_undo(self, repo: Path) -> None:
        """Undo compares captured content, so a truncated *display* diff is fine."""
        (repo / "big.py").write_text("new\n")
        snap = UndoSnapshot(
            task="t",
            files=(_diff("big.py", before="old\n", after="new\n", truncated=True),),
        )
        assert plan_undo(snap, repo).is_clean
        apply_undo(snap, repo)
        assert (repo / "big.py").read_text() == "old\n"


class TestReviewCapturesBothSides:
    """build_review must record what undo depends on."""

    def _pair(self, tmp_path: Path) -> tuple[Path, Path]:
        source, shadow = tmp_path / "src", tmp_path / "shadow"
        source.mkdir()
        shadow.mkdir()
        return source, shadow

    def test_before_and_after_are_captured(self, tmp_path: Path) -> None:
        source, shadow = self._pair(tmp_path)
        (source / "a.py").write_text("original\n")
        (shadow / "a.py").write_text("changed\n")

        review = build_review(
            task="t", mode_used="fastpath", reasoning="r",
            changed_files=frozenset({"a.py"}), source=source, shadow=shadow,
        )
        d = review.diffs[0]
        assert d.before_exists and d.before_text == "original\n"
        assert d.after_exists and d.after_text == "changed\n"
        assert d.before_is_restorable

    def test_added_file_has_no_before(self, tmp_path: Path) -> None:
        source, shadow = self._pair(tmp_path)
        (shadow / "new.py").write_text("fresh\n")

        review = build_review(
            task="t", mode_used="fastpath", reasoning="r",
            changed_files=frozenset({"new.py"}), source=source, shadow=shadow,
        )
        d = review.diffs[0]
        assert d.status == "added"
        assert not d.before_exists
        assert d.before_is_restorable        # deleting it is a faithful reversal

    def test_round_trip_through_undo(self, tmp_path: Path) -> None:
        """The real path: review a change, apply it, then reverse it exactly."""
        source, shadow = self._pair(tmp_path)
        (source / "a.py").write_text("original\n")
        (shadow / "a.py").write_text("changed\n")

        review = build_review(
            task="t", mode_used="fastpath", reasoning="r",
            changed_files=frozenset({"a.py"}), source=source, shadow=shadow,
        )
        # Simulate finalize writing the shadow into the source.
        (source / "a.py").write_text("changed\n")

        apply_undo(UndoSnapshot(task="t", files=review.diffs), source)
        assert (source / "a.py").read_text() == "original\n"
