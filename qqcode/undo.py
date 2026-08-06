"""Per-turn undo: restoring the files one accepted turn changed.

Why not git. The obvious undo is `git checkout <base_commit> -- .`, but that
fails on the case that matters most: a conversation run against a repository
with uncommitted work. finalize() writes the working tree without committing,
so there is no commit to return to, and checking out the session's base commit
would also discard everything the user did by hand before the session started.

So undo is snapshot-based. When a turn is accepted, `build_review` has already
read every changed file on both sides — the source's prior content to produce
the diff, and the shadow's new content. Keeping both makes undo exact, scoped
to a single turn, and independent of version control.

What this deliberately does not do: it does not merge. If a file changed after
the turn was applied, restoring it would discard that later edit, so `plan_undo`
reports the conflict and `apply_undo` refuses unless the caller passes force.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from qqcode.review import FileDiff


@dataclass(frozen=True)
class UndoSnapshot:
    """The captured state of one accepted turn, enough to reverse it."""

    task: str
    files: tuple[FileDiff, ...] = ()

    @property
    def is_restorable(self) -> bool:
        """Whether every file in this turn can be put back faithfully."""
        return all(f.before_is_restorable for f in self.files)


@dataclass(frozen=True)
class UndoPlan:
    """What undoing a turn would do, computed before anything is written."""

    task: str
    to_restore: tuple[str, ...] = ()   # files whose prior content is rewritten
    to_delete: tuple[str, ...] = ()    # files the turn created
    conflicts: tuple[str, ...] = field(default_factory=tuple)
    unrestorable: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_empty(self) -> bool:
        return not self.to_restore and not self.to_delete

    @property
    def is_clean(self) -> bool:
        """Whether undo can proceed without discarding anything unexpected."""
        return not self.conflicts and not self.unrestorable


class UndoConflictError(RuntimeError):
    """A file changed after the turn was applied; undoing would discard that."""

    def __init__(self, plan: UndoPlan):
        self.plan = plan
        super().__init__(
            f"{', '.join(plan.conflicts)} changed after this turn was applied; "
            f"undoing would discard those edits"
        )


def plan_undo(snapshot: UndoSnapshot, repo: Path) -> UndoPlan:
    """Work out what undoing this turn would change, without touching disk.

    Conflicts come from comparing what is on disk now against what the turn
    left behind. A file that no longer matches was edited since — by a later
    turn or by hand — and restoring it would silently throw that away.
    """
    to_restore: list[str] = []
    to_delete: list[str] = []
    conflicts: list[str] = []
    unrestorable: list[str] = []

    for f in snapshot.files:
        if not f.before_is_restorable:
            unrestorable.append(f.path)
            continue

        if _changed_since(f, repo / f.path):
            conflicts.append(f.path)

        if f.before_exists:
            to_restore.append(f.path)
        else:
            to_delete.append(f.path)

    return UndoPlan(
        task=snapshot.task,
        to_restore=tuple(to_restore),
        to_delete=tuple(to_delete),
        conflicts=tuple(conflicts),
        unrestorable=tuple(unrestorable),
    )


def apply_undo(snapshot: UndoSnapshot, repo: Path, *, force: bool = False) -> UndoPlan:
    """Restore the files this turn changed. Returns the plan that was applied.

    Nothing is written until the whole plan is known to be applicable, so a
    refused undo leaves the tree exactly as it was.

    Args:
        force: Proceed even when a file changed after the turn was applied.

    Raises:
        UndoConflictError: A file changed since the turn and `force` is False.
    """
    plan = plan_undo(snapshot, repo)

    if plan.conflicts and not force:
        raise UndoConflictError(plan)

    for f in snapshot.files:
        if not f.before_is_restorable:
            continue
        target = repo / f.path
        if f.before_exists:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f.before_text or "", encoding="utf-8")
        elif target.is_file():
            target.unlink()

    return plan


def _changed_since(f: FileDiff, target: Path) -> bool:
    """Whether `target` differs from what the turn left behind."""
    if not f.after_exists:
        # The turn deleted this file; finding one means someone recreated it.
        return target.is_file()

    if not target.is_file():
        return True  # the turn wrote it, and it is gone now

    if f.after_text is None:
        # Binary on the way in, so there is no text to compare. Treat it as
        # changed rather than assume it is untouched.
        return True

    try:
        return target.read_text(encoding="utf-8") != f.after_text
    except (UnicodeDecodeError, OSError):
        return True
