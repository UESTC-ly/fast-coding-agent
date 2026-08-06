"""Change review: turning a shadow workspace's diff into a human verdict.

The batch model decides success objectively — hidden acceptance tests pass or
they do not. Conversation has no such oracle: the criterion is whether the person
asking is satisfied. This module supplies the second verdict source without
disturbing the first.

Conditions 1 (valid finish state) and 2 (diff within the expected file set) are
unaffected. Only the third condition gains a human variant, and only when no
acceptance harness was supplied.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

# Files above this size are summarized rather than diffed. A review a person
# cannot read is not a review, and a generated lockfile is the usual culprit.
MAX_DIFF_LINES_PER_FILE = 400


@dataclass(frozen=True)
class FileDiff:
    """One file's change, rendered for human reading."""

    path: str
    status: str          # "added" | "modified" | "deleted"
    diff_text: str       # unified diff, or a summary when too large
    truncated: bool = False
    # Pre-change state, captured while the source is still untouched. This is
    # what makes a per-turn undo possible without relying on git, which cannot
    # help when the change was never committed.
    before_exists: bool = False
    before_text: str | None = None   # None with before_exists=True => binary
    # Post-change state, read from the shadow before it is finalized. Undo
    # compares this against what is on disk to notice later edits; capturing it
    # here is exact, where reconstructing it from `diff_text` would not be.
    after_exists: bool = False
    after_text: str | None = None

    @property
    def before_is_restorable(self) -> bool:
        """Whether undo can faithfully put this file back.

        False for binary or undecodable files: their prior bytes were never
        captured, so "restoring" them would write mojibake or delete a file
        that did exist.
        """
        return not self.before_exists or self.before_text is not None


@dataclass(frozen=True)
class ChangeReview:
    """Everything a person needs to accept or reject one turn's output."""

    task: str
    mode_used: str
    reasoning: str
    diffs: tuple[FileDiff, ...] = field(default_factory=tuple)

    @property
    def changed_files(self) -> tuple[str, ...]:
        return tuple(d.path for d in self.diffs)

    def is_empty(self) -> bool:
        """Whether the agent claimed success without changing anything."""
        return not self.diffs


# Returns True to accept the change (finalize it), False to discard it.
ConfirmCallback = Callable[[ChangeReview], bool]


def build_review(
    *,
    task: str,
    mode_used: str,
    reasoning: str,
    changed_files: frozenset[str],
    source: Path,
    shadow: Path,
) -> ChangeReview:
    """Render a shadow workspace's changes against the untouched source.

    Safe to call before `finalize`, and only meaningful there: the source still
    holds the pre-change content, which is exactly what the diff needs.
    """
    diffs = tuple(
        _diff_one(rel, source / rel, shadow / rel)
        for rel in sorted(changed_files)
    )
    return ChangeReview(task=task, mode_used=mode_used, reasoning=reasoning, diffs=diffs)


def _diff_one(rel: str, before: Path, after: Path) -> FileDiff:
    old = _read(before)
    new = _read(after)
    # Existence is checked separately from readability: `_read` returns None for
    # both "absent" and "binary", but undo must tell them apart — one means
    # delete, the other means do not touch.
    before_exists = before.is_file()
    after_exists = after.is_file()

    if not before_exists and after_exists:
        status = "added"
    elif before_exists and not after_exists:
        status = "deleted"
    else:
        status = "modified"

    lines = list(
        difflib.unified_diff(
            (old or "").splitlines(keepends=True),
            (new or "").splitlines(keepends=True),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
            n=3,
        )
    )

    if len(lines) > MAX_DIFF_LINES_PER_FILE:
        kept = "".join(lines[:MAX_DIFF_LINES_PER_FILE])
        dropped = len(lines) - MAX_DIFF_LINES_PER_FILE
        return FileDiff(
            path=rel,
            status=status,
            diff_text=f"{kept}\n... {dropped} more diff lines omitted ...",
            truncated=True,
            before_exists=before_exists,
            before_text=old,
            after_exists=after_exists,
            after_text=new,
        )

    return FileDiff(
        path=rel,
        status=status,
        diff_text="".join(lines),
        before_exists=before_exists,
        before_text=old,
        after_exists=after_exists,
        after_text=new,
    )


def _read(path: Path) -> str | None:
    """File content, or None when absent or not decodable as text.

    Binary files return None on both sides, which renders as an empty diff rather
    than a wall of mojibake. The path still appears in the review, so the change
    is visible even though its content is not.
    """
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None
