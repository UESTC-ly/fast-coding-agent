"""Skill definition and parsing.

A skill is knowledge, not capability: it carries no side effects and the model
never "calls" it. It is injected into context when relevant. Implementing a
skill as a tool would cost an extra round trip for nothing.

Skills live at `.qqcode/skills/<name>/SKILL.md` as YAML frontmatter plus a
Markdown body:

    ---
    name: run-tests
    description: How tests run in this repo and how to read failures
    globs: ["tests/**", "**/*_test.py"]
    keywords: ["test", "pytest", "coverage"]
    fastpath_safe: true
    routing_hint: fast
    ---
    (body: concrete commands, common failure modes, conventions)

Beyond instructions, a skill is routing evidence. A `fastpath_safe` skill whose
globs match the task supplies concrete steps and file anchors, which raises the
classifier's confidence that evidence is sufficient. A `routing_hint: full`
skill does the opposite — it forces escalation no matter how simple the task
looks, which is the point for workflows like database migrations.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

SKILL_FILENAME = "SKILL.md"
FRONTMATTER_DELIMITER = "---"


class RoutingHint(StrEnum):
    """A skill's prior on which tier should handle a matching task."""

    NONE = "none"  # No opinion
    FAST = "fast"  # Task shape is well understood; FastPath is plausible
    FULL = "full"  # Always escalate; the workflow needs the full tool loop


@dataclass(frozen=True)
class Skill:
    """A parsed instruction pack.

    `body` is the payload that costs tokens; `name` and `description` form the
    cheap index entry used to decide whether the body is worth loading.
    """

    name: str
    description: str
    body: str
    path: Path | None = None
    globs: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    # Whether injecting this skill's body is cheap enough for FastPath.
    fastpath_safe: bool = False
    routing_hint: RoutingHint = RoutingHint.NONE

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Skill.name must be non-empty")
        if not self.description:
            raise ValueError(f"{self.name}: description must be non-empty")
        if not self.body.strip():
            raise ValueError(f"{self.name}: body must be non-empty")
        # A FULL hint means "never FastPath", so also claiming fastpath_safe is
        # contradictory and would let the skill load on the tier it forbids.
        if self.fastpath_safe and self.routing_hint is RoutingHint.FULL:
            raise ValueError(f"{self.name}: fastpath_safe conflicts with routing_hint=full")

    @property
    def index_entry(self) -> str:
        """The always-resident one-liner. Loading the body is a separate step."""
        return f"- {self.name}: {self.description}"

    def matches_path(self, path: str) -> bool:
        """Whether any glob matches this repo-relative path."""
        return any(_glob_match(path, g) for g in self.globs)

    def matches_text(self, text: str) -> bool:
        """Whether any keyword appears in the text (case-insensitive)."""
        lowered = text.lower()
        return any(k.lower() in lowered for k in self.keywords)


def _glob_match(path: str, pattern: str) -> bool:
    """Match a path against a glob, treating `**` as crossing directories.

    fnmatch's `*` already spans separators, so `tests/**` and `**/*_test.py`
    both behave as expected once a bare `dir/**` is also allowed to match `dir`
    itself.
    """
    if fnmatch.fnmatch(path, pattern):
        return True
    if pattern.endswith("/**"):
        return fnmatch.fnmatch(path, pattern[:-3])
    return False


def parse_skill(text: str, *, path: Path | None = None) -> Skill:
    """Parse SKILL.md content into a Skill.

    Raises:
        ValueError: Missing or malformed frontmatter, or invalid field values.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != FRONTMATTER_DELIMITER:
        raise ValueError(f"{path or '<string>'}: missing frontmatter opening '---'")

    try:
        close = next(
            i for i, ln in enumerate(lines[1:], start=1) if ln.strip() == FRONTMATTER_DELIMITER
        )
    except StopIteration:
        raise ValueError(f"{path or '<string>'}: unterminated frontmatter") from None

    raw = yaml.safe_load("\n".join(lines[1:close])) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path or '<string>'}: frontmatter must be a mapping")
    body = "\n".join(lines[close + 1 :]).strip()

    return Skill(
        name=str(raw.get("name", "")),
        description=str(raw.get("description", "")),
        body=body,
        path=path,
        globs=_str_tuple(raw, "globs", path),
        keywords=_str_tuple(raw, "keywords", path),
        fastpath_safe=bool(raw.get("fastpath_safe", False)),
        routing_hint=_routing_hint(raw.get("routing_hint", "none"), path),
    )


def _str_tuple(raw: dict[str, Any], key: str, path: Path | None) -> tuple[str, ...]:
    value = raw.get(key, [])
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(str(v) for v in value)
    raise ValueError(f"{path or '<string>'}: {key} must be a string or list of strings")


def _routing_hint(value: object, path: Path | None) -> RoutingHint:
    try:
        return RoutingHint(str(value))
    except ValueError:
        valid = [h.value for h in RoutingHint]
        raise ValueError(
            f"{path or '<string>'}: routing_hint must be one of {valid}, got {value!r}"
        ) from None


def load_skill(skill_dir: Path) -> Skill:
    """Load a skill from a directory containing SKILL.md.

    Raises:
        FileNotFoundError: No SKILL.md in the directory.
        ValueError: Malformed skill definition.
    """
    md = skill_dir / SKILL_FILENAME
    if not md.is_file():
        raise FileNotFoundError(f"No {SKILL_FILENAME} in {skill_dir}")
    return parse_skill(md.read_text(encoding="utf-8"), path=md)
