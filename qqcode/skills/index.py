"""Skill index: discovery, matching, and tier-aware selection.

Loading rules, in cost order:

- FastPath gets no index. 50 skills' name+description lines run ~1.5k tokens,
  which Full Agent absorbs and FastPath cannot. The one exception is a single
  pinned body: when static features already anchor the task to a matching
  `fastpath_safe` skill, injecting that body alone is net positive — it replaces
  exploration turns that would cost far more than the skill's tokens.

- Full Agent gets the index resident and loads bodies on demand.

- Sub-agents get only what their spec pins. Inheriting a full index would undo
  the context isolation that justifies spawning one.

Placement matters as much as size. Skill text belongs *after* the cache
breakpoint, in `system → repo card → [BREAKPOINT] → skills → task` order. Put a
per-task skill body before the breakpoint and every task presents a different
cache prefix, so prompt caching stops hitting — the recomputed prefix costs more
than the skill ever saved.
"""

from __future__ import annotations

from pathlib import Path

from qqcode.skills.skill import RoutingHint, Skill, load_skill

SKILLS_DIRNAME = "skills"
# Built-in skills shipped with the package; project skills override these.
_BUILTIN_DIR = Path(__file__).parent / "builtin"


class SkillIndex:
    """Registry of available skills with tier-aware selection."""

    def __init__(self, skills: list[Skill] | None = None):
        self._skills: dict[str, Skill] = {}
        for s in skills or []:
            self.add(s)

    @classmethod
    def _load_builtin(cls) -> list[Skill]:
        """Load skills bundled with the qqcode package."""
        if not _BUILTIN_DIR.is_dir():
            return []
        return [
            load_skill(d)
            for d in sorted(_BUILTIN_DIR.iterdir())
            if (d / "SKILL.md").is_file()
        ]

    @classmethod
    def discover(cls, root: Path) -> SkillIndex:
        """Load built-in skills plus every skill under `<root>/.qqcode/skills/*/SKILL.md`.

        Built-in skills are always loaded. Project-local skills take precedence:
        a project skill with the same name silently overrides the built-in.
        A missing `.qqcode/skills/` directory is not an error.
        Malformed skills propagate.
        """
        merged: dict[str, Skill] = {s.name: s for s in cls._load_builtin()}
        base = root / ".qqcode" / SKILLS_DIRNAME
        if base.is_dir():
            for d in sorted(base.iterdir()):
                if (d / "SKILL.md").is_file():
                    s = load_skill(d)
                    merged[s.name] = s  # project skill overrides built-in
        return cls(list(merged.values()))

    def add(self, skill: Skill) -> None:
        """Register a skill.

        Raises:
            ValueError: Name already registered.
        """
        if skill.name in self._skills:
            raise ValueError(f"Skill already registered: {skill.name}")
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill:
        """Look up a skill.

        Raises:
            KeyError: Not registered.
        """
        try:
            return self._skills[name]
        except KeyError:
            raise KeyError(f"Unknown skill: {name}. Available: {sorted(self._skills)}") from None

    def all(self) -> list[Skill]:
        """Every skill, ordered by name."""
        return [self._skills[n] for n in sorted(self._skills)]

    def __len__(self) -> int:
        return len(self._skills)

    def index_text(self) -> str:
        """The resident index block: one line per skill, bodies excluded."""
        return "\n".join(s.index_entry for s in self.all())

    def match(self, *, task: str = "", paths: tuple[str, ...] = ()) -> list[Skill]:
        """Skills relevant to a task, by keyword or path glob.

        Args:
            task: Task text, matched against keywords.
            paths: Repo-relative paths, matched against globs.
        """
        return [
            s
            for s in self.all()
            if (task and s.matches_text(task)) or any(s.matches_path(p) for p in paths)
        ]

    def routing_hint(self, matched: list[Skill]) -> RoutingHint:
        """Combined hint from matched skills.

        FULL wins over everything: one workflow demanding the full tool loop
        overrides any number of skills that consider the task simple.
        """
        hints = {s.routing_hint for s in matched}
        if RoutingHint.FULL in hints:
            return RoutingHint.FULL
        if RoutingHint.FAST in hints:
            return RoutingHint.FAST
        return RoutingHint.NONE

    def pin_for_fastpath(self, *, task: str = "", paths: tuple[str, ...] = ()) -> Skill | None:
        """The single skill body FastPath may carry, if any.

        Returns a skill only when exactly one `fastpath_safe` skill matches. Two
        matches mean the task is not as well anchored as it looked, so nothing is
        pinned and the tier decision falls to the normal gate.
        """
        candidates = [s for s in self.match(task=task, paths=paths) if s.fastpath_safe]
        return candidates[0] if len(candidates) == 1 else None

    def select(
        self,
        tier: str,
        *,
        task: str = "",
        paths: tuple[str, ...] = (),
        pinned: tuple[str, ...] = (),
    ) -> tuple[str, list[Skill]]:
        """Skill context for a tier.

        Args:
            tier: "fastpath", "fullagent", or "subagent".
            task: Task text for keyword matching.
            paths: Repo-relative paths for glob matching.
            pinned: Skill names to force-load (a sub-agent spec's pins).

        Returns:
            `(index_text, bodies_to_inject)`. `index_text` is empty on tiers
            that get no resident index.

        Raises:
            KeyError: A pinned name is not registered.
        """
        forced = [self.get(n) for n in pinned]

        if tier == "fastpath":
            if forced:
                return "", forced
            auto = self.pin_for_fastpath(task=task, paths=paths)
            return "", [auto] if auto else []

        if tier == "subagent":
            return "", forced

        return self.index_text(), forced
