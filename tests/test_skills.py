"""Tests for skill parsing, matching, and tier-aware selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from qqcode.skills.index import SkillIndex
from qqcode.skills.skill import RoutingHint, Skill, load_skill, parse_skill

SKILL_MD = """---
name: run-tests
description: How tests run in this repo
globs: ["tests/**", "**/*_test.py"]
keywords: ["test", "pytest"]
fastpath_safe: true
routing_hint: fast
---
Run `pytest -q` from the repo root.
"""


def _skill(name: str = "s", **kw: object) -> Skill:
    base: dict[str, object] = {"name": name, "description": "d", "body": "b"}
    return Skill(**{**base, **kw})  # type: ignore[arg-type]


def _write_skill(root: Path, name: str, content: str) -> Path:
    d = root / ".qqcode" / "skills" / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(content, encoding="utf-8")
    return d


class TestParsing:
    def test_parses_all_frontmatter_fields(self) -> None:
        s = parse_skill(SKILL_MD)
        assert s.name == "run-tests"
        assert s.description == "How tests run in this repo"
        assert s.globs == ("tests/**", "**/*_test.py")
        assert s.keywords == ("test", "pytest")
        assert s.fastpath_safe
        assert s.routing_hint is RoutingHint.FAST

    def test_body_excludes_frontmatter(self) -> None:
        assert parse_skill(SKILL_MD).body == "Run `pytest -q` from the repo root."

    def test_defaults_when_optional_fields_absent(self) -> None:
        s = parse_skill("---\nname: n\ndescription: d\n---\nbody")
        assert s.globs == ()
        assert s.keywords == ()
        assert not s.fastpath_safe
        assert s.routing_hint is RoutingHint.NONE

    def test_scalar_glob_is_accepted(self) -> None:
        s = parse_skill("---\nname: n\ndescription: d\nglobs: src/**\n---\nbody")
        assert s.globs == ("src/**",)

    def test_rejects_missing_frontmatter(self) -> None:
        with pytest.raises(ValueError, match="missing frontmatter"):
            parse_skill("just a body")

    def test_rejects_unterminated_frontmatter(self) -> None:
        with pytest.raises(ValueError, match="unterminated frontmatter"):
            parse_skill("---\nname: n\n")

    def test_rejects_non_mapping_frontmatter(self) -> None:
        with pytest.raises(ValueError, match="must be a mapping"):
            parse_skill("---\n- a\n- b\n---\nbody")

    def test_rejects_bad_routing_hint(self) -> None:
        with pytest.raises(ValueError, match="routing_hint must be one of"):
            parse_skill("---\nname: n\ndescription: d\nrouting_hint: maybe\n---\nbody")

    def test_rejects_bad_globs_type(self) -> None:
        with pytest.raises(ValueError, match="globs must be a string or list"):
            parse_skill("---\nname: n\ndescription: d\nglobs: {a: 1}\n---\nbody")

    def test_rejects_empty_body(self) -> None:
        with pytest.raises(ValueError, match="body must be non-empty"):
            parse_skill("---\nname: n\ndescription: d\n---\n")

    def test_rejects_missing_description(self) -> None:
        with pytest.raises(ValueError, match="description must be non-empty"):
            parse_skill("---\nname: n\n---\nbody")

    def test_rejects_fastpath_safe_with_full_hint(self) -> None:
        with pytest.raises(ValueError, match="conflicts with routing_hint=full"):
            _skill(fastpath_safe=True, routing_hint=RoutingHint.FULL)


class TestLoading:
    def test_loads_from_directory(self, tmp_path: Path) -> None:
        d = _write_skill(tmp_path, "run-tests", SKILL_MD)
        assert load_skill(d).name == "run-tests"

    def test_records_source_path(self, tmp_path: Path) -> None:
        d = _write_skill(tmp_path, "run-tests", SKILL_MD)
        assert load_skill(d).path == d / "SKILL.md"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No SKILL.md"):
            load_skill(tmp_path)


class TestMatching:
    def test_glob_matches_nested_path(self) -> None:
        assert _skill(globs=("tests/**",)).matches_path("tests/unit/test_x.py")

    def test_glob_matches_directory_itself(self) -> None:
        assert _skill(globs=("tests/**",)).matches_path("tests")

    def test_double_star_prefix_matches_suffix_pattern(self) -> None:
        assert _skill(globs=("**/*_test.py",)).matches_path("src/deep/thing_test.py")

    def test_non_matching_path_is_rejected(self) -> None:
        assert not _skill(globs=("tests/**",)).matches_path("src/main.py")

    def test_keyword_match_is_case_insensitive(self) -> None:
        assert _skill(keywords=("pytest",)).matches_text("Run PyTest now")

    def test_absent_keyword_does_not_match(self) -> None:
        assert not _skill(keywords=("pytest",)).matches_text("build the docs")

    def test_index_entry_omits_body(self) -> None:
        s = _skill(name="run-tests", description="how tests run", body="LONG BODY TEXT")
        assert s.index_entry == "- run-tests: how tests run"
        assert "LONG BODY TEXT" not in s.index_entry


class TestIndexBasics:
    def test_add_and_get(self) -> None:
        idx = SkillIndex([_skill("a")])
        assert idx.get("a").name == "a"

    def test_rejects_duplicate_name(self) -> None:
        idx = SkillIndex([_skill("a")])
        with pytest.raises(ValueError, match="already registered"):
            idx.add(_skill("a"))

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown skill"):
            SkillIndex().get("nope")

    def test_all_is_sorted(self) -> None:
        idx = SkillIndex([_skill("z"), _skill("a")])
        assert [s.name for s in idx.all()] == ["a", "z"]

    def test_index_text_has_one_line_per_skill(self) -> None:
        idx = SkillIndex([_skill("a"), _skill("b")])
        assert idx.index_text().splitlines() == ["- a: d", "- b: d"]


class TestDiscovery:
    def test_discovers_skills(self, tmp_path: Path) -> None:
        _write_skill(tmp_path, "run-tests", SKILL_MD)
        idx = SkillIndex.discover(tmp_path)
        assert "run-tests" in {s.name for s in idx.all()}

    def test_missing_directory_yields_only_builtins(self, tmp_path: Path) -> None:
        # No project skills → only package built-ins loaded
        idx = SkillIndex.discover(tmp_path)
        names = {s.name for s in idx.all()}
        assert "python-docstrings" in names
        assert "pytest-patterns" in names

    def test_directory_without_skill_md_is_skipped(self, tmp_path: Path) -> None:
        (tmp_path / ".qqcode" / "skills" / "junk").mkdir(parents=True)
        idx = SkillIndex.discover(tmp_path)
        assert "junk" not in {s.name for s in idx.all()}

    def test_malformed_skill_propagates(self, tmp_path: Path) -> None:
        """Silently dropping a skill would leave the agent ignoring conventions."""
        _write_skill(tmp_path, "broken", "no frontmatter here")
        with pytest.raises(ValueError, match="missing frontmatter"):
            SkillIndex.discover(tmp_path)


class TestRoutingHints:
    def test_full_hint_wins_over_fast(self) -> None:
        idx = SkillIndex()
        matched = [
            _skill("a", routing_hint=RoutingHint.FAST),
            _skill("b", routing_hint=RoutingHint.FULL),
        ]
        assert idx.routing_hint(matched) is RoutingHint.FULL

    def test_fast_hint_when_no_full(self) -> None:
        idx = SkillIndex()
        matched = [_skill("a", routing_hint=RoutingHint.FAST), _skill("b")]
        assert idx.routing_hint(matched) is RoutingHint.FAST

    def test_none_when_no_opinions(self) -> None:
        assert SkillIndex().routing_hint([_skill("a")]) is RoutingHint.NONE

    def test_none_when_nothing_matched(self) -> None:
        assert SkillIndex().routing_hint([]) is RoutingHint.NONE


class TestFastPathPinning:
    def test_single_matching_safe_skill_is_pinned(self) -> None:
        idx = SkillIndex([_skill("t", keywords=("test",), fastpath_safe=True)])
        pinned = idx.pin_for_fastpath(task="add a test")
        assert pinned is not None
        assert pinned.name == "t"

    def test_two_matches_pin_nothing(self) -> None:
        """Ambiguity means the task is less anchored than it looked."""
        idx = SkillIndex(
            [
                _skill("a", keywords=("test",), fastpath_safe=True),
                _skill("b", keywords=("test",), fastpath_safe=True),
            ]
        )
        assert idx.pin_for_fastpath(task="add a test") is None

    def test_unsafe_skill_is_never_pinned(self) -> None:
        idx = SkillIndex([_skill("t", keywords=("test",), fastpath_safe=False)])
        assert idx.pin_for_fastpath(task="add a test") is None

    def test_no_match_pins_nothing(self) -> None:
        idx = SkillIndex([_skill("t", keywords=("deploy",), fastpath_safe=True)])
        assert idx.pin_for_fastpath(task="add a test") is None

    def test_path_match_can_pin(self) -> None:
        idx = SkillIndex([_skill("t", globs=("tests/**",), fastpath_safe=True)])
        assert idx.pin_for_fastpath(paths=("tests/test_x.py",)) is not None


class TestTierSelection:
    def _index(self) -> SkillIndex:
        return SkillIndex(
            [
                _skill("t", keywords=("test",), fastpath_safe=True),
                _skill("other", keywords=("deploy",)),
            ]
        )

    def test_fastpath_gets_no_resident_index(self) -> None:
        """The index alone would exceed FastPath's budget."""
        index_text, _ = self._index().select("fastpath", task="add a test")
        assert index_text == ""

    def test_fastpath_gets_the_auto_pinned_body(self) -> None:
        _, bodies = self._index().select("fastpath", task="add a test")
        assert [s.name for s in bodies] == ["t"]

    def test_fastpath_gets_nothing_without_a_match(self) -> None:
        _, bodies = self._index().select("fastpath", task="unrelated work")
        assert bodies == []

    def test_explicit_pin_overrides_auto_selection(self) -> None:
        _, bodies = self._index().select("fastpath", task="add a test", pinned=("other",))
        assert [s.name for s in bodies] == ["other"]

    def test_fullagent_gets_resident_index(self) -> None:
        index_text, _ = self._index().select("fullagent")
        assert index_text.splitlines() == ["- other: d", "- t: d"]

    def test_fullagent_loads_no_bodies_by_default(self) -> None:
        _, bodies = self._index().select("fullagent", task="add a test")
        assert bodies == []

    def test_subagent_gets_no_index(self) -> None:
        """Inheriting an index would undo the isolation that justifies spawning."""
        index_text, _ = self._index().select("subagent", task="add a test")
        assert index_text == ""

    def test_subagent_gets_only_pinned_bodies(self) -> None:
        _, bodies = self._index().select("subagent", task="add a test", pinned=("t",))
        assert [s.name for s in bodies] == ["t"]

    def test_subagent_without_pins_gets_nothing(self) -> None:
        _, bodies = self._index().select("subagent", task="add a test")
        assert bodies == []

    def test_unknown_pinned_name_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown skill"):
            self._index().select("fullagent", pinned=("ghost",))
