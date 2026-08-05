"""Tests for sub-agent specs and the preset registry."""

from __future__ import annotations

import pytest

from qqcode.agents.subagent import (
    ALL_TOOLS,
    READ_ONLY_TOOLS,
    TOOL_RUN,
    TOOL_WRITE,
    WRITE_TOOLS,
    Isolation,
    ModelTier,
    SubAgentResult,
    SubAgentSpec,
    get_preset,
    list_presets,
    register_preset,
)


def _spec(**overrides: object) -> SubAgentSpec:
    base = {
        "name": "custom",
        "description": "d",
        "system_prompt": "p",
    }
    return SubAgentSpec(**{**base, **overrides})  # type: ignore[arg-type]


class TestSpecValidation:
    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            _spec(name="")

    def test_rejects_zero_max_turns(self) -> None:
        with pytest.raises(ValueError, match="max_turns"):
            _spec(max_turns=0)

    def test_rejects_unknown_tool(self) -> None:
        with pytest.raises(ValueError, match="unknown tools"):
            _spec(allowed_tools=frozenset({"launch_missiles"}))

    def test_read_only_spec_cannot_grant_write_tools(self) -> None:
        with pytest.raises(ValueError, match="read_only spec cannot grant write"):
            _spec(isolation=Isolation.READ_ONLY, allowed_tools=frozenset({TOOL_WRITE}))

    def test_read_only_spec_may_run_commands(self) -> None:
        spec = _spec(isolation=Isolation.READ_ONLY, allowed_tools=READ_ONLY_TOOLS | {TOOL_RUN})
        assert TOOL_RUN in spec.allowed_tools

    def test_shadow_write_spec_may_grant_write_tools(self) -> None:
        spec = _spec(isolation=Isolation.SHADOW_WRITE, allowed_tools=READ_ONLY_TOOLS | WRITE_TOOLS)
        assert spec.allowed_tools >= WRITE_TOOLS

    def test_spec_is_immutable(self) -> None:
        spec = _spec()
        with pytest.raises(AttributeError):
            spec.name = "mutated"  # type: ignore[misc]


class TestDerive:
    def test_derive_overrides_field(self) -> None:
        assert _spec(max_turns=5).derive(max_turns=9).max_turns == 9

    def test_derive_leaves_original_unchanged(self) -> None:
        spec = _spec(max_turns=5)
        spec.derive(max_turns=9)
        assert spec.max_turns == 5

    def test_derive_revalidates(self) -> None:
        spec = _spec(isolation=Isolation.READ_ONLY)
        with pytest.raises(ValueError, match="read_only spec cannot grant write"):
            spec.derive(allowed_tools=frozenset({TOOL_WRITE}))

    def test_derive_can_relax_isolation_and_tools_together(self) -> None:
        spec = _spec().derive(
            isolation=Isolation.SHADOW_WRITE,
            allowed_tools=READ_ONLY_TOOLS | WRITE_TOOLS,
        )
        assert spec.isolation is Isolation.SHADOW_WRITE

    def test_derive_preserves_mcp_and_skill_grants(self) -> None:
        spec = _spec(mcp_servers=frozenset({"github"}), pinned_skills=("run-tests",))
        derived = spec.derive(max_turns=3)
        assert derived.mcp_servers == frozenset({"github"})
        assert derived.pinned_skills == ("run-tests",)


class TestMCPAndSkillGrants:
    def test_no_mcp_servers_by_default(self) -> None:
        """Inheriting a parent's MCP schemas would cancel out context isolation."""
        assert _spec().mcp_servers == frozenset()

    def test_no_pinned_skills_by_default(self) -> None:
        assert _spec().pinned_skills == ()

    def test_mcp_servers_can_be_granted_explicitly(self) -> None:
        assert _spec(mcp_servers=frozenset({"github"})).mcp_servers == frozenset({"github"})

    def test_pinned_skills_can_be_granted_explicitly(self) -> None:
        assert _spec(pinned_skills=("run-tests", "conventions")).pinned_skills == (
            "run-tests",
            "conventions",
        )

    def test_rejects_duplicate_pinned_skills(self) -> None:
        with pytest.raises(ValueError, match="pinned_skills contains duplicates"):
            _spec(pinned_skills=("a", "a"))

    def test_builtin_presets_grant_no_mcp(self) -> None:
        for spec in BUILTIN_PRESETS:
            assert spec.mcp_servers == frozenset(), spec.name

    def test_builtin_presets_pin_no_skills(self) -> None:
        for spec in BUILTIN_PRESETS:
            assert spec.pinned_skills == (), spec.name


class TestRegistry:
    def test_expected_presets_are_registered(self) -> None:
        names = {s.name for s in list_presets()}
        assert names >= {
            "explorer",
            "reviewer",
            "security-auditor",
            "planner",
            "test-writer",
            "build-fixer",
            "doc-writer",
        }

    def test_get_preset_returns_spec(self) -> None:
        assert get_preset("explorer").name == "explorer"

    def test_get_preset_rejects_unknown_name(self) -> None:
        with pytest.raises(KeyError, match="Unknown sub-agent preset"):
            get_preset("nope")

    def test_list_presets_is_sorted(self) -> None:
        names = [s.name for s in list_presets()]
        assert names == sorted(names)

    def test_register_custom_preset(self) -> None:
        register_preset(_spec(name="test-only-custom"), overwrite=True)
        assert get_preset("test-only-custom").name == "test-only-custom"

    def test_register_rejects_duplicate_without_overwrite(self) -> None:
        with pytest.raises(ValueError, match="already registered"):
            register_preset(_spec(name="explorer"))


BUILTIN_PRESETS = [
    get_preset(n)
    for n in (
        "build-fixer",
        "doc-writer",
        "explorer",
        "planner",
        "reviewer",
        "security-auditor",
        "test-writer",
    )
]


class TestPresetInvariants:
    @pytest.mark.parametrize("spec", BUILTIN_PRESETS, ids=lambda s: s.name)
    def test_preset_has_substantive_prompt(self, spec: SubAgentSpec) -> None:
        assert len(spec.system_prompt) > 100

    @pytest.mark.parametrize("spec", BUILTIN_PRESETS, ids=lambda s: s.name)
    def test_preset_has_description(self, spec: SubAgentSpec) -> None:
        assert spec.description

    @pytest.mark.parametrize("spec", BUILTIN_PRESETS, ids=lambda s: s.name)
    def test_preset_tools_are_known(self, spec: SubAgentSpec) -> None:
        assert spec.allowed_tools <= ALL_TOOLS

    @pytest.mark.parametrize("spec", BUILTIN_PRESETS, ids=lambda s: s.name)
    def test_preset_can_read(self, spec: SubAgentSpec) -> None:
        """Every sub-agent needs to read the code it works on."""
        assert spec.allowed_tools >= READ_ONLY_TOOLS

    @pytest.mark.parametrize("spec", BUILTIN_PRESETS, ids=lambda s: s.name)
    def test_preset_turn_budget_is_bounded(self, spec: SubAgentSpec) -> None:
        assert 1 <= spec.max_turns <= 30

    def test_analysis_presets_are_read_only(self) -> None:
        for name in ("explorer", "reviewer", "security-auditor", "planner"):
            assert get_preset(name).isolation is Isolation.READ_ONLY

    def test_editing_presets_can_write(self) -> None:
        for name in ("test-writer", "build-fixer", "doc-writer"):
            spec = get_preset(name)
            assert spec.isolation is Isolation.SHADOW_WRITE
            assert spec.allowed_tools >= WRITE_TOOLS

    def test_presets_needing_verification_can_run_commands(self) -> None:
        """Writing tests or fixing builds is unverifiable without executing them."""
        for name in ("test-writer", "build-fixer"):
            assert TOOL_RUN in get_preset(name).allowed_tools

    def test_structured_presets_declare_schema(self) -> None:
        for name in ("explorer", "reviewer", "security-auditor", "planner"):
            schema = get_preset(name).output_schema
            assert schema is not None
            assert schema["type"] == "object"
            assert schema["required"]

    def test_deep_tier_reserved_for_hard_reasoning(self) -> None:
        deep = {s.name for s in BUILTIN_PRESETS if s.model_tier is ModelTier.DEEP}
        assert deep == {"planner", "security-auditor"}


class TestSubAgentResult:
    def test_success_result_defaults_to_no_files_touched(self) -> None:
        result = SubAgentResult(
            spec_name="explorer", success=True, output="found it", turns_used=3, tokens_used=1200
        )
        assert result.files_touched == []
        assert result.error is None

    def test_failure_result_carries_error(self) -> None:
        result = SubAgentResult(
            spec_name="build-fixer",
            success=False,
            output="",
            turns_used=20,
            tokens_used=9000,
            error="turn budget exhausted",
        )
        assert not result.success
        assert result.error == "turn budget exhausted"

    def test_result_records_cost_for_ledger(self) -> None:
        """Sub-agent token spend must be reportable to the parent's ledger."""
        result = SubAgentResult(
            spec_name="reviewer", success=True, output={}, turns_used=5, tokens_used=4321
        )
        assert result.tokens_used == 4321
