"""Tests for the tool registry, MCP admission rules, and result compression."""

from __future__ import annotations

import pytest

from qqcode.models.protocol import ALL_TIERS, OutputSpec, ToolSpec
from qqcode.tools.artifacts import (
    InMemoryArtifactStore,
    ResultPolicy,
    build_tool_result,
)
from qqcode.tools.mcp import MCPCapability, MCPServerConfig
from qqcode.tools.registry import ToolRegistry, ToolSource, mcp_tool_name


def _tool(name: str = "read_file") -> ToolSpec:
    return ToolSpec(name=name, description="d", input_schema={"type": "object"})


class TestBuiltinRegistration:
    def test_registers_under_bare_name(self) -> None:
        reg = ToolRegistry()
        reg.register_builtin(_tool())
        assert reg.get("read_file").source is ToolSource.BUILTIN

    def test_visible_on_all_tiers_by_default(self) -> None:
        reg = ToolRegistry()
        reg.register_builtin(_tool())
        for tier in ("fastpath", "fullagent", "subagent"):
            assert [t.name for t in reg.visible(tier)] == ["read_file"]  # type: ignore[arg-type]

    def test_rejects_duplicate_name(self) -> None:
        reg = ToolRegistry()
        reg.register_builtin(_tool())
        with pytest.raises(ValueError, match="already registered"):
            reg.register_builtin(_tool())

    def test_rejects_mcp_namespace_squatting(self) -> None:
        reg = ToolRegistry()
        with pytest.raises(ValueError, match="may not use the MCP namespace"):
            reg.register_builtin(_tool("mcp__evil__read_file"))

    def test_rejects_unknown_tier(self) -> None:
        reg = ToolRegistry()
        with pytest.raises(ValueError, match="unknown tiers"):
            reg.register_builtin(_tool(), tiers=frozenset({"routing"}))

    def test_rejects_empty_tiers(self) -> None:
        reg = ToolRegistry()
        with pytest.raises(ValueError, match="tiers must be non-empty"):
            reg.register_builtin(_tool(), tiers=frozenset())

    def test_tier_restriction_is_honoured(self) -> None:
        reg = ToolRegistry()
        reg.register_builtin(_tool("run_command"), tiers=frozenset({"fullagent"}))
        assert reg.visible("fastpath") == []
        assert [t.name for t in reg.visible("fullagent")] == ["run_command"]


class TestMCPRegistration:
    def test_namespaces_tool_name(self) -> None:
        assert mcp_tool_name("github", "create_issue") == "mcp__github__create_issue"

    def test_rejects_server_name_with_separator(self) -> None:
        with pytest.raises(ValueError, match="may not contain"):
            mcp_tool_name("bad__name", "t")

    def test_registered_under_namespaced_name(self) -> None:
        reg = ToolRegistry()
        name = reg.register_mcp("github", _tool("create_issue"), mutates=True)
        assert name == "mcp__github__create_issue"
        assert reg.get(name).server == "github"

    def test_namespacing_prevents_builtin_shadowing(self) -> None:
        """A server exposing read_file must not capture the builtin's name."""
        reg = ToolRegistry()
        reg.register_builtin(_tool("read_file"))
        reg.register_mcp("filesystem", _tool("read_file"), mutates=False)

        assert reg.get("read_file").source is ToolSource.BUILTIN
        assert reg.get("mcp__filesystem__read_file").source is ToolSource.MCP

    def test_rejects_fastpath_exposure(self) -> None:
        reg = ToolRegistry()
        with pytest.raises(ValueError, match="cannot be exposed to FastPath"):
            reg.register_mcp(
                "github", _tool("x"), mutates=False, tiers=frozenset({"fastpath", "fullagent"})
            )


class TestMCPVisibilityGating:
    def _registry(self) -> ToolRegistry:
        reg = ToolRegistry()
        reg.register_builtin(_tool("read_file"))
        reg.register_mcp("github", _tool("create_issue"), mutates=True)
        return reg

    def test_hidden_from_fastpath_even_when_enabled(self) -> None:
        reg = self._registry()
        names = [t.name for t in reg.visible("fastpath", mcp_servers=frozenset({"github"}))]
        assert names == ["read_file"]

    def test_hidden_when_server_not_enabled(self) -> None:
        reg = self._registry()
        assert [t.name for t in reg.visible("fullagent")] == ["read_file"]

    def test_visible_when_server_enabled(self) -> None:
        reg = self._registry()
        names = [t.name for t in reg.visible("fullagent", mcp_servers=frozenset({"github"}))]
        assert names == ["mcp__github__create_issue", "read_file"]

    def test_subagent_gets_nothing_without_explicit_grant(self) -> None:
        reg = self._registry()
        assert [t.name for t in reg.visible("subagent")] == ["read_file"]

    def test_allowed_tools_restricts_surface(self) -> None:
        reg = self._registry()
        visible = reg.visible(
            "fullagent",
            mcp_servers=frozenset({"github"}),
            allowed_tools=frozenset({"read_file"}),
        )
        assert [t.name for t in visible] == ["read_file"]


class TestStructuredOutputExclusivity:
    def test_output_spec_alone_is_fine(self) -> None:
        reg = ToolRegistry()
        reg.register_builtin(_tool(), tiers=frozenset({"fullagent"}))
        spec = OutputSpec(tool_name="emit", schema={"type": "object"})
        assert reg.specs_for("fastpath", output_spec=spec) == []

    def test_output_spec_with_visible_tools_raises(self) -> None:
        reg = ToolRegistry()
        reg.register_builtin(_tool())
        spec = OutputSpec(tool_name="emit", schema={"type": "object"})
        with pytest.raises(ValueError, match="cannot be combined with"):
            reg.specs_for("fastpath", output_spec=spec)

    def test_tools_without_output_spec_are_returned(self) -> None:
        reg = ToolRegistry()
        reg.register_builtin(_tool())
        assert [s.name for s in reg.specs_for("fastpath")] == ["read_file"]


class TestMCPServerConfig:
    def _stdio(self, **kw: object) -> MCPServerConfig:
        base: dict[str, object] = {
            "name": "fs",
            "transport": "stdio",
            "capabilities": frozenset({MCPCapability.READ}),
            "command": ["mcp-fs"],
        }
        return MCPServerConfig(**{**base, **kw})  # type: ignore[arg-type]

    def test_read_only_stdio_server_is_valid(self) -> None:
        assert not self._stdio().can_write

    def test_capabilities_must_be_declared(self) -> None:
        with pytest.raises(ValueError, match="capabilities must be declared"):
            self._stdio(capabilities=frozenset())

    def test_server_name_may_not_contain_separator(self) -> None:
        with pytest.raises(ValueError, match="may not contain"):
            self._stdio(name="bad__fs")

    def test_stdio_requires_command(self) -> None:
        with pytest.raises(ValueError, match="requires command"):
            self._stdio(command=None)

    def test_stdio_rejects_url(self) -> None:
        with pytest.raises(ValueError, match="must not set url"):
            self._stdio(url="https://x")

    def test_sse_requires_url(self) -> None:
        with pytest.raises(ValueError, match="requires url"):
            MCPServerConfig(
                name="remote",
                transport="sse",
                capabilities=frozenset({MCPCapability.READ}),
            )

    def test_sse_rejects_command(self) -> None:
        with pytest.raises(ValueError, match="must not set command"):
            MCPServerConfig(
                name="remote",
                transport="sse",
                url="https://x",
                command=["y"],
                capabilities=frozenset({MCPCapability.READ}),
            )

    def test_fastpath_cannot_be_enabled(self) -> None:
        with pytest.raises(ValueError, match="cannot be enabled on"):
            self._stdio(enabled_tiers=frozenset({"fastpath"}))

    def test_routing_cannot_be_enabled(self) -> None:
        with pytest.raises(ValueError, match="cannot be enabled on"):
            self._stdio(enabled_tiers=frozenset({"routing"}))

    def test_enabled_tiers_must_be_non_empty(self) -> None:
        with pytest.raises(ValueError, match="must be non-empty"):
            self._stdio(enabled_tiers=frozenset())

    def test_write_server_without_shadow_arg_is_rejected(self) -> None:
        """Unconfined writes would bypass path guards and the snapshot diff."""
        with pytest.raises(ValueError, match="must declare shadow_root_arg"):
            self._stdio(capabilities=frozenset({MCPCapability.WRITE}))

    def test_write_server_with_shadow_arg_is_admitted(self) -> None:
        cfg = self._stdio(capabilities=frozenset({MCPCapability.WRITE}), shadow_root_arg="--root")
        assert cfg.can_write

    def test_launch_command_confines_write_server_to_shadow(self) -> None:
        cfg = self._stdio(capabilities=frozenset({MCPCapability.WRITE}), shadow_root_arg="--root")
        assert cfg.launch_command("/tmp/shadow") == ["mcp-fs", "--root", "/tmp/shadow"]

    def test_launch_command_leaves_read_server_unchanged(self) -> None:
        assert self._stdio().launch_command("/tmp/shadow") == ["mcp-fs"]

    def test_launch_command_rejects_sse_server(self) -> None:
        cfg = MCPServerConfig(
            name="remote",
            transport="sse",
            url="https://x",
            capabilities=frozenset({MCPCapability.READ}),
        )
        with pytest.raises(ValueError, match="requires stdio"):
            cfg.launch_command("/tmp/shadow")

    def test_network_server_is_not_replay_safe(self) -> None:
        cfg = self._stdio(capabilities=frozenset({MCPCapability.NETWORK}))
        assert not cfg.replay_safe

    def test_local_server_is_replay_safe(self) -> None:
        assert self._stdio().replay_safe

    def test_allowlist_gates_tools(self) -> None:
        cfg = self._stdio(tool_allowlist=frozenset({"read_file"}))
        assert cfg.allows_tool("read_file")
        assert not cfg.allows_tool("delete_everything")

    def test_absent_allowlist_admits_all(self) -> None:
        assert self._stdio().allows_tool("anything")

    def test_rejects_non_positive_timeout(self) -> None:
        with pytest.raises(ValueError, match="startup_timeout must be positive"):
            self._stdio(startup_timeout=0)


class TestResultPolicy:
    def test_rejects_non_positive_limits(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            ResultPolicy(max_inline_chars=0)

    def test_rejects_excerpt_larger_than_budget(self) -> None:
        with pytest.raises(ValueError, match="must be\n?\\s*below max_inline_chars"):
            ResultPolicy(max_inline_chars=100, head_chars=80, tail_chars=40)


class TestToolResultCompression:
    def test_small_output_passes_through_verbatim(self) -> None:
        store = InMemoryArtifactStore()
        result = build_tool_result("tu_1", "short", store=store)
        assert result.content == "short"
        assert len(store) == 0

    def test_oversized_output_is_excerpted(self) -> None:
        store = InMemoryArtifactStore()
        policy = ResultPolicy(max_inline_chars=200, head_chars=50, tail_chars=30)
        result = build_tool_result("tu_1", "x" * 5000, store=store, policy=policy)

        assert len(result.content) < 5000
        assert "art_0001" in result.content
        assert "chars omitted" in result.content

    def test_full_output_is_recoverable_from_store(self) -> None:
        store = InMemoryArtifactStore()
        policy = ResultPolicy(max_inline_chars=200, head_chars=50, tail_chars=30)
        original = "y" * 5000
        build_tool_result("tu_1", original, store=store, policy=policy)
        assert store.get("art_0001") == original

    def test_excerpt_keeps_head_and_tail(self) -> None:
        store = InMemoryArtifactStore()
        policy = ResultPolicy(max_inline_chars=100, head_chars=20, tail_chars=20)
        content = "HEAD" + "m" * 500 + "TAIL"
        result = build_tool_result("tu_1", content, store=store, policy=policy)
        assert result.content.startswith("HEAD")
        assert result.content.endswith("TAIL")

    def test_error_flag_survives_compression(self) -> None:
        store = InMemoryArtifactStore()
        policy = ResultPolicy(max_inline_chars=100, head_chars=20, tail_chars=20)
        result = build_tool_result("tu_1", "e" * 500, store=store, policy=policy, is_error=True)
        assert result.is_error

    def test_tool_use_id_is_preserved(self) -> None:
        store = InMemoryArtifactStore()
        assert build_tool_result("tu_42", "ok", store=store).tool_use_id == "tu_42"

    def test_unknown_artifact_id_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown artifact id"):
            InMemoryArtifactStore().get("art_9999")

    def test_ids_are_distinct_across_store_entries(self) -> None:
        store = InMemoryArtifactStore()
        assert store.put("a") != store.put("b")


def test_all_tiers_excludes_routing() -> None:
    """Routing runs with a forced output schema and no tools."""
    assert "routing" not in ALL_TIERS
    assert sorted(ALL_TIERS) == ["fastpath", "fullagent", "subagent"]
