"""Tool registry: single source of truth for tool schemas.

One JSON Schema per tool, registered once. Provider adapters wrap these into
Anthropic or OpenAI shapes; nothing else defines a tool.

Two rules the registry enforces rather than documents:

- MCP tools are namespaced `mcp__<server>__<tool>`. Without the prefix a
  third-party server exposing `read_file` would quietly shadow the builtin and
  route reads around PathGuard.
- Structured output and real tools are mutually exclusive. Forcing structured
  output is implemented as a forced single tool call, so any other visible tool
  makes the request ill-formed. Asking for both raises instead of silently
  dropping one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from qqcode.models.protocol import ALL_TIERS, OutputSpec, Tier, ToolSpec

MCP_PREFIX = "mcp__"


class ToolSource(StrEnum):
    """Where a tool's implementation lives."""

    BUILTIN = "builtin"
    MCP = "mcp"


def mcp_tool_name(server: str, tool: str) -> str:
    """Build the namespaced registry name for an MCP tool."""
    if "__" in server:
        raise ValueError(f"MCP server name may not contain '__': {server!r}")
    return f"{MCP_PREFIX}{server}__{tool}"


@dataclass(frozen=True)
class RegisteredTool:
    """A tool plus the metadata the registry gates on."""

    spec: ToolSpec
    source: ToolSource
    mutates: bool
    tiers: frozenset[str]
    server: str | None = None

    @property
    def name(self) -> str:
        return self.spec.name


class ToolRegistry:
    """Registry of every tool the agent can reach."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register_builtin(
        self,
        spec: ToolSpec,
        *,
        mutates: bool = False,
        tiers: frozenset[str] = ALL_TIERS,
    ) -> None:
        """Register a first-party tool under its bare name.

        Raises:
            ValueError: Name collides, claims the MCP namespace, or names an
                unknown tier.
        """
        if spec.name.startswith(MCP_PREFIX):
            raise ValueError(f"Builtin tool may not use the MCP namespace: {spec.name}")
        self._insert(
            RegisteredTool(
                spec=spec, source=ToolSource.BUILTIN, mutates=mutates, tiers=tiers, server=None
            )
        )

    def register_mcp(
        self,
        server: str,
        spec: ToolSpec,
        *,
        mutates: bool,
        tiers: frozenset[str] = frozenset({"fullagent"}),
    ) -> str:
        """Register an MCP tool, prefixing its name with the server namespace.

        Args:
            server: Server name from its MCPServerConfig.
            spec: Tool schema as reported by the server.
            mutates: Whether the tool writes; drives write-budget accounting.
            tiers: Surfaces the tool is visible on. FastPath is rejected.

        Returns:
            The namespaced registry name.

        Raises:
            ValueError: Name collides, FastPath is requested, or a tier is unknown.
        """
        if "fastpath" in tiers:
            raise ValueError(
                f"{server}: MCP tools cannot be exposed to FastPath; their schemas "
                "exceed the entire FastPath budget"
            )
        namespaced = mcp_tool_name(server, spec.name)
        self._insert(
            RegisteredTool(
                spec=ToolSpec(
                    name=namespaced,
                    description=spec.description,
                    input_schema=spec.input_schema,
                ),
                source=ToolSource.MCP,
                mutates=mutates,
                tiers=tiers,
                server=server,
            )
        )
        return namespaced

    def _insert(self, tool: RegisteredTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        if not tool.tiers:
            raise ValueError(f"{tool.name}: tiers must be non-empty")
        unknown = tool.tiers - ALL_TIERS
        if unknown:
            raise ValueError(f"{tool.name}: unknown tiers {sorted(unknown)}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> RegisteredTool:
        """Look up a registered tool.

        Raises:
            KeyError: Not registered.
        """
        try:
            return self._tools[name]
        except KeyError:
            raise KeyError(f"Unknown tool: {name}") from None

    def names(self) -> list[str]:
        """Every registered tool name, sorted."""
        return sorted(self._tools)

    def visible(
        self,
        tier: Tier,
        *,
        mcp_servers: frozenset[str] = frozenset(),
        allowed_tools: frozenset[str] | None = None,
    ) -> list[RegisteredTool]:
        """Tools reachable on `tier`, after MCP and per-agent gating.

        Args:
            tier: Execution surface being assembled.
            mcp_servers: Servers explicitly enabled for this call. MCP tools
                from unlisted servers stay hidden, and all MCP stays hidden on
                FastPath regardless of this argument.
            allowed_tools: Restrict to these names (a sub-agent's tool surface).
                None means no per-agent restriction.
        """
        out = []
        for name in sorted(self._tools):
            tool = self._tools[name]
            if tier not in tool.tiers:
                continue
            if tool.source is ToolSource.MCP:
                if tier == "fastpath":
                    continue
                if tool.server not in mcp_servers:
                    continue
            if allowed_tools is not None and name not in allowed_tools:
                continue
            out.append(tool)
        return out

    def specs_for(
        self,
        tier: Tier,
        *,
        mcp_servers: frozenset[str] = frozenset(),
        allowed_tools: frozenset[str] | None = None,
        output_spec: OutputSpec | None = None,
    ) -> list[ToolSpec]:
        """Tool schemas to send with a request on `tier`.

        Raises:
            ValueError: `output_spec` is set while real tools would be visible.
        """
        tools = self.visible(tier, mcp_servers=mcp_servers, allowed_tools=allowed_tools)
        if output_spec is not None and tools:
            raise ValueError(
                f"Structured output ({output_spec.tool_name}) cannot be combined with "
                f"real tools {[t.name for t in tools]}; forcing structured output "
                "consumes the single permitted tool call"
            )
        return [t.spec for t in tools]
