"""MCP server configuration.

Configuration and admission rules only; the client implementation lands with
M4.5. The rules encoded here protect the core guarantees:

1. MCP is invisible to FastPath. A single server's schemas (~6-9k tokens) cost
   more than FastPath's entire budget, so `enabled_tiers` may not name it.

2. A write-capable server bypasses shadow isolation unless it can be pointed at
   the shadow root. Its file handles are real paths, so PathGuard, WriteQuota
   and the snapshot diff never see the write — silently breaking the
   "no unexpected modifications" condition. Such a server must declare the
   argument used to confine it, or it is not admitted.

3. Network-capable servers make a run unreproducible, so tasks that touch one
   are excluded from the routing-trace replay dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

# Tiers an MCP server may be enabled on. FastPath is deliberately absent.
MCP_ELIGIBLE_TIERS = frozenset({"fullagent", "subagent"})


class MCPCapability(StrEnum):
    """What a server is allowed to do. Must be declared explicitly."""

    READ = "read"
    WRITE = "write"
    NETWORK = "network"


@dataclass(frozen=True)
class MCPServerConfig:
    """Admission record for one MCP server.

    `capabilities` has no default: an operator states what the server can do
    before it runs.
    """

    name: str
    transport: Literal["stdio", "sse"]
    capabilities: frozenset[MCPCapability]
    command: list[str] | None = None
    url: str | None = None
    # None means every tool the server exposes; otherwise an explicit allowlist.
    tool_allowlist: frozenset[str] | None = None
    enabled_tiers: frozenset[str] = frozenset({"fullagent"})
    # CLI argument used to confine a write-capable server to the shadow root,
    # e.g. "--root". Required when WRITE is declared.
    shadow_root_arg: str | None = None
    startup_timeout: float = 15.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("MCPServerConfig.name must be non-empty")
        # The registry encodes MCP tools as mcp__<server>__<tool>; a server name
        # containing the separator would make that encoding ambiguous.
        if "__" in self.name:
            raise ValueError(f"{self.name}: server name may not contain '__'")

        if not self.capabilities:
            raise ValueError(f"{self.name}: capabilities must be declared explicitly")

        if self.transport == "stdio":
            if not self.command:
                raise ValueError(f"{self.name}: stdio transport requires command")
            if self.url is not None:
                raise ValueError(f"{self.name}: stdio transport must not set url")
        else:
            if not self.url:
                raise ValueError(f"{self.name}: sse transport requires url")
            if self.command is not None:
                raise ValueError(f"{self.name}: sse transport must not set command")

        if not self.enabled_tiers:
            raise ValueError(f"{self.name}: enabled_tiers must be non-empty")
        ineligible = self.enabled_tiers - MCP_ELIGIBLE_TIERS
        if ineligible:
            raise ValueError(
                f"{self.name}: MCP cannot be enabled on {sorted(ineligible)}; "
                f"eligible tiers are {sorted(MCP_ELIGIBLE_TIERS)}"
            )

        if MCPCapability.WRITE in self.capabilities and not self.shadow_root_arg:
            raise ValueError(
                f"{self.name}: write-capable server must declare shadow_root_arg so it "
                "can be confined to the shadow workspace; otherwise its writes bypass "
                "path guards and the snapshot diff"
            )

        if self.startup_timeout <= 0:
            raise ValueError(f"{self.name}: startup_timeout must be positive")

    @property
    def can_write(self) -> bool:
        return MCPCapability.WRITE in self.capabilities

    @property
    def replay_safe(self) -> bool:
        """Whether runs using this server can enter the offline replay dataset."""
        return MCPCapability.NETWORK not in self.capabilities

    def allows_tool(self, tool: str) -> bool:
        """Whether the server's allowlist admits this tool name."""
        return self.tool_allowlist is None or tool in self.tool_allowlist

    def launch_command(self, shadow_root: str) -> list[str]:
        """Full stdio command, confined to the shadow root when write-capable.

        Raises:
            ValueError: Called on a non-stdio server.
        """
        if self.transport != "stdio" or not self.command:
            raise ValueError(f"{self.name}: launch_command requires stdio transport")
        if self.can_write and self.shadow_root_arg:
            return [*self.command, self.shadow_root_arg, shadow_root]
        return list(self.command)
