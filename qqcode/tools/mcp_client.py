"""MCP client: lazy connection management and tool delegation.

Connects to MCP servers via stdio or SSE, fetches tool schemas on first access,
and routes tool calls through the active session. Crashes are isolated: a server
that dies mid-session returns an error for that call without killing the agent.

Design constraints:
- Lazy startup: servers are launched on first tool invocation, not at
  orchestrator boot. A task that routes to FastPath pays zero MCP cost.
- Write confinement: write-capable servers receive their `shadow_root_arg` so
  their file operations stay within the workspace snapshot boundary.
- Crash isolation: a server crash surfaces as a tool error, not a process exit.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from typing import Any

from qqcode.models.protocol import ToolSpec
from qqcode.tools.mcp import MCPServerConfig


@dataclass
class MCPSession:
    """Active connection to one MCP server."""

    config: MCPServerConfig
    process: subprocess.Popen[bytes] | None
    tools: dict[str, ToolSpec]
    started_at: float

    @property
    def alive(self) -> bool:
        """Whether the server process is still running."""
        if self.process is None:
            return True  # SSE transport has no managed process
        return self.process.poll() is None


class MCPClient:
    """Manages connections to MCP servers and routes tool calls."""

    def __init__(self, shadow_root: str | None = None):
        """
        Args:
            shadow_root: Workspace root passed to write-capable stdio servers.
        """
        self._shadow_root = shadow_root
        self._sessions: dict[str, MCPSession] = {}
        self._tool_to_server: dict[str, str] = {}

    def register_server(self, config: MCPServerConfig) -> list[ToolSpec]:
        """Lazily start a server and fetch its tool schemas.

        Called by the registry on first access for this server. The server is
        launched if stdio, or the SSE endpoint is probed for its schema.

        Args:
            config: Server configuration validated at construction.

        Returns:
            Tool schemas reported by the server.

        Raises:
            RuntimeError: Server failed to start or timeout elapsed.
        """
        if config.name in self._sessions:
            return list(self._sessions[config.name].tools.values())

        if config.transport == "stdio":
            return self._start_stdio(config)
        return self._fetch_sse_schema(config)

    def _start_stdio(self, config: MCPServerConfig) -> list[ToolSpec]:
        """Launch a stdio server and read its tool list."""
        if not config.command:
            raise RuntimeError(f"{config.name}: stdio transport requires command")

        cmd = config.launch_command(self._shadow_root or "")
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=False,
            )
        except Exception as exc:
            raise RuntimeError(f"{config.name}: failed to launch: {exc}") from exc

        # Wait for initialization message
        deadline = time.monotonic() + config.startup_timeout
        init_msg = None
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                _, stderr = proc.communicate(timeout=1.0)
                raise RuntimeError(
                    f"{config.name}: process exited during startup: "
                    f"{stderr.decode('utf-8', errors='replace')}"
                )
            # Non-blocking read attempt
            try:
                line = proc.stdout.readline() if proc.stdout else b""
                if line:
                    init_msg = json.loads(line.decode("utf-8"))
                    break
            except (json.JSONDecodeError, UnicodeDecodeError):
                time.sleep(0.05)
                continue

        if init_msg is None:
            proc.kill()
            raise RuntimeError(f"{config.name}: no init message within {config.startup_timeout}s")

        # Extract tools from init message (MCP protocol structure)
        tools_data = init_msg.get("tools", [])
        tools: dict[str, ToolSpec] = {}
        for t in tools_data:
            name = t.get("name", "")
            if not name or not config.allows_tool(name):
                continue
            tools[name] = ToolSpec(
                name=name,
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {}),
            )

        session = MCPSession(
            config=config, process=proc, tools=tools, started_at=time.monotonic()
        )
        self._sessions[config.name] = session

        for tool_name in tools:
            self._tool_to_server[tool_name] = config.name

        return list(tools.values())

    def _fetch_sse_schema(self, config: MCPServerConfig) -> list[ToolSpec]:
        """Fetch tool schemas from an SSE server (not yet implemented)."""
        raise NotImplementedError(f"{config.name}: SSE transport not yet implemented")

    def call_tool(self, server: str, tool: str, args: dict[str, Any]) -> str:
        """Execute a tool call against the named server.

        Args:
            server: Server name from MCPServerConfig.
            tool: Tool name as reported by the server.
            args: JSON-serializable input parameters.

        Returns:
            Tool output as a string.

        Raises:
            RuntimeError: Server not registered, died, or returned an error.
        """
        session = self._sessions.get(server)
        if session is None:
            raise RuntimeError(f"MCP server {server!r} not registered")

        if not session.alive:
            raise RuntimeError(
                f"MCP server {server!r} died; restart required "
                f"(ran for {time.monotonic() - session.started_at:.1f}s)"
            )

        if session.config.transport == "stdio":
            return self._call_stdio(session, tool, args)
        return self._call_sse(session, tool, args)

    def _call_stdio(self, session: MCPSession, tool: str, args: dict[str, Any]) -> str:
        """Send a tool call over stdio and read the response."""
        if session.process is None or session.process.stdin is None:
            raise RuntimeError(f"{session.config.name}: no stdin handle")

        request = json.dumps({"tool": tool, "arguments": args}) + "\n"
        try:
            session.process.stdin.write(request.encode("utf-8"))
            session.process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError(f"{session.config.name}: stdin write failed: {exc}") from exc

        # Read response line
        if session.process.stdout is None:
            raise RuntimeError(f"{session.config.name}: no stdout handle")

        try:
            line = session.process.stdout.readline()
            response = json.loads(line.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError(f"{session.config.name}: invalid response: {exc}") from exc

        if "error" in response:
            raise RuntimeError(f"{session.config.name}.{tool}: {response['error']}")

        return str(response.get("content", ""))

    def _call_sse(self, session: MCPSession, tool: str, args: dict[str, Any]) -> str:
        """Call a tool over SSE (not yet implemented)."""
        raise NotImplementedError(f"{session.config.name}: SSE calls not yet implemented")

    def shutdown_all(self) -> None:
        """Terminate all stdio servers gracefully."""
        for session in self._sessions.values():
            if session.process is not None:
                try:
                    session.process.terminate()
                    session.process.wait(timeout=2.0)
                except Exception:
                    session.process.kill()
        self._sessions.clear()
        self._tool_to_server.clear()
