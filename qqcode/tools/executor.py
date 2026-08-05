"""Tool executor: maps tool calls to workspace operations.

Executes builtin tools against a workspace, enforces guards, and compresses
oversized results. Every tool returns a `ToolResultContent` ready for the model.

Spawn is handled via callback to avoid circular imports between agents and tools.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from qqcode.models.protocol import ToolResultContent, ToolUseContent
from qqcode.safety.guards import CommandGuard, PathGuard
from qqcode.skills.index import SkillIndex
from qqcode.tools.artifacts import (
    DEFAULT_RESULT_POLICY,
    ArtifactStore,
    ResultPolicy,
    build_tool_result,
)
from qqcode.tools.builtins import (
    MAX_COMMAND_TIMEOUT,
    TOOL_ARTIFACT,
    TOOL_EDIT,
    TOOL_FINISH,
    TOOL_GREP,
    TOOL_LIST,
    TOOL_READ,
    TOOL_RUN,
    TOOL_SKILL,
    TOOL_SPAWN,
    TOOL_WRITE,
)
from qqcode.tools.mcp_client import MCPClient
from qqcode.tools.registry import MCP_PREFIX
from qqcode.workspace.protocol import Workspace

SpawnCallback = Callable[[str, str], str]


class ToolExecutor:
    """Executes builtin tools with guard enforcement and result compression."""

    def __init__(
        self,
        workspace: Workspace,
        store: ArtifactStore,
        skill_index: SkillIndex,
        *,
        policy: ResultPolicy = DEFAULT_RESULT_POLICY,
        spawn_callback: SpawnCallback | None = None,
        mcp_client: MCPClient | None = None,
    ):
        self._workspace = workspace
        self._store = store
        self._skill_index = skill_index
        self._policy = policy
        self._spawn_callback = spawn_callback
        self._mcp_client = mcp_client
        self._path_guard = PathGuard(Path(workspace.root))
        self._command_guard = CommandGuard()
        self._files_touched: set[str] = set()

    @property
    def files_touched(self) -> frozenset[str]:
        """Repo-relative paths modified this run."""
        return frozenset(self._files_touched)

    def execute(self, call: ToolUseContent) -> ToolResultContent:
        """Execute one tool call, returning a compressed result."""
        try:
            text, is_error = self._dispatch(call)
        except Exception as exc:
            text = f"Internal error: {type(exc).__name__}: {exc}"
            is_error = True
        return build_tool_result(
            call.id, text, store=self._store, policy=self._policy, is_error=is_error
        )

    def _dispatch(self, call: ToolUseContent) -> tuple[str, bool]:
        # MCP tools are namespaced as mcp__<server>__<tool>
        if call.name.startswith(MCP_PREFIX):
            return self._call_mcp_tool(call.name, call.input)

        handlers = {
            TOOL_READ: self._read_file,
            TOOL_LIST: self._list_files,
            TOOL_GREP: self._grep,
            TOOL_WRITE: self._write_file,
            TOOL_EDIT: self._edit_file,
            TOOL_RUN: self._run_command,
            TOOL_ARTIFACT: self._read_artifact,
            TOOL_SKILL: self._read_skill,
            TOOL_FINISH: self._finish,
            TOOL_SPAWN: self._spawn_subagent,
        }
        handler = handlers.get(call.name)
        if handler is None:
            return f"Unknown tool: {call.name}", True
        return handler(call.input)

    def _read_file(self, args: dict[str, Any]) -> tuple[str, bool]:
        path = str(args.get("path", ""))
        if not path:
            return "Missing required argument: path", True
        try:
            self._path_guard.validate(path)
            content = self._workspace.read_file(path)
            return content, False
        except Exception as exc:
            return f"Cannot read {path}: {exc}", True

    def _list_files(self, args: dict[str, Any]) -> tuple[str, bool]:
        pattern = str(args.get("pattern", "*"))
        try:
            paths = sorted(self._workspace.list_files(pattern))
            return "\n".join(paths) if paths else "(no files matched)", False
        except Exception as exc:
            return f"list_files error: {exc}", True

    def _grep(self, args: dict[str, Any]) -> tuple[str, bool]:
        pattern_str = str(args.get("pattern", ""))
        glob_pat = str(args.get("glob", "*"))
        if not pattern_str:
            return "Missing required argument: pattern", True
        try:
            regex = re.compile(pattern_str)
        except re.error as exc:
            return f"Invalid regex: {exc}", True
        try:
            paths = self._workspace.list_files(glob_pat)
            matches: list[str] = []
            for path in paths:
                try:
                    content = self._workspace.read_file(path)
                    for i, line in enumerate(content.splitlines(), 1):
                        if regex.search(line):
                            matches.append(f"{path}:{i}: {line}")
                            if len(matches) >= 500:
                                matches.append("... (500+ matches, refine your pattern)")
                                return "\n".join(matches), False
                except Exception:
                    continue
            return "\n".join(matches) if matches else "(no matches)", False
        except Exception as exc:
            return f"grep error: {exc}", True

    def _write_file(self, args: dict[str, Any]) -> tuple[str, bool]:
        path = str(args.get("path", ""))
        content = str(args.get("content", ""))
        if not path:
            return "Missing required argument: path", True
        try:
            self._path_guard.validate(path)
            self._workspace.write_file(path, content)
            self._files_touched.add(path)
            return f"Wrote {len(content)} chars to {path}", False
        except Exception as exc:
            return f"Cannot write {path}: {exc}", True

    def _edit_file(self, args: dict[str, Any]) -> tuple[str, bool]:
        path = str(args.get("path", ""))
        old = str(args.get("old_string", ""))
        new = str(args.get("new_string", ""))
        if not path or not old:
            return "Missing required arguments: path, old_string", True
        try:
            self._path_guard.validate(path)
            content = self._workspace.read_file(path)
            count = content.count(old)
            if count == 0:
                return f"old_string not found in {path}", True
            if count > 1:
                return f"old_string appears {count} times in {path}, must be unique", True
            updated = content.replace(old, new, 1)
            self._workspace.write_file(path, updated)
            self._files_touched.add(path)
            return f"Edited {path}", False
        except Exception as exc:
            return f"Cannot edit {path}: {exc}", True

    def _run_command(self, args: dict[str, Any]) -> tuple[str, bool]:
        cmd = args.get("cmd")
        if not isinstance(cmd, list) or not cmd:
            return "cmd must be a non-empty array of strings", True
        timeout = min(float(args.get("timeout", 30.0)), MAX_COMMAND_TIMEOUT)
        try:
            argv = [str(c) for c in cmd]
            self._command_guard.validate(argv)
            exit_code, stdout, stderr = self._workspace.run_command(argv, timeout=timeout)
            out = f"exit={exit_code}\n{stdout}\n{stderr}".strip()
            return out, exit_code != 0
        except Exception as exc:
            return f"Command failed: {exc}", True

    def _read_artifact(self, args: dict[str, Any]) -> tuple[str, bool]:
        artifact_id = str(args.get("artifact_id", ""))
        if not artifact_id:
            return "Missing required argument: artifact_id", True
        try:
            return self._store.get(artifact_id), False
        except KeyError:
            return f"No artifact with id {artifact_id}", True

    def _read_skill(self, args: dict[str, Any]) -> tuple[str, bool]:
        name = str(args.get("name", ""))
        if not name:
            return "Missing required argument: name", True
        skill = self._skill_index._skills.get(name)  # noqa: SLF001
        if skill is None:
            available = sorted(self._skill_index._skills)  # noqa: SLF001
            return f"No skill named {name!r}. Available: {available}", True
        return skill.body, False

    def _finish(self, args: dict[str, Any]) -> tuple[str, bool]:
        summary = str(args.get("summary", ""))
        return f"Task marked complete: {summary}", False

    def _spawn_subagent(self, args: dict[str, Any]) -> tuple[str, bool]:
        preset = str(args.get("preset", ""))
        task = str(args.get("task", ""))
        if not preset or not task:
            return "Missing required arguments: preset, task", True
        if self._spawn_callback is None:
            return "spawn_subagent is not available in this context", True
        try:
            return self._spawn_callback(preset, task), False
        except Exception as exc:
            return f"Sub-agent failed: {exc}", True

    def _call_mcp_tool(self, namespaced_name: str, args: dict[str, Any]) -> tuple[str, bool]:
        """Route a tool call to its MCP server.

        Args:
            namespaced_name: Full registry name like mcp__filesystem__read_file.
            args: Tool input parameters.

        Returns:
            (output, is_error) tuple.
        """
        if self._mcp_client is None:
            return "MCP tools are not available in this context", True

        # Parse mcp__<server>__<tool>
        parts = namespaced_name.split("__")
        if len(parts) < 3:
            return f"Malformed MCP tool name: {namespaced_name}", True

        server = parts[1]
        tool = "__".join(parts[2:])  # Tool name may contain __ itself

        try:
            result = self._mcp_client.call_tool(server, tool, args)
            return result, False
        except Exception as exc:
            return f"MCP call failed: {exc}", True
