"""Builtin tool schemas: the wire contract the model sees.

One `ToolSpec` per capability. Implementations live in
`qqcode.tools.executor`; this module holds only names, schemas, and tier gating.

Tool names are defined here rather than in the agent layer so the registry and
sub-agent specs reference a single definition instead of drifting copies.

No builtin includes `"fastpath"` in its tiers, and that is deliberate rather
than incidental: FastPath runs on forced structured output, and
`ToolRegistry.specs_for` raises when an output spec is combined with visible
tools. A tool leaking onto the FastPath surface would break every FastPath call.
"""

from __future__ import annotations

from qqcode.models.protocol import ToolSpec
from qqcode.tools.registry import ToolRegistry

# --- Tool names ----------------------------------------------------------------

TOOL_READ = "read_file"
TOOL_LIST = "list_files"
TOOL_GREP = "grep"
TOOL_WRITE = "write_file"
TOOL_EDIT = "edit_file"
TOOL_RUN = "run_command"
TOOL_ARTIFACT = "read_artifact"
TOOL_SKILL = "read_skill"
TOOL_FINISH = "finish"
TOOL_SPAWN = "spawn_subagent"

READ_ONLY_TOOLS = frozenset({TOOL_READ, TOOL_LIST, TOOL_GREP})
WRITE_TOOLS = frozenset({TOOL_WRITE, TOOL_EDIT})
ALL_TOOLS = READ_ONLY_TOOLS | WRITE_TOOLS | {TOOL_RUN}

# Surfaces. FastPath is absent from both — see the module docstring.
AGENT_TIERS = frozenset({"fullagent", "subagent"})
FULLAGENT_ONLY = frozenset({"fullagent"})

# A command may not run longer than this regardless of what the model asks for.
MAX_COMMAND_TIMEOUT = 120.0


# --- Schemas -------------------------------------------------------------------

READ_FILE = ToolSpec(
    name=TOOL_READ,
    description=(
        "Read a file's contents. Path is relative to the repo root. "
        "Prefer grep to locate the region you need before reading a large file."
    ),
    input_schema={
        "type": "object",
        "required": ["path"],
        "properties": {"path": {"type": "string", "description": "Repo-relative file path"}},
    },
)

LIST_FILES = ToolSpec(
    name=TOOL_LIST,
    description="List files matching a glob pattern, relative to the repo root.",
    input_schema={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern, e.g. '**/*.py'. Defaults to all files.",
            }
        },
    },
)

GREP = ToolSpec(
    name=TOOL_GREP,
    description=(
        "Search file contents with a regular expression. Returns matching lines "
        "prefixed by path and line number."
    ),
    input_schema={
        "type": "object",
        "required": ["pattern"],
        "properties": {
            "pattern": {"type": "string", "description": "Python regular expression"},
            "glob": {
                "type": "string",
                "description": "Restrict to files matching this glob, e.g. '**/*.py'",
            },
        },
    },
)

WRITE_FILE = ToolSpec(
    name=TOOL_WRITE,
    description=(
        "Write a file, replacing it entirely and creating parent directories as "
        "needed. Use edit_file for a targeted change to an existing file."
    ),
    input_schema={
        "type": "object",
        "required": ["path", "content"],
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string", "description": "Complete new file contents"},
        },
    },
)

EDIT_FILE = ToolSpec(
    name=TOOL_EDIT,
    description=(
        "Replace an exact string in a file. old_string must appear exactly once, "
        "so include surrounding context when the target text is not unique."
    ),
    input_schema={
        "type": "object",
        "required": ["path", "old_string", "new_string"],
        "properties": {
            "path": {"type": "string"},
            "old_string": {"type": "string", "description": "Exact text to replace, must be unique"},
            "new_string": {"type": "string", "description": "Replacement text"},
        },
    },
)

RUN_COMMAND = ToolSpec(
    name=TOOL_RUN,
    description=(
        "Run a command in the workspace. Pass argv as an array. Network tools "
        "and shell interpreters are blocked; run test and build binaries directly."
    ),
    input_schema={
        "type": "object",
        "required": ["cmd"],
        "properties": {
            "cmd": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Argv array, e.g. ['pytest', '-q', 'tests/']",
            },
            "timeout": {
                "type": "number",
                "description": f"Seconds to wait, capped at {MAX_COMMAND_TIMEOUT}",
            },
        },
    },
)

READ_ARTIFACT = ToolSpec(
    name=TOOL_ARTIFACT,
    description=(
        "Read the full output of an earlier tool call that was truncated. "
        "Use the artifact id from the truncation notice."
    ),
    input_schema={
        "type": "object",
        "required": ["artifact_id"],
        "properties": {"artifact_id": {"type": "string"}},
    },
)

READ_SKILL = ToolSpec(
    name=TOOL_SKILL,
    description=(
        "Read a skill's full instructions by name. The available skills are "
        "listed in your context; load one when its subject matches your task."
    ),
    input_schema={
        "type": "object",
        "required": ["name"],
        "properties": {"name": {"type": "string"}},
    },
)

FINISH = ToolSpec(
    name=TOOL_FINISH,
    description=(
        "Declare the task complete. Call this only when the work is done and "
        "verified — it is the signal that your run finished validly rather than "
        "running out of turns."
    ),
    input_schema={
        "type": "object",
        "required": ["summary"],
        "properties": {
            "summary": {"type": "string", "description": "What you changed and why"},
            "files_changed": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Repo-relative paths you modified",
            },
        },
    },
)

SPAWN_SUBAGENT = ToolSpec(
    name=TOOL_SPAWN,
    description=(
        "Delegate a bounded piece of work to a sub-agent. The sub-agent runs in "
        "its own context and returns only its conclusion, so use this for "
        "exploration or review that would otherwise fill your context with "
        "intermediate output."
    ),
    input_schema={
        "type": "object",
        "required": ["preset", "task"],
        "properties": {
            "preset": {
                "type": "string",
                "description": "Working mode, e.g. explorer, reviewer, planner, test-writer",
            },
            "task": {
                "type": "string",
                "description": "Self-contained instruction; the sub-agent sees none of your context",
            },
        },
    },
)


# Every builtin with the surfaces it appears on.
BUILTIN_TOOLS: tuple[tuple[ToolSpec, bool, frozenset[str]], ...] = (
    # (spec, mutates, tiers)
    (READ_FILE, False, AGENT_TIERS),
    (LIST_FILES, False, AGENT_TIERS),
    (GREP, False, AGENT_TIERS),
    (READ_ARTIFACT, False, AGENT_TIERS),
    (READ_SKILL, False, AGENT_TIERS),
    (WRITE_FILE, True, AGENT_TIERS),
    (EDIT_FILE, True, AGENT_TIERS),
    (RUN_COMMAND, False, AGENT_TIERS),
    (FINISH, False, AGENT_TIERS),
    (SPAWN_SUBAGENT, False, FULLAGENT_ONLY),
)


def register_builtins(registry: ToolRegistry) -> None:
    """Register every builtin tool into `registry`.

    Raises:
        ValueError: A name is already registered.
    """
    for spec, mutates, tiers in BUILTIN_TOOLS:
        registry.register_builtin(spec, mutates=mutates, tiers=tiers)


def default_registry() -> ToolRegistry:
    """A registry populated with the builtins and nothing else."""
    registry = ToolRegistry()
    register_builtins(registry)
    return registry
