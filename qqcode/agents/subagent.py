"""Sub-agent specifications and built-in working modes.

A sub-agent is a bounded, single-purpose worker the Full Agent spawns to keep
expensive exploration out of the main context. It runs as an isolated subgraph:
the parent sends a task, the sub-agent burns its own turns, and only the final
structured result returns. Intermediate tool output never enters parent context.

Two dials define a sub-agent's blast radius:
- `isolation`  — whether it may write at all
- `allowed_tools` — the exact tool surface it can reach

Built-in presets cover the common shapes (explore, review, test, fix, plan).
Register custom specs with `register_preset` for project-specific workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from qqcode.models.protocol import ModelTier

__all__ = [
    "ALL_TOOLS",
    "READ_ONLY_TOOLS",
    "WRITE_TOOLS",
    "Isolation",
    "ModelTier",
    "SubAgentResult",
    "SubAgentSpec",
    "get_preset",
    "list_presets",
    "register_preset",
]

# Tool names available to sub-agents. The Full Agent's tool registry (M4)
# resolves these to callables; specs only reference them by name.
TOOL_READ = "read_file"
TOOL_LIST = "list_files"
TOOL_GREP = "grep"
TOOL_WRITE = "write_file"
TOOL_EDIT = "edit_file"
TOOL_RUN = "run_command"

READ_ONLY_TOOLS = frozenset({TOOL_READ, TOOL_LIST, TOOL_GREP})
WRITE_TOOLS = frozenset({TOOL_WRITE, TOOL_EDIT})
ALL_TOOLS = READ_ONLY_TOOLS | WRITE_TOOLS | {TOOL_RUN}


class Isolation(StrEnum):
    """Write permission for a sub-agent."""

    READ_ONLY = "read_only"  # Cannot mutate the workspace at all
    SHADOW_WRITE = "shadow_write"  # May write inside the shadow workspace


@dataclass(frozen=True)
class SubAgentSpec:
    """Definition of a sub-agent working mode.

    Frozen so a preset cannot be mutated by one caller and observed changed by
    another. Use `derive` to build a variant.
    """

    name: str
    description: str
    system_prompt: str
    model_tier: ModelTier = ModelTier.BALANCED
    isolation: Isolation = Isolation.READ_ONLY
    allowed_tools: frozenset[str] = READ_ONLY_TOOLS
    max_turns: int = 12
    # JSON Schema forcing structured output; None means free-form text.
    output_schema: dict[str, Any] | None = None
    # MCP servers this sub-agent may reach. Empty by default and never
    # inherited from the parent: carrying a parent's MCP schemas would cancel
    # out the context isolation that justifies spawning a sub-agent.
    mcp_servers: frozenset[str] = frozenset()
    # Skill names whose bodies load unconditionally. Sub-agents get no resident
    # skill index, so anything they need to know must be pinned here.
    pinned_skills: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("SubAgentSpec.name must be non-empty")
        if self.max_turns < 1:
            raise ValueError(f"{self.name}: max_turns must be >= 1, got {self.max_turns}")

        unknown = self.allowed_tools - ALL_TOOLS
        if unknown:
            raise ValueError(f"{self.name}: unknown tools {sorted(unknown)}")

        if self.isolation is Isolation.READ_ONLY:
            writers = self.allowed_tools & WRITE_TOOLS
            if writers:
                raise ValueError(
                    f"{self.name}: read_only spec cannot grant write tools {sorted(writers)}"
                )

        if len(set(self.pinned_skills)) != len(self.pinned_skills):
            raise ValueError(f"{self.name}: pinned_skills contains duplicates")

    def derive(self, **overrides: Any) -> SubAgentSpec:
        """Return a copy with fields replaced, re-running validation."""
        current = {
            "name": self.name,
            "description": self.description,
            "system_prompt": self.system_prompt,
            "model_tier": self.model_tier,
            "isolation": self.isolation,
            "allowed_tools": self.allowed_tools,
            "max_turns": self.max_turns,
            "output_schema": self.output_schema,
            "mcp_servers": self.mcp_servers,
            "pinned_skills": self.pinned_skills,
        }
        return SubAgentSpec(**{**current, **overrides})


@dataclass
class SubAgentResult:
    """What a sub-agent returns to its parent.

    Deliberately narrow: the parent sees the conclusion and cost, never the
    intermediate tool traffic that produced it.
    """

    spec_name: str
    success: bool
    output: str | dict[str, Any]
    turns_used: int
    tokens_used: int
    files_touched: list[str] = field(default_factory=list)
    error: str | None = None


# --- Structured output schemas -------------------------------------------------

FINDINGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "line": {"type": "integer"},
                    "severity": {"enum": ["critical", "high", "medium", "low"]},
                    "issue": {"type": "string"},
                    "suggestion": {"type": "string"},
                },
                "required": ["file", "severity", "issue"],
            },
        },
        "summary": {"type": "string"},
    },
    "required": ["findings", "summary"],
}

CODE_MAP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entry_points": {"type": "array", "items": {"type": "string"}},
        "relevant_files": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "role": {"type": "string"},
                    "key_symbols": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["path", "role"],
            },
        },
        "call_flow": {"type": "string"},
        "conventions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["relevant_files", "call_flow"],
}

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "order": {"type": "integer"},
                    "action": {"type": "string"},
                    "files": {"type": "array", "items": {"type": "string"}},
                    "rationale": {"type": "string"},
                },
                "required": ["order", "action", "files"],
            },
        },
        "risks": {"type": "array", "items": {"type": "string"}},
        "verification": {"type": "string"},
    },
    "required": ["steps", "verification"],
}


# --- Built-in presets ----------------------------------------------------------

EXPLORER = SubAgentSpec(
    name="explorer",
    description="定位实现位置、理解调用链，返回结构化代码地图",
    model_tier=ModelTier.FAST,
    isolation=Isolation.READ_ONLY,
    allowed_tools=READ_ONLY_TOOLS,
    max_turns=15,
    output_schema=CODE_MAP_SCHEMA,
    system_prompt=(
        "You map unfamiliar code. Given a question about where something lives or how "
        "a flow works, search the repository and return a structured map.\n\n"
        "Search broadly before reading deeply — grep for symbols and filenames first, "
        "then read only the regions that matter. Do not read entire large files.\n\n"
        "Report what the code actually does, not what its names suggest. If you cannot "
        "locate something, say so explicitly rather than guessing at a plausible path."
    ),
)

REVIEWER = SubAgentSpec(
    name="reviewer",
    description="审查代码正确性、安全性与可维护性，返回分级 findings",
    model_tier=ModelTier.BALANCED,
    isolation=Isolation.READ_ONLY,
    allowed_tools=READ_ONLY_TOOLS,
    max_turns=12,
    output_schema=FINDINGS_SCHEMA,
    system_prompt=(
        "You review code for defects. Report only issues you can demonstrate with a "
        "concrete failure scenario: specific inputs or state leading to a wrong result, "
        "crash, or security exposure.\n\n"
        "Severity is about consequence, not style. Data loss and security holes are "
        "critical; naming preferences are not findings at all.\n\n"
        "Read enough surrounding code to confirm a suspected issue is real before "
        "reporting it. An unverified guess costs more than a missed nitpick."
    ),
)

SECURITY_AUDITOR = SubAgentSpec(
    name="security-auditor",
    description="安全专项审查：注入、认证、密钥、路径遍历、不安全反序列化",
    model_tier=ModelTier.DEEP,
    isolation=Isolation.READ_ONLY,
    allowed_tools=READ_ONLY_TOOLS,
    max_turns=15,
    output_schema=FINDINGS_SCHEMA,
    system_prompt=(
        "You audit code for security vulnerabilities. Focus on: injection (SQL, command, "
        "template), authentication and authorization gaps, hardcoded credentials, path "
        "traversal, unsafe deserialization, SSRF, and missing input validation at trust "
        "boundaries.\n\n"
        "For each finding, trace the path from attacker-controlled input to the dangerous "
        "sink. If you cannot trace that path, it is not a finding.\n\n"
        "Note where existing controls already mitigate a theoretical issue — a defended "
        "pattern reported as a vulnerability wastes the reader's time."
    ),
)

PLANNER = SubAgentSpec(
    name="planner",
    description="产出有序实现计划，标注涉及文件、风险与验证方式",
    model_tier=ModelTier.DEEP,
    isolation=Isolation.READ_ONLY,
    allowed_tools=READ_ONLY_TOOLS,
    max_turns=12,
    output_schema=PLAN_SCHEMA,
    system_prompt=(
        "You turn a task into an ordered implementation plan. Read the code first — a "
        "plan that names files that do not exist is worse than no plan.\n\n"
        "Each step states a concrete action and the exact files it touches. Order steps "
        "so each one leaves the codebase in a working state where possible.\n\n"
        "State how the result will be verified. Flag risks that could invalidate the "
        "plan, not generic cautions."
    ),
)

TEST_WRITER = SubAgentSpec(
    name="test-writer",
    description="为指定代码补测试，覆盖正常路径、边界与错误分支",
    model_tier=ModelTier.BALANCED,
    isolation=Isolation.SHADOW_WRITE,
    allowed_tools=READ_ONLY_TOOLS | WRITE_TOOLS | {TOOL_RUN},
    max_turns=20,
    system_prompt=(
        "You write tests for existing code. Match the project's existing test framework, "
        "file layout, and naming — read a neighbouring test file before writing.\n\n"
        "Cover the normal path, boundaries, and error branches. Each test asserts one "
        "behaviour and names it in the test title.\n\n"
        "Run the tests you write. A test that fails against correct code is a bug in the "
        "test; a test that passes against broken code is worthless. If a test reveals a "
        "real defect in the code under test, report it rather than weakening the test."
    ),
)

BUILD_FIXER = SubAgentSpec(
    name="build-fixer",
    description="修复构建/类型/lint 错误，最小 diff，不做架构改动",
    model_tier=ModelTier.FAST,
    isolation=Isolation.SHADOW_WRITE,
    allowed_tools=READ_ONLY_TOOLS | WRITE_TOOLS | {TOOL_RUN},
    max_turns=20,
    system_prompt=(
        "You fix build, type, and lint errors with the smallest change that resolves "
        "them. No refactoring, no architectural edits, no drive-by improvements.\n\n"
        "Work one error at a time: read the message, read the code it points at, fix it, "
        "re-run the build. Errors often cascade — the first fix may clear several.\n\n"
        "If an error signals a real design problem rather than a mechanical slip, fix "
        "the build minimally and report the underlying issue instead of silencing it "
        "with a cast or an ignore directive."
    ),
)

DOC_WRITER = SubAgentSpec(
    name="doc-writer",
    description="撰写或更新文档，与代码实际行为保持一致",
    model_tier=ModelTier.FAST,
    isolation=Isolation.SHADOW_WRITE,
    allowed_tools=READ_ONLY_TOOLS | WRITE_TOOLS,
    max_turns=12,
    system_prompt=(
        "You write documentation that matches what the code actually does. Read the "
        "implementation before describing it.\n\n"
        "Match the surrounding docs in tone, structure, and level of detail. Document "
        "the contract — arguments, return values, raised errors, side effects — not the "
        "line-by-line implementation.\n\n"
        "Do not document intentions you cannot verify in code. If behaviour looks like a "
        "bug, report it rather than documenting the bug as the contract."
    ),
)


_PRESETS: dict[str, SubAgentSpec] = {
    spec.name: spec
    for spec in (
        EXPLORER,
        REVIEWER,
        SECURITY_AUDITOR,
        PLANNER,
        TEST_WRITER,
        BUILD_FIXER,
        DOC_WRITER,
    )
}


def get_preset(name: str) -> SubAgentSpec:
    """Look up a built-in or registered sub-agent spec.

    Raises:
        KeyError: No spec registered under that name.
    """
    try:
        return _PRESETS[name]
    except KeyError:
        raise KeyError(
            f"Unknown sub-agent preset {name!r}. Available: {sorted(_PRESETS)}"
        ) from None


def list_presets() -> list[SubAgentSpec]:
    """All registered specs, ordered by name."""
    return [_PRESETS[n] for n in sorted(_PRESETS)]


def register_preset(spec: SubAgentSpec, *, overwrite: bool = False) -> None:
    """Add a custom sub-agent spec to the registry.

    Raises:
        ValueError: Name already taken and overwrite is False.
    """
    if spec.name in _PRESETS and not overwrite:
        raise ValueError(f"Preset {spec.name!r} already registered; pass overwrite=True")
    _PRESETS[spec.name] = spec
