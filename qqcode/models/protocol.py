"""Canonical message protocol and ModelClient interface.

All provider adapters must implement ModelClient to provide:
- Unified message format (canonical Msg)
- Tool calls and structured output
- Prompt caching control
- Token usage tracking (feeding into CostLedger)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal, Protocol


class Role(StrEnum):
    """Canonical message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ModelTier(StrEnum):
    """Model capability tier; each adapter maps this to a concrete model id.

    Keeping tiers abstract lets cost tuning happen in one place instead of
    scattering model names across call sites and presets.
    """

    FAST = "fast"  # Cheap, high-volume work: mechanical edits, simple lookups
    BALANCED = "balanced"  # Default: most implementation and review work
    DEEP = "deep"  # Hardest reasoning: architecture, subtle bug hunts


@dataclass(frozen=True)
class TextContent:
    """Plain text content block."""

    text: str


@dataclass(frozen=True)
class ToolUseContent:
    """Tool invocation by the model."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class ToolResultContent:
    """Tool execution result returned to model."""

    tool_use_id: str
    content: str
    is_error: bool = False


ContentBlock = TextContent | ToolUseContent | ToolResultContent

# Execution phase a provider call is billed against.
Phase = Literal["routing", "fastpath", "fullagent", "subagent"]

# Execution surface a tool or skill is exposed on. Routing carries no tools —
# it runs with a forced output schema only — so it is absent here.
Tier = Literal["fastpath", "fullagent", "subagent"]
ALL_TIERS: frozenset[str] = frozenset({"fastpath", "fullagent", "subagent"})


@dataclass(frozen=True)
class Msg:
    """Canonical message format.

    Unified across Anthropic (content blocks) and OpenAI (role + content/tool_calls).
    Adapters translate between this and provider-specific formats.
    """

    role: Role
    content: list[ContentBlock]
    # For caching: mark this message as a cache breakpoint
    cache_breakpoint: bool = False


@dataclass(frozen=True)
class ToolSpec:
    """Tool definition (JSON Schema + metadata)."""

    name: str
    description: str
    input_schema: dict[str, Any]  # JSON Schema


@dataclass(frozen=True)
class OutputSpec:
    """Structured output constraint (forces model to call a specific tool)."""

    tool_name: str
    schema: dict[str, Any]  # JSON Schema


@dataclass
class Budget:
    """Token budget for a single request."""

    max_tokens: int = 4096
    # If set, stop generating when cumulative input + output exceeds this
    total_limit: int | None = None


@dataclass
class Usage:
    """Token usage归一化 across providers."""

    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class Completion:
    """Unified model response."""

    content: list[ContentBlock]
    stop_reason: Literal["end_turn", "max_tokens", "tool_use", "stop_sequence"] | str
    usage: Usage
    raw: Any  # Provider-specific response object for debugging


class ModelClient(Protocol):
    """Protocol for model providers.

    All adapters (Anthropic, OpenAI) must implement this interface.
    """

    def invoke(
        self,
        messages: list[Msg],
        tools: list[ToolSpec] | None = None,
        output_spec: OutputSpec | None = None,
        budget: Budget | None = None,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> Completion:
        """Invoke model with canonical messages and tools.

        Args:
            messages: Conversation history in canonical format
            tools: Available tools (if any)
            output_spec: Force structured output via tool call
            budget: Token limits
            temperature: Sampling temperature
            **kwargs: Provider-specific overrides

        Returns:
            Unified completion with usage tracking
        """
        ...


@dataclass
class CostLedger:
    """Cumulative cost tracking across all provider calls.

    Single source of truth for Automatic cost accounting.
    All token spend (FastPath attempts, retries, sub-agents, Full Agent)
    accumulates here.
    """

    total_input: int = 0
    total_output: int = 0
    cache_creation: int = 0
    cache_read: int = 0
    calls: int = 0
    # Calls that were retried after a transient provider failure. Their tokens
    # are already inside the totals; this counts how often it happened.
    retried_calls: int = 0
    # Track by phase
    routing_tokens: int = 0
    fastpath_tokens: int = 0
    fullagent_tokens: int = 0
    subagent_tokens: int = 0

    def add(self, usage: Usage, phase: Phase, *, retried: bool = False) -> None:
        """Record usage from a single call."""
        self.total_input += usage.input_tokens
        self.total_output += usage.output_tokens
        self.cache_creation += usage.cache_creation_tokens
        self.cache_read += usage.cache_read_tokens
        self.calls += 1
        if retried:
            self.retried_calls += 1

        total = usage.total
        if phase == "routing":
            self.routing_tokens += total
        elif phase == "fastpath":
            self.fastpath_tokens += total
        elif phase == "subagent":
            self.subagent_tokens += total
        else:
            self.fullagent_tokens += total

    @property
    def automatic_total(self) -> int:
        """Total tokens spent in Automatic mode across every phase."""
        return self.total_input + self.total_output

    def summary(self) -> dict[str, Any]:
        """Export ledger for logging/analysis."""
        return {
            "total_input": self.total_input,
            "total_output": self.total_output,
            "cache_creation": self.cache_creation,
            "cache_read": self.cache_read,
            "calls": self.calls,
            "retried_calls": self.retried_calls,
            "automatic_total": self.automatic_total,
            "by_phase": {
                "routing": self.routing_tokens,
                "fastpath": self.fastpath_tokens,
                "fullagent": self.fullagent_tokens,
                "subagent": self.subagent_tokens,
            },
        }
