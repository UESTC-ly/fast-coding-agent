"""Anthropic adapter.

Differences this layer absorbs:
- Tools arrive as `tool_use` / `tool_result` content blocks, not a separate field.
- The system prompt is a top-level parameter, not a message.
- Structured output is a forced single tool call (`tool_choice={"type": "tool"}`).
- Prompt caching needs explicit `cache_control` breakpoints.
- `usage.input_tokens` EXCLUDES cache reads, unlike OpenAI's `prompt_tokens`.
  `to_usage` keeps the canonical `Usage.input_tokens` cache-exclusive so both
  providers report the same thing.
"""

from __future__ import annotations

from typing import Any

from qqcode.models.errors import ProviderError
from qqcode.models.protocol import (
    Budget,
    Completion,
    ContentBlock,
    ModelTier,
    Msg,
    OutputSpec,
    Role,
    TextContent,
    ToolSpec,
    ToolUseContent,
    Usage,
)

# Tier to model id. One place to retune cost.
TIER_MODELS: dict[ModelTier, str] = {
    ModelTier.FAST: "claude-haiku-4-5-20251001",
    ModelTier.BALANCED: "claude-sonnet-5",
    ModelTier.DEEP: "claude-opus-5",
}

CACHE_CONTROL: dict[str, str] = {"type": "ephemeral"}


def _block_to_api(block: ContentBlock) -> dict[str, Any]:
    """Convert one canonical block to Anthropic wire format."""
    if isinstance(block, TextContent):
        return {"type": "text", "text": block.text}
    if isinstance(block, ToolUseContent):
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    return {
        "type": "tool_result",
        "tool_use_id": block.tool_use_id,
        "content": block.content,
        "is_error": block.is_error,
    }


def to_api_messages(messages: list[Msg]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split canonical messages into (system blocks, conversation messages).

    Anthropic takes the system prompt as its own parameter. Tool results are
    user-role blocks, so a TOOL message becomes a user message.

    A message flagged `cache_breakpoint` gets `cache_control` on its final
    block, which caches the whole prefix up to that point.
    """
    system: list[dict[str, Any]] = []
    convo: list[dict[str, Any]] = []

    for msg in messages:
        blocks = [_block_to_api(b) for b in msg.content]
        if msg.cache_breakpoint and blocks:
            blocks[-1] = {**blocks[-1], "cache_control": CACHE_CONTROL}

        if msg.role is Role.SYSTEM:
            system.extend(blocks)
            continue

        role = "user" if msg.role in (Role.USER, Role.TOOL) else "assistant"
        # Anthropic rejects consecutive same-role messages; merging keeps a
        # tool_result next to a preceding user turn from breaking the call.
        if convo and convo[-1]["role"] == role:
            convo[-1]["content"].extend(blocks)
        else:
            convo.append({"role": role, "content": blocks})

    return system, convo


def to_api_tools(tools: list[ToolSpec], *, cache_last: bool = True) -> list[dict[str, Any]]:
    """Convert tool specs, marking the last one as a cache breakpoint.

    Tool definitions are stable across a task, so caching them is free savings.
    """
    out: list[dict[str, Any]] = [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in tools
    ]
    if out and cache_last:
        out[-1] = {**out[-1], "cache_control": CACHE_CONTROL}
    return out


def from_api_content(blocks: list[Any]) -> list[ContentBlock]:
    """Convert an Anthropic response's content blocks to canonical form.

    Thinking blocks are dropped from canonical content: replaying them requires
    provider-specific ordering rules, and `Completion.raw` retains the original
    for callers that need to echo them back.
    """
    out: list[ContentBlock] = []
    for b in blocks:
        kind = getattr(b, "type", None)
        if kind == "text":
            out.append(TextContent(text=b.text))
        elif kind == "tool_use":
            out.append(ToolUseContent(id=b.id, name=b.name, input=dict(b.input)))
    return out


def to_usage(raw_usage: Any) -> Usage:
    """Normalize Anthropic usage.

    `input_tokens` already excludes cache reads, which is the canonical
    convention, so it passes through unchanged.
    """
    return Usage(
        input_tokens=getattr(raw_usage, "input_tokens", 0) or 0,
        output_tokens=getattr(raw_usage, "output_tokens", 0) or 0,
        cache_creation_tokens=getattr(raw_usage, "cache_creation_input_tokens", 0) or 0,
        cache_read_tokens=getattr(raw_usage, "cache_read_input_tokens", 0) or 0,
    )


class AnthropicAdapter:
    """ModelClient implementation over the Anthropic SDK."""

    def __init__(self, client: Any, *, tier_models: dict[ModelTier, str] | None = None):
        """
        Args:
            client: An `anthropic.Anthropic` instance, or any object exposing
                `messages.create`.
            tier_models: Override the tier-to-model mapping.
        """
        self._client = client
        self._tier_models = dict(tier_models or TIER_MODELS)

    def model_for(self, tier: ModelTier) -> str:
        return self._tier_models[tier]

    def invoke(
        self,
        messages: list[Msg],
        tools: list[ToolSpec] | None = None,
        output_spec: OutputSpec | None = None,
        budget: Budget | None = None,
        temperature: float = 0.0,
        *,
        tier: ModelTier = ModelTier.BALANCED,
        **kwargs: Any,
    ) -> Completion:
        """Invoke the model with canonical inputs.

        Raises:
            ValueError: Both real tools and `output_spec` were supplied.
            ProviderError: The SDK call failed.
        """
        if output_spec is not None and tools:
            raise ValueError(
                "Structured output is a forced single tool call and cannot be "
                "combined with real tools"
            )

        system, convo = to_api_messages(messages)
        budget = budget or Budget()

        params: dict[str, Any] = {
            "model": self.model_for(tier),
            "messages": convo,
            "max_tokens": budget.max_tokens,
            "temperature": temperature,
        }
        if system:
            params["system"] = system

        if output_spec is not None:
            params["tools"] = to_api_tools(
                [
                    ToolSpec(
                        name=output_spec.tool_name,
                        description="Emit the structured result.",
                        input_schema=output_spec.schema,
                    )
                ]
            )
            params["tool_choice"] = {"type": "tool", "name": output_spec.tool_name}
        elif tools:
            params["tools"] = to_api_tools(tools)

        params.update(kwargs)

        try:
            response = self._client.messages.create(**params)
        except Exception as exc:
            raise _as_provider_error(exc) from exc

        return Completion(
            content=from_api_content(list(response.content)),
            stop_reason=getattr(response, "stop_reason", "end_turn") or "end_turn",
            usage=to_usage(getattr(response, "usage", None)),
            raw=response,
        )


def _as_provider_error(exc: Exception) -> ProviderError:
    """Wrap an SDK exception, preserving its status code when present."""
    if isinstance(exc, ProviderError):
        return exc
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    return ProviderError(f"Anthropic call failed: {exc}", status_code=status)
