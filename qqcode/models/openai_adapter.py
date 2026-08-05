"""OpenAI adapter.

Differences this layer absorbs:
- Tool calls live in `message.tool_calls`; results are `role: "tool"` messages
  keyed by `tool_call_id`.
- The system prompt is an ordinary message.
- Structured output uses `response_format` with a strict `json_schema`, which
  returns JSON as message content rather than a tool call. `from_api_message`
  re-wraps it as a `ToolUseContent` so callers see one shape across providers.
- Prefix caching is automatic, so `cache_breakpoint` needs no wire encoding.
- `usage.prompt_tokens` INCLUDES cached tokens, unlike Anthropic. `to_usage`
  subtracts the cached count to keep `Usage.input_tokens` cache-exclusive;
  without this the same task would report materially different input totals
  depending on provider.
"""

from __future__ import annotations

import json
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
    ToolResultContent,
    ToolSpec,
    ToolUseContent,
    Usage,
)

TIER_MODELS: dict[ModelTier, str] = {
    ModelTier.FAST: "gpt-5-mini",
    ModelTier.BALANCED: "gpt-5",
    ModelTier.DEEP: "gpt-5",
}

# Stop-reason names differ; map to the canonical set.
FINISH_REASONS: dict[str, str] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "stop_sequence",
}


def to_api_messages(messages: list[Msg]) -> list[dict[str, Any]]:
    """Convert canonical messages to OpenAI chat format.

    An assistant turn carrying tool calls folds its text and `tool_calls` into
    one message; each tool result becomes its own `role: "tool"` message.
    """
    out: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role is Role.TOOL:
            for block in msg.content:
                if isinstance(block, ToolResultContent):
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": block.tool_use_id,
                            "content": block.content,
                        }
                    )
            continue

        # A tool result on any other role cannot be encoded: OpenAI only accepts
        # them as standalone `role: "tool"` messages. Dropping them silently
        # produces an empty message and a misleading "No tool output found for
        # function call" 400 from the server, so fail loudly at the boundary.
        stray = [b for b in msg.content if isinstance(b, ToolResultContent)]
        if stray:
            raise ProviderError(
                f"ToolResultContent on role={msg.role.value!r}; tool results must "
                f"use Role.TOOL to be encoded for OpenAI "
                f"(tool_use_ids={[b.tool_use_id for b in stray]})"
            )

        texts = [b.text for b in msg.content if isinstance(b, TextContent)]
        tool_uses = [b for b in msg.content if isinstance(b, ToolUseContent)]

        entry: dict[str, Any] = {"role": msg.role.value, "content": "\n".join(texts)}
        if tool_uses:
            entry["tool_calls"] = [
                {
                    "id": tu.id,
                    "type": "function",
                    "function": {"name": tu.name, "arguments": json.dumps(tu.input)},
                }
                for tu in tool_uses
            ]
            # OpenAI expects null content, not "", alongside tool_calls.
            if not entry["content"]:
                entry["content"] = None
        out.append(entry)

    return out


def to_api_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    """Convert tool specs to OpenAI function-tool format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]


def _strictify(node: Any) -> Any:
    """Return a copy of a JSON schema that satisfies OpenAI's strict mode.

    Strict mode requires every object to carry `additionalProperties: false`
    explicitly. Canonical specs are written to Anthropic's `input_schema`
    conventions, which have no such requirement, so the constraint is injected
    here rather than duplicated into every OutputSpec.
    """
    if isinstance(node, dict):
        out = {k: _strictify(v) for k, v in node.items()}
        if out.get("type") == "object":
            out.setdefault("additionalProperties", False)
        return out
    if isinstance(node, list):
        return [_strictify(v) for v in node]
    return node


def to_response_format(output_spec: OutputSpec) -> dict[str, Any]:
    """Build a strict `json_schema` response format."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": output_spec.tool_name,
            "schema": _strictify(output_spec.schema),
            "strict": True,
        },
    }


def from_api_message(message: Any, *, output_spec: OutputSpec | None = None) -> list[ContentBlock]:
    """Convert an OpenAI response message to canonical content.

    With `output_spec` set, the JSON body is wrapped as a `ToolUseContent` named
    after the schema so structured results look identical on both providers.

    Raises:
        ProviderError: Structured output was requested but the body is empty or
            not valid JSON — returning it as text would push a malformed
            payload downstream.
    """
    content = getattr(message, "content", None)

    if output_spec is not None:
        if not content:
            raise ProviderError("Structured output requested but response body was empty")
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"Structured output was not valid JSON: {exc}") from exc
        return [
            ToolUseContent(
                id=f"so_{output_spec.tool_name}",
                name=output_spec.tool_name,
                input=payload,
            )
        ]

    out: list[ContentBlock] = []
    if content:
        out.append(TextContent(text=content))
    for call in getattr(message, "tool_calls", None) or []:
        fn = call.function
        out.append(
            ToolUseContent(
                id=call.id,
                name=fn.name,
                input=json.loads(fn.arguments) if fn.arguments else {},
            )
        )
    return out


def to_usage(raw_usage: Any) -> Usage:
    """Normalize OpenAI usage to the cache-exclusive canonical convention.

    `prompt_tokens` counts cached tokens, so the cached portion is subtracted to
    match Anthropic's `input_tokens`.
    """
    prompt = getattr(raw_usage, "prompt_tokens", 0) or 0
    completion = getattr(raw_usage, "completion_tokens", 0) or 0

    details = getattr(raw_usage, "prompt_tokens_details", None)
    cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0

    return Usage(
        input_tokens=max(0, prompt - cached),
        output_tokens=completion,
        cache_creation_tokens=0,  # Not reported; prefix caching is implicit.
        cache_read_tokens=cached,
    )


class OpenAIAdapter:
    """ModelClient implementation over the OpenAI SDK."""

    def __init__(
        self,
        client: Any,
        *,
        tier_models: dict[ModelTier, str] | None = None,
        reasoning_effort: str | None = None,
    ):
        """
        Args:
            client: An `openai.OpenAI` instance, or any object exposing
                `chat.completions.create`.
            tier_models: Override the tier-to-model mapping.
            reasoning_effort: Sent as `reasoning_effort` on every request when
                set ("low"/"medium"/"high"). Omitted entirely when None so the
                provider default applies and non-reasoning models stay working.
        """
        self._client = client
        self._tier_models = dict(tier_models or TIER_MODELS)
        self._reasoning_effort = reasoning_effort

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
            ProviderError: The SDK call failed or returned a malformed payload.
        """
        if output_spec is not None and tools:
            raise ValueError(
                "Structured output cannot be combined with real tools; the "
                "response format leaves no room for a tool call"
            )

        budget = budget or Budget()
        params: dict[str, Any] = {
            "model": self.model_for(tier),
            "messages": to_api_messages(messages),
            "max_completion_tokens": budget.max_tokens,
            "temperature": temperature,
        }
        if self._reasoning_effort is not None:
            params["reasoning_effort"] = self._reasoning_effort

        if output_spec is not None:
            params["response_format"] = to_response_format(output_spec)
        elif tools:
            params["tools"] = to_api_tools(tools)

        params.update(kwargs)

        try:
            response = self._client.chat.completions.create(**params)
        except Exception as exc:
            raise _as_provider_error(exc) from exc

        choice = response.choices[0]
        finish = getattr(choice, "finish_reason", "stop") or "stop"

        return Completion(
            content=from_api_message(choice.message, output_spec=output_spec),
            stop_reason=FINISH_REASONS.get(finish, finish),
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
    return ProviderError(f"OpenAI call failed: {exc}", status_code=status)
