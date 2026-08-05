"""Contract tests for the provider adapters.

No network and no API key: each test drives the adapter with a fake SDK client
that records the params it received and returns a hand-built response object.
The adapters only ever use `getattr`, so plain namespaces are enough.

The point of these tests is the *contract* — the same canonical input must
produce the same canonical output on both providers, and the two providers'
different usage conventions must normalize to one.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from qqcode.models import anthropic_adapter as anth
from qqcode.models import openai_adapter as oai
from qqcode.models.errors import ProviderError
from qqcode.models.protocol import (
    Budget,
    ModelTier,
    Msg,
    OutputSpec,
    Role,
    TextContent,
    ToolResultContent,
    ToolSpec,
    ToolUseContent,
)

PROBE_SPEC = OutputSpec(
    tool_name="report",
    schema={
        "type": "object",
        "required": ["value"],
        "properties": {"value": {"type": "string"}},
    },
)

WEATHER_TOOL = ToolSpec(
    name="get_weather",
    description="Look up weather.",
    input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
)


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------


class FakeAnthropicClient:
    """Records `messages.create` params; returns a canned response."""

    def __init__(self, response: Any = None, raises: Exception | None = None):
        self.captured: dict[str, Any] = {}
        self._response = response
        self._raises = raises
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **params: Any) -> Any:
        self.captured = params
        if self._raises is not None:
            raise self._raises
        return self._response


class FakeOpenAIClient:
    """Records `chat.completions.create` params; returns a canned response."""

    def __init__(self, response: Any = None, raises: Exception | None = None):
        self.captured: dict[str, Any] = {}
        self._response = response
        self._raises = raises
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **params: Any) -> Any:
        self.captured = params
        if self._raises is not None:
            raise self._raises
        return self._response


def anth_response(
    blocks: list[Any],
    *,
    stop_reason: str = "end_turn",
    input_tokens: int = 100,
    output_tokens: int = 20,
    cache_creation: int = 0,
    cache_read: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=blocks,
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation,
            cache_read_input_tokens=cache_read,
        ),
    )


def oai_response(
    *,
    content: str | None = None,
    tool_calls: list[Any] | None = None,
    finish_reason: str = "stop",
    prompt_tokens: int = 100,
    completion_tokens: int = 20,
    cached_tokens: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
        ),
    )


def oai_tool_call(call_id: str, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=arguments)
    )


# --------------------------------------------------------------------------
# The headline invariant: usage normalization across providers
# --------------------------------------------------------------------------


def test_anthropic_input_tokens_pass_through_cache_exclusive() -> None:
    """Anthropic already excludes cache reads from input_tokens."""
    usage = anth.to_usage(
        SimpleNamespace(
            input_tokens=200,
            output_tokens=50,
            cache_creation_input_tokens=1000,
            cache_read_input_tokens=8000,
        )
    )
    assert usage.input_tokens == 200
    assert usage.cache_read_tokens == 8000
    assert usage.cache_creation_tokens == 1000


def test_openai_prompt_tokens_have_cache_subtracted() -> None:
    """OpenAI counts cached tokens inside prompt_tokens; they must come out."""
    usage = oai.to_usage(
        SimpleNamespace(
            prompt_tokens=8200,  # 200 fresh + 8000 cached
            completion_tokens=50,
            prompt_tokens_details=SimpleNamespace(cached_tokens=8000),
        )
    )
    assert usage.input_tokens == 200
    assert usage.cache_read_tokens == 8000


def test_both_providers_normalize_identical_workload_identically() -> None:
    """The whole reason the subtraction exists: one workload, one number.

    Same 200 fresh + 8000 cached input and 50 output on each provider must
    yield the same canonical Usage, or per-provider cost reports diverge for
    reasons that have nothing to do with the task.
    """
    a = anth.to_usage(
        SimpleNamespace(
            input_tokens=200,
            output_tokens=50,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=8000,
        )
    )
    o = oai.to_usage(
        SimpleNamespace(
            prompt_tokens=8200,
            completion_tokens=50,
            prompt_tokens_details=SimpleNamespace(cached_tokens=8000),
        )
    )
    assert (a.input_tokens, a.output_tokens, a.cache_read_tokens) == (
        o.input_tokens,
        o.output_tokens,
        o.cache_read_tokens,
    )
    assert a.total == o.total == 250


def test_openai_usage_never_goes_negative_on_inconsistent_report() -> None:
    """A proxy reporting cached > prompt must not produce negative input."""
    usage = oai.to_usage(
        SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=10,
            prompt_tokens_details=SimpleNamespace(cached_tokens=500),
        )
    )
    assert usage.input_tokens == 0


def test_usage_tolerates_missing_fields() -> None:
    """A response with no usage object at all must not crash the call path."""
    assert anth.to_usage(None).total == 0
    assert oai.to_usage(None).total == 0


# --------------------------------------------------------------------------
# Structured output ⊥ real tools, on both providers
# --------------------------------------------------------------------------


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_structured_output_and_tools_are_mutually_exclusive(provider: str) -> None:
    """Forcing structured output consumes the one permitted tool call."""
    if provider == "anthropic":
        adapter: Any = anth.AnthropicAdapter(FakeAnthropicClient())
    else:
        adapter = oai.OpenAIAdapter(FakeOpenAIClient())

    with pytest.raises(ValueError):
        adapter.invoke(
            [Msg(role=Role.USER, content=[TextContent(text="hi")])],
            tools=[WEATHER_TOOL],
            output_spec=PROBE_SPEC,
        )


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_mutual_exclusion_rejects_before_reaching_the_sdk(provider: str) -> None:
    """The check must fire locally — a rejected call must cost nothing."""
    if provider == "anthropic":
        client: Any = FakeAnthropicClient()
        adapter: Any = anth.AnthropicAdapter(client)
    else:
        client = FakeOpenAIClient()
        adapter = oai.OpenAIAdapter(client)

    with pytest.raises(ValueError):
        adapter.invoke(
            [Msg(role=Role.USER, content=[TextContent(text="hi")])],
            tools=[WEATHER_TOOL],
            output_spec=PROBE_SPEC,
        )
    assert client.captured == {}


# --------------------------------------------------------------------------
# Structured output produces one shape on both providers
# --------------------------------------------------------------------------


def test_anthropic_structured_output_forces_the_tool() -> None:
    client = FakeAnthropicClient(
        anth_response(
            [SimpleNamespace(type="tool_use", id="tu_1", name="report", input={"value": "x"})],
            stop_reason="tool_use",
        )
    )
    adapter = anth.AnthropicAdapter(client)
    completion = adapter.invoke(
        [Msg(role=Role.USER, content=[TextContent(text="go")])], output_spec=PROBE_SPEC
    )

    assert client.captured["tool_choice"] == {"type": "tool", "name": "report"}
    assert [t["name"] for t in client.captured["tools"]] == ["report"]
    assert completion.content == [ToolUseContent(id="tu_1", name="report", input={"value": "x"})]


def test_openai_structured_output_is_rewrapped_as_tool_use() -> None:
    """OpenAI returns JSON as message content; callers must still see a tool use."""
    client = FakeOpenAIClient(oai_response(content='{"value": "x"}'))
    adapter = oai.OpenAIAdapter(client)
    completion = adapter.invoke(
        [Msg(role=Role.USER, content=[TextContent(text="go")])], output_spec=PROBE_SPEC
    )

    fmt = client.captured["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    assert completion.content == [
        ToolUseContent(id="so_report", name="report", input={"value": "x"})
    ]


def test_openai_strict_schema_gets_additional_properties_false_everywhere() -> None:
    """`strict: true` is rejected unless EVERY object forbids extra properties.

    Schemas are authored in Anthropic's `input_schema` style, which has no such
    requirement, so the adapter injects it — including on nested objects, where
    a missing flag 400s just as hard as one on the root.
    """
    spec = OutputSpec(
        tool_name="submit_patch",
        schema={
            "type": "object",
            "required": ["files"],
            "properties": {
                "files": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["path"],
                        "properties": {"path": {"type": "string"}},
                    },
                }
            },
        },
    )
    fmt = oai.to_response_format(spec)
    schema = fmt["json_schema"]["schema"]

    assert schema["additionalProperties"] is False
    assert schema["properties"]["files"]["items"]["additionalProperties"] is False
    # The caller's spec is module-level shared state; mutating it would leak.
    assert "additionalProperties" not in spec.schema


def test_both_providers_yield_the_same_structured_payload() -> None:
    """Different wire encodings, one canonical result."""
    a = anth.AnthropicAdapter(
        FakeAnthropicClient(
            anth_response(
                [
                    SimpleNamespace(
                        type="tool_use", id="tu_1", name="report", input={"value": "x"}
                    )
                ],
                stop_reason="tool_use",
            )
        )
    ).invoke([Msg(role=Role.USER, content=[TextContent(text="go")])], output_spec=PROBE_SPEC)

    o = oai.OpenAIAdapter(FakeOpenAIClient(oai_response(content='{"value": "x"}'))).invoke(
        [Msg(role=Role.USER, content=[TextContent(text="go")])], output_spec=PROBE_SPEC
    )

    assert isinstance(a.content[0], ToolUseContent)
    assert isinstance(o.content[0], ToolUseContent)
    assert a.content[0].name == o.content[0].name == "report"
    assert a.content[0].input == o.content[0].input == {"value": "x"}


def test_openai_empty_structured_body_raises_rather_than_returning_text() -> None:
    adapter = oai.OpenAIAdapter(FakeOpenAIClient(oai_response(content="")))
    with pytest.raises(ProviderError, match="empty"):
        adapter.invoke(
            [Msg(role=Role.USER, content=[TextContent(text="go")])], output_spec=PROBE_SPEC
        )


def test_openai_malformed_structured_body_raises() -> None:
    adapter = oai.OpenAIAdapter(FakeOpenAIClient(oai_response(content="{not json")))
    with pytest.raises(ProviderError, match="valid JSON"):
        adapter.invoke(
            [Msg(role=Role.USER, content=[TextContent(text="go")])], output_spec=PROBE_SPEC
        )


# --------------------------------------------------------------------------
# stop_reason canonicalization
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("finish", "expected"),
    [
        ("stop", "end_turn"),
        ("length", "max_tokens"),
        ("tool_calls", "tool_use"),
        ("function_call", "tool_use"),
        ("content_filter", "stop_sequence"),
    ],
)
def test_openai_finish_reason_maps_to_canonical(finish: str, expected: str) -> None:
    adapter = oai.OpenAIAdapter(FakeOpenAIClient(oai_response(content="hi", finish_reason=finish)))
    completion = adapter.invoke([Msg(role=Role.USER, content=[TextContent(text="go")])])
    assert completion.stop_reason == expected


def test_openai_unknown_finish_reason_passes_through() -> None:
    """An unmapped reason must survive verbatim rather than be silently renamed."""
    adapter = oai.OpenAIAdapter(
        FakeOpenAIClient(oai_response(content="hi", finish_reason="something_new"))
    )
    completion = adapter.invoke([Msg(role=Role.USER, content=[TextContent(text="go")])])
    assert completion.stop_reason == "something_new"


def test_anthropic_stop_reason_is_already_canonical() -> None:
    adapter = anth.AnthropicAdapter(
        FakeAnthropicClient(
            anth_response([SimpleNamespace(type="text", text="hi")], stop_reason="max_tokens")
        )
    )
    completion = adapter.invoke([Msg(role=Role.USER, content=[TextContent(text="go")])])
    assert completion.stop_reason == "max_tokens"


# --------------------------------------------------------------------------
# Message translation
# --------------------------------------------------------------------------


def test_anthropic_lifts_system_out_of_the_conversation() -> None:
    system, convo = anth.to_api_messages(
        [
            Msg(role=Role.SYSTEM, content=[TextContent(text="be brief")]),
            Msg(role=Role.USER, content=[TextContent(text="hi")]),
        ]
    )
    assert system == [{"type": "text", "text": "be brief"}]
    assert [m["role"] for m in convo] == ["user"]


def test_anthropic_cache_breakpoint_marks_the_final_block() -> None:
    """The breakpoint caches the whole prefix up to that message."""
    system, _ = anth.to_api_messages(
        [
            Msg(
                role=Role.SYSTEM,
                content=[TextContent(text="a"), TextContent(text="b")],
                cache_breakpoint=True,
            )
        ]
    )
    assert "cache_control" not in system[0]
    assert system[1]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_merges_consecutive_same_role_messages() -> None:
    """Anthropic rejects two user turns in a row; a tool result follows a user turn."""
    _, convo = anth.to_api_messages(
        [
            Msg(role=Role.USER, content=[TextContent(text="hi")]),
            Msg(
                role=Role.TOOL,
                content=[ToolResultContent(tool_use_id="tu_1", content="42")],
            ),
        ]
    )
    assert len(convo) == 1
    assert convo[0]["role"] == "user"
    assert [b["type"] for b in convo[0]["content"]] == ["text", "tool_result"]


def test_anthropic_caches_the_last_tool_definition() -> None:
    """Tool schemas are stable across a task, so caching them is free savings."""
    tools = anth.to_api_tools([WEATHER_TOOL])
    assert tools[-1]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_drops_thinking_blocks_from_canonical_content() -> None:
    """Replaying thinking needs provider-specific ordering; `raw` keeps it."""
    content = anth.from_api_content(
        [
            SimpleNamespace(type="thinking", thinking="hmm"),
            SimpleNamespace(type="text", text="answer"),
        ]
    )
    assert content == [TextContent(text="answer")]


def test_openai_tool_results_become_their_own_messages() -> None:
    msgs = oai.to_api_messages(
        [
            Msg(
                role=Role.TOOL,
                content=[ToolResultContent(tool_use_id="call_1", content="42")],
            )
        ]
    )
    assert msgs == [{"role": "tool", "tool_call_id": "call_1", "content": "42"}]


def test_openai_rejects_tool_results_on_a_non_tool_role() -> None:
    """A misrouted tool result must fail loudly, not vanish.

    Anthropic carries tool results on a `user` turn, so mislabelling them
    Role.USER worked there while silently producing an empty user message
    here — the gateway then correctly reported "No tool output found for
    function call ...", which read like a provider bug for two sessions.
    """
    with pytest.raises(ProviderError, match="Role.TOOL"):
        oai.to_api_messages(
            [
                Msg(
                    role=Role.USER,
                    content=[ToolResultContent(tool_use_id="call_1", content="42")],
                )
            ]
        )


def test_openai_sends_null_content_alongside_tool_calls() -> None:
    """OpenAI rejects an empty string where it expects null."""
    msgs = oai.to_api_messages(
        [
            Msg(
                role=Role.ASSISTANT,
                content=[ToolUseContent(id="call_1", name="get_weather", input={"city": "Kyoto"})],
            )
        ]
    )
    assert msgs[0]["content"] is None
    assert msgs[0]["tool_calls"][0]["function"]["name"] == "get_weather"


def test_openai_parses_tool_call_arguments() -> None:
    client = FakeOpenAIClient(
        oai_response(
            tool_calls=[oai_tool_call("call_1", "get_weather", '{"city": "Kyoto"}')],
            finish_reason="tool_calls",
        )
    )
    completion = oai.OpenAIAdapter(client).invoke(
        [Msg(role=Role.USER, content=[TextContent(text="weather?")])], tools=[WEATHER_TOOL]
    )
    assert completion.content == [
        ToolUseContent(id="call_1", name="get_weather", input={"city": "Kyoto"})
    ]
    assert completion.stop_reason == "tool_use"


def test_openai_handles_tool_call_with_empty_arguments() -> None:
    client = FakeOpenAIClient(
        oai_response(tool_calls=[oai_tool_call("call_1", "ping", "")], finish_reason="tool_calls")
    )
    completion = oai.OpenAIAdapter(client).invoke(
        [Msg(role=Role.USER, content=[TextContent(text="ping")])], tools=[WEATHER_TOOL]
    )
    assert isinstance(completion.content[0], ToolUseContent)
    assert completion.content[0].input == {}


# --------------------------------------------------------------------------
# Tier mapping and request params
# --------------------------------------------------------------------------


def test_tier_models_are_overridable_per_adapter() -> None:
    """Pinning every tier to one model is how comparable runs are set up."""
    pinned = dict.fromkeys(ModelTier, "some-model")
    assert anth.AnthropicAdapter(FakeAnthropicClient(), tier_models=pinned).model_for(
        ModelTier.FAST
    ) == "some-model"
    assert oai.OpenAIAdapter(FakeOpenAIClient(), tier_models=pinned).model_for(
        ModelTier.DEEP
    ) == "some-model"


def test_adapters_send_their_provider_specific_token_param() -> None:
    """Anthropic takes max_tokens; OpenAI takes max_completion_tokens."""
    ac = FakeAnthropicClient(anth_response([SimpleNamespace(type="text", text="hi")]))
    anth.AnthropicAdapter(ac).invoke(
        [Msg(role=Role.USER, content=[TextContent(text="go")])], budget=Budget(max_tokens=321)
    )
    assert ac.captured["max_tokens"] == 321

    oc = FakeOpenAIClient(oai_response(content="hi"))
    oai.OpenAIAdapter(oc).invoke(
        [Msg(role=Role.USER, content=[TextContent(text="go")])], budget=Budget(max_tokens=321)
    )
    assert oc.captured["max_completion_tokens"] == 321


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_kwargs_override_computed_params(provider: str) -> None:
    """An escape hatch for provider-specific knobs must actually win."""
    if provider == "anthropic":
        client: Any = FakeAnthropicClient(anth_response([SimpleNamespace(type="text", text="hi")]))
        adapter: Any = anth.AnthropicAdapter(client)
    else:
        client = FakeOpenAIClient(oai_response(content="hi"))
        adapter = oai.OpenAIAdapter(client)

    adapter.invoke([Msg(role=Role.USER, content=[TextContent(text="go")])], model="forced-model")
    assert client.captured["model"] == "forced-model"


def test_anthropic_omits_system_param_when_there_is_none() -> None:
    client = FakeAnthropicClient(anth_response([SimpleNamespace(type="text", text="hi")]))
    anth.AnthropicAdapter(client).invoke([Msg(role=Role.USER, content=[TextContent(text="go")])])
    assert "system" not in client.captured


# --------------------------------------------------------------------------
# Error translation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_sdk_exception_becomes_provider_error_preserving_status(provider: str) -> None:
    boom = RuntimeError("upstream exploded")
    boom.status_code = 503  # type: ignore[attr-defined]

    if provider == "anthropic":
        adapter: Any = anth.AnthropicAdapter(FakeAnthropicClient(raises=boom))
    else:
        adapter = oai.OpenAIAdapter(FakeOpenAIClient(raises=boom))

    with pytest.raises(ProviderError) as info:
        adapter.invoke([Msg(role=Role.USER, content=[TextContent(text="go")])])
    assert info.value.status_code == 503
    assert info.value.retryable is True


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_status_is_read_from_a_nested_response_object(provider: str) -> None:
    """Some SDKs put the status on `exc.response`, not on the exception."""
    boom = RuntimeError("rate limited")
    boom.response = SimpleNamespace(status_code=429)  # type: ignore[attr-defined]

    if provider == "anthropic":
        adapter: Any = anth.AnthropicAdapter(FakeAnthropicClient(raises=boom))
    else:
        adapter = oai.OpenAIAdapter(FakeOpenAIClient(raises=boom))

    with pytest.raises(ProviderError) as info:
        adapter.invoke([Msg(role=Role.USER, content=[TextContent(text="go")])])
    assert info.value.status_code == 429
    assert info.value.retryable is True


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_statusless_failure_is_not_retryable(provider: str) -> None:
    """Unknown failures fail fast instead of burning budget on repeats."""
    if provider == "anthropic":
        adapter: Any = anth.AnthropicAdapter(FakeAnthropicClient(raises=RuntimeError("???")))
    else:
        adapter = oai.OpenAIAdapter(FakeOpenAIClient(raises=RuntimeError("???")))

    with pytest.raises(ProviderError) as info:
        adapter.invoke([Msg(role=Role.USER, content=[TextContent(text="go")])])
    assert info.value.status_code is None
    assert info.value.retryable is False


@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_provider_error_is_not_rewrapped(provider: str) -> None:
    original = ProviderError("already classified", status_code=429)

    if provider == "anthropic":
        adapter: Any = anth.AnthropicAdapter(FakeAnthropicClient(raises=original))
    else:
        adapter = oai.OpenAIAdapter(FakeOpenAIClient(raises=original))

    with pytest.raises(ProviderError) as info:
        adapter.invoke([Msg(role=Role.USER, content=[TextContent(text="go")])])
    assert info.value is original
