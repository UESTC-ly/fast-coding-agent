"""Live smoke test against real provider endpoints.

Verifies the M2 adapter layer and M3 routing path end to end with the two models
the user pinned:

- OpenAI  : gpt-5.6-luna
- Anthropic: claude-sonnet-5

Every tier is pinned to the same model per provider (`uniform_tiers`) so the two
providers' numbers stay comparable — otherwise FAST would silently pick a cheaper
model on one side and the token totals would mean different things.

What this covers, and what it does not:

covered      — credentials load, base_url routing, structured output round-trip,
               canonical stop_reason mapping, usage normalization, ledger
               accounting by phase, L1 classifier + L2 hard gate on live output.
NOT covered  — hidden acceptance execution. `CommandGuard.ALLOW_LIST` has no
               `sh`, so FastPath's acceptance hook cannot run yet (ledger risk
               R7). This script therefore exercises routing and patch generation
               only, and says so in its output rather than implying a full
               three-condition pass.

Usage:
    .venv/bin/python scripts/smoke_test.py             # both providers
    .venv/bin/python scripts/smoke_test.py anthropic   # one provider
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass

from qqcode.config import Config
from qqcode.models.factory import build_client, uniform_tiers
from qqcode.models.protocol import (
    Budget,
    CostLedger,
    ModelTier,
    Msg,
    OutputSpec,
    Role,
    TextContent,
    ToolSpec,
    ToolUseContent,
)
from qqcode.routing import RoutingDecision, route_task
from qqcode.skills import SkillIndex

# Models pinned by the user for this run.
SMOKE_MODELS: dict[str, str] = {
    "openai": "gpt-5.6-luna",
    "anthropic": "claude-sonnet-5",
}

PROBE_SPEC = OutputSpec(
    tool_name="report",
    schema={
        "type": "object",
        "required": ["language", "confident"],
        "properties": {
            "language": {"type": "string", "description": "Programming language named in the text"},
            "confident": {"type": "boolean"},
        },
    },
)

WEATHER_TOOL = ToolSpec(
    name="get_weather",
    description="Look up current weather for a city.",
    input_schema={
        "type": "object",
        "required": ["city"],
        "properties": {"city": {"type": "string"}},
    },
)


@dataclass
class CheckResult:
    """Outcome of one named check."""

    name: str
    ok: bool
    detail: str


def _ok(name: str, detail: str) -> CheckResult:
    return CheckResult(name=name, ok=True, detail=detail)


def _fail(name: str, detail: str) -> CheckResult:
    return CheckResult(name=name, ok=False, detail=detail)


def check_plain_text(client: object, ledger: CostLedger) -> CheckResult:
    """A bare completion returns text and bills to the ledger."""
    name = "plain text + ledger"
    before = ledger.automatic_total
    try:
        completion = client.invoke(  # type: ignore[attr-defined]
            messages=[
                Msg(role=Role.SYSTEM, content=[TextContent(text="Answer in one word.")]),
                Msg(role=Role.USER, content=[TextContent(text="Capital of France?")]),
            ],
            budget=Budget(max_tokens=64),
            tier=ModelTier.BALANCED,
            phase="fullagent",
        )
    except Exception as exc:
        return _fail(name, f"{type(exc).__name__}: {exc}")

    text = " ".join(b.text for b in completion.content if isinstance(b, TextContent)).strip()
    spent = ledger.automatic_total - before
    if not text:
        return _fail(name, f"empty text; stop_reason={completion.stop_reason}")
    if spent <= 0:
        return _fail(name, "ledger did not move")
    return _ok(
        name,
        f"{text[:40]!r} · stop={completion.stop_reason} · "
        f"in={completion.usage.input_tokens} out={completion.usage.output_tokens}",
    )


def check_structured_output(client: object, ledger: CostLedger) -> CheckResult:
    """Forced structured output arrives as ToolUseContent on both providers."""
    name = "structured output"
    try:
        completion = client.invoke(  # type: ignore[attr-defined]
            messages=[
                Msg(role=Role.SYSTEM, content=[TextContent(text="Call the report tool.")]),
                Msg(
                    role=Role.USER,
                    content=[TextContent(text="I have been writing Rust all afternoon.")],
                ),
            ],
            output_spec=PROBE_SPEC,
            budget=Budget(max_tokens=256),
            tier=ModelTier.BALANCED,
            phase="routing",
        )
    except Exception as exc:
        return _fail(name, f"{type(exc).__name__}: {exc}")

    blocks = [b for b in completion.content if isinstance(b, ToolUseContent)]
    if not blocks:
        return _fail(name, f"no ToolUseContent; got {completion.content!r}")
    block = blocks[0]
    if block.name != PROBE_SPEC.tool_name:
        return _fail(name, f"wrong tool name: {block.name}")
    if "language" not in block.input:
        return _fail(name, f"schema not honored: {block.input!r}")
    return _ok(name, f"{block.input!r} · stop={completion.stop_reason}")


def check_real_tool(client: object, ledger: CostLedger) -> CheckResult:
    """A real tool offer produces a tool_use stop reason and a parsed call."""
    name = "real tool call"
    try:
        completion = client.invoke(  # type: ignore[attr-defined]
            messages=[
                Msg(role=Role.USER, content=[TextContent(text="What is the weather in Kyoto?")]),
            ],
            tools=[WEATHER_TOOL],
            budget=Budget(max_tokens=256),
            tier=ModelTier.BALANCED,
            phase="fullagent",
        )
    except Exception as exc:
        return _fail(name, f"{type(exc).__name__}: {exc}")

    calls = [b for b in completion.content if isinstance(b, ToolUseContent)]
    if not calls:
        return _fail(name, f"model declined the tool; stop={completion.stop_reason}")
    if completion.stop_reason != "tool_use":
        return _fail(name, f"stop_reason not canonicalized: {completion.stop_reason}")
    return _ok(name, f"{calls[0].name}({calls[0].input!r}) · stop={completion.stop_reason}")


def check_mutual_exclusion(client: object, ledger: CostLedger) -> CheckResult:
    """Tools + output_spec is rejected locally, before any network call."""
    name = "structured ⊥ tools"
    before = ledger.calls
    try:
        client.invoke(  # type: ignore[attr-defined]
            messages=[Msg(role=Role.USER, content=[TextContent(text="hi")])],
            tools=[WEATHER_TOOL],
            output_spec=PROBE_SPEC,
            budget=Budget(max_tokens=64),
            tier=ModelTier.BALANCED,
            phase="fullagent",
        )
    except ValueError as exc:
        if ledger.calls != before:
            return _fail(name, "raised but still billed a call")
        return _ok(name, f"rejected pre-send: {str(exc)[:60]}")
    except Exception as exc:
        return _fail(name, f"wrong exception type {type(exc).__name__}: {exc}")
    return _fail(name, "no exception raised")


def check_routing(client: object, ledger: CostLedger) -> CheckResult:
    """L1 classifier runs live and L2 gates its output."""
    name = "L1 + L2 routing"
    index = SkillIndex()
    before = ledger.routing_tokens
    try:
        result = route_task("add a docstring to the parse_args function in cli.py", index, client)  # type: ignore[arg-type]
    except Exception as exc:
        return _fail(name, f"{type(exc).__name__}: {exc}")

    if ledger.routing_tokens == before:
        return _fail(name, "no routing tokens billed — L1 was skipped or failed silently")
    if result.decision not in (RoutingDecision.FASTPATH, RoutingDecision.FULLAGENT):
        return _fail(name, f"invalid decision {result.decision!r}")
    return _ok(
        name,
        f"{result.decision.value} conf={result.confidence:.2f} "
        f"files={list(result.files_hint)} · {result.reasoning[:60]}",
    )


CHECKS = (
    check_plain_text,
    check_structured_output,
    check_real_tool,
    check_mutual_exclusion,
    check_routing,
)


def run_provider(config: Config, provider: str) -> tuple[list[CheckResult], CostLedger]:
    """Run every check against one provider."""
    model = SMOKE_MODELS[provider]
    client, ledger = build_client(
        config, provider=provider, tier_models=uniform_tiers(model)
    )

    results = []
    for check in CHECKS:
        try:
            results.append(check(client, ledger))
        except Exception:
            results.append(_fail(check.__name__, traceback.format_exc(limit=2)))
    return results, ledger


def main(argv: list[str]) -> int:
    """Run the smoke test; returns a process exit code."""
    requested = argv[1:] or ["anthropic", "openai"]
    unknown = [p for p in requested if p not in SMOKE_MODELS]
    if unknown:
        print(f"unknown provider(s): {unknown}; valid: {sorted(SMOKE_MODELS)}")
        return 2

    config = Config.from_env()
    all_ok = True

    for provider in requested:
        model = SMOKE_MODELS[provider]
        print(f"\n{'=' * 68}\n{provider}  ·  {model}\n{'=' * 68}")

        try:
            results, ledger = run_provider(config, provider)
        except Exception as exc:
            print(f"  SETUP FAILED  {type(exc).__name__}: {exc}")
            all_ok = False
            continue

        for r in results:
            mark = "PASS" if r.ok else "FAIL"
            print(f"  [{mark}] {r.name}")
            for line in r.detail.splitlines():
                print(f"         {line}")
            all_ok = all_ok and r.ok

        s = ledger.summary()
        print(
            f"\n  ledger: calls={s['calls']} retried={s['retried_calls']} "
            f"in={s['total_input']} out={s['total_output']} "
            f"cache_read={s['cache_read']} total={s['automatic_total']}"
        )
        print(f"  by_phase: {s['by_phase']}")

    print(
        "\nNOTE: hidden acceptance execution is NOT covered — CommandGuard has no "
        "`sh`, so FastPath's acceptance hook cannot run (ledger risk R7).\n"
        "This run verifies adapters, billing, and routing only."
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
