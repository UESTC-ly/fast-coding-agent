"""LangGraph StateGraph for Full Agent and Sub-Agent execution.

The ReAct loop runs as a two-node graph:

    START → call_model ─(tool calls)──► run_tools → call_model → ...
                        └─(finished)──► END

Termination is set inside `call_model` (explicit finish, stuck, budget, max_turns)
and inside `run_tools` (stuck-on-same-error). The conditional edge reads
`finish_reason`; anything non-empty routes to END.

Sub-agents are compiled as separate, smaller graphs with the same structure but
restricted tools and turn budgets. They run synchronously inside the parent's
`run_tools` node via a `SpawnCallback` closure.
"""

from __future__ import annotations

import operator
import uuid
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from qqcode.models.billing import BilledClient
from qqcode.models.errors import BudgetExhaustedError
from qqcode.models.protocol import (
    Budget,
    ContentBlock,
    ModelTier,
    Msg,
    Phase,
    Role,
    TextContent,
    ToolSpec,
    ToolUseContent,
)
from qqcode.skills.index import SkillIndex
from qqcode.tools.artifacts import InMemoryArtifactStore
from qqcode.tools.builtins import TOOL_FINISH
from qqcode.tools.executor import SpawnCallback, ToolExecutor
from qqcode.tools.registry import ToolRegistry
from qqcode.workspace.protocol import Workspace

# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    messages: Annotated[list[Msg], operator.add]   # append-only reducer
    turns_used: int
    max_turns: int
    finish_reason: str    # "" = still running
    finish_summary: str
    files_touched: list[str]
    consecutive_empty: int
    last_error_key: str   # first 200 chars of last error, for streak detection
    error_streak: int


# ---------------------------------------------------------------------------
# Node builders (closures capture mutable context)
# ---------------------------------------------------------------------------


def _make_call_model(
    client: BilledClient,
    tools: list[ToolSpec],
    model_tier: ModelTier,
    phase: Phase = "fullagent",
) -> Any:
    def call_model(state: AgentState) -> dict[str, Any]:
        if state["finish_reason"]:
            return {}
        if state["turns_used"] >= state["max_turns"]:
            return {"finish_reason": "max_turns"}

        try:
            completion = client.invoke(
                messages=state["messages"],
                tools=tools,
                budget=Budget(max_tokens=4096),
                tier=model_tier,
                phase=phase,
            )
        except BudgetExhaustedError:
            return {"finish_reason": "budget"}
        except Exception as exc:
            return {"finish_reason": "error", "finish_summary": f"{type(exc).__name__}: {exc}"}

        assistant_msg = Msg(role=Role.ASSISTANT, content=completion.content)
        tool_calls = [b for b in completion.content if isinstance(b, ToolUseContent)]

        # Explicit finish via finish tool
        for call in tool_calls:
            if call.name == TOOL_FINISH:
                summary = str(call.input.get("summary", ""))
                return {
                    "messages": [assistant_msg],
                    "turns_used": state["turns_used"] + 1,
                    "finish_reason": "explicit",
                    "finish_summary": summary,
                    "consecutive_empty": 0,
                }

        # No tool calls: stuck detection
        if not tool_calls:
            new_empty = state["consecutive_empty"] + 1
            reason = "stuck" if new_empty >= 2 else ""
            return {
                "messages": [assistant_msg],
                "turns_used": state["turns_used"] + 1,
                "consecutive_empty": new_empty,
                "finish_reason": reason,
            }

        return {
            "messages": [assistant_msg],
            "turns_used": state["turns_used"] + 1,
            "consecutive_empty": 0,
        }

    return call_model


def _make_run_tools(executor: ToolExecutor) -> Any:
    def run_tools(state: AgentState) -> dict[str, Any]:
        last_msg = state["messages"][-1]
        tool_calls = [
            b for b in last_msg.content
            if isinstance(b, ToolUseContent) and b.name != TOOL_FINISH
        ]

        results: list[ContentBlock] = []
        last_error_key = state["last_error_key"]
        error_streak = state["error_streak"]
        finish_reason = ""

        for call in tool_calls:
            result = executor.execute(call)
            results.append(result)

            if result.is_error:
                key = (result.content[:200]
                       if isinstance(result.content, str)
                       else str(result.content)[:200])
                if key == last_error_key:
                    error_streak += 1
                    if error_streak >= 3:
                        finish_reason = "stuck"
                        break
                else:
                    last_error_key = key
                    error_streak = 1
            else:
                last_error_key = ""
                error_streak = 0

        # Role.TOOL, not Role.USER: the Anthropic adapter folds both to wire role
        # "user", but the OpenAI adapter only emits `{"role": "tool",
        # "tool_call_id": ...}` for Role.TOOL and silently drops
        # ToolResultContent from any other role.
        tool_msg = Msg(role=Role.TOOL, content=list[ContentBlock](results))
        files = list(state["files_touched"]) + list(executor.files_touched)
        return {
            "messages": [tool_msg],
            "files_touched": files,
            "last_error_key": last_error_key,
            "error_streak": error_streak,
            "finish_reason": finish_reason,
        }

    return run_tools


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def _route_after_model(state: AgentState) -> str:
    if state["finish_reason"]:
        return END
    last = state["messages"][-1]
    has_real_tools = any(
        isinstance(b, ToolUseContent) and b.name != TOOL_FINISH
        for b in last.content
    )
    return "run_tools" if has_real_tools else "call_model"


def _route_after_tools(state: AgentState) -> str:
    return END if state["finish_reason"] else "call_model"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def _compile_graph(
    client: BilledClient,
    executor: ToolExecutor,
    tools: list[ToolSpec],
    model_tier: ModelTier,
    phase: Phase = "fullagent",
) -> Any:
    graph: StateGraph[AgentState] = StateGraph(AgentState)
    graph.add_node("call_model", _make_call_model(client, tools, model_tier, phase))
    graph.add_node("run_tools", _make_run_tools(executor))
    graph.add_edge(START, "call_model")
    graph.add_conditional_edges(
        "call_model", _route_after_model,
        {"run_tools": "run_tools", END: END, "call_model": "call_model"},
    )
    graph.add_conditional_edges(
        "run_tools", _route_after_tools,
        {"call_model": "call_model", END: END},
    )
    return graph.compile(checkpointer=MemorySaver())


def build_full_agent_graph(
    client: BilledClient,
    executor: ToolExecutor,
    tools: list[ToolSpec],
    model_tier: ModelTier,
) -> Any:
    """Compile the Full Agent ReAct graph. Returns a runnable LangGraph graph."""
    return _compile_graph(client, executor, tools, model_tier)


# ---------------------------------------------------------------------------
# Sub-agent spawn callback
# ---------------------------------------------------------------------------


def make_spawn_callback(
    workspace: Workspace,
    client: BilledClient,
    skill_index: SkillIndex,
    registry: ToolRegistry,
) -> SpawnCallback:
    """Return a SpawnCallback that runs sub-agents as nested LangGraph graphs."""
    from qqcode.agents.subagent import get_preset  # avoid circular import at module level

    def spawn(preset_name: str, task: str) -> str:
        try:
            spec = get_preset(preset_name)
        except KeyError:
            return f"Unknown sub-agent preset: {preset_name!r}"

        store = InMemoryArtifactStore()
        sub_executor = ToolExecutor(
            workspace, store, skill_index,
            spawn_callback=None,  # sub-agents cannot spawn further
        )

        sub_tools = registry.specs_for(tier="subagent", allowed_tools=spec.allowed_tools)

        # Build system prompt, optionally with pinned skill bodies
        system = spec.system_prompt
        if spec.pinned_skills:
            bodies = []
            for name in spec.pinned_skills:
                skill = skill_index._skills.get(name)  # noqa: SLF001
                if skill:
                    bodies.append(f"## {name}\n{skill.body}")
            if bodies:
                system = system + "\n\n" + "\n\n".join(bodies)

        initial: AgentState = {
            "messages": [
                Msg(role=Role.SYSTEM, content=[TextContent(text=system)], cache_breakpoint=True),
                Msg(role=Role.USER, content=[TextContent(text=task)]),
            ],
            "turns_used": 0,
            "max_turns": spec.max_turns,
            "finish_reason": "",
            "finish_summary": "",
            "files_touched": [],
            "consecutive_empty": 0,
            "last_error_key": "",
            "error_streak": 0,
        }

        sub_graph = _compile_graph(client, sub_executor, sub_tools, spec.model_tier, phase="subagent")
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        final: AgentState = sub_graph.invoke(initial, config=config)

        return (
            final.get("finish_summary")
            or f"({spec.name} completed, reason={final.get('finish_reason')})"
        )

    return spawn
