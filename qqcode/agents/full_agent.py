"""Full Agent execution: LangGraph-powered ReAct loop.

Replaces the hand-written while loop with a LangGraph StateGraph so that:
- Each node (call_model, run_tools) is an isolated, testable unit
- State is checkpointed in memory per run (MemorySaver)
- Sub-agents run as nested compiled graphs via the spawn callback

The external interface is unchanged: execute_full_agent(inp, workspace, client)
→ FullAgentResult. All existing tests pass without modification.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from qqcode.agents.graph import AgentState, build_full_agent_graph, make_spawn_callback
from qqcode.events import EventCallback
from qqcode.models.billing import BilledClient
from qqcode.models.protocol import (
    ModelTier,
    Msg,
    Role,
    TextContent,
)
from qqcode.skills.index import SkillIndex
from qqcode.tools.artifacts import InMemoryArtifactStore
from qqcode.tools.builtins import default_registry
from qqcode.tools.executor import ToolExecutor
from qqcode.tools.registry import ToolRegistry
from qqcode.workspace.protocol import Workspace, WorkspaceSnapshot


@dataclass(frozen=True)
class FullAgentInput:
    """Task and context for Full Agent execution."""

    task: str
    baseline: WorkspaceSnapshot
    skill_index: SkillIndex
    tool_registry: ToolRegistry | None = None  # None = use default_registry()
    escalation_context: str = ""
    max_turns: int = 30
    model_tier: ModelTier = ModelTier.BALANCED
    # Digest of earlier turns in the same conversation. Empty in batch mode,
    # where each task is independent by definition.
    history: str = ""
    # Progress callback. None keeps the loop silent, which is what batch mode
    # and every existing test expect.
    on_event: EventCallback | None = None


@dataclass
class FullAgentResult:
    """What the Full Agent hands back after N turns."""

    success: bool
    final_snapshot: WorkspaceSnapshot
    changed_files: frozenset[str]
    reasoning: str
    turns_used: int
    finish_reason: str  # "explicit" | "max_turns" | "budget" | "stuck" | "error"
    messages: list[Msg] = field(default_factory=list)
    error: str | None = None


def execute_full_agent(
    inp: FullAgentInput,
    workspace: Workspace,
    client: BilledClient,
) -> FullAgentResult:
    """Run the Full Agent loop via LangGraph until termination."""
    registry = inp.tool_registry or default_registry()
    store = InMemoryArtifactStore()

    spawn_cb = make_spawn_callback(workspace, client, inp.skill_index, registry)
    executor = ToolExecutor(workspace, store, inp.skill_index, spawn_callback=spawn_cb)

    tools = registry.specs_for(tier="fullagent")
    graph = build_full_agent_graph(client, executor, tools, inp.model_tier, inp.on_event)

    system = _build_system_prompt(inp)
    initial: AgentState = {
        "messages": [
            Msg(role=Role.SYSTEM, content=[TextContent(text=system)], cache_breakpoint=True),
            Msg(role=Role.USER, content=[TextContent(text=f"Task: {inp.task}")]),
        ],
        "turns_used": 0,
        "max_turns": inp.max_turns,
        "finish_reason": "",
        "finish_summary": "",
        "files_touched": [],
        "consecutive_empty": 0,
        "last_error_key": "",
        "error_streak": 0,
    }

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    final: AgentState = graph.invoke(initial, config=config)

    return _build_result(final, workspace, inp.baseline)


def _build_system_prompt(inp: FullAgentInput) -> str:
    parts = [
        "You are a Full Agent that solves coding tasks through tool use.",
        "",
        "Call tools to read code, write changes, and verify they work. "
        "When the task is complete and verified, call the `finish` tool.",
        "",
        "Rules:",
        "- Read before writing. Understand existing conventions before changing them.",
        "- Run tests or checks after making changes. Broken code is never done.",
        "- If stuck, explain what blocks you rather than looping on the same error.",
        "",
        "Scope:",
        "- Change only what the task requires. Do not refactor, reformat, or "
        "'improve' surrounding code — unrelated edits are indistinguishable "
        "from bugs to whoever reviews this.",
        "- Do not add, edit, or delete test files unless the task explicitly asks "
        "for tests. This work is verified by tests you cannot see; editing tests "
        "cannot make the task pass and may make correct work score as a failure.",
        "- Verify with the project's existing test command. A test you wrote "
        "yourself proves nothing about the real criteria.",
        "",
        "Efficiency:",
        "- Do not re-read a file you have already read; its content is above.",
        "- If the same call fails twice the same way, the approach is wrong. "
        "Change it instead of retrying.",
        "- Prefer targeted edits over rewriting a whole file.",
        "- Call `finish` as soon as the change is made and verified. Extra turns "
        "spent re-checking finished work are pure cost.",
    ]
    if inp.history:
        # Before the escalation context: the conversation is the frame the task
        # sits in, whereas an escalation is a detail about this attempt.
        parts += ["", inp.history]
    if inp.escalation_context:
        parts += ["", "## Previous FastPath Attempt", inp.escalation_context]
    return "\n".join(parts)


def _build_result(
    final: AgentState,
    workspace: Workspace,
    baseline: WorkspaceSnapshot,
) -> FullAgentResult:
    snapshot = workspace.snapshot()
    changed = baseline.changed_files(snapshot)
    finish_reason = final.get("finish_reason", "error")
    summary = final.get("finish_summary", "")
    error: str | None = None

    if finish_reason == "error":
        error = summary
        summary = ""

    success = finish_reason == "explicit"

    return FullAgentResult(
        success=success,
        final_snapshot=snapshot,
        changed_files=frozenset(changed),
        reasoning=summary,
        turns_used=final.get("turns_used", 0),
        finish_reason=finish_reason,
        messages=final.get("messages", []),
        error=error,
    )
