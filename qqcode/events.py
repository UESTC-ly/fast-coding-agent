"""Agent progress events: what the agent is doing, while it does it.

The Full Agent runs a ReAct loop that can span dozens of tool calls. Batch mode
only needs the outcome, so the loop was silent by design. A conversation needs
the opposite: seeing which files were read and which commands ran is how a person
judges whether the agent understood the task — and how they know to interrupt
before it spends thirty turns down the wrong path.

Events are emitted through a plain callback rather than a queue or a generator:
the graph is synchronous, so a callback keeps the emission point in the same
stack frame as the work it describes. A slow or throwing consumer is the
consumer's problem — `emit` isolates exceptions so a rendering bug in the
terminal can never abort an agent run.

`on_event=None` means the same code path stays silent, which is what batch mode
and every existing test rely on.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

# What kind of thing happened. A closed set so renderers can switch on it
# exhaustively rather than pattern-matching free text.
EventKind = Literal[
    "turn_start",     # the model is about to be called
    "assistant_text", # the model said something between tool calls
    "tool_start",     # a tool call is about to run
    "tool_end",       # a tool call returned
    "finish",         # the loop terminated
]


@dataclass(frozen=True)
class AgentEvent:
    """One observable step in an agent run.

    Args:
        kind: Which stage this describes.
        turn: 1-based turn index, 0 when not tied to a turn.
        tool: Tool name for `tool_start` / `tool_end`, else "".
        detail: Human-readable summary — the file path, the command, the
            assistant's text. Already truncated for display.
        is_error: Only meaningful for `tool_end`.
        meta: Extra structured fields for renderers that want them. Kept
            separate from `detail` so a renderer never has to parse prose.
    """

    kind: EventKind
    turn: int = 0
    tool: str = ""
    detail: str = ""
    is_error: bool = False
    meta: dict[str, str] = field(default_factory=dict)


EventCallback = Callable[[AgentEvent], None]

# Tool inputs can hold whole file bodies. Events are for display, so they carry a
# short label rather than the payload; the full content is already in the diff
# the person reviews at the end of the turn.
MAX_DETAIL_CHARS = 120


def emit(on_event: EventCallback | None, event: AgentEvent) -> None:
    """Deliver an event, swallowing consumer failures.

    A renderer that raises must not kill the agent run that was merely
    describing itself. This is the one place in the codebase where catching
    broadly is correct: the exception belongs to the consumer, the work it
    describes has already happened, and there is nothing for the agent to
    recover from.
    """
    if on_event is None:
        return
    with contextlib.suppress(Exception):  # see docstring
        on_event(event)


def describe_tool_call(name: str, tool_input: dict[str, object]) -> str:
    """A one-line label for a tool call, in the terms the person cares about.

    Reads and writes are identified by path, commands by their argv. Anything
    unrecognised falls back to its key names rather than its values, because an
    unknown tool's values may be arbitrarily large.
    """
    # Keys are the builtin tools' actual parameter names — `cmd`, not `command`;
    # see qqcode/tools/builtins.py. Getting one wrong is invisible in tests that
    # only assert "some detail was produced", because the fallback still returns
    # something: the key names instead of the value.
    for key in ("path", "cmd", "pattern", "name", "preset", "artifact_id", "summary"):
        if key not in tool_input:
            continue
        value = tool_input[key]
        text = " ".join(str(v) for v in value) if isinstance(value, list) else str(value)
        if text.strip():
            return _clip(text)

    if not tool_input:
        return ""
    return _clip(", ".join(sorted(tool_input)))


def _clip(text: str) -> str:
    """Collapse to one line and cap the length."""
    flat = " ".join(text.split())
    if len(flat) <= MAX_DETAIL_CHARS:
        return flat
    return flat[:MAX_DETAIL_CHARS] + "…"
