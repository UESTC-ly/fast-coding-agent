"""Artifact store and tool-result compression.

Every tool result — builtin or MCP — passes through `build_tool_result` before
it reaches model context. Output above the inline budget is parked in the store
and replaced by a head/tail excerpt plus an artifact id the agent can re-read
on demand.

This is the only channel, and MCP gets no exemption: a single tool returning a
50k-token JSON blob would otherwise blow the window on its own.
"""

from __future__ import annotations

from typing import Protocol

from qqcode.models.protocol import ToolResultContent


class ArtifactStore(Protocol):
    """Out-of-context storage for oversized tool output."""

    def put(self, content: str) -> str:
        """Store content and return its artifact id."""
        ...

    def get(self, artifact_id: str) -> str:
        """Retrieve stored content.

        Raises:
            KeyError: No artifact under that id.
        """
        ...


class InMemoryArtifactStore:
    """Process-local artifact store, scoped to a single task run."""

    def __init__(self) -> None:
        self._items: dict[str, str] = {}

    def put(self, content: str) -> str:
        artifact_id = f"art_{len(self._items) + 1:04d}"
        self._items[artifact_id] = content
        return artifact_id

    def get(self, artifact_id: str) -> str:
        try:
            return self._items[artifact_id]
        except KeyError:
            raise KeyError(f"Unknown artifact id: {artifact_id}") from None

    def __len__(self) -> int:
        return len(self._items)


class ResultPolicy:
    """Inline budget for tool results.

    Immutable by convention; construct a new instance to change limits.
    """

    __slots__ = ("head_chars", "max_inline_chars", "tail_chars")

    def __init__(
        self,
        max_inline_chars: int = 4000,
        head_chars: int = 1500,
        tail_chars: int = 1000,
    ) -> None:
        if min(max_inline_chars, head_chars, tail_chars) <= 0:
            raise ValueError("ResultPolicy limits must be positive")
        # Otherwise the "compressed" form could exceed the budget it exists to enforce.
        if head_chars + tail_chars >= max_inline_chars:
            raise ValueError(
                f"head_chars + tail_chars ({head_chars + tail_chars}) must be "
                f"below max_inline_chars ({max_inline_chars})"
            )
        self.max_inline_chars = max_inline_chars
        self.head_chars = head_chars
        self.tail_chars = tail_chars


DEFAULT_RESULT_POLICY = ResultPolicy()


def build_tool_result(
    tool_use_id: str,
    content: str,
    *,
    store: ArtifactStore,
    policy: ResultPolicy = DEFAULT_RESULT_POLICY,
    is_error: bool = False,
) -> ToolResultContent:
    """Wrap raw tool output for model consumption, compressing when oversized.

    Args:
        tool_use_id: Id of the tool call this answers.
        content: Raw tool output.
        store: Where full output goes when it exceeds the inline budget.
        policy: Inline budget and excerpt sizes.
        is_error: Whether the tool failed.

    Returns:
        A result block within the inline budget. Oversized output is excerpted
        and its artifact id embedded in the text.
    """
    if len(content) <= policy.max_inline_chars:
        return ToolResultContent(tool_use_id=tool_use_id, content=content, is_error=is_error)

    artifact_id = store.put(content)
    omitted = len(content) - policy.head_chars - policy.tail_chars
    excerpt = (
        f"{content[: policy.head_chars]}\n\n"
        f"[... {omitted} chars omitted. Full output stored as {artifact_id} — "
        f"read it with read_artifact if you need the rest ...]\n\n"
        f"{content[-policy.tail_chars :]}"
    )
    return ToolResultContent(tool_use_id=tool_use_id, content=excerpt, is_error=is_error)
