"""Cross-turn conversation context.

Without this, every turn is a fresh model session. The agent could see the
*files* an earlier turn changed (the shadow is seeded from the working tree),
but not what it had said about them — so a follow-up like "that approach was
wrong, try something simpler" had no referent. The person is talking about a
conversation; the model was only ever handed one isolated task.

**What carries across turns, and what deliberately does not.**

Not the ReAct transcript. A single Full Agent turn is 5-8 model calls with full
file contents in the tool results; replaying that verbatim would make turn 5 cost
more than turns 1-4 combined, in a project whose entire premise is spending fewer
tokens than a naive agent. And most of it is dead weight — intermediate reads and
retries say nothing about what the person now wants changed.

What carries is a digest: for each prior turn, the request, whether it was
applied, what the agent said it did, and which files moved. That is what a
pronoun in "make that simpler" actually points at. It costs tens of tokens per
turn instead of thousands.

The digest is capped two ways — a turn count and a character budget — because an
unbounded history turns a long conversation into a slow, expensive one. Oldest
turns drop first; the recent ones are what follow-ups refer to.
"""

from __future__ import annotations

from dataclasses import dataclass

from qqcode.memory.session import TurnRecord

# How many prior turns can appear in the digest. Chosen for what a pronoun can
# plausibly reach: "that fix" means the last turn or two, never the twelfth one
# back. Raising this trades tokens for reach that people rarely use.
MAX_CONTEXT_TURNS = 6

# Hard ceiling on rendered digest size. The turn cap alone is not enough — one
# turn that touched 200 files would blow past any sane prompt budget on its own.
MAX_CONTEXT_CHARS = 4_000

# Per-turn caps, applied before the global budget. Keeps one verbose summary from
# crowding out five useful ones.
MAX_SUMMARY_CHARS = 300
MAX_FILES_LISTED = 8


@dataclass(frozen=True)
class ConversationContext:
    """Prior turns, rendered for injection into a prompt.

    `text` is empty for the first turn of a session, which is the signal for
    callers to inject nothing at all rather than a "no history" header.
    """

    text: str
    turns_included: int

    def is_empty(self) -> bool:
        return not self.text

    def __len__(self) -> int:
        return len(self.text)


def build_context(
    turns: list[TurnRecord],
    *,
    max_turns: int = MAX_CONTEXT_TURNS,
    max_chars: int = MAX_CONTEXT_CHARS,
) -> ConversationContext:
    """Render recent turns into a compact digest.

    Args:
        turns: The session's turn log, oldest first.
        max_turns: How many recent turns to consider.
        max_chars: Ceiling on the rendered text.

    Returns:
        A context whose `text` is empty when there is no usable history, so the
        first turn of a session behaves exactly as it did before this existed.
    """
    usable = [t for t in turns if _is_informative(t)]
    if not usable:
        return ConversationContext(text="", turns_included=0)

    recent = usable[-max_turns:]

    # Render newest-first so that truncation drops the oldest, then flip back to
    # chronological order for the prompt — models read a conversation forwards.
    blocks: list[str] = []
    total = 0
    for turn in reversed(recent):
        block = _render_turn(turn)
        if total + len(block) > max_chars:
            break
        blocks.append(block)
        total += len(block)

    if not blocks:
        return ConversationContext(text="", turns_included=0)

    body = "\n\n".join(reversed(blocks))
    return ConversationContext(
        text=f"## Earlier in this conversation\n\n{body}",
        turns_included=len(blocks),
    )


def _is_informative(turn: TurnRecord) -> bool:
    """Whether a turn tells the model anything worth paying for.

    Interrupted turns are dropped: the person cancelled mid-flight, so whatever
    the agent was doing was neither finished nor endorsed. Presenting it as
    history invites the model to resume abandoned work.

    Rejected and failed turns are kept, and are among the most valuable entries —
    "that approach was wrong" only makes sense if the wrong approach is on
    record.
    """
    return turn.outcome != "interrupted"


def _render_turn(turn: TurnRecord) -> str:
    lines = [f"### Request: {turn.task}", f"Outcome: {_outcome_phrase(turn)}"]

    if turn.summary:
        summary = turn.summary[:MAX_SUMMARY_CHARS]
        if len(turn.summary) > MAX_SUMMARY_CHARS:
            summary += "…"
        lines.append(f"What the agent reported: {summary}")

    if turn.changed_files:
        shown = list(turn.changed_files[:MAX_FILES_LISTED])
        listed = ", ".join(shown)
        if len(turn.changed_files) > MAX_FILES_LISTED:
            listed += f", and {len(turn.changed_files) - MAX_FILES_LISTED} more"
        lines.append(f"Files: {listed}")

    return "\n".join(lines)


def _outcome_phrase(turn: TurnRecord) -> str:
    """Plain-language outcome.

    Spelled out rather than passed through as a bare enum value: the model has to
    infer what it may build on, and "rejected" alone does not say who rejected it
    or whether the code still exists.
    """
    if turn.outcome == "accepted":
        return "applied to the repository (the change is present in the files now)"
    if turn.outcome == "rejected":
        return "the person declined this change (it was discarded, not in the files)"
    if turn.outcome == "failed":
        return "failed, no change was made"
    return turn.outcome
