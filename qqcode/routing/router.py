"""M3 Routing: three-layer intelligent task classification.

L0 (Static Features) → L1 (Cheap Classifier) → L2 (Hard Gate)

L0: Zero-cost heuristics (file count, task length, keywords).
L1: Single cheap model call with structured output.
L2: Deterministic code gates (no model judgment).

Output: FastPath or FullAgent decision + file hints for shadow workspace.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from qqcode.models.billing import BilledClient
from qqcode.models.protocol import (
    Budget,
    ModelTier,
    Msg,
    OutputSpec,
    Role,
    TextContent,
    ToolUseContent,
)
from qqcode.skills import RoutingHint, SkillIndex


class RoutingDecision(StrEnum):
    """Routing outcome."""

    FASTPATH = "fastpath"
    FULLAGENT = "fullagent"


@dataclass(frozen=True)
class RoutingResult:
    """Classification result with file hints."""

    decision: RoutingDecision
    confidence: float  # 0.0-1.0 from L1, or 1.0 from L0/L2
    files_hint: tuple[str, ...]  # Expected touched files
    reasoning: str = ""  # L1 explanation or L0/L2 rule trigger
    # Trace metadata — which layer decided and what the raw L1 said.
    layer: str = ""             # "l0" | "l1_l2" | "fallback"
    l0_triggered: bool = False
    l1_decision: str = ""       # Raw L1 verdict; empty when L0 fired
    l1_confidence: float = 0.0  # Raw L1 confidence; 0.0 when L0 fired
    l2_override: bool = False   # True when L2 overruled L1


# L0 static thresholds — defaults; override via RoutingThresholds for replay.
MAX_FASTPATH_TASK_LENGTH = 500  # chars
MAX_FASTPATH_FILES = 3
FULLMUST_KEYWORDS = frozenset({
    "refactor",
    "architecture",
    "migrate",
    "redesign",
    "investigate",
    "debug complex",
    "multiple modules",
})


@dataclass(frozen=True)
class RoutingThresholds:
    """Calibration knobs for the routing pipeline.

    Thread through route_task() to simulate different settings without
    changing production defaults. Default values mirror the module constants
    so production behaviour is unchanged when this object is not supplied.
    """

    confidence: float = 0.7        # τ — L2 escalates when L1 conf < this
    max_task_length: int = 500     # L — L0 escalates when task longer than this
    max_files: int = 3             # K — L2 escalates when files_hint > this


DEFAULT_THRESHOLDS = RoutingThresholds()


L1_CLASSIFIER_SPEC = OutputSpec(
    tool_name="classify_task",
    schema={
        "type": "object",
        "required": ["decision", "confidence", "files", "reasoning"],
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["fastpath", "fullagent"],
                "description": "Routing decision",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence in decision",
            },
            "files": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Expected file paths to modify (relative to repo root)",
            },
            "reasoning": {
                "type": "string",
                "description": "Why this routing decision",
            },
        },
    },
)


def route_task(
    task: str,
    skill_index: SkillIndex,
    client: BilledClient | None = None,
    thresholds: RoutingThresholds | None = None,
) -> RoutingResult:
    """Route task through L0 → L1 → L2 pipeline.

    Args:
        task: User's task description.
        skill_index: Skills for routing hint extraction.
        client: Billed model client for L1 (if None, L0+L2 only).
        thresholds: Calibration overrides; defaults to DEFAULT_THRESHOLDS.

    Returns:
        RoutingResult with decision, file hints, and trace metadata.
    """
    t = thresholds or DEFAULT_THRESHOLDS

    # L0: Static features
    l0_result = _l0_classify(task, skill_index, t)
    if l0_result:
        return l0_result

    # L1: Cheap classifier (requires client)
    if client:
        l1_result = _l1_classify(task, client)
        if l1_result:
            # L2: Hard gate on L1 output
            return _l2_gate(l1_result, t)

    # Fallback: no client or L1 failed → default to FastPath
    return RoutingResult(
        decision=RoutingDecision.FASTPATH,
        confidence=0.5,
        files_hint=(),
        reasoning="L1 unavailable, defaulting to FastPath",
        layer="fallback",
    )


def _l0_classify(
    task: str, skill_index: SkillIndex, thresholds: RoutingThresholds
) -> RoutingResult | None:
    """L0: Zero-cost static classification.

    Returns routing decision if confident, None if evidence insufficient.
    """
    # Force FullAgent for complex keywords
    task_lower = task.lower()
    if any(kw in task_lower for kw in FULLMUST_KEYWORDS):
        return RoutingResult(
            decision=RoutingDecision.FULLAGENT,
            confidence=1.0,
            files_hint=(),
            reasoning="L0: Complex keyword detected",
            layer="l0",
            l0_triggered=True,
        )

    # Force FullAgent if task is too long
    if len(task) > thresholds.max_task_length:
        return RoutingResult(
            decision=RoutingDecision.FULLAGENT,
            confidence=1.0,
            files_hint=(),
            reasoning=f"L0: Task length {len(task)} exceeds FastPath budget",
            layer="l0",
            l0_triggered=True,
        )

    # Skill routing hint
    matched_skills = skill_index.match(task=task, paths=())
    hint = skill_index.routing_hint(matched_skills)
    if hint == RoutingHint.FULL:
        return RoutingResult(
            decision=RoutingDecision.FULLAGENT,
            confidence=1.0,
            files_hint=(),
            reasoning="L0: Skill routing hint demands FullAgent",
            layer="l0",
            l0_triggered=True,
        )
    if hint == RoutingHint.FAST:
        return RoutingResult(
            decision=RoutingDecision.FASTPATH,
            confidence=0.85,
            files_hint=(),
            reasoning="L0: Skill routing hint suggests FastPath",
            layer="l0",
            l0_triggered=True,
        )

    # Insufficient evidence → proceed to L1
    return None


def _l1_classify(task: str, client: BilledClient) -> RoutingResult | None:
    """L1: Cheap model call with structured output.

    Returns classification or None on error. The returned RoutingResult carries
    l1_decision and l1_confidence so replay can re-apply L2 without another call.
    """
    system = """You are a task routing classifier. Analyze the task and decide:
- FastPath: Well-defined, 1-3 files, clear requirements, no exploration needed
- FullAgent: Complex, multi-file, requires investigation or iteration

Output ONLY the classify_task tool call."""

    messages = [
        Msg(role=Role.SYSTEM, content=[TextContent(text=system)]),
        Msg(role=Role.USER, content=[TextContent(text=f"Task: {task}")]),
    ]

    try:
        completion = client.invoke(
            messages=messages,
            output_spec=L1_CLASSIFIER_SPEC,
            budget=Budget(max_tokens=512),
            tier=ModelTier.FAST,
            phase="routing",
        )
    except Exception:
        return None

    # Extract classification
    for block in completion.content:
        if isinstance(block, ToolUseContent) and block.name == "classify_task":
            data = block.input
            raw_decision = data["decision"]
            raw_conf = float(data["confidence"])
            return RoutingResult(
                decision=RoutingDecision(raw_decision),
                confidence=raw_conf,
                files_hint=tuple(data.get("files", [])),
                reasoning=data.get("reasoning", ""),
                layer="l1_l2",
                l1_decision=raw_decision,
                l1_confidence=raw_conf,
            )

    return None


def _l2_gate(l1_result: RoutingResult, thresholds: RoutingThresholds) -> RoutingResult:
    """L2: Deterministic hard gate on L1 classification.

    Rules:
    - If L1 says FastPath but files > thresholds.max_files → override to FullAgent
    - If L1 confidence < thresholds.confidence → override to FullAgent
    - Otherwise: accept L1 decision
    """
    if l1_result.decision == RoutingDecision.FASTPATH:
        if len(l1_result.files_hint) > thresholds.max_files:
            return RoutingResult(
                decision=RoutingDecision.FULLAGENT,
                confidence=1.0,
                files_hint=l1_result.files_hint,
                reasoning=f"L2: {len(l1_result.files_hint)} files exceed FastPath limit",
                layer="l1_l2",
                l1_decision=l1_result.l1_decision,
                l1_confidence=l1_result.l1_confidence,
                l2_override=True,
            )
        if l1_result.confidence < thresholds.confidence:
            return RoutingResult(
                decision=RoutingDecision.FULLAGENT,
                confidence=1.0,
                files_hint=l1_result.files_hint,
                reasoning=f"L2: Low confidence {l1_result.confidence:.2f}, escalating to FullAgent",
                layer="l1_l2",
                l1_decision=l1_result.l1_decision,
                l1_confidence=l1_result.l1_confidence,
                l2_override=True,
            )

    return l1_result
