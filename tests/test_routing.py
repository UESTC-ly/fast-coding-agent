"""Tests for routing: L0/L1/L2 classification and FastPath execution."""

from __future__ import annotations

from unittest.mock import Mock

from qqcode.models.billing import BilledClient
from qqcode.models.protocol import (
    Completion,
    ModelTier,
    ToolUseContent,
    Usage,
)
from qqcode.routing import RoutingDecision, route_task
from qqcode.skills import RoutingHint, Skill, SkillIndex


def test_l0_forces_fullagent_on_complex_keyword() -> None:
    """L0 detects complex keywords and routes to FullAgent."""
    index = SkillIndex()
    result = route_task("refactor the authentication module", index, client=None)

    assert result.decision == RoutingDecision.FULLAGENT
    assert result.confidence == 1.0
    assert "L0: Complex keyword" in result.reasoning


def test_l0_forces_fullagent_on_long_task() -> None:
    """L0 detects task length exceeding FastPath budget."""
    index = SkillIndex()
    long_task = "x" * 600  # > MAX_FASTPATH_TASK_LENGTH (500)
    result = route_task(long_task, index, client=None)

    assert result.decision == RoutingDecision.FULLAGENT
    assert result.confidence == 1.0
    assert "exceeds FastPath budget" in result.reasoning


def test_l0_respects_skill_routing_hint_full() -> None:
    """L0 honors skill routing hint demanding FullAgent."""
    skill = Skill(
        name="test",
        description="test skill",
        body="test",
        keywords=("special",),
        routing_hint=RoutingHint.FULL,
    )
    index = SkillIndex([skill])
    result = route_task("do something special", index, client=None)

    assert result.decision == RoutingDecision.FULLAGENT
    assert result.confidence == 1.0
    assert "routing hint demands FullAgent" in result.reasoning


def test_l0_respects_skill_routing_hint_fast() -> None:
    """L0 honors skill routing hint suggesting FastPath."""
    skill = Skill(
        name="test",
        description="test skill",
        body="test",
        keywords=("simple",),
        routing_hint=RoutingHint.FAST,
    )
    index = SkillIndex([skill])
    result = route_task("do something simple", index, client=None)

    assert result.decision == RoutingDecision.FASTPATH
    assert result.confidence == 0.85
    assert "routing hint suggests FastPath" in result.reasoning


def test_l1_classifier_called_when_l0_insufficient() -> None:
    """L1 classifier is invoked when L0 cannot decide."""
    index = SkillIndex()

    # Mock BilledClient returning L1 classification
    mock_client = Mock(spec=BilledClient)
    mock_client.invoke.return_value = Completion(
        content=[
            ToolUseContent(
                id="call_1",
                name="classify_task",
                input={
                    "decision": "fastpath",
                    "confidence": 0.9,
                    "files": ["foo.py"],
                    "reasoning": "Simple task",
                },
            )
        ],
        stop_reason="tool_use",
        usage=Usage(input_tokens=100, output_tokens=50),
        raw={},
    )

    result = route_task("fix typo in foo.py", index, client=mock_client)

    # L1 was called
    assert mock_client.invoke.called
    call_kwargs = mock_client.invoke.call_args.kwargs
    assert call_kwargs["phase"] == "routing"
    assert call_kwargs["tier"] == ModelTier.FAST

    # L2 accepted L1 decision
    assert result.decision == RoutingDecision.FASTPATH
    assert result.confidence == 0.9
    assert result.files_hint == ("foo.py",)


def test_l2_overrides_fastpath_when_too_many_files() -> None:
    """L2 gate overrides FastPath when file count exceeds limit."""
    index = SkillIndex()

    mock_client = Mock(spec=BilledClient)
    mock_client.invoke.return_value = Completion(
        content=[
            ToolUseContent(
                id="call_1",
                name="classify_task",
                input={
                    "decision": "fastpath",
                    "confidence": 0.95,
                    "files": ["a.py", "b.py", "c.py", "d.py"],  # 4 files > MAX (3)
                    "reasoning": "Looks simple",
                },
            )
        ],
        stop_reason="tool_use",
        usage=Usage(input_tokens=100, output_tokens=50),
        raw={},
    )

    result = route_task("update four files", index, client=mock_client)

    assert result.decision == RoutingDecision.FULLAGENT
    assert result.confidence == 1.0
    assert "exceed FastPath limit" in result.reasoning


def test_l2_overrides_fastpath_when_low_confidence() -> None:
    """L2 gate escalates to FullAgent when L1 confidence is low."""
    index = SkillIndex()

    mock_client = Mock(spec=BilledClient)
    mock_client.invoke.return_value = Completion(
        content=[
            ToolUseContent(
                id="call_1",
                name="classify_task",
                input={
                    "decision": "fastpath",
                    "confidence": 0.6,  # < 0.7 threshold
                    "files": ["foo.py"],
                    "reasoning": "Not sure",
                },
            )
        ],
        stop_reason="tool_use",
        usage=Usage(input_tokens=100, output_tokens=50),
        raw={},
    )

    result = route_task("ambiguous task", index, client=mock_client)

    assert result.decision == RoutingDecision.FULLAGENT
    assert result.confidence == 1.0
    assert "Low confidence" in result.reasoning


def test_fallback_to_fastpath_when_no_client() -> None:
    """When client is None and L0 cannot decide, default to FastPath."""
    index = SkillIndex()
    result = route_task("simple task without obvious signals", index, client=None)

    assert result.decision == RoutingDecision.FASTPATH
    assert result.confidence == 0.5
    assert "L1 unavailable" in result.reasoning


def test_fallback_to_fastpath_when_l1_fails() -> None:
    """When L1 model call fails, fallback to FastPath."""
    index = SkillIndex()

    mock_client = Mock(spec=BilledClient)
    mock_client.invoke.side_effect = RuntimeError("API error")

    result = route_task("task with L1 error", index, client=mock_client)

    assert result.decision == RoutingDecision.FASTPATH
    assert result.confidence == 0.5
    assert "L1 unavailable" in result.reasoning
