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


# ---------------------------------------------------------------------------
# L0 → L1 prefetch-hint recovery (方案一)
#
# L0 decides *that* a task is simple without saying *which* files it touches, so
# FastPath arrived with nothing to inline. Recovering names from the task text
# only works when the text names one; measured on 5 SWE-bench statements, zero
# did, and all 4 such runs declined at 23k-44k tokens against 5.8k for the same
# task when a hint existed. L1 already names files as a by-product of
# classifying, and one L1 call is cheaper than one wasted FastPath call.
# ---------------------------------------------------------------------------


def _fast_skill_index() -> SkillIndex:
    """A skill whose FAST hint makes L0 decide, so L1 is never consulted."""
    return SkillIndex([
        Skill(
            name="test",
            description="test skill",
            body="test",
            keywords=("simple",),
            routing_hint=RoutingHint.FAST,
        )
    ])


def _l1_naming(files: list[str], decision: str = "fastpath") -> Mock:
    client = Mock(spec=BilledClient)
    client.invoke.return_value = Completion(
        content=[
            ToolUseContent(
                id="call_1",
                name="classify_task",
                input={
                    "decision": decision,
                    "confidence": 0.9,
                    "files": files,
                    "reasoning": "Simple task",
                },
            )
        ],
        stop_reason="tool_use",
        usage=Usage(input_tokens=100, output_tokens=50),
        raw={},
    )
    return client


def test_l0_fastpath_without_hint_asks_l1_which_files_to_prefetch() -> None:
    client = _l1_naming(["src/thing.py"])
    result = route_task("do something simple", _fast_skill_index(), client=client)

    assert client.invoke.called, "L0 FastPath with no hint must consult L1 for files"
    assert result.prefetch_hint == ("src/thing.py",)


def test_recovered_files_never_become_the_condition_3_contract() -> None:
    """`files_hint` is enforced as `diff ⊆ files_hint`.

    A guess placed there turns a decline into a wrong rejection of a correct
    patch that touches a file the classifier failed to name — strictly worse
    than sending no context. The recovered names must stay advisory.
    """
    client = _l1_naming(["src/thing.py"])
    result = route_task("do something simple", _fast_skill_index(), client=client)

    assert result.files_hint == (), "recovered names must not become the contract"


def test_l0_decision_survives_a_disagreeing_l1() -> None:
    """L0 already decided; L1 is consulted for filenames, not for a verdict.

    Letting the verdict through would allow a probabilistic layer to silently
    overturn a deterministic one.
    """
    client = _l1_naming(["src/thing.py"], decision="fullagent")
    result = route_task("do something simple", _fast_skill_index(), client=client)

    assert result.decision == RoutingDecision.FASTPATH
    assert result.layer == "l0"
    assert result.confidence == 0.85


def test_no_call_is_spent_when_the_task_already_names_a_file() -> None:
    """Text extraction gets the name for free and is better evidence than a
    classifier that never saw the repository."""
    client = _l1_naming(["wrong.py"])
    result = route_task("fix divide in calc.py, simple", _fast_skill_index(), client=client)

    assert not client.invoke.called, "no call may be spent when the text names a file"
    assert result.prefetch_hint == ()


def test_no_call_is_spent_when_l0_chose_fullagent() -> None:
    """Full Agent explores with tools; it needs no prefetch."""
    index = SkillIndex([
        Skill(
            name="test",
            description="test skill",
            body="test",
            keywords=("simple",),
            routing_hint=RoutingHint.FULL,
        )
    ])
    client = _l1_naming(["src/thing.py"])
    result = route_task("do something simple", index, client=client)

    assert not client.invoke.called
    assert result.decision == RoutingDecision.FULLAGENT


def test_l1_failure_leaves_the_l0_decision_intact() -> None:
    """A dead classifier must cost the run nothing beyond the failed call."""
    client = Mock(spec=BilledClient)
    client.invoke.side_effect = RuntimeError("provider down")

    result = route_task("do something simple", _fast_skill_index(), client=client)

    assert result.decision == RoutingDecision.FASTPATH
    assert result.prefetch_hint == ()
