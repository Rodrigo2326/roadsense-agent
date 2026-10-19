"""The action contract is closed, and policy enforces what the model cannot."""

from __future__ import annotations

import pytest

from roadsense.decision_schema import (
    ACTIONS,
    TERMINAL_ACTIONS,
    TOOL_ACTIONS,
    InvalidAction,
    validate_action_name,
)
from roadsense.policy import apply, legality
from roadsense.state import AgentState, Evidence


def test_action_space_is_exactly_the_four_documented_actions():
    assert set(ACTIONS) == {
        "detect_objects",
        "read_sign_text",
        "request_better_evidence",
        "answer",
    }
    assert TOOL_ACTIONS | TERMINAL_ACTIONS == set(ACTIONS)
    assert not TOOL_ACTIONS & TERMINAL_ACTIONS


def test_every_action_has_a_description_for_the_pointer_head():
    # The descriptions are the prompt. An empty one would silently degrade scoring.
    for name, desc in ACTIONS.items():
        assert desc.strip(), f"{name} has no description"
        assert len(desc) > 40, f"{name} description is too thin to discriminate"


@pytest.mark.parametrize("bad", ["", "ANSWER", "detect objects", "rm -rf /", "tool_0"])
def test_unknown_action_names_are_rejected(bad):
    with pytest.raises(InvalidAction):
        validate_action_name(bad)


def test_answer_is_always_legal(image):
    state = AgentState(question="q", image_path=image)
    assert legality(state, "answer") is None
    state_no_image = AgentState(question="q", image_path=None)
    assert legality(state_no_image, "answer") is None


def test_vision_tools_are_illegal_without_a_usable_image():
    state = AgentState(question="q", image_path=None)
    for action in TOOL_ACTIONS:
        assert legality(state, action) is not None


def test_a_tool_cannot_run_twice(image):
    state = AgentState(question="q", image_path=image)
    state.evidence.append(Evidence(tool="detect_objects", ok=True, summary="x"))
    assert legality(state, "detect_objects") is not None
    assert legality(state, "read_sign_text") is None


def test_policy_falls_back_to_deciders_next_preference_not_a_hardcoded_action(image):
    """The override must still come from the model's ranking."""
    state = AgentState(question="q", image_path=image)
    state.evidence.append(Evidence(tool="detect_objects", ok=True, summary="x"))
    verdict = apply(state, ["detect_objects", "read_sign_text", "answer"])
    assert verdict.action == "read_sign_text"
    assert verdict.rejected and "already ran" in verdict.rejected

    # Same illegal first pick, different ranking -> different legal outcome.
    verdict2 = apply(state, ["detect_objects", "answer", "read_sign_text"])
    assert verdict2.action == "answer"
