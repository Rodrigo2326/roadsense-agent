"""The second decision must be made *after* seeing the first tool's output."""

from __future__ import annotations

from conftest import FakeBedrock, FakeRekognition, ScriptedDecider

from roadsense.agent import run


def _run(decider, image, **kw):
    return run(
        kw.pop("question", "What vehicles appear and what does the sign say?"),
        image,
        decider=decider,
        rekognition=kw.pop("rekognition", FakeRekognition()),
        bedrock=kw.pop("bedrock", FakeBedrock()),
        pricing={},
        **kw,
    )


def test_two_tools_run_in_the_order_decider_chose(image):
    rek = FakeRekognition()
    result = _run(ScriptedDecider(["detect_objects", "read_sign_text", "answer"]), image, rekognition=rek)
    assert rek.calls == ["detect_labels", "detect_text"]
    assert [e.tool for e in result.state.evidence] == ["detect_objects", "read_sign_text"]
    assert [d.action for d in result.state.decisions] == [
        "detect_objects",
        "read_sign_text",
        "answer",
    ]


def test_the_state_shown_to_the_second_decision_contains_the_first_tools_result(image):
    decider = ScriptedDecider(["detect_objects", "read_sign_text", "answer"])
    _run(decider, image)

    first, second = decider.states[0], decider.states[1]
    assert "No tool has been run yet" in first
    assert "detect_objects" not in first.split("Tools not yet run")[0]

    # The second call genuinely sees new information, including the tool's own number.
    assert "detect_objects ran successfully" in second
    assert "Car" in second
    assert "0.98" in second
    assert "Steps used: 1 of 4" in second


def test_state_grows_monotonically_across_steps(image):
    decider = ScriptedDecider(["detect_objects", "read_sign_text", "answer"])
    _run(decider, image)
    sizes = [s.count("\n") for s in decider.states]
    assert sizes == sorted(sizes)
    assert len(set(decider.states)) == len(decider.states), "a step saw an unchanged state"


def test_step_budget_is_enforced_and_the_forced_step_is_marked_as_forced(image):
    """With a budget of 2, a decider that always asks for more is cut off."""
    decider = ScriptedDecider(["detect_objects", "read_sign_text", "detect_objects"])
    result = _run(decider, image, max_steps=2)

    assert decider.calls == 2, "Decider was consulted after the budget ran out"
    assert len(result.state.decisions) == 3
    forced = result.state.decisions[-1]
    assert forced.action == "answer"
    assert forced.decider_called is False
    assert forced.confidence is None, "a forced step must not report a confidence"
    assert "budget" in (forced.rejected or "")


def test_bedrock_is_called_once_with_every_piece_of_evidence(image):
    bedrock = FakeBedrock()
    _run(ScriptedDecider(["detect_objects", "read_sign_text", "answer"]), image, bedrock=bedrock)
    assert len(bedrock.prompts) == 1
    prompt = bedrock.prompts[0]
    assert "detect_objects" in prompt and "read_sign_text" in prompt
    assert "STOP" in prompt and "Car" in prompt


def test_bedrock_is_told_not_to_invent_and_not_to_give_safety_advice(image):
    bedrock = FakeBedrock()
    _run(ScriptedDecider(["answer"]), image, bedrock=bedrock)
    system = bedrock.systems[0]
    assert "Never add" in system
    assert "collision risk" in system
