"""Decider's pick must actually change which tool runs. That is the whole claim."""

from __future__ import annotations

from conftest import FakeBedrock, FakeRekognition, ScriptedDecider

from roadsense.agent import run


def _run(decider, image, rekognition=None, bedrock=None, **kw):
    return run(
        kw.pop("question", "What is in this scene?"),
        image,
        decider=decider,
        rekognition=rekognition or FakeRekognition(),
        bedrock=bedrock or FakeBedrock(),
        pricing={},
        **kw,
    )


def test_choosing_detect_objects_calls_the_detector_and_not_ocr(image):
    rek = FakeRekognition()
    _run(ScriptedDecider(["detect_objects", "answer"]), image, rek)
    assert rek.calls == ["detect_labels"]


def test_choosing_read_sign_text_calls_ocr_and_not_the_detector(image):
    rek = FakeRekognition()
    _run(ScriptedDecider(["read_sign_text", "answer"]), image, rek)
    assert rek.calls == ["detect_text"]


def test_the_same_input_routes_differently_when_decider_decides_differently(image):
    """Identical question and image; only the decision changes. So does the tool."""
    rek_a, rek_b = FakeRekognition(), FakeRekognition()
    _run(ScriptedDecider(["detect_objects", "answer"]), image, rek_a)
    _run(ScriptedDecider(["read_sign_text", "answer"]), image, rek_b)
    assert rek_a.calls != rek_b.calls


def test_answer_on_the_first_decision_runs_no_tool_at_all(image):
    rek = FakeRekognition()
    result = _run(ScriptedDecider(["answer"]), image, rek)
    assert rek.calls == []
    assert result.state.evidence == []
    assert result.outcome == "answered"


def test_decider_never_receives_the_image(image):
    """Decider sees structured state only -- no path, no bytes."""
    decider = ScriptedDecider(["detect_objects", "answer"])
    _run(decider, image, question="What does the sign say?")
    for state_text in decider.states:
        assert image not in state_text
        assert "\xff\xd8" not in state_text


def test_tool_confidence_reaches_the_state_verbatim(image):
    rek = FakeRekognition(labels=[{"Name": "Car", "Confidence": 73.4, "Instances": []}])
    result = _run(ScriptedDecider(["detect_objects", "answer"]), image, rek)
    evidence = result.state.evidence[0]
    assert evidence.quality == 0.734
    assert "0.73" in result.state.render_for_decider()
