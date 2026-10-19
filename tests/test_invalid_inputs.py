"""Bad inputs, failing tools, and malformed decider responses must not crash the loop."""

from __future__ import annotations

import json

import pytest
from conftest import FakeBedrock, FakeRekognition, ScriptedDecider

from roadsense.agent import OUTCOME_ERROR, run
from roadsense.decider_client import DeciderUnavailable, HttpDecider
from roadsense.state import AgentState


def _run(decider, image, **kw):
    return run(
        kw.pop("question", "What is here?"),
        image,
        decider=decider,
        rekognition=kw.pop("rekognition", FakeRekognition()),
        bedrock=kw.pop("bedrock", FakeBedrock()),
        pricing={},
        **kw,
    )


@pytest.mark.parametrize(
    "path,expected",
    [
        (None, "no image was provided"),
        ("does_not_exist.jpg", "not found"),
        ("notes.txt", "unsupported image format"),
    ],
)
def test_unusable_images_are_described_precisely(tmp_path, path, expected):
    if path == "notes.txt":
        p = tmp_path / "notes.txt"
        p.write_text("hello")
        path = str(p)
    state = AgentState(question="q", image_path=path)
    usable, why = state.image_is_usable()
    assert usable is False
    assert expected in why


def test_empty_file_is_rejected(tmp_path):
    p = tmp_path / "empty.jpg"
    p.write_bytes(b"")
    assert AgentState(question="q", image_path=str(p)).image_is_usable() == (False, "image file is empty")


def test_missing_image_never_reaches_a_vision_tool():
    """With masking on, the vision tools are not even offered."""
    rek = FakeRekognition()
    result = _run(ScriptedDecider(["read_sign_text", "detect_objects", "answer"]), None, rekognition=rek)
    assert rek.calls == [], "a vision tool ran without a usable image"
    offered = result.state.decisions[0].offered_actions
    assert offered == ["answer", "request_better_evidence"]


def test_without_masking_the_illegal_pick_is_vetoed_and_recorded():
    """The guardrail still holds when the options are not masked."""
    rek = FakeRekognition()
    result = run(
        "What is here?",
        None,
        decider=ScriptedDecider(["read_sign_text", "answer"]),
        rekognition=rek,
        bedrock=FakeBedrock(),
        pricing={},
        mask_illegal_actions=False,
    )
    assert rek.calls == []
    first = result.state.decisions[0]
    assert first.offered_actions == sorted(["answer", "detect_objects", "read_sign_text", "request_better_evidence"])
    assert first.rejected is not None and "needs a readable image" in first.rejected


def test_a_failing_tool_is_recorded_as_evidence_and_the_loop_continues(image):
    rek = FakeRekognition(raises=RuntimeError("Rekognition is down"))
    result = _run(ScriptedDecider(["detect_objects", "read_sign_text", "answer"]), image, rekognition=rek)
    failures = [e for e in result.state.evidence if not e.ok]
    assert len(failures) == 2
    assert "Rekognition is down" in failures[0].error
    assert result.outcome == "answered", "a tool outage must not abort the run"
    assert any("detect_objects" in e for e in result.state.errors)


def test_a_failing_tool_appears_in_the_state_shown_to_the_next_decision(image):
    decider = ScriptedDecider(["detect_objects", "read_sign_text", "answer"])
    _run(decider, image, rekognition=FakeRekognition(raises=RuntimeError("boom")))
    assert "detect_objects FAILED" in decider.states[1]


def test_bedrock_failure_yields_an_error_outcome_but_still_a_full_trace(image):
    result = _run(
        ScriptedDecider(["detect_objects", "answer"]),
        image,
        bedrock=FakeBedrock(raises=RuntimeError("throttled")),
    )
    assert result.outcome == OUTCOME_ERROR
    assert result.answer == ""
    assert len(result.trace["steps"]) == 2
    assert len(result.trace["evidence"]) == 1
    assert any("throttled" in e for e in result.state.errors)
    json.dumps(result.trace)  # the trace must stay serialisable on the error path


def test_malformed_decider_http_response_is_raised_not_silently_defaulted(monkeypatch):
    class FakeResponse:
        def read(self):
            return b'{"answers": {"next_action": {"nonsense": true}}}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: FakeResponse())
    with pytest.raises(DeciderUnavailable, match="malformed"):
        HttpDecider("http://decider.invalid").decide("state", {"answer": "d", "detect_objects": "d"})


def test_unreachable_decider_reports_the_endpoint(monkeypatch):
    with pytest.raises(DeciderUnavailable, match="could not reach"):
        HttpDecider("http://127.0.0.1:9", timeout=1).decide("s", {"answer": "d", "detect_objects": "d"})
