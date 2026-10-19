"""Recognising that the evidence is not enough, and saying so."""

from __future__ import annotations

from conftest import FakeBedrock, FakeRekognition, ScriptedDecider

from roadsense.agent import OUTCOME_INSUFFICIENT, run


def _run(decider, image, **kw):
    return run(
        kw.pop("question", "What does the sign say?"),
        image,
        decider=decider,
        rekognition=kw.pop("rekognition", FakeRekognition()),
        bedrock=kw.pop("bedrock", FakeBedrock()),
        pricing={},
        **kw,
    )


def test_requesting_better_evidence_ends_the_run_as_insufficient(image):
    result = _run(ScriptedDecider(["request_better_evidence"]), image)
    assert result.outcome == OUTCOME_INSUFFICIENT
    assert result.state.evidence[-1].tool == "request_better_evidence"


def test_the_request_names_the_concrete_reason_not_a_vague_hedge(image):
    """Low OCR confidence must surface as a number in the reason."""
    rek = FakeRekognition(
        text=[
            {"DetectedText": "ST0P", "Type": "LINE", "Confidence": 61.0},
            {"DetectedText": "ST0P", "Type": "WORD", "Confidence": 61.0, "Geometry": {}},
        ]
    )
    result = _run(ScriptedDecider(["read_sign_text", "request_better_evidence"]), image, rekognition=rek)
    reason = result.state.evidence[-1].summary
    assert "read_sign_text confidence was only 0.61" in reason


def test_low_confidence_ocr_is_reported_as_unconfirmable_not_as_fact(image):
    rek = FakeRekognition(
        text=[
            {"DetectedText": "ST0P", "Type": "LINE", "Confidence": 55.0},
            {"DetectedText": "ST0P", "Type": "WORD", "Confidence": 55.0, "Geometry": {}},
        ]
    )
    result = _run(ScriptedDecider(["read_sign_text", "answer"]), image, rekognition=rek)
    summary = result.state.evidence[0].summary
    assert "cannot be confirmed" in summary
    assert "0.55" in summary


def test_empty_ocr_result_reports_no_text_rather_than_inventing_one(image):
    rek = FakeRekognition(text=[])
    result = _run(ScriptedDecider(["read_sign_text", "request_better_evidence"]), image, rekognition=rek)
    evidence = result.state.evidence[0]
    assert evidence.ok is True
    assert evidence.items == []
    assert evidence.quality is None
    assert "no readable text" in evidence.summary
    assert "read_sign_text returned nothing" in result.state.evidence[-1].summary


def test_the_unconfirmable_reading_still_reaches_bedrock_flagged(image):
    """The writer must see both the text and the fact that it is not confirmable."""
    bedrock = FakeBedrock()
    rek = FakeRekognition(
        text=[
            {"DetectedText": "ST0P", "Type": "LINE", "Confidence": 52.0},
            {"DetectedText": "ST0P", "Type": "WORD", "Confidence": 52.0, "Geometry": {}},
        ]
    )
    _run(ScriptedDecider(["read_sign_text", "answer"]), image, rekognition=rek, bedrock=bedrock)
    prompt = bedrock.prompts[0]
    assert "cannot be confirmed" in prompt
    assert "tool confidence 0.52" in prompt


def test_the_summary_never_quotes_text_that_is_not_in_the_items(image):
    """Regression: a LINE above the floor whose WORD is below it must not leak.

    Rekognition can return a LINE with acceptable confidence while the word inside it
    falls under the threshold. Building the readable summary from LINE entries let
    that text reach Bedrock as evidence, which then repeated it. Caught by the
    benchmark fidelity check on rs-001.
    """
    rek = FakeRekognition(
        text=[
            {"DetectedText": "STOP", "Type": "LINE", "Confidence": 99.0},
            {"DetectedText": "the", "Type": "LINE", "Confidence": 72.0},
            {"DetectedText": "STOP", "Type": "WORD", "Confidence": 99.0, "ParentId": 0, "Geometry": {}},
            {"DetectedText": "the", "Type": "WORD", "Confidence": 31.0, "ParentId": 1, "Geometry": {}},
        ]
    )
    result = _run(ScriptedDecider(["read_sign_text", "answer"]), image, rekognition=rek)
    evidence = result.state.evidence[0]

    assert "the" not in evidence.summary.split("'")[1]
    assert evidence.summary.count("STOP") == 1
    assert [i["text"] for i in evidence.items] == ["STOP"]


def test_words_are_regrouped_into_their_original_lines(image):
    rek = FakeRekognition(
        text=[
            {"DetectedText": "SPEED LIMIT", "Type": "LINE", "Confidence": 99.0},
            {"DetectedText": "50", "Type": "LINE", "Confidence": 99.0},
            {"DetectedText": "SPEED", "Type": "WORD", "Confidence": 99.0, "ParentId": 0, "Geometry": {}},
            {"DetectedText": "LIMIT", "Type": "WORD", "Confidence": 98.0, "ParentId": 0, "Geometry": {}},
            {"DetectedText": "50", "Type": "WORD", "Confidence": 97.0, "ParentId": 1, "Geometry": {}},
        ]
    )
    result = _run(ScriptedDecider(["read_sign_text", "answer"]), image, rekognition=rek)
    assert "SPEED LIMIT | 50" in result.state.evidence[0].summary
