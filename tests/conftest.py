"""Test doubles.

These stand in for Decider, Rekognition and Bedrock so the loop's control flow can be
tested deterministically and offline. They are not a substitute for the real thing:
the integrations themselves are exercised by ``scripts/smoke_decider.py`` and
``scripts/smoke_aws.py``, which call the live services and are reported separately.
"""

from __future__ import annotations

import pathlib

import pytest

from roadsense.decider_client import DeciderResult


class ScriptedDecider:
    """Replays a fixed list of picks, so a test can pin an exact decision path."""

    def __init__(self, picks: list[str | dict[str, float]]) -> None:
        self.picks = list(picks)
        self.states: list[str] = []
        self.calls = 0

    def decide(self, state_text: str, actions: dict[str, str]) -> DeciderResult:
        self.states.append(state_text)
        pick = self.picks[min(self.calls, len(self.picks) - 1)]
        self.calls += 1
        if isinstance(pick, str):
            probs = {a: (0.7 if a == pick else 0.1) for a in actions}
        else:
            probs = {a: pick.get(a, 0.0) for a in actions}
        total = sum(probs.values()) or 1.0
        probs = {k: v / total for k, v in probs.items()}
        best = max(probs, key=lambda k: probs[k])
        return DeciderResult(
            action=best,
            confidence=probs[best],
            probabilities=probs,
            evidence_sufficient=0.5,
            latency_ms=1.0,
            input_tokens=10,
            output_tokens=2,
            backend="scripted",
        )


class FakeRekognition:
    """Returns canned Rekognition payloads, or raises to simulate an outage."""

    def __init__(self, *, labels=None, text=None, raises: Exception | None = None) -> None:
        self._labels = labels if labels is not None else [
            {"Name": "Car", "Confidence": 98.5, "Instances": [{}]},
            {"Name": "Traffic Sign", "Confidence": 88.1, "Instances": []},
        ]
        self._text = text if text is not None else [
            {"DetectedText": "STOP", "Type": "LINE", "Confidence": 99.0},
            {"DetectedText": "STOP", "Type": "WORD", "Confidence": 99.0, "Geometry": {"BoundingBox": {}}},
        ]
        self._raises = raises
        self.calls: list[str] = []

    def detect_labels(self, **kwargs):
        self.calls.append("detect_labels")
        if self._raises:
            raise self._raises
        return {"Labels": self._labels}

    def detect_text(self, **kwargs):
        self.calls.append("detect_text")
        if self._raises:
            raise self._raises
        return {"TextDetections": self._text}


class FakeBedrock:
    def __init__(self, text: str = "Fake grounded answer.", raises: Exception | None = None) -> None:
        self.text = text
        self._raises = raises
        self.prompts: list[str] = []
        self.systems: list[str] = []

    def converse(self, **kwargs):
        if self._raises:
            raise self._raises
        self.prompts.append(kwargs["messages"][0]["content"][0]["text"])
        self.systems.append(kwargs["system"][0]["text"])
        return {
            "output": {"message": {"content": [{"text": self.text}]}},
            "usage": {"inputTokens": 100, "outputTokens": 20},
        }


@pytest.fixture
def image(tmp_path) -> str:
    """A real file with a .jpg name. Rekognition is faked, so bytes do not matter."""
    p = tmp_path / "scene.jpg"
    p.write_bytes(b"\xff\xd8\xff\xe0 not a real jpeg, the detector is faked")
    return str(p)


@pytest.fixture
def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]
