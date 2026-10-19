"""Talking to Strands Decider 2B.

Two backends behind one interface, because the model runs where the weights are and
that is not always where the agent runs:

* ``InProcessDecider`` loads the checkpoint with ``strands_decider.infer.load_engine``.
  This is what runs on the EC2 box that has the 4.5 GB of weights.
* ``HttpDecider`` posts to ``strands-decider serve``'s ``POST /v1/systemone``. This is
  what lets a laptop drive a decider hosted elsewhere.

Both ask the same two questions per step, in a single call. That is not a convenience
— it is the property the model is built around: the state is encoded once and its KV
cache is broadcast across the question batch, so the second question is close to free.

The API shape below was read off the installed package (``strands_decider/schema.py``),
not off documentation. Notably ``ChoiceQuestion`` takes ``criteria``, not ``options``.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

NEXT_ACTION = "next_action"
EVIDENCE_SUFFICIENT = "evidence_sufficient"

ACTION_INSTRUCTIONS = (
    "Which single action should the agent take next to answer the question with "
    "evidence it can actually verify?"
)
SUFFICIENCY_INSTRUCTIONS = (
    "Is the evidence collected so far already sufficient to answer the question "
    "without guessing?"
)


@dataclass
class DeciderResult:
    """One Decider call. Every number here came from the model."""

    action: str
    confidence: float
    probabilities: dict[str, float]
    evidence_sufficient: float | None
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    backend: str = ""
    ranking: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.ranking:
            self.ranking = sorted(
                self.probabilities, key=lambda k: self.probabilities[k], reverse=True
            )


class Decider(Protocol):
    def decide(self, state_text: str, actions: dict[str, str]) -> DeciderResult: ...


class DeciderUnavailable(RuntimeError):
    """The decider could not be reached or loaded. Never swallowed silently."""


class InProcessDecider:
    """Loads the checkpoint into this process. First call pays the load cost."""

    def __init__(
        self,
        checkpoint: str = "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610",
        device: str = "cpu",
        ask_sufficiency: bool = True,
    ) -> None:
        self.checkpoint = checkpoint
        # ``load_engine`` defaults to device="cuda" and raises on a CPU-only host,
        # so this is always passed explicitly.
        self.device = device
        # The sufficiency question is telemetry: it is recorded and displayed, but
        # nothing in the control flow reads it. On CPU it is not free -- measured at
        # ~0.8-1.0 s per call regardless of state size (scripts/profile_latency.py),
        # which is roughly a third of a step's wall time. Worth paying for a demo
        # that shows the agent's own sense of progress; worth questioning in a
        # benchmark that reports latency.
        self.ask_sufficiency = ask_sufficiency
        self._engine: Any | None = None

    def _engine_or_load(self) -> Any:
        if self._engine is None:
            self._engine = self._load_engine()(self.checkpoint, device=self.device)
        return self._engine

    @staticmethod
    def _import(name: str) -> Any:
        """Import a strands_decider symbol, or explain why it is missing."""
        module, _, attr = name.rpartition(".")
        try:
            mod = __import__(module, fromlist=[attr])
        except ImportError as exc:
            raise DeciderUnavailable(
                "strands-decider is not installed in this interpreter. Install it with "
                "`pip install strands-decider` (the weights need ~5 GB), or point the "
                "agent at a remote decider with --decider-endpoint."
            ) from exc
        return getattr(mod, attr)

    def _load_engine(self) -> Any:
        return self._import("strands_decider.infer.load_engine")

    def decide(self, state_text: str, actions: dict[str, str]) -> DeciderResult:
        ChoiceQuestion = self._import("strands_decider.schema.ChoiceQuestion")
        NoulQuestion = self._import("strands_decider.schema.NoulQuestion")
        engine = self._engine_or_load()
        questions: dict[str, Any] = {
            NEXT_ACTION: ChoiceQuestion(instructions=ACTION_INSTRUCTIONS, criteria=actions)
        }
        if self.ask_sufficiency:
            questions[EVIDENCE_SUFFICIENT] = NoulQuestion(instructions=SUFFICIENCY_INSTRUCTIONS)
        t0 = time.perf_counter()
        resp = engine.ask(state=state_text, questions=questions)
        dt = (time.perf_counter() - t0) * 1000
        a = resp.answers[NEXT_ACTION]
        n = resp.answers.get(EVIDENCE_SUFFICIENT)
        return DeciderResult(
            action=a.choice,
            confidence=a.confidence,
            probabilities=dict(a.probabilities),
            evidence_sufficient=None if n is None else n.noul,
            latency_ms=dt,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            backend=f"in-process:{self.checkpoint}",
        )


class HttpDecider:
    """Posts to ``strands-decider serve``. Same wire types, different transport."""

    def __init__(self, endpoint: str, timeout: float = 120.0, ask_sufficiency: bool = True) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.ask_sufficiency = ask_sufficiency

    def decide(self, state_text: str, actions: dict[str, str]) -> DeciderResult:
        payload = {
            "state": state_text,
            "questions": {
                NEXT_ACTION: {
                    "type": "choice",
                    "instructions": ACTION_INSTRUCTIONS,
                    "criteria": actions,
                },
            },
        }
        if self.ask_sufficiency:
            payload["questions"][EVIDENCE_SUFFICIENT] = {
                "type": "noul",
                "instructions": SUFFICIENCY_INSTRUCTIONS,
            }
        req = urllib.request.Request(
            f"{self.endpoint}/v1/systemone",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                body = json.loads(r.read())
        except urllib.error.HTTPError as exc:
            raise DeciderUnavailable(
                f"decider server returned HTTP {exc.code}: {exc.read()[:400]!r}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise DeciderUnavailable(
                f"could not reach decider server at {self.endpoint}: {exc}"
            ) from exc
        dt = (time.perf_counter() - t0) * 1000

        try:
            a = body["answers"][NEXT_ACTION]
            probs = dict(a["probabilities"])
            n = body["answers"].get(EVIDENCE_SUFFICIENT)
        except (KeyError, TypeError) as exc:
            raise DeciderUnavailable(
                f"malformed decider response: {json.dumps(body)[:400]}"
            ) from exc

        usage = body.get("usage") or {}
        return DeciderResult(
            action=a["choice"],
            confidence=float(a["confidence"]),
            probabilities=probs,
            evidence_sufficient=None if n is None else float(n["noul"]),
            latency_ms=dt,
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
            backend=f"http:{self.endpoint}",
        )
