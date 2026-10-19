"""Baseline tool selectors.

Each one is a drop-in replacement for the Decider client: same
``decide(state_text, options) -> DeciderResult`` signature, same loop around it, same
tools, same writer, same guardrails. Only the thing choosing the next action changes.
That is what makes the comparison about selection policy rather than about plumbing.

Four arms:

``keyword``     cheap, deterministic, and the honest bar to clear. Most "agentic tool
                routing" in production is this, written as an if-statement.
``exhaustive``  run every tool that is still legal, then answer. Never misses
                evidence; pays for everything, every time.
``llm``         ask Bedrock to pick the next action from the same menu. A generative
                model doing the job the pointer model was built for. Write it as
                ``llm:<model-id>`` to pin a specific one -- running the arm across a
                price tier is the only way to answer "would a better model just win?",
                and comparing a 2B pointer model against only the cheapest frontier
                option would be rigging the baseline.
``decider``     Strands Decider 2B (``roadsense.decider_client``).
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from roadsense.decider_client import DeciderResult

#: Words that, in the *question*, call for each tool. Deliberately simple: this is
#: the baseline, and making it cleverer would be making it a different thing.
OCR_WORDS = ("text", "sign say", "says", "read", "written", "word", "letter", "label")
DETECT_WORDS = ("object", "vehicle", "car", "truck", "people", "person", "see", "visible", "contain")


def _result(pick: str, options: dict[str, str], *, latency_ms: float, backend: str,
            probabilities: dict[str, float] | None = None,
            input_tokens: int = 0, output_tokens: int = 0) -> DeciderResult:
    if probabilities is None:
        # A rule has no distribution. Saying 1.0 would claim a certainty the rule
        # never computed, so the mass is spread and the pick simply leads.
        probabilities = {k: (0.9 if k == pick else 0.1 / max(1, len(options) - 1)) for k in options}
    best = max(probabilities, key=lambda k: probabilities[k])
    n = len(probabilities)
    confidence = 1.0 if n <= 1 else max(0.0, min(1.0, (n * probabilities[best] - 1.0) / (n - 1.0)))
    return DeciderResult(
        action=best,
        confidence=confidence,
        probabilities=probabilities,
        evidence_sufficient=None,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        backend=backend,
    )


def _question_of(state_text: str) -> str:
    for line in state_text.splitlines():
        if line.startswith("Question:"):
            return line[len("Question:"):].strip().lower()
    return state_text.lower()


class KeywordDecider:
    """An if-statement over the question. No model, no network, ~0 ms."""

    name = "keyword"

    def decide(self, state_text: str, options: dict[str, str]) -> DeciderResult:
        t0 = time.perf_counter()
        q = _question_of(state_text)
        wants_ocr = any(w in q for w in OCR_WORDS)
        wants_detect = any(w in q for w in DETECT_WORDS)

        pick = "answer"
        if wants_detect and "detect_objects" in options:
            pick = "detect_objects"
        elif wants_ocr and "read_sign_text" in options:
            pick = "read_sign_text"
        elif "request_better_evidence" in options and not {"detect_objects", "read_sign_text"} & set(options):
            # Nothing can be gathered. The rule has no way to know whether that is
            # because the tools ran or because the image is unusable, so it answers.
            pick = "answer"

        if pick not in options:
            pick = "answer"
        return _result(pick, options, latency_ms=(time.perf_counter() - t0) * 1000, backend="keyword")


class ExhaustiveDecider:
    """Run everything still available, then stop. The cost ceiling."""

    name = "exhaustive"

    def decide(self, state_text: str, options: dict[str, str]) -> DeciderResult:
        t0 = time.perf_counter()
        for tool in ("detect_objects", "read_sign_text"):
            if tool in options:
                return _result(tool, options, latency_ms=(time.perf_counter() - t0) * 1000,
                               backend="exhaustive")
        pick = "answer" if "answer" in options else next(iter(options))
        return _result(pick, options, latency_ms=(time.perf_counter() - t0) * 1000,
                       backend="exhaustive")


SELECTOR_SYSTEM = (
    "You select the next action for an evidence-gathering agent. You are given the "
    "agent's current state and a menu of allowed actions. Reply with a JSON object "
    'of exactly the form {"action": "<name>"} where <name> is one of the menu keys. '
    "No other text."
)


class LlmDecider:
    """Bedrock picks the action from the same menu. Temperature 0 for reproducibility."""

    name = "llm"

    def __init__(self, bedrock: Any, model_id: str) -> None:
        self.bedrock = bedrock
        self.model_id = model_id
        self.parse_failures = 0

    def decide(self, state_text: str, options: dict[str, str]) -> DeciderResult:
        menu = "\n".join(f"- {k}: {v}" for k, v in options.items())
        prompt = f"{state_text}\n\nAllowed actions:\n{menu}\n\nWhich action next?"
        t0 = time.perf_counter()
        try:
            resp = self.bedrock.converse(
                modelId=self.model_id,
                system=[{"text": SELECTOR_SYSTEM}],
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": 40, "temperature": 0.0},
            )
        except Exception:  # noqa: BLE001 - an outage is a measurable failure, not a crash
            self.parse_failures += 1
            return _result("answer", options, latency_ms=(time.perf_counter() - t0) * 1000,
                           backend="llm:error")

        dt = (time.perf_counter() - t0) * 1000
        usage = resp.get("usage", {})
        text = "".join(b.get("text", "") for b in resp["output"]["message"]["content"])

        pick = None
        match = re.search(r'"action"\s*:\s*"([a-z_]+)"', text)
        if match and match.group(1) in options:
            pick = match.group(1)
        else:
            # Generative models drift off the format. Falling back to a substring
            # scan is generous to this baseline on purpose -- the point is to compare
            # selection quality, not to score it on JSON discipline.
            for name in options:
                if name in text:
                    pick = name
                    break
        if pick is None:
            self.parse_failures += 1
            pick = "answer" if "answer" in options else next(iter(options))

        return _result(
            pick, options, latency_ms=dt, backend=f"llm:{self.model_id}",
            input_tokens=int(usage.get("inputTokens", 0)),
            output_tokens=int(usage.get("outputTokens", 0)),
        )


def build(name: str, *, bedrock: Any = None, model_id: str = "", **decider_kw) -> Any:
    if name == "keyword":
        return KeywordDecider()
    if name == "exhaustive":
        return ExhaustiveDecider()
    if name == "llm" or name.startswith("llm:"):
        if bedrock is None:
            raise ValueError("the llm arm needs a bedrock client")
        # "llm:<model-id>" pins the selector model independently of the writer's.
        selector_model = name.split(":", 1)[1] if ":" in name else model_id
        return LlmDecider(bedrock, selector_model)
    if name == "decider":
        from roadsense.decider_client import HttpDecider, InProcessDecider

        endpoint = decider_kw.get("endpoint")
        ask_sufficiency = decider_kw.get("ask_sufficiency", True)
        if endpoint:
            return HttpDecider(endpoint, ask_sufficiency=ask_sufficiency)
        return InProcessDecider(
            decider_kw.get("checkpoint", "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"),
            device=decider_kw.get("device", "cpu"),
            ask_sufficiency=ask_sufficiency,
        )
    raise ValueError(f"unknown arm {name!r}")


ARMS = ("keyword", "exhaustive", "llm", "decider")
