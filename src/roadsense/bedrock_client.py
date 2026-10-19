"""Bedrock writes the answer. It does not get to decide anything.

Bedrock is given the question and the tool outputs as text, and nothing else -- in
particular it never receives the image. That is the point: if the writer cannot see
the scene, it cannot describe anything the tools did not report, and "grounded in the
evidence" stops being a promise in a prompt and becomes a property of the plumbing.

The system prompt still does its share of the work, because a model handed thin
evidence will reach for plausible filler unless told what to do instead. It is told
to name the gap.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .state import AgentState

DEFAULT_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

SYSTEM = (
    "You write the final answer for RoadSense, an agent that inspects road-scene "
    "photographs with vision tools.\n"
    "\n"
    "You cannot see the image. You only receive the question and the output of the "
    "tools that were run. Follow these rules exactly:\n"
    "1. State only what the tool output supports. Never add a vehicle, a word of sign "
    "text, a colour, a count, a distance or a measurement that is not in the evidence.\n"
    "2. When a tool reported low confidence, say the reading is not confirmable rather "
    "than repeating it as fact.\n"
    "3. When the evidence does not answer the question, say plainly what is missing and "
    "what kind of image would settle it. Do not guess to fill the gap.\n"
    "4. Do not comment on road safety, collision risk, braking distance or what a driver "
    "should do. RoadSense is a visual-evidence demo, not a safety system.\n"
    "5. Be brief: at most four sentences, plain prose, no headings or bullet points."
)


@dataclass
class BedrockAnswer:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    model_id: str
    error: str | None = None


def render_evidence(state: AgentState) -> str:
    """The evidence block handed to the writer. Mirrors what Decider saw."""
    lines = [f"Question: {state.question}", ""]

    if not state.evidence:
        lines.append("No tool produced any evidence.")
    else:
        lines.append("Tool output:")
        for e in state.evidence:
            q = "not reported" if e.quality is None else f"{e.quality:.2f}"
            head = f"- {e.tool}: {'ok' if e.ok else 'FAILED'} (tool confidence {q})"
            lines.append(head)
            lines.append(f"    {e.summary}")
            if e.error:
                lines.append(f"    error: {e.error}")
            for item in e.items[:12]:
                lines.append(f"    item: {item}")

    if state.errors:
        lines += ["", "Problems encountered: " + "; ".join(state.errors)]

    decisions = ", ".join(f"{d.step}:{d.action}" for d in state.decisions)
    lines += ["", f"Actions taken in order: {decisions or 'none'}."]
    return "\n".join(lines)


def write_answer(
    client: Any, state: AgentState, model_id: str = DEFAULT_MODEL_ID
) -> BedrockAnswer:
    prompt = render_evidence(state)
    t0 = time.perf_counter()
    try:
        resp = client.converse(
            modelId=model_id,
            system=[{"text": SYSTEM}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 400, "temperature": 0.0},
        )
    except Exception as exc:  # noqa: BLE001 - a Bedrock outage must not lose the trace
        return BedrockAnswer(
            text="",
            input_tokens=0,
            output_tokens=0,
            latency_ms=(time.perf_counter() - t0) * 1000,
            model_id=model_id,
            error=f"{type(exc).__name__}: {exc}",
        )

    dt = (time.perf_counter() - t0) * 1000
    usage = resp.get("usage", {})
    blocks = resp["output"]["message"]["content"]
    text = "".join(b.get("text", "") for b in blocks).strip()
    return BedrockAnswer(
        text=text,
        input_tokens=int(usage.get("inputTokens", 0)),
        output_tokens=int(usage.get("outputTokens", 0)),
        latency_ms=dt,
        model_id=model_id,
    )
