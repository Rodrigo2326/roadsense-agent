"""RoadSense, assembled from the generic pattern.

This module is deliberately thin. The loop itself -- ask, veto, execute, repeat,
stop -- lives in :mod:`roadsense.pattern.evidence_seeking` and knows nothing about
roads, images or AWS. What is left here is only the domain: which four actions exist,
what makes one illegal, how to run it, and how to turn the finished episode into a
trace and a written answer.

If that split looks like over-engineering for one agent, it is the opposite: it is
the evidence that the pattern is reusable. The same ``EvidenceSeeker`` drives
``examples/triage_example.py``, which has no image in it at all.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from . import policy
from .bedrock_client import DEFAULT_MODEL_ID, BedrockAnswer, write_answer
from .decider_client import Decider
from .decision_schema import ACTIONS, TERMINAL_ACTIONS, TOOL_ACTIONS
from .pattern import Action, EvidenceSeeker
from .state import AgentState, Decision, Evidence
from .telemetry import CostReport, build_trace, compute_cost, load_pricing
from .tools import detect_objects, read_sign_text, request_better_evidence

#: Outcomes a run can end in. ``insufficient_evidence`` is a success of the design,
#: not a failure: it means the agent declined to invent an answer.
OUTCOME_ANSWERED = "answered"
OUTCOME_INSUFFICIENT = "insufficient_evidence"
OUTCOME_ERROR = "error"

#: The domain action set, in the pattern's vocabulary.
ROADSENSE_ACTIONS = [
    Action(name=name, description=desc, terminal=name in TERMINAL_ACTIONS)
    for name, desc in ACTIONS.items()
]


@dataclass
class AgentResult:
    state: AgentState
    answer: str
    outcome: str
    trace: dict[str, Any]
    cost: CostReport
    bedrock: BedrockAnswer


def _read_image(state: AgentState) -> bytes | None:
    usable, why = state.image_is_usable()
    if not usable:
        state.errors.append(why)
        return None
    try:
        return Path(state.image_path or "").read_bytes()
    except OSError as exc:
        state.errors.append(f"could not read image: {exc}")
        return None


def _make_executor(
    rekognition: Any,
    counter: dict[str, int],
    emit: Callable[[dict[str, Any]], None],
):
    """Build the ``execute`` callback the seeker calls for each chosen action."""

    def _report(evidence: Evidence) -> None:
        state_evidence = asdict(evidence)
        state_evidence.pop("raw", None)  # the raw API payload is for the trace, not the wire
        emit({"type": "evidence", "evidence": state_evidence})

    def execute(state: AgentState, action: str) -> None:
        if action == "answer":
            return  # Bedrock writes the answer after the episode ends.

        if action == "request_better_evidence":
            evidence = request_better_evidence(state)
            state.evidence.append(evidence)
            _report(evidence)
            return

        if action not in TOOL_ACTIONS:
            return

        image = _read_image(state)
        if image is None:
            # Legality should have caught this. If the file vanished between the
            # check and the read, record it as evidence rather than crashing.
            unreadable = Evidence(
                tool=action,
                ok=False,
                summary="The tool could not run because the image was unreadable.",
                error=state.errors[-1] if state.errors else "unreadable image",
            )
            state.evidence.append(unreadable)
            _report(unreadable)
            return

        fn = detect_objects if action == "detect_objects" else read_sign_text
        evidence = fn(rekognition, image)
        counter["rekognition"] += 1
        state.evidence.append(evidence)
        if not evidence.ok and evidence.error:
            state.errors.append(f"{action}: {evidence.error}")
        _report(evidence)

    return execute


def run(
    question: str,
    image_path: str | None,
    *,
    decider: Decider,
    rekognition: Any,
    bedrock: Any,
    max_steps: int = 4,
    bedrock_model: str = DEFAULT_MODEL_ID,
    pricing: dict[str, Any] | None = None,
    mask_illegal_actions: bool = True,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> AgentResult:
    """Run one episode.

    ``on_event`` receives dicts as the episode unfolds -- ``decision`` when a step is
    chosen, ``evidence`` when a tool reports back, ``answer`` at the end. It is what
    the live web view consumes; everything else reads the finished trace.
    """
    t_start = time.perf_counter()
    state = AgentState(question=question, image_path=image_path, max_steps=max_steps)
    counter = {"rekognition": 0}

    def emit(event: dict[str, Any]) -> None:
        if on_event is not None:
            on_event(event)

    seeker: EvidenceSeeker[AgentState] = EvidenceSeeker(
        actions=ROADSENSE_ACTIONS,
        render_state=lambda s, progress: s.render_for_decider(progress.steps_used),
        is_legal=policy.legality,
        execute=_make_executor(rekognition, counter, emit),
        max_steps=max_steps,
        fallback_action="answer",
        mask_illegal=mask_illegal_actions,
    )

    # ``evidence_sufficient`` is RoadSense's own extra signal, not part of the
    # pattern. The decider answers it alongside the choice in the same call, so it
    # costs nothing extra; it is read back off the client here.
    sufficiency: list[float | None] = []
    inner_decide = decider.decide

    def decide_and_capture(state_text: str, options: dict[str, str]):
        result = inner_decide(state_text, options)
        sufficiency.append(result.evidence_sufficient)
        decide_and_capture.backend = result.backend  # type: ignore[attr-defined]
        return result

    decide_and_capture.backend = ""  # type: ignore[attr-defined]

    class _Backend:
        decide = staticmethod(decide_and_capture)

    def report_step(step) -> None:
        emit({
            "type": "decision",
            "step": step.index,
            "action": step.action,
            "confidence": step.confidence,
            "probabilities": step.probabilities,
            "offered": step.offered,
            "rejected": step.rejected,
            "latency_ms": round(step.latency_ms),
            "decider_called": step.decider_called,
            "evidence_sufficient": sufficiency[-1] if sufficiency else None,
        })

    episode = seeker.run(state, _Backend(), on_step=report_step)

    for i, step in enumerate(episode.steps):
        state.decisions.append(
            Decision(
                step=step.index,
                action=step.action,
                confidence=step.confidence,
                probabilities=step.probabilities,
                evidence_sufficient=sufficiency[i] if i < len(sufficiency) else None,
                latency_ms=step.latency_ms,
                rejected=step.rejected,
                decider_called=step.decider_called,
                offered_actions=step.offered,
            )
        )

    answer = write_answer(bedrock, state, model_id=bedrock_model)
    if answer.error:
        state.errors.append(f"bedrock: {answer.error}")

    last = state.decisions[-1].action if state.decisions else "answer"
    if answer.error:
        outcome = OUTCOME_ERROR
    elif last == "request_better_evidence":
        outcome = OUTCOME_INSUFFICIENT
    else:
        outcome = OUTCOME_ANSWERED

    cost = compute_cost(
        rekognition_calls=counter["rekognition"],
        bedrock_model=bedrock_model,
        bedrock_input_tokens=answer.input_tokens,
        bedrock_output_tokens=answer.output_tokens,
        decider_input_tokens=episode.input_tokens,
        decider_output_tokens=episode.output_tokens,
        pricing=pricing if pricing is not None else load_pricing(),
    )
    trace = build_trace(
        state,
        answer=answer.text,
        # The model that answered, not the one requested -- they differ when Bedrock
        # had no capacity for the first choice and the client fell back.
        bedrock_model=answer.model_id,
        bedrock_usage={
            "input_tokens": answer.input_tokens,
            "output_tokens": answer.output_tokens,
            "latency_ms": round(answer.latency_ms, 1),
        },
        decider_backend=decide_and_capture.backend,  # type: ignore[attr-defined]
        cost=cost,
        wall_ms=(time.perf_counter() - t_start) * 1000,
        outcome=outcome,
    )
    trace["stop_reason"] = episode.stop_reason
    trace["policy_overrides"] = len(episode.overrides)
    trace["bedrock_fell_back_from"] = answer.fell_back_from
    emit({"type": "answer", "answer": answer.text, "outcome": outcome, "trace": trace})
    return AgentResult(
        state=state, answer=answer.text, outcome=outcome, trace=trace, cost=cost, bedrock=answer
    )
