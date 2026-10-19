"""Guardrails between Decider's pick and what actually runs.

Decider decides; policy decides whether that decision is *legal right now*. The two
jobs are kept apart on purpose, because conflating them is how agents end up with a
model that has been prompt-engineered into obeying rules it cannot actually enforce.
Here the rules are enforced in code and the model never gets a vote on them.

When a pick is illegal we do not substitute a hardcoded action. We walk Decider's own
probability ranking and take the best option that *is* legal. The model still chooses;
policy only removes what it was never allowed to pick. That keeps the override
honest — and it keeps the trace honest too, because the rejection is recorded with
its reason rather than silently rewritten.
"""

from __future__ import annotations

from dataclasses import dataclass

from .decision_schema import ACTIONS, IMAGE_REQUIRED_ACTIONS, TOOL_ACTIONS
from .state import AgentState


@dataclass
class PolicyVerdict:
    action: str
    rejected: str | None = None
    """``None`` when Decider's first pick stood. Otherwise why it was overruled."""


def legality(state: AgentState, action: str) -> str | None:
    """Return the reason ``action`` is illegal in ``state``, or ``None`` if it is fine."""
    if action not in ACTIONS:
        return f"{action!r} is not an allowed action"

    if action in IMAGE_REQUIRED_ACTIONS:
        usable, why = state.image_is_usable()
        if not usable:
            return f"{action} needs a readable image but {why}"

    if action in TOOL_ACTIONS and action in state.tools_run:
        return f"{action} already ran; repeating it would add no new evidence"

    return None


def available_actions(state: AgentState) -> dict[str, str]:
    """The subset of ``ACTIONS`` that is legal right now, with its descriptions.

    Offering an illegal option and then vetoing the model for taking it is a design
    smell: it wastes a slot in the pointer head and it makes the model look worse than
    it is. Masking removes the option instead. ``answer`` is always legal, so the
    result can never fall below the two options a choice question requires.

    Masking is a *default*, not a law -- :func:`apply` still re-checks legality,
    because an HTTP decider could return anything, and because the benchmark runs an
    unmasked arm to measure how often masking actually changes the outcome.
    """
    legal = {name: desc for name, desc in ACTIONS.items() if legality(state, name) is None}
    assert "answer" in legal, "answer must always be legal"
    return legal


def apply(state: AgentState, ranking: list[str]) -> PolicyVerdict:
    """Pick the highest-ranked legal action.

    ``ranking`` is Decider's actions ordered by probability, best first.
    """
    # Budget is checked before anything else: once it is gone the only honest move
    # is to answer from what we have, even if the model wants more evidence.
    if state.budget_exhausted:
        return PolicyVerdict(
            "answer",
            f"step budget of {state.max_steps} is exhausted; forced to answer with the evidence at hand",
        )

    first = ranking[0] if ranking else "answer"
    why_first = legality(state, first)
    if why_first is None:
        return PolicyVerdict(first)

    for alternative in ranking[1:]:
        if legality(state, alternative) is None:
            return PolicyVerdict(
                alternative,
                f"Decider picked {first} but {why_first}; fell back to its next legal preference",
            )

    # Every action Decider ranked is illegal. ``answer`` is always legal, so this is
    # only reachable if the ranking was empty or malformed.
    return PolicyVerdict("answer", f"no legal action available ({why_first})")
