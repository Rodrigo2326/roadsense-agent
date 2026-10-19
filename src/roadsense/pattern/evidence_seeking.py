"""Evidence-Seeking Tool Selection -- the pattern, with the domain removed.

The shape of the problem this solves is not specific to images. You have a question,
a set of actions that each buy you some evidence, and no way to know up front which
ones you need. Running them all is wasteful and often impossible; guessing from the
first one is how agents end up confidently wrong.

The pattern is four moving parts:

1. **A typed, closed action set.** Every option is named and described up front.
   A pointer-head decision model like Strands Decider 2B scores these options rather
   than generating a name, so "the model emitted an action that does not exist" is
   not a failure mode that can occur.
2. **A state rendered as text.** The decision model sees what has been learned so
   far, never the raw artefact. That is what makes the selector portable: a decision
   over an evidence summary works the same whether the evidence came from OCR, a
   database, or a colleague.
3. **Legality enforced in code.** Preconditions ("this tool already ran", "this needs
   a file that is missing") are checked by a function, not requested in a prompt. By
   default the illegal options are *masked out* of the menu rather than offered and
   then vetoed -- measurably better, because the model stops spending probability
   mass on things it was never allowed to pick.
4. **A budget, and an honest end.** The loop always terminates. When the budget runs
   out the step is recorded as forced, with no confidence attached, because nothing
   decided it.

Everything here is generic over your state type ``S``. See
``examples/triage_example.py`` for a complete, image-free use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Generic, Protocol, Sequence, TypeVar

S = TypeVar("S")


@dataclass(frozen=True)
class Progress:
    """How much of the budget is gone, handed to ``render_state`` on every step.

    The budget is part of the decision, not bookkeeping around it: an agent with one
    step left should weigh "gather more" differently from one with four. Passing it
    explicitly is what stops the state renderer from having to guess where in the
    loop it is.
    """

    steps_used: int
    max_steps: int

    @property
    def remaining(self) -> int:
        return max(0, self.max_steps - self.steps_used)


@dataclass(frozen=True)
class Action:
    """One option the selector may choose.

    ``description`` is not documentation -- it is the text the decision model scores.
    Write it as a plain statement of what the action does and when it applies.
    """

    name: str
    description: str
    terminal: bool = False
    """Terminal actions end the episode once executed."""


class SelectorChoice(Protocol):
    """What a decision backend must return.

    ``roadsense.decider_client.DeciderResult`` satisfies this structurally, so the
    Strands Decider client drops in with no adapter.
    """

    action: str
    confidence: float
    probabilities: dict[str, float]
    ranking: list[str]
    latency_ms: float
    input_tokens: int
    output_tokens: int


class DecisionBackend(Protocol):
    def decide(self, state_text: str, options: dict[str, str]) -> SelectorChoice: ...


class StopReason:
    TERMINAL = "terminal_action"
    BUDGET = "budget_exhausted"


@dataclass
class Step:
    index: int
    action: str
    confidence: float | None
    probabilities: dict[str, float]
    offered: list[str]
    rejected: str | None
    latency_ms: float
    decider_called: bool


@dataclass
class Episode:
    steps: list[Step] = field(default_factory=list)
    stop_reason: str = StopReason.TERMINAL
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def actions_taken(self) -> list[str]:
        return [s.action for s in self.steps]

    @property
    def overrides(self) -> list[Step]:
        """Steps where legality overruled the model.

        A high count means the action descriptions or the state rendering are not
        carrying the preconditions well enough -- it is a diagnostic, not just a log.
        """
        return [s for s in self.steps if s.rejected]


class EvidenceSeeker(Generic[S]):
    """Drives the loop. Owns no domain knowledge; everything arrives as callbacks.

    Args:
        actions: the closed action set. Needs at least two options, and the fallback
            must be terminal, or the loop could not end.
        render_state: turns your state plus the budget progress into the text the
            model reads.
        is_legal: returns ``None`` if the action is allowed in this state, or a short
            reason why it is not.
        execute: performs the action, mutating your state. Called for terminal
            actions too, so a "give up and explain" action can record why.
        max_steps: hard ceiling on decisions.
        fallback_action: what a forced or stuck episode ends with. Must be terminal,
            and must be legal in every state.
        mask_illegal: when true (default) illegal actions are removed from the menu
            instead of being offered and then vetoed.
    """

    def __init__(
        self,
        *,
        actions: Sequence[Action],
        render_state: Callable[[S, Progress], str],
        is_legal: Callable[[S, str], str | None],
        execute: Callable[[S, str], None],
        max_steps: int = 4,
        fallback_action: str = "answer",
        mask_illegal: bool = True,
    ) -> None:
        if len(actions) < 2:
            raise ValueError("an action set needs at least 2 options to be a choice")
        by_name = {a.name: a for a in actions}
        if len(by_name) != len(actions):
            raise ValueError("action names must be unique")
        if fallback_action not in by_name:
            raise ValueError(f"fallback_action {fallback_action!r} is not in the action set")
        if not by_name[fallback_action].terminal:
            raise ValueError(f"fallback_action {fallback_action!r} must be terminal")

        self.actions = by_name
        self.render_state = render_state
        self.is_legal = is_legal
        self.execute = execute
        self.max_steps = max_steps
        self.fallback_action = fallback_action
        self.mask_illegal = mask_illegal

    def menu(self, state: S) -> dict[str, str]:
        """The options offered for this state."""
        if not self.mask_illegal:
            return {a.name: a.description for a in self.actions.values()}
        legal = {
            a.name: a.description
            for a in self.actions.values()
            if self.is_legal(state, a.name) is None
        }
        # A choice question needs two options; the fallback is legal by contract.
        legal.setdefault(self.fallback_action, self.actions[self.fallback_action].description)
        return legal

    def resolve(self, state: S, ranking: Sequence[str]) -> tuple[str, str | None]:
        """Highest-ranked legal action, and why the first pick was dropped if it was."""
        if not ranking:
            return self.fallback_action, "the backend returned no ranking"
        first = ranking[0]
        why = self.is_legal(state, first)
        if why is None:
            return first, None
        for alt in ranking[1:]:
            if self.is_legal(state, alt) is None:
                return alt, f"picked {first} but {why}; fell back to its next legal preference"
        return self.fallback_action, f"no legal action available ({why})"

    def run(
        self,
        state: S,
        decider: DecisionBackend,
        on_step: Callable[[Step], None] | None = None,
    ) -> Episode:
        """Drive the loop to a terminal action or the budget.

        ``on_step`` is called with each step *after* it is decided and *before* it is
        executed. That ordering is what lets a live view show the decision and then
        the evidence it produced, in the order they actually happened, rather than
        replaying a finished episode.
        """
        episode = Episode()

        while True:
            index = len(episode.steps) + 1

            if index > self.max_steps:
                episode.steps.append(
                    Step(
                        index=index,
                        action=self.fallback_action,
                        confidence=None,
                        probabilities={},
                        offered=[],
                        rejected=f"step budget of {self.max_steps} is exhausted",
                        latency_ms=0.0,
                        decider_called=False,
                    )
                )
                episode.stop_reason = StopReason.BUDGET
                if on_step is not None:
                    on_step(episode.steps[-1])
                self.execute(state, self.fallback_action)
                return episode

            options = self.menu(state)
            progress = Progress(steps_used=len(episode.steps), max_steps=self.max_steps)
            choice = decider.decide(self.render_state(state, progress), options)
            episode.input_tokens += choice.input_tokens
            episode.output_tokens += choice.output_tokens

            action, rejected = self.resolve(state, choice.ranking)
            episode.steps.append(
                Step(
                    index=index,
                    action=action,
                    confidence=choice.confidence,
                    probabilities=dict(choice.probabilities),
                    offered=sorted(options),
                    rejected=rejected,
                    latency_ms=choice.latency_ms,
                    decider_called=True,
                )
            )

            if on_step is not None:
                on_step(episode.steps[-1])

            self.execute(state, action)
            if self.actions[action].terminal:
                episode.stop_reason = StopReason.TERMINAL
                return episode
