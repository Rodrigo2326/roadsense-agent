"""The reusable pattern, tested without any road, image or AWS call."""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from roadsense.pattern import Action, EvidenceSeeker, Progress, StopReason


@dataclass
class Box:
    """A minimal state: a bag of facts and which tools already ran."""

    facts: list[str] = field(default_factory=list)
    ran: list[str] = field(default_factory=list)
    allow_b: bool = True


ACTIONS = [
    Action("tool_a", "Gather fact A."),
    Action("tool_b", "Gather fact B."),
    Action("give_up", "Stop and explain that the facts are not enough.", terminal=True),
    Action("answer", "Stop and answer from the facts gathered.", terminal=True),
]


def render(box: Box, progress: Progress) -> str:
    return f"facts={box.facts} ran={box.ran} step={progress.steps_used}/{progress.max_steps}"


def is_legal(box: Box, action: str) -> str | None:
    if action in box.ran:
        return f"{action} already ran"
    if action == "tool_b" and not box.allow_b:
        return "tool_b is not available in this state"
    return None


def execute(box: Box, action: str) -> None:
    if action in ("answer", "give_up"):
        return
    box.ran.append(action)
    box.facts.append(f"fact from {action}")


class Stub:
    """Returns a fixed ranking each call, filtered to whatever is on the menu."""

    def __init__(self, rankings: list[list[str]]) -> None:
        self.rankings = rankings
        self.calls = 0
        self.menus: list[list[str]] = []
        self.states: list[str] = []

    def decide(self, state_text: str, options: dict[str, str]):
        self.states.append(state_text)
        self.menus.append(sorted(options))
        ranking = self.rankings[min(self.calls, len(self.rankings) - 1)]
        self.calls += 1
        ranked = [a for a in ranking if a in options] or sorted(options)
        probs = {a: 0.0 for a in options}
        for i, a in enumerate(ranked):
            probs[a] = 1.0 / (i + 1)
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}

        @dataclass
        class Choice:
            action: str
            confidence: float
            probabilities: dict
            ranking: list
            latency_ms: float = 1.0
            input_tokens: int = 7
            output_tokens: int = 2

        return Choice(ranked[0], probs[ranked[0]], probs, ranked)


def seeker(**kw) -> EvidenceSeeker[Box]:
    return EvidenceSeeker(
        actions=ACTIONS,
        render_state=render,
        is_legal=is_legal,
        execute=execute,
        max_steps=kw.pop("max_steps", 4),
        fallback_action="answer",
        **kw,
    )


def test_a_terminal_action_ends_the_episode():
    episode = seeker().run(Box(), Stub([["answer"]]))
    assert episode.actions_taken == ["answer"]
    assert episode.stop_reason == StopReason.TERMINAL


def test_non_terminal_actions_keep_the_loop_going_and_accumulate_state():
    box = Box()
    episode = seeker().run(box, Stub([["tool_a"], ["tool_b"], ["answer"]]))
    assert episode.actions_taken == ["tool_a", "tool_b", "answer"]
    assert box.facts == ["fact from tool_a", "fact from tool_b"]


def test_illegal_options_are_masked_out_of_the_menu():
    box = Box(allow_b=False)
    stub = Stub([["tool_b", "tool_a"], ["answer"]])
    seeker().run(box, stub)
    assert "tool_b" not in stub.menus[0]
    assert stub.menus[0] == ["answer", "give_up", "tool_a"]


def test_a_tool_disappears_from_the_menu_once_it_has_run():
    stub = Stub([["tool_a"], ["answer"]])
    seeker().run(Box(), stub)
    assert "tool_a" in stub.menus[0]
    assert "tool_a" not in stub.menus[1]


def test_without_masking_an_illegal_pick_is_vetoed_and_the_reason_recorded():
    box = Box(allow_b=False)
    episode = seeker(mask_illegal=False).run(box, Stub([["tool_b", "tool_a"], ["answer"]]))
    first = episode.steps[0]
    assert first.action == "tool_a"
    assert "tool_b" in (first.rejected or "")
    assert len(episode.overrides) == 1
    assert sorted(first.offered) == ["answer", "give_up", "tool_a", "tool_b"]


def test_masking_removes_the_overrides_that_vetoing_would_have_produced():
    """The same scenario, both arms -- this is the measurement, not just a setting."""
    unmasked = seeker(mask_illegal=False).run(Box(allow_b=False), Stub([["tool_b", "tool_a"], ["answer"]]))
    masked = seeker(mask_illegal=True).run(Box(allow_b=False), Stub([["tool_b", "tool_a"], ["answer"]]))
    assert len(unmasked.overrides) == 1
    assert len(masked.overrides) == 0
    assert unmasked.actions_taken == masked.actions_taken


def test_the_budget_forces_a_terminal_step_that_is_marked_as_not_decided():
    stub = Stub([["tool_a"], ["tool_b"], ["give_up"]])
    episode = seeker(max_steps=2).run(Box(), stub)
    assert stub.calls == 2, "the decider was consulted after the budget ran out"
    assert episode.stop_reason == StopReason.BUDGET
    forced = episode.steps[-1]
    assert forced.action == "answer"
    assert forced.decider_called is False
    assert forced.confidence is None
    assert forced.probabilities == {}
    assert "budget" in (forced.rejected or "")


def test_the_loop_terminates_even_if_the_decider_only_ever_asks_for_more():
    """Every non-terminal action is exhausted, then the budget closes it out."""
    episode = seeker(max_steps=10).run(Box(), Stub([["tool_a", "tool_b"]]))
    assert episode.steps[-1].action == "answer"
    assert len(episode.steps) <= 4


def test_render_state_receives_the_live_step_count():
    stub = Stub([["tool_a"], ["tool_b"], ["answer"]])
    seeker().run(Box(), stub)
    assert "step=0/4" in stub.states[0]
    assert "step=1/4" in stub.states[1]
    assert "step=2/4" in stub.states[2]


def test_progress_reports_remaining_budget():
    assert Progress(steps_used=1, max_steps=4).remaining == 3
    assert Progress(steps_used=9, max_steps=4).remaining == 0


def test_token_usage_is_summed_across_the_episode():
    episode = seeker().run(Box(), Stub([["tool_a"], ["answer"]]))
    assert episode.input_tokens == 14
    assert episode.output_tokens == 4


def test_an_empty_ranking_falls_back_instead_of_crashing():
    class Empty:
        def decide(self, state_text, options):
            @dataclass
            class Choice:
                action: str = ""
                confidence: float = 0.0
                probabilities: dict = field(default_factory=dict)
                ranking: list = field(default_factory=list)
                latency_ms: float = 0.0
                input_tokens: int = 0
                output_tokens: int = 0

            return Choice()

    episode = seeker().run(Box(), Empty())
    assert episode.actions_taken == ["answer"]
    assert "no ranking" in (episode.steps[0].rejected or "")


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"actions": [Action("only", "d", terminal=True)]}, "at least 2 options"),
        ({"actions": [Action("a", "d"), Action("a", "d", terminal=True)]}, "unique"),
        ({"fallback_action": "nope"}, "not in the action set"),
        ({"fallback_action": "tool_a"}, "must be terminal"),
    ],
)
def test_a_misconfigured_action_set_fails_loudly_at_construction(kwargs, message):
    args = {
        "actions": ACTIONS,
        "render_state": render,
        "is_legal": is_legal,
        "execute": execute,
        "fallback_action": "answer",
    }
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        EvidenceSeeker(**args)
