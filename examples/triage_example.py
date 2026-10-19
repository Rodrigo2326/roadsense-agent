"""The pattern with no images, no AWS, and no road.

A support agent is handed a ticket and has to decide what to look up before it
replies. It can check the order, check the refund window, ask the customer for the
one fact nobody recorded, or answer. Same four moving parts as RoadSense: a closed
action set, a text state, legality in code, a budget.

Run it against a real decider:

    python examples/triage_example.py --decider-endpoint http://localhost:8000

or against the bundled scripted stub, which needs nothing installed:

    python examples/triage_example.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadsense.pattern import Action, EvidenceSeeker, Progress  # noqa: E402

ACTIONS = [
    Action(
        "lookup_order",
        "Look up the order in the orders database to find its status, ship date and "
        "current location.",
    ),
    Action(
        "check_refund_window",
        "Check the refund policy to find out whether this order is still inside the "
        "window during which a refund can be issued.",
    ),
    Action(
        "ask_customer",
        "Stop and ask the customer for information that no internal system holds, "
        "because the answer cannot be determined without it.",
        terminal=True,
    ),
    Action(
        "answer",
        "Stop looking things up and reply to the customer using only the facts that "
        "have already been gathered.",
        terminal=True,
    ),
]

#: A tiny stand-in for the systems a real agent would query.
ORDERS = {
    "A-1001": {"status": "delivered", "shipped": "2026-09-02", "days_since_delivery": 41},
}


@dataclass
class Ticket:
    text: str
    order_id: str | None
    facts: list[str] = field(default_factory=list)
    tools_run: list[str] = field(default_factory=list)


def render(ticket: Ticket, progress: Progress) -> str:
    lines = [
        "Task: decide what to check before replying to a customer support ticket.",
        f"Ticket: {ticket.text}",
        f"Order id on the ticket: {ticket.order_id or 'MISSING - the customer did not give one'}",
    ]
    if ticket.facts:
        lines.append("Facts gathered so far:")
        lines += [f"  - {f}" for f in ticket.facts]
    else:
        lines.append("Facts gathered so far: none. Nothing has been looked up yet.")

    not_run = [a.name for a in ACTIONS if not a.terminal and a.name not in ticket.tools_run]
    lines.append(
        f"Lookups not yet run: {', '.join(not_run)}." if not_run else "All lookups have been run."
    )
    lines.append(f"Steps used: {progress.steps_used} of {progress.max_steps}.")
    return "\n".join(lines)


def is_legal(ticket: Ticket, action: str) -> str | None:
    """Preconditions, in code. Never asked of the model in a prompt."""
    if action in ticket.tools_run:
        return f"{action} already ran; repeating it would add no new facts"
    if action == "lookup_order" and not ticket.order_id:
        return "lookup_order needs an order id and the ticket does not have one"
    if action == "check_refund_window" and "delivered" not in " ".join(ticket.facts):
        return "the refund window depends on the delivery date, which is not known yet"
    return None


def execute(ticket: Ticket, action: str) -> None:
    if action in ("answer", "ask_customer"):
        if action == "ask_customer":
            ticket.facts.append(
                "Asked the customer for the order id, because no lookup is possible without it."
            )
        return

    ticket.tools_run.append(action)
    if action == "lookup_order":
        order = ORDERS.get(ticket.order_id or "")
        ticket.facts.append(
            f"Order {ticket.order_id} is {order['status']}, delivered "
            f"{order['days_since_delivery']} days ago."
            if order
            else f"Order {ticket.order_id} was not found in the orders database."
        )
    elif action == "check_refund_window":
        ticket.facts.append(
            "The refund window is 30 days after delivery, so this order is outside it."
        )


class ScriptedBackend:
    """A stand-in decider, so the example runs with nothing installed."""

    def __init__(self, picks: list[str]) -> None:
        self.picks, self.calls = picks, 0

    def decide(self, state_text: str, options: dict[str, str]):
        pick = next((p for p in self.picks[self.calls :] if p in options), "answer")
        self.calls += 1
        probs = {k: (0.85 if k == pick else 0.15 / max(1, len(options) - 1)) for k in options}

        @dataclass
        class Choice:
            action: str
            confidence: float
            probabilities: dict
            ranking: list
            latency_ms: float = 0.0
            input_tokens: int = 0
            output_tokens: int = 0

        return Choice(
            action=pick,
            confidence=0.85,
            probabilities=probs,
            ranking=sorted(probs, key=lambda k: probs[k], reverse=True),
        )


def build_decider(endpoint: str | None, picks: list[str]):
    if not endpoint:
        return ScriptedBackend(picks)
    from roadsense.decider_client import HttpDecider

    return HttpDecider(endpoint)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--decider-endpoint", default=None)
    args = ap.parse_args()

    seeker: EvidenceSeeker[Ticket] = EvidenceSeeker(
        actions=ACTIONS,
        render_state=render,
        is_legal=is_legal,
        execute=execute,
        max_steps=4,
        fallback_action="answer",
    )

    scenarios = [
        (
            "ticket with an order id -- needs two lookups in sequence",
            Ticket("I want a refund for order A-1001, it arrived ages ago.", "A-1001"),
            ["lookup_order", "check_refund_window", "answer"],
        ),
        (
            "ticket with no order id -- the only honest move is to ask",
            Ticket("I want a refund, this is unacceptable.", None),
            ["lookup_order", "ask_customer"],
        ),
    ]

    for title, ticket, picks in scenarios:
        print(f"\n=== {title} ===")
        episode = seeker.run(ticket, build_decider(args.decider_endpoint, picks))
        for step in episode.steps:
            conf = "forced" if step.confidence is None else f"{step.confidence:.3f}"
            print(f"  step {step.index}: {step.action:22} conf={conf}  offered={step.offered}")
            if step.rejected:
                print(f"      policy: {step.rejected}")
        print(f"  stop_reason: {episode.stop_reason}")
        for fact in ticket.facts:
            print(f"  fact: {fact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
