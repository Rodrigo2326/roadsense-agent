"""request_better_evidence -- the honest dead end.

This does not capture anything. It cannot: RoadSense reads static files, there is no
camera in the architecture. All it does is record *why* the current evidence was not
good enough, so the final answer can say so concretely instead of vaguely hedging.
"""

from __future__ import annotations

from ..state import AgentState, Evidence


def request_better_evidence(state: AgentState) -> Evidence:
    reasons: list[str] = []

    usable, why = state.image_is_usable()
    if not usable:
        reasons.append(why)

    for e in state.evidence:
        if not e.ok:
            reasons.append(f"{e.tool} failed ({e.error})")
        elif not e.items:
            reasons.append(f"{e.tool} returned nothing")
        elif e.quality is not None and e.quality < 0.80:
            reasons.append(f"{e.tool} confidence was only {e.quality:.2f}")

    if not reasons:
        reasons.append("the evidence gathered does not cover what the question asks")

    return Evidence(
        tool="request_better_evidence",
        ok=True,
        summary="A clearer image is needed because " + "; ".join(reasons) + ".",
        items=[{"reason": r} for r in reasons],
        quality=None,
    )
