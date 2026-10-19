"""The reusable half of RoadSense, with no road in it.

``Evidence-Seeking Tool Selection``: a loop that asks a decision model which action
to take next given what is already known, enforces which actions are legal in code
rather than in a prompt, and stops when the model says it has enough -- or when the
budget says it must.
"""

from .evidence_seeking import (
    Action,
    Episode,
    EvidenceSeeker,
    Progress,
    SelectorChoice,
    Step,
    StopReason,
)

__all__ = [
    "Action",
    "Episode",
    "EvidenceSeeker",
    "Progress",
    "SelectorChoice",
    "Step",
    "StopReason",
]
