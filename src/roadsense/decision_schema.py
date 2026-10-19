"""The typed action contract.

Decider 2B is a *pointer* model: it never writes text, it scores a fixed set of
options. That is the whole reason the action space here is a closed dict instead of
a JSON blob we would have to parse and repair. There is no "malformed action" failure
mode to defend against -- the model physically cannot emit a name that is not a key
of ``ACTIONS``.

What we still validate is everything the model is *not* responsible for: that the
action is legal given the state (no OCR without an image), that we are not repeating
a tool that already ran, and that the step budget holds. Those are policy, not
parsing. See ``roadsense.policy``.
"""

from __future__ import annotations

from typing import Final

# Action name -> the description Decider scores against. These strings are the
# prompt: they are what the pointer head actually reads, so they are written as
# plain declarative descriptions of the effect, not as instructions.
ACTIONS: Final[dict[str, str]] = {
    "detect_objects": (
        "Run an object detector on the image to list the vehicles, people, signs and "
        "other objects that are present in the scene."
    ),
    "read_sign_text": (
        "Run optical character recognition on the image to read the text written on "
        "road signs and other surfaces."
    ),
    "request_better_evidence": (
        "Stop and ask for a clearer image, because the evidence gathered so far is too "
        "poor or too blurry to answer the question honestly."
    ),
    "answer": (
        "Stop gathering evidence and write the final answer using only the evidence "
        "that has already been collected."
    ),
}

# Actions that consume a vision tool call. The other two are terminal.
TOOL_ACTIONS: Final[frozenset[str]] = frozenset({"detect_objects", "read_sign_text"})
TERMINAL_ACTIONS: Final[frozenset[str]] = frozenset({"answer", "request_better_evidence"})

# Actions that need a readable image on disk. Asking for OCR on a missing file is a
# guardrail violation (brief section 5), not a tool error.
IMAGE_REQUIRED_ACTIONS: Final[frozenset[str]] = TOOL_ACTIONS


class InvalidAction(ValueError):
    """Raised when an action name is not part of the closed contract."""


def validate_action_name(name: str) -> str:
    """Reject anything outside ``ACTIONS``. Returns the name unchanged if legal."""
    if name not in ACTIONS:
        raise InvalidAction(
            f"action {name!r} is not in the allowed set {sorted(ACTIONS)}"
        )
    return name
