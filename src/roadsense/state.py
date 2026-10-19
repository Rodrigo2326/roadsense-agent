"""Agent state, and how it is rendered for Decider.

Two rules from the brief drive this file.

**Decider never sees the image.** It receives a text rendering of what is known so
far. That is a deliberate constraint, not a limitation we are working around: the
point of the pattern is that tool selection is a decision over *evidence state*, and
a decision over evidence state is reusable in domains that have no images at all.

**Tool confidence is the tool's, never the agent's.** ``Evidence.quality`` holds the
number Rekognition actually returned. Nothing in this module invents a score. When a
tool gives us no usable signal we record ``None`` and render it as unknown, because
"we don't know how good this is" is itself evidence the decider should weigh.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Evidence:
    """One tool's output, normalised. ``raw`` keeps the untouched API response."""

    tool: str
    ok: bool
    summary: str
    """One-line description of what this tool found, in the words of the tool."""
    items: list[dict[str, Any]] = field(default_factory=list)
    quality: float | None = None
    """Top confidence reported *by the tool*, 0..1. ``None`` when the tool gave none."""
    error: str | None = None
    duration_ms: float = 0.0
    raw: dict[str, Any] | None = None


@dataclass
class Decision:
    """One step of the loop: what was chosen, and how sure the model said it was."""

    step: int
    action: str
    confidence: float | None
    """Decider's calibrated confidence. ``None`` when no Decider call was made --
    a forced step is not a decision, and reporting 0.0 would read like one."""
    probabilities: dict[str, float]
    evidence_sufficient: float | None
    """P(true) from the noul question. ``None`` if that question was not asked."""
    latency_ms: float
    rejected: str | None = None
    """Set when policy overrode the model's pick, with the reason."""
    decider_called: bool = True
    offered_actions: list[str] = field(default_factory=list)
    """Which actions were on the menu for this step. Empty on a forced step."""


@dataclass
class AgentState:
    """Everything the agent knows. Serialisable as-is into the trace."""

    question: str
    image_path: str | None
    max_steps: int = 4
    evidence: list[Evidence] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def steps_used(self) -> int:
        return len(self.decisions)

    @property
    def tools_run(self) -> list[str]:
        return [e.tool for e in self.evidence]

    @property
    def budget_exhausted(self) -> bool:
        return self.steps_used >= self.max_steps

    def image_is_usable(self) -> tuple[bool, str]:
        """Can a vision tool run at all? Checked before any OCR/detection call."""
        if not self.image_path:
            return False, "no image was provided"
        p = Path(self.image_path)
        if not p.is_file():
            return False, f"image file not found: {self.image_path}"
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            return False, f"unsupported image format: {p.suffix!r} (need .jpg/.jpeg/.png)"
        if p.stat().st_size == 0:
            return False, "image file is empty"
        return True, "ok"

    def render_for_decider(self, steps_used: int | None = None) -> str:
        """The text Decider actually reads. No image, no invented confidence.

        ``steps_used`` is supplied by the loop while an episode is running, because
        decisions are only appended to ``self.decisions`` once the episode ends.
        """
        lines = [
            "Task: answer a question about a road scene photograph using vision tools.",
            f"Question: {self.question}",
        ]

        usable, why = self.image_is_usable()
        lines.append(f"Image: {'provided and readable' if usable else 'UNUSABLE - ' + why}.")

        if not self.evidence:
            lines.append("Evidence collected so far: none. No tool has been run yet.")
        else:
            lines.append("Evidence collected so far:")
            for e in self.evidence:
                q = "unknown" if e.quality is None else f"{e.quality:.2f}"
                status = "ran successfully" if e.ok else "FAILED"
                lines.append(f"  - {e.tool} {status}. {e.summary} (tool confidence: {q})")

        not_run = sorted({"detect_objects", "read_sign_text"} - set(self.tools_run))
        if not_run:
            lines.append(f"Tools not yet run: {', '.join(not_run)}.")
        else:
            lines.append("All vision tools have already been run.")

        if self.errors:
            lines.append(f"Errors so far: {'; '.join(self.errors)}.")

        used = self.steps_used if steps_used is None else steps_used
        lines.append(f"Steps used: {used} of {self.max_steps}.")
        return "\n".join(lines)
