"""The trace, and what it costs.

Every run writes a JSON document that is enough to reconstruct the episode: what was
asked, what Decider scored at each step, what policy did about it, what each tool
actually returned, and what Bedrock wrote. The trace is the deliverable, not a debug
aid -- the whole claim of this project is that the decision path is inspectable.

On cost, this module is deliberately stingy about what it is willing to assert.
Token and image counts are always real. A dollar figure appears only when a rate was
loaded from ``pricing.json``; otherwise the field is ``null`` and ``cost_note`` says
why. Rekognition's rate is in the shipped defaults because it came from the AWS
Pricing API. Bedrock's is not: the Pricing API does not publish a rate for
``claude-haiku-4-5`` and the public pricing page does not list it either, so quoting
one would be a number we made up. Fill it in ``pricing.json`` if you have it.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .state import AgentState

#: Verified 2026-10-16 against the AWS Pricing API, ServiceCode=AmazonRekognition,
#: location "US East (N. Virginia)", group "Number of images processed using Image
#: Async API", first tier (0-1M images/month): USD 0.0010000000 per image.
#: DetectLabels and DetectText are both billed from this group.
REKOGNITION_USD_PER_IMAGE = 0.001

#: No default. See the module docstring.
BEDROCK_RATES: dict[str, dict[str, float]] = {}


@dataclass
class CostReport:
    rekognition_calls: int = 0
    bedrock_input_tokens: int = 0
    bedrock_output_tokens: int = 0
    decider_input_tokens: int = 0
    decider_output_tokens: int = 0
    rekognition_usd: float | None = None
    bedrock_usd: float | None = None
    total_usd: float | None = None
    notes: list[str] = field(default_factory=list)


def load_pricing(path: str | Path | None = None) -> dict[str, Any]:
    """Load rate overrides. Absent file is normal, not an error."""
    p = Path(path or os.environ.get("ROADSENSE_PRICING", "pricing.json"))
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"_error": f"could not read {p}: {exc}"}


def compute_cost(
    *,
    rekognition_calls: int,
    bedrock_model: str,
    bedrock_input_tokens: int,
    bedrock_output_tokens: int,
    decider_input_tokens: int = 0,
    decider_output_tokens: int = 0,
    pricing: dict[str, Any] | None = None,
) -> CostReport:
    pricing = pricing or {}
    rep = CostReport(
        rekognition_calls=rekognition_calls,
        bedrock_input_tokens=bedrock_input_tokens,
        bedrock_output_tokens=bedrock_output_tokens,
        decider_input_tokens=decider_input_tokens,
        decider_output_tokens=decider_output_tokens,
    )
    if "_error" in pricing:
        rep.notes.append(pricing["_error"])

    rek_rate = pricing.get("rekognition_usd_per_image", REKOGNITION_USD_PER_IMAGE)
    rep.rekognition_usd = round(rekognition_calls * rek_rate, 6)
    rep.notes.append(
        f"Rekognition billed at ${rek_rate}/image (AWS Pricing API, us-east-1, first tier)."
    )

    rates = {**BEDROCK_RATES, **(pricing.get("bedrock_usd_per_1k_tokens") or {})}
    rate = rates.get(bedrock_model)
    if rate and "input" in rate and "output" in rate:
        rep.bedrock_usd = round(
            bedrock_input_tokens / 1000 * rate["input"]
            + bedrock_output_tokens / 1000 * rate["output"],
            6,
        )
        rep.notes.append(f"Bedrock billed from pricing.json rates for {bedrock_model}.")
    else:
        rep.notes.append(
            f"Bedrock cost omitted: no verified rate for {bedrock_model!r}. Token counts "
            "above are exact; add 'bedrock_usd_per_1k_tokens' to pricing.json to price them."
        )

    # Decider runs on hardware we already pay for by the hour, so its tokens are not a
    # per-call charge at all. The EC2 hourly rate is the real cost and it is reported
    # separately in the benchmark, not amortised into a fake per-token number.
    rep.notes.append(
        "Decider 2B tokens are not billed per token: the model runs on self-managed "
        "compute. See the benchmark report for the hourly instance cost."
    )

    if rep.bedrock_usd is None:
        rep.total_usd = None
    else:
        rep.total_usd = round((rep.rekognition_usd or 0.0) + rep.bedrock_usd, 6)
    return rep


def build_trace(
    state: AgentState,
    *,
    answer: str,
    bedrock_model: str,
    bedrock_usage: dict[str, int],
    decider_backend: str,
    cost: CostReport,
    wall_ms: float,
    outcome: str,
) -> dict[str, Any]:
    """Assemble the run document. ``raw`` tool payloads are kept for auditability."""
    return {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "question": state.question,
        "image_path": state.image_path,
        "max_steps": state.max_steps,
        "outcome": outcome,
        "answer": answer,
        "decider_backend": decider_backend,
        "bedrock_model": bedrock_model,
        "bedrock_usage": bedrock_usage,
        "wall_ms": round(wall_ms, 1),
        "steps": [asdict(d) for d in state.decisions],
        "evidence": [asdict(e) for e in state.evidence],
        "errors": state.errors,
        "cost": asdict(cost),
    }


def write_trace(trace: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(trace, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return p
