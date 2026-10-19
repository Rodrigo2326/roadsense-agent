"""Command line entry point: one question, one image, one trace."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from .agent import run
from .bedrock_client import DEFAULT_MODEL_ID
from .decider_client import DeciderUnavailable, HttpDecider, InProcessDecider

DEFAULT_CHECKPOINT = "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="roadsense",
        description="Ask a question about a road-scene image. The agent decides what "
        "evidence to gather, gathers it, and answers only from what it found.",
    )
    p.add_argument("--question", "-q", required=True)
    p.add_argument("--image", "-i", default=None, help="path to a .jpg/.jpeg/.png file")
    p.add_argument("--max-steps", type=int, default=int(os.environ.get("ROADSENSE_MAX_STEPS", 4)))
    p.add_argument("--region", default=os.environ.get("AWS_REGION", "us-east-1"))
    p.add_argument("--bedrock-model", default=os.environ.get("ROADSENSE_BEDROCK_MODEL_ID", DEFAULT_MODEL_ID))
    p.add_argument(
        "--decider-endpoint",
        default=os.environ.get("ROADSENSE_DECIDER_ENDPOINT"),
        help="URL of a `strands-decider serve` instance. Omit to load the model in-process.",
    )
    p.add_argument("--decider-checkpoint", default=os.environ.get("ROADSENSE_DECIDER_CHECKPOINT", DEFAULT_CHECKPOINT))
    p.add_argument("--decider-device", default=os.environ.get("ROADSENSE_DECIDER_DEVICE", "cpu"))
    p.add_argument("--trace", default=None, help="where to write the JSON trace")
    p.add_argument("--quiet", action="store_true", help="print only the answer")
    return p


def _print_report(result: Any) -> None:
    print("\n=== decision trace ===")
    for d in result.state.decisions:
        conf = "n/a (forced)" if d.confidence is None else f"{d.confidence:.3f}"
        suff = "n/a" if d.evidence_sufficient is None else f"{d.evidence_sufficient:.3f}"
        print(f"  step {d.step}: {d.action}  confidence={conf}  P(evidence sufficient)={suff}  {d.latency_ms:.0f}ms")
        if d.probabilities:
            ranked = sorted(d.probabilities.items(), key=lambda kv: kv[1], reverse=True)
            print("      " + "  ".join(f"{k}={v:.3f}" for k, v in ranked))
        if d.rejected:
            print(f"      policy: {d.rejected}")

    print("\n=== evidence ===")
    for e in result.state.evidence:
        q = "not reported" if e.quality is None else f"{e.quality:.2f}"
        print(f"  {e.tool}: ok={e.ok} confidence={q} ({e.duration_ms:.0f}ms)")
        print(f"      {e.summary}")

    c = result.cost
    total = "omitted (see notes)" if c.total_usd is None else f"${c.total_usd:.6f}"
    print(f"\n=== cost ===\n  rekognition calls: {c.rekognition_calls}  bedrock tokens: "
          f"{c.bedrock_input_tokens} in / {c.bedrock_output_tokens} out  total: {total}")
    for n in c.notes:
        print(f"      - {n}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import boto3

    decider = (
        HttpDecider(args.decider_endpoint)
        if args.decider_endpoint
        else InProcessDecider(args.decider_checkpoint, device=args.decider_device)
    )

    try:
        result = run(
            args.question,
            args.image,
            decider=decider,
            rekognition=boto3.client("rekognition", region_name=args.region),
            bedrock=boto3.client("bedrock-runtime", region_name=args.region),
            max_steps=args.max_steps,
            bedrock_model=args.bedrock_model,
        )
    except DeciderUnavailable as exc:
        print(f"decider unavailable: {exc}", file=sys.stderr)
        return 2

    if not args.quiet:
        _print_report(result)
    print(f"\n=== answer ({result.outcome}) ===\n{result.answer}")

    if args.trace:
        from .telemetry import write_trace

        print(f"\ntrace written to {write_trace(result.trace, Path(args.trace))}")
    return 0 if result.outcome != "error" else 1


if __name__ == "__main__":
    sys.exit(main())
