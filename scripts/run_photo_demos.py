"""Run the agent over the real photographs and save the traces for the demo page.

The benchmark runs on synthetic images on purpose -- degradation you can dial is
worth more than realism when the point is reproducibility. These runs are the other
half: the same agent, unchanged, on photographs a person actually took, so a reader
can see it is not a trick that only works on drawings.

Output is shaped like a benchmark result file, so `scripts/make_replay.py` can read
it with no special case.

    python scripts/run_photo_demos.py --device cuda
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PHOTOS = [
    ("stop-sign.jpg", "What text can be read on the sign?"),
    ("speed-limit.jpg", "What vehicles appear and what text can be read on the sign?"),
    ("intersection.jpg", "What text can be read on the sign?"),
    ("street-traffic.jpg", "Are there any vehicles in this photo?"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(ROOT / "eval/results/results_photos.json"))
    args = ap.parse_args()

    import boto3

    from roadsense.agent import run
    from roadsense.decider_client import InProcessDecider

    decider = InProcessDecider(device=args.device)

    # The first CUDA inference costs ~4 s while kernels are selected; every one after
    # is ~160 ms. Pay it here, or the first photograph in the demo appears to be 25x
    # slower than the rest for a reason that has nothing to do with the photograph.
    decider.decide(
        "Task: warm-up.\nEvidence collected so far: none.",
        {"answer": "Stop and answer.", "detect_objects": "Run a detector."},
    )
    rekognition = boto3.client("rekognition", region_name="us-east-1")
    bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

    traces = []
    for name, question in PHOTOS:
        path = ROOT / "demo" / "photos" / name
        if not path.is_file():
            print(f"  missing {name}, skipping")
            continue
        print(f"  {name}: {question}", flush=True)
        result = run(question, str(path), decider=decider, rekognition=rekognition,
                     bedrock=bedrock, pricing={})
        trace = result.trace
        trace["case_id"] = pathlib.Path(name).stem
        traces.append(trace)
        for step in result.state.decisions:
            conf = "forced" if step.confidence is None else f"{step.confidence:.3f}"
            print(f"      step {step.step}: {step.action:24} conf={conf} {step.latency_ms:.0f}ms")
        print(f"      -> {result.outcome}: {result.answer[:90]}", flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"meta": {"source": "real photographs, Wikimedia Commons", "device": args.device},
         "results": [{"arm": "decider", "summary": {}, "scores": [], "traces": traces}]},
        indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out} ({len(traces)} traces)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
