"""Where the time actually goes in a Decider 2B decision.

The published headline for this model is ~115 ms (RTX 3090) and ~153 ms (M3
MacBook). RoadSense measures ~3.2 s on an EC2 c7i.2xlarge. Before blaming the
deployment it is worth reading what the package itself says, in
``strands_decider/infer.py``:

    "on an M3 Pro a 256-token v19 question takes 7.7 s in bf16 and 3.5 s in fp32"

So 3.5 s on Apple Silicon is the *package authors'* own PyTorch-CPU number. Our 3.2 s
is in line with it. The 153 ms figure almost certainly comes from the MLX backend the
repo ships for Apple Silicon, not from PyTorch — and the 115 ms figure is a GPU.

This script measures the three things that actually move the number on CPU, so the
benchmark can report a latency that is honest about its hardware class rather than
implying the model is slow:

1. **Thread count.** torch defaults to half the vCPUs on this box.
2. **State size.** The published numbers are for "small tasks"; RoadSense states run
   300-480 tokens.
3. **Marginal cost of a second question.** The model's headline property is that the
   state is encoded once and its KV cache is broadcast across the question batch, so
   question two should be nearly free. That is a claim worth checking, not repeating.

Run:  python scripts/profile_latency.py [--threads 4 8] [--repeats 3]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time

CHECKPOINT = os.environ.get(
    "ROADSENSE_DECIDER_CHECKPOINT", "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"
)
DEVICE = os.environ.get("ROADSENSE_DECIDER_DEVICE", "cpu")

ACTIONS = {
    "detect_objects": "Run an object detector on the image to list the vehicles, people, signs and other objects present.",
    "read_sign_text": "Run optical character recognition on the image to read the text written on road signs.",
    "request_better_evidence": "Stop and ask for a clearer image, because the evidence gathered so far is too poor to answer honestly.",
    "answer": "Stop gathering evidence and write the final answer using only the evidence already collected.",
}

SHORT = "Question: What does the sign say?\nEvidence: none yet.\nSteps used: 0 of 4."

MEDIUM = (
    "Task: answer a question about a road scene photograph using vision tools.\n"
    "Question: What vehicles appear and what text can be read on the sign?\n"
    "Image: provided and readable.\n"
    "Evidence collected so far: none. No tool has been run yet.\n"
    "Tools not yet run: detect_objects, read_sign_text.\n"
    "Steps used: 0 of 4."
)

LONG = MEDIUM + "\n" + (
    "Evidence collected so far:\n"
    "  - detect_objects ran successfully. Objects detected: Road (1.00), Sign (1.00), "
    "Symbol (1.00), Bus Stop (0.96), Outdoors (0.96), Road Sign (0.85), City (0.78), "
    "Text (0.69). (tool confidence: 1.00)\n"
    "  - read_sign_text ran successfully. OCR read the text: 'STOP | AHEAD'. "
    "(tool confidence: 1.00)\n"
) * 3

STATES = {"short": SHORT, "medium": MEDIUM, "long": LONG}


def time_calls(engine, state: str, questions: dict, repeats: int) -> list[float]:
    engine.ask(state=state, questions=questions)  # warm-up, not measured
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        engine.ask(state=state, questions=questions)
        out.append((time.perf_counter() - t0) * 1000)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threads", nargs="+", type=int, default=[4, 8])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="eval/results/latency_profile.json")
    args = ap.parse_args()

    import torch
    from strands_decider.infer import load_engine
    from strands_decider.schema import ChoiceQuestion, NoulQuestion

    one = {"next_action": ChoiceQuestion(instructions="Which action next?", criteria=ACTIONS)}
    two = {
        **one,
        "evidence_sufficient": NoulQuestion(
            instructions="Is the evidence collected so far already sufficient?"
        ),
    }

    env = {
        "checkpoint": CHECKPOINT,
        "device": DEVICE,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
    }
    print(json.dumps(env, indent=2), flush=True)

    t0 = time.perf_counter()
    engine = load_engine(CHECKPOINT, device=DEVICE)
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"\nmodel load: {load_ms / 1000:.1f}s", flush=True)

    torso_dtype = str(next(engine.model.torso.parameters()).dtype)
    print(f"torso dtype on {DEVICE}: {torso_dtype}", flush=True)

    results = []
    for threads in args.threads:
        torch.set_num_threads(threads)
        print(f"\n=== torch threads = {threads} (of {os.cpu_count()} vCPU) ===", flush=True)
        for name, state in STATES.items():
            for label, qs in (("1 question", one), ("2 questions", two)):
                times = time_calls(engine, state, qs, args.repeats)
                med = statistics.median(times)
                results.append(
                    {
                        "threads": threads,
                        "state": name,
                        "state_chars": len(state),
                        "questions": label,
                        "median_ms": round(med, 1),
                        "all_ms": [round(t, 1) for t in times],
                    }
                )
                print(f"  {name:7} ({len(state):4} chars)  {label:12}  median {med:8.0f} ms", flush=True)

    best = min(results, key=lambda r: r["median_ms"])
    worst = max(results, key=lambda r: r["median_ms"])
    print(f"\nfastest: {best['median_ms']:.0f} ms  ({best['threads']} threads, "
          f"{best['state']} state, {best['questions']})")
    print(f"slowest: {worst['median_ms']:.0f} ms  ({worst['threads']} threads, "
          f"{worst['state']} state, {worst['questions']})")

    payload = {"env": env, "load_ms": round(load_ms, 1), "torso_dtype": torso_dtype,
               "results": results}
    import pathlib
    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
