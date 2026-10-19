"""Smoke test Fase 0: una decision REAL de Strands Decider 2B.

Contrato verificado leyendo el codigo fuente del wheel strands-decider 0.1.0:
  strands_decider.infer.load_engine(checkpoint, *, device, use_prefix_cache, attn_implementation)
  SystemOneEngine.ask(state, questions) -> SystemOneResponse(model, answers, usage)
  ChoiceAnswer -> .choice (str), .probabilities (dict), .confidence (float calibrado)

No simula nada: si falla, el script sale con codigo != 0 y muestra el error real.
"""

import json
import os
import sys
import time

CHECKPOINT = os.environ.get(
    "ROADSENSE_DECIDER_CHECKPOINT", "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"
)
DEVICE = os.environ.get("ROADSENSE_DECIDER_DEVICE", "cpu")

# Las 4 acciones permitidas del brief (seccion 5). Decider elige entre estas y solo estas.
ACTIONS = {
    "detect_objects": "Run an object detector on the image to list the vehicles, people and other objects that are present.",
    "read_sign_text": "Run OCR on the image to read the text written on road signs.",
    "request_better_evidence": "Ask the user for a clearer image, because the current evidence is too poor to answer.",
    "answer": "Stop gathering evidence and write the final answer from the evidence already collected.",
}


def main() -> int:
    from strands_decider.infer import load_engine
    from strands_decider.schema import ChoiceQuestion, NoulQuestion

    print(f"[1/3] loading checkpoint={CHECKPOINT} device={DEVICE} ...", flush=True)
    t0 = time.time()
    engine = load_engine(CHECKPOINT, device=DEVICE)
    print(f"      loaded in {time.time() - t0:.1f}s", flush=True)

    # Estado estructurado. Decider NO ve la imagen, solo el estado textual (brief, seccion 4).
    state_step1 = (
        "Task: answer a question about a road scene photo using vision tools.\n"
        "Question: What vehicles appear and what text can be read on the sign?\n"
        "Image: provided, valid JPEG, 1920x1080.\n"
        "Evidence collected so far: none. No tool has been run yet.\n"
        "Steps used: 0 of 4."
    )

    cases = [
        ("step1_no_evidence", state_step1),
        (
            "step2_after_detection",
            "Task: answer a question about a road scene photo using vision tools.\n"
            "Question: What vehicles appear and what text can be read on the sign?\n"
            "Image: provided, valid JPEG, 1920x1080.\n"
            "Evidence collected so far:\n"
            "  - detect_objects already ran. It found: Car (confidence 0.98), "
            "Truck (confidence 0.91), Traffic Sign (confidence 0.88).\n"
            "  - No OCR has been run, so the text on the sign is still unknown.\n"
            "Steps used: 1 of 4.",
        ),
        (
            "step3_blurry_ocr_failed",
            "Task: answer a question about a road scene photo using vision tools.\n"
            "Question: What text can be read on the sign?\n"
            "Image: provided, but heavily motion-blurred and low resolution.\n"
            "Evidence collected so far:\n"
            "  - read_sign_text already ran. It returned no readable text; the highest "
            "OCR confidence was 0.11, below the usable threshold.\n"
            "  - detect_objects already ran. It found Traffic Sign (confidence 0.62).\n"
            "Steps used: 2 of 4.",
        ),
    ]

    for name, state in cases:
        print(f"\n[2/3] decision: {name}", flush=True)
        t0 = time.time()
        resp = engine.ask(
            state=state,
            questions={
                "next_action": ChoiceQuestion(
                    instructions=(
                        "Which single action should the agent take next to answer the "
                        "question with evidence it can actually verify?"
                    ),
                    criteria=ACTIONS,
                ),
                "evidence_sufficient": NoulQuestion(
                    instructions=(
                        "Is the evidence collected so far already sufficient to answer "
                        "the question without guessing?"
                    ),
                ),
            },
        )
        dt = (time.time() - t0) * 1000
        a = resp.answers["next_action"]
        n = resp.answers["evidence_sufficient"]
        print(f"      model={resp.model}  latency={dt:.0f}ms  usage={resp.usage.model_dump()}")
        print(f"      choice={a.choice!r}  confidence={a.confidence:.3f}")
        probs = json.dumps({k: round(v, 4) for k, v in a.probabilities.items()})
        print(f"      probabilities={probs}")
        print(f"      evidence_sufficient P(true)={n.noul:.3f}")

    print("\n[3/3] SMOKE TEST DECIDER: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
