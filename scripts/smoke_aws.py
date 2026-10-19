"""Smoke test Fase 0: Bedrock + Rekognition reales. Sin mocks."""

import pathlib
import sys

import boto3

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from roadsense.tools import detect_objects, read_sign_text  # noqa: E402

REGION = "us-east-1"
MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def main() -> int:
    rek = boto3.client("rekognition", region_name=REGION)
    brt = boto3.client("bedrock-runtime", region_name=REGION)
    base = pathlib.Path(__file__).resolve().parents[1] / "demo" / "images"

    for name in ("synthetic_sign_clear.jpg", "synthetic_sign_blurry.jpg"):
        img = (base / name).read_bytes()
        print(f"\n=== {name} ({len(img)} bytes) ===")
        for ev in (detect_objects(rek, img), read_sign_text(rek, img)):
            q = "None" if ev.quality is None else f"{ev.quality:.3f}"
            print(f"  {ev.tool}: ok={ev.ok} quality={q} ({ev.duration_ms:.0f}ms)")
            print(f"    {ev.summary}")
            if ev.error:
                print(f"    ERROR: {ev.error}")

    r = brt.converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": "Reply with exactly: BEDROCK_OK"}]}],
        inferenceConfig={"maxTokens": 10, "temperature": 0.0},
    )
    print(f"\n=== bedrock {MODEL_ID} ===")
    print("  ", r["output"]["message"]["content"][0]["text"].strip(), "| usage:", r["usage"])
    print("\nSMOKE TEST AWS: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
