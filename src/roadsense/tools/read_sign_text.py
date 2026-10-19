"""read_sign_text -- Amazon Rekognition DetectText.

DetectText, not Textract: Textract is built for documents, this is text embedded in
a photographed scene.

The quality signal here is the one the agent leans on hardest, because it is what
separates "the sign says STOP" from "there is a sign and I cannot read it". We report
the *highest* word-level confidence Rekognition returned. If that number is low, the
state will say so in plain words and Decider gets to act on it -- that is the whole
blurry-image case.
"""

from __future__ import annotations

import time
from typing import Any

from ..state import Evidence

#: Rekognition's floor for returning a detection at all.
MIN_CONFIDENCE = 50.0
#: Below this, we still report the text but flag it as not confirmable. Chosen to sit
#: well above Rekognition's floor; it is a reporting threshold, not a filter.
LEGIBLE_THRESHOLD = 0.80


def read_sign_text(client: Any, image_bytes: bytes) -> Evidence:
    t0 = time.perf_counter()
    try:
        resp = client.detect_text(Image={"Bytes": image_bytes})
    except Exception as exc:  # noqa: BLE001 - tool boundary
        return Evidence(
            tool="read_sign_text",
            ok=False,
            summary="The OCR tool failed and returned no text.",
            error=f"{type(exc).__name__}: {exc}",
            duration_ms=(time.perf_counter() - t0) * 1000,
        )

    dt = (time.perf_counter() - t0) * 1000
    detections = resp.get("TextDetections", [])

    # WORD detections carry per-word confidence and a ``ParentId`` pointing at the
    # LINE they belong to. The human-readable reading is rebuilt from the words we
    # kept, grouped back into lines -- never from the raw LINE entries.
    #
    # Reading the LINE text directly is the obvious shortcut and it is wrong: a line
    # can clear the confidence floor while one of its words does not, so the summary
    # ends up quoting text that is in no structured item. The writer then repeats it
    # as evidence. That exact bug shipped here and was caught by the benchmark's
    # fidelity check on case rs-001, where the summary read 'STOP | AHEAD | the'
    # while the items held only STOP and AHEAD.
    kept = [
        d
        for d in detections
        if d.get("Type") == "WORD" and d.get("Confidence", 0) >= MIN_CONFIDENCE
    ]

    items = [
        {
            "text": d["DetectedText"],
            "confidence": round(d["Confidence"] / 100.0, 4),
            "line_id": d.get("ParentId"),
            "box": d.get("Geometry", {}).get("BoundingBox"),
        }
        for d in kept
    ]

    if not items:
        return Evidence(
            tool="read_sign_text",
            ok=True,
            summary="OCR ran but found no readable text in the image.",
            items=[],
            quality=None,
            duration_ms=dt,
            raw=resp,
        )

    best = max(i["confidence"] for i in items)

    grouped: dict[Any, list[str]] = {}
    for item in items:
        grouped.setdefault(item["line_id"], []).append(item["text"])
    reading = " | ".join(" ".join(words) for words in grouped.values())
    if best < LEGIBLE_THRESHOLD:
        summary = (
            f"OCR returned low-confidence text that cannot be confirmed: {reading!r}. "
            f"Highest word confidence was only {best:.2f}."
        )
    else:
        summary = f"OCR read the text: {reading!r}."

    return Evidence(
        tool="read_sign_text",
        ok=True,
        summary=summary,
        items=items,
        quality=best,
        duration_ms=dt,
        raw=resp,
    )
