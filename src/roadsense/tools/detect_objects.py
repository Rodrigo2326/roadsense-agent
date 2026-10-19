"""detect_objects -- Amazon Rekognition DetectLabels.

The confidence we hand back up is Rekognition's own, rescaled from its 0..100 to the
0..1 the rest of the agent speaks. We do not blend it, floor it, or substitute a
guess when it is missing: a tool that reports nothing reports ``None``.
"""

from __future__ import annotations

import time
from typing import Any

from ..state import Evidence

#: Rekognition returns a long tail of low-confidence labels. Below this they are
#: noise for our purpose, so they are dropped from ``items`` -- but the raw response
#: is still kept in ``Evidence.raw`` so the trace stays auditable.
MIN_CONFIDENCE = 60.0
MAX_LABELS = 15


def detect_objects(client: Any, image_bytes: bytes) -> Evidence:
    t0 = time.perf_counter()
    try:
        resp = client.detect_labels(
            Image={"Bytes": image_bytes},
            MaxLabels=MAX_LABELS,
            MinConfidence=MIN_CONFIDENCE,
        )
    except Exception as exc:  # noqa: BLE001 - tool boundary: never kill the loop
        return Evidence(
            tool="detect_objects",
            ok=False,
            summary="The object detector failed and returned no objects.",
            error=f"{type(exc).__name__}: {exc}",
            duration_ms=(time.perf_counter() - t0) * 1000,
        )

    dt = (time.perf_counter() - t0) * 1000
    labels = resp.get("Labels", [])
    items = [
        {
            "name": lb["Name"],
            "confidence": round(lb["Confidence"] / 100.0, 4),
            "instances": len(lb.get("Instances", [])),
        }
        for lb in labels
    ]

    if not items:
        return Evidence(
            tool="detect_objects",
            ok=True,
            summary="The object detector ran but recognised no objects above its confidence threshold.",
            items=[],
            quality=None,
            duration_ms=dt,
            raw=resp,
        )

    named = ", ".join(f"{i['name']} ({i['confidence']:.2f})" for i in items[:8])
    return Evidence(
        tool="detect_objects",
        ok=True,
        summary=f"Objects detected: {named}.",
        items=items,
        quality=max(i["confidence"] for i in items),
        duration_ms=dt,
        raw=resp,
    )
