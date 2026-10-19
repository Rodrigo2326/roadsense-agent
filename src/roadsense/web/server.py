"""A live view of the agent deciding.

Upload a photograph, ask a question, and watch the loop happen: the menu of allowed
actions, the probability Decider assigns to each, the evidence that comes back, and
the menu shrinking as tools get used up. The JSON trace says all of this already, but
nobody reads a JSON trace and *sees* a model change its mind.

The agent runs on a worker thread and pushes events into a queue; the request streams
them out as newline-delimited JSON as they happen. Nothing is buffered until the end,
because the whole point is watching a 94 ms decision land before the tool it chose has
even been called.

    python -m roadsense.web.server --host 0.0.0.0 --port 8080

The model loads once at startup (~2 minutes, mostly weights) and is reused for every
request. Bind to 0.0.0.0 only behind a security group that restricts the source: this
server has no authentication and will happily spend your Bedrock budget.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import queue
import tempfile
import threading
import time
from typing import Any

import boto3
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from ..agent import run as run_agent
from ..bedrock_client import DEFAULT_MODEL_ID
from ..decider_client import InProcessDecider

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SAMPLES = ROOT / "demo" / "images"

REGION = os.environ.get("AWS_REGION", "us-east-1")
CHECKPOINT = os.environ.get(
    "ROADSENSE_DECIDER_CHECKPOINT", "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"
)
DEVICE = os.environ.get("ROADSENSE_DECIDER_DEVICE", "cuda")
MODEL_ID = os.environ.get("ROADSENSE_BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)

#: Shown as clickable examples. Each is chosen to make a different thing visible.
SAMPLE_CAPTIONS = {
    "synthetic_sign_clear.jpg": "readable sign",
    "synthetic_sign_and_vehicles.jpg": "sign and vehicles",
    "synthetic_speed_limit.jpg": "speed limit",
    "synthetic_roadwork.jpg": "road work",
    "synthetic_sign_occluded.jpg": "sign half hidden",
    "synthetic_sign_partial.jpg": "degraded, partly readable",
    "synthetic_sign_garbled.jpg": "degraded, OCR goes wrong",
    "synthetic_sign_illegible.jpg": "degraded past reading",
    "synthetic_empty_road.jpg": "nothing to find",
}

app = FastAPI(title="RoadSense")

_state: dict[str, Any] = {"decider": None, "rekognition": None, "bedrock": None, "ready": False}
_lock = threading.Lock()


def warm_up() -> None:
    """Load the model and the AWS clients once, before the first request."""
    t0 = time.perf_counter()
    decider = InProcessDecider(CHECKPOINT, device=DEVICE)
    decider._engine_or_load()  # pay the weight load here, not on someone's first click

    # And pay the *first inference* here too. On CUDA the opening call runs ~4 s while
    # kernels are selected and compiled; every call after it is ~160 ms. Without this
    # the very first thing a viewer sees is the slowest thing the agent ever does.
    decider.decide(
        "Task: warm-up.\nQuestion: none.\nEvidence collected so far: none.",
        {"answer": "Stop and answer.", "detect_objects": "Run an object detector."},
    )
    _state.update(
        decider=decider,
        rekognition=boto3.client("rekognition", region_name=REGION),
        bedrock=boto3.client("bedrock-runtime", region_name=REGION),
        ready=True,
        load_seconds=round(time.perf_counter() - t0, 1),
    )
    print(f"[roadsense] model ready in {_state['load_seconds']}s on {DEVICE}", flush=True)


@app.on_event("startup")
def _startup() -> None:
    threading.Thread(target=warm_up, daemon=True).start()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (HERE / "ui.html").read_text(encoding="utf-8")


@app.get("/api/status")
def status() -> dict[str, Any]:
    return {
        "ready": _state["ready"],
        "device": DEVICE,
        "checkpoint": CHECKPOINT,
        "bedrock_model": MODEL_ID,
        "load_seconds": _state.get("load_seconds"),
        "samples": [
            {"name": n, "caption": c}
            for n, c in SAMPLE_CAPTIONS.items()
            if (SAMPLES / n).is_file()
        ],
    }


@app.get("/api/sample/{name}")
def sample(name: str) -> FileResponse:
    # Resolve and confine to the samples directory: the name arrives from the client.
    path = (SAMPLES / name).resolve()
    if not path.is_file() or SAMPLES.resolve() not in path.parents:
        return FileResponse(SAMPLES / "synthetic_sign_clear.jpg")
    return FileResponse(path)


def _resolve_image(sample_name: str | None, upload: UploadFile | None) -> tuple[str | None, Any]:
    """Return a path for the agent, plus a cleanup handle for an uploaded temp file."""
    if sample_name:
        path = (SAMPLES / sample_name).resolve()
        if path.is_file() and SAMPLES.resolve() in path.parents:
            return str(path), None
        return None, None
    if upload is not None and upload.filename:
        suffix = pathlib.Path(upload.filename).suffix.lower() or ".jpg"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(upload.file.read())
        tmp.close()
        return tmp.name, tmp.name
    return None, None


@app.post("/api/run")
async def api_run(
    question: str = Form(...),
    sample: str | None = Form(None),
    max_steps: int = Form(4),
    ask_sufficiency: bool = Form(True),
    image: UploadFile | None = File(None),
):
    if not _state["ready"]:
        return StreamingResponse(
            iter([json.dumps({"type": "error", "message": "model is still loading"}) + "\n"]),
            media_type="application/x-ndjson",
        )

    image_path, temp_path = _resolve_image(sample, image)
    events: queue.Queue = queue.Queue()

    def worker() -> None:
        try:
            # One engine, one GPU: serialise requests rather than racing them.
            with _lock:
                decider = _state["decider"]
                decider.ask_sufficiency = ask_sufficiency
                run_agent(
                    question,
                    image_path,
                    decider=decider,
                    rekognition=_state["rekognition"],
                    bedrock=_state["bedrock"],
                    max_steps=max_steps,
                    bedrock_model=MODEL_ID,
                    on_event=events.put,
                )
        except Exception as exc:  # noqa: BLE001 - surface it in the UI, do not 500
            events.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
            events.put(None)

    threading.Thread(target=worker, daemon=True).start()

    async def stream():
        loop = asyncio.get_running_loop()
        started = time.perf_counter()
        yield json.dumps({"type": "start", "question": question, "device": DEVICE}) + "\n"
        while True:
            event = await loop.run_in_executor(None, events.get)
            if event is None:
                break
            event["elapsed_ms"] = round((time.perf_counter() - started) * 1000)
            yield json.dumps(event, default=str) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
