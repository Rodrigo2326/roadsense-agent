"""Synthesise the demo narration with Amazon Polly and measure what it actually runs to.

The script in Documentacion/GUION_CRONOMETRADO.md carries an *intended* duration per
segment, written by ear. What matters for assembling the video is the duration Polly
actually produces, which is never quite the same. So every clip is synthesised, then
measured, and the measurements are written out for the recorder to size its windows
against -- audio first, screen second, which is the only order that syncs without
hand-nudging clips in an editor afterwards.

    python scripts/make_voiceover.py
    python scripts/make_voiceover.py --voice Ruth

Writes demo/audio/<id>.mp3 and demo/audio/segments.json.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess

import boto3

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "demo" / "audio"

# Keyed to the segment letters in GUION_CRONOMETRADO.md. The gap between B and C is
# the animation, which is deliberately silent: it is the moment the whole demo exists
# to show, and narrating over it buys nothing.
SEGMENTS: list[tuple[str, str]] = [
    ("A", "This is RoadSense. Ask it a question about a road photograph and it does not "
          "guess from a single look. It has four actions, and it picks one at a time. A "
          "two billion parameter model called Strands Decider 2B makes every one of those "
          "choices, without writing a single word."),
    ("B", "A real highway. Watch it decide."),
    ("C", "Step one. Decider scores all four actions and picks the detector, but at only "
          "zero point seven seven confidence. It is saying both openings are reasonable, "
          "because they are. The tool reports back: a road, a freeway, cars. And step two "
          "is zero point nine eight on reading the sign. The decision changed because the "
          "evidence changed. The menu is shorter too: the detector already ran, so it is "
          "withdrawn."),
    ("D", "Same sign, degraded past reading. O C R comes back with nothing."),
    ("E", "It does not invent the word. The menu is down to two, and it answers: zero point "
          "eight six, against zero point one four for asking you for a better photograph. "
          "But read the line underneath. Confidence is zero point seven two, and the model "
          "puts the chance that this evidence is enough at zero point two seven. It answers, "
          "and it tells you the ground is thin. Amazon Bedrock writes that answer, and never "
          "sees the image."),
    ("F", "Now, no image at all."),
    ("G", "The menu starts with only two options. Both vision tools were withdrawn before "
          "the model saw them. That is the reusable idea: do not ask a model to respect a "
          "rule you can enforce by not offering the option."),
    ("H", "Thirty seven cases, seven selectors, one loop. Haiku, Sonnet and Opus all land "
          "on eighty nine point two percent. The same number, across an order of magnitude "
          "in price. Scaling buys nothing. Decider does not win every column, and a keyword "
          "rule picks tools better. But that rule cannot notice the file is missing. Decider "
          "gets five out of five on invalid inputs, at ninety four milliseconds a decision, "
          "with zero billable tokens."),
    ("I", "That is Evidence Seeking Tool Selection, running on Strands Decider 2B. The demo "
          "you just watched is live, the code is on GitHub under Apache two point oh, and "
          "every number in it came from a recorded run. Thanks for watching."),
]

DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)")


def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def duration_of(path: pathlib.Path, ffmpeg: str) -> float:
    """Read a clip's length off ffmpeg's own report rather than trusting the script."""
    proc = subprocess.run([ffmpeg, "-i", str(path)], capture_output=True, text=True)
    m = DURATION_RE.search(proc.stderr)
    if not m:
        raise RuntimeError(f"could not read duration of {path}:\n{proc.stderr[-500:]}")
    h, mnt, s = m.groups()
    return int(h) * 3600 + int(mnt) * 60 + float(s)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--voice", default="Matthew", help="Polly voice id")
    ap.add_argument("--engine", default="generative")
    ap.add_argument("--region", default="us-east-1")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    polly = boto3.client("polly", region_name=args.region)
    ffmpeg = ffmpeg_exe()

    rows, chars = [], 0
    for seg_id, text in SEGMENTS:
        resp = polly.synthesize_speech(
            Text=text, VoiceId=args.voice, Engine=args.engine, OutputFormat="mp3",
        )
        path = OUT / f"{seg_id}.mp3"
        path.write_bytes(resp["AudioStream"].read())
        chars += len(text)
        secs = duration_of(path, ffmpeg)
        rows.append({"id": seg_id, "file": path.name, "duration": round(secs, 2),
                     "chars": len(text), "text": text})
        print(f"  {seg_id}  {secs:6.2f}s  {len(text):4d} chars  -> {path.name}")

    manifest = {"voice": args.voice, "engine": args.engine, "segments": rows}
    (OUT / "segments.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    total = sum(r["duration"] for r in rows)
    print(f"\n{len(rows)} clips, {total:.1f}s of speech, {chars} characters billed")
    print(f"wrote {OUT / 'segments.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
