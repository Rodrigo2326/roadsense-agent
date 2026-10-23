"""Lay the Polly narration onto the recorded screen capture.

The recorder stamps the real time of every narration beat against its own clock and
writes them to demo/video/timeline.json, so each clip is placed at a measured offset
rather than nudged into place by hand in an editor. The same file carries the pre-roll
-- the seconds the browser spent loading before the first beat -- which is trimmed off
the front so the finished video starts on the first word.

Before muxing, every segment's window is checked against the clip that has to fit in
it. A clip that overruns its window would talk over the next segment's screen, so that
is an error rather than something to discover on YouTube.

    python scripts/mux_demo.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import imageio_ffmpeg

ROOT = pathlib.Path(__file__).resolve().parents[1]
VIDEO_DIR = ROOT / "demo" / "video"
AUDIO_DIR = ROOT / "demo" / "audio"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", default=str(VIDEO_DIR / "roadsense-demo.webm"))
    ap.add_argument("--out", default=str(VIDEO_DIR / "roadsense-demo-voiced.webm"))
    args = ap.parse_args()

    timeline = json.loads((VIDEO_DIR / "timeline.json").read_text(encoding="utf-8"))
    clips = json.loads((AUDIO_DIR / "segments.json").read_text(encoding="utf-8"))
    preroll = timeline["preroll"]
    beats = {b["id"]: b["at"] for b in timeline["beats"]}
    order = [b["id"] for b in timeline["beats"]]
    length = {c["id"]: c["duration"] for c in clips["segments"]}

    # A clip must finish before the screen moves on to the next segment.
    print(f"{'seg':<5}{'at':>9}{'window':>9}{'clip':>8}   fit")
    problems = []
    for i, seg in enumerate(order[:-1]):
        if seg not in length:
            continue
        window = beats[order[i + 1]] - beats[seg]
        clip = length[seg]
        ok = window >= clip
        if not ok:
            problems.append((seg, window, clip))
        m, s = divmod(beats[seg], 60)
        print(f"{seg:<5}{int(m)}:{s:05.2f}{window:>8.1f}s{clip:>7.1f}s   "
              f"{'OK' if ok else 'OVERRUNS by %.1fs' % (clip - window)}")
    if problems:
        print("\nclips overrun their windows; re-record with the current segments.json first")
        return 1

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    present = [s for s in order if s in length and (AUDIO_DIR / f"{s}.mp3").is_file()]

    cmd = [ffmpeg, "-y", "-loglevel", "error", "-stats",
           "-ss", f"{preroll:.3f}", "-i", args.video]
    for seg in present:
        cmd += ["-i", str(AUDIO_DIR / f"{seg}.mp3")]

    # adelay slides each clip to its measured beat; normalize=0 on amix keeps each at
    # full level instead of dividing the volume by the number of inputs.
    parts = []
    for i, seg in enumerate(present, start=1):
        ms = round(beats[seg] * 1000)
        parts.append(f"[{i}:a]adelay={ms}|{ms}[a{i}]")
    mix = "".join(f"[a{i}]" for i in range(1, len(present) + 1))
    parts.append(f"{mix}amix=inputs={len(present)}:duration=longest:normalize=0[aout]")

    cmd += ["-filter_complex", ";".join(parts),
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "libvpx", "-b:v", "1400k", "-crf", "30",
            "-deadline", "good", "-cpu-used", "2",
            "-c:a", "libopus", "-b:a", "96k",
            args.out]

    print(f"\nmuxing {len(present)} clips onto {pathlib.Path(args.video).name} "
          f"(trimming {preroll:.2f}s of pre-roll)")
    proc = subprocess.run(cmd, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print("ffmpeg failed")
        return proc.returncode

    out = pathlib.Path(args.out)
    probe = subprocess.run([ffmpeg, "-i", str(out)], capture_output=True, text=True)
    for line in probe.stderr.splitlines():
        if "Duration" in line or "Stream" in line:
            print(" ", line.strip())
    print(f"\nwrote {out}  {out.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
