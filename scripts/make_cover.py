"""Draw the article cover: the menu shrinking, as actual probabilities.

Builder Center asks for 1200x675 and advises against text in the image, which rules
out a screenshot of the page.

The first version was bars on a plain ground, and it read as a skeleton loader --
grey rounded rectangles on light grey is exactly what a site shows while content is
still arriving, so in a feed it looks like a broken image. The fix is to put the
photograph underneath: now the picture says "a real road scene, and the machine's
decision about it", which is the project, and nothing about it looks unfinished.

The bars are read out of a real trace rather than drawn by eye, so the picture is the
data. Withdrawn actions keep their slot as an outline, because the shrinking menu is
the point and it is invisible if the rows simply vanish.

    python scripts/make_cover.py --case speed-limit
"""

from __future__ import annotations

import argparse
import json
import pathlib

from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).resolve().parents[1]

W, H = 1200, 675
GROUND = (241, 243, 246)
INK = (20, 23, 28)
CHOSEN = (86, 104, 221)      # validated against the dataviz palette checker
DIM = (191, 199, 216)
TRACK = (227, 231, 238)
OUTLINE = (209, 215, 226)

ACTION_ORDER = ["detect_objects", "read_sign_text", "request_better_evidence", "answer"]


def load_steps(results: pathlib.Path, case_id: str) -> list[dict]:
    data = json.loads(results.read_text(encoding="utf-8"))
    decider = next(r for r in data["results"] if r["arm"] == "decider")
    trace = next(t for t in decider["traces"] if t["case_id"] == case_id)
    return trace["steps"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(ROOT / "eval/results/results_photos.json"))
    ap.add_argument("--case", default="speed-limit")
    ap.add_argument("--out", default=str(ROOT / "demo/cover.jpg"))
    args = ap.parse_args()

    steps = load_steps(pathlib.Path(args.results), args.case)[:3]

    photo_path = ROOT / "demo" / "photos" / f"{args.case}.jpg"
    if photo_path.is_file():
        photo = Image.open(photo_path).convert("RGB")
        # Cover-crop to 16:9 rather than squashing the road out of proportion.
        target = W / H
        pw, ph = photo.size
        if pw / ph > target:
            new_w = int(ph * target)
            photo = photo.crop(((pw - new_w) // 2, 0, (pw + new_w) // 2, ph))
        else:
            new_h = int(pw / target)
            photo = photo.crop((0, (ph - new_h) // 2, pw, (ph + new_h) // 2))
        img = photo.resize((W, H), Image.LANCZOS)
        # Darken, so the panel on top keeps its contrast wherever the sky is bright.
        img = Image.blend(img, Image.new("RGB", (W, H), (14, 17, 24)), 0.52)
    else:
        img = Image.new("RGB", (W, H), GROUND)

    d = ImageDraw.Draw(img, "RGBA")

    # One translucent card holds the decision, so the photograph reads as background
    # and the data reads as an instrument laid over it.
    card = [64, 86, W - 64, H - 86]
    d.rounded_rectangle(card, radius=18, fill=(244, 246, 250, 238))

    bar_h, gap = 40, 26
    block_h = 4 * bar_h + 3 * gap
    top = card[1] + (card[3] - card[1] - block_h) // 2 - 16

    pad_x, col_gap = card[0] + 46, 56
    n = len(steps)
    bar_w = (card[2] - card[0] - 92 - col_gap * (n - 1)) // n

    for i, step in enumerate(steps):
        x = pad_x + i * (bar_w + col_gap)
        offered = set(step["offered_actions"])
        probs = step["probabilities"]

        for j, action in enumerate(ACTION_ORDER):
            y = top + j * (bar_h + gap)
            if action not in offered:
                d.rounded_rectangle([x, y, x + bar_w, y + bar_h], radius=6,
                                    outline=OUTLINE, width=2)
                continue
            d.rounded_rectangle([x, y, x + bar_w, y + bar_h], radius=6, fill=TRACK)
            p = probs.get(action, 0.0)
            fill_w = max(8, int(bar_w * p))
            colour = CHOSEN if action == step["action"] else DIM
            d.rounded_rectangle([x, y, x + fill_w, y + bar_h], radius=6, fill=colour)

    dot_y = top + block_h + 42
    for i in range(n):
        cx = pad_x + i * (bar_w + col_gap) + 18
        d.ellipse([cx - 8, dot_y - 8, cx + 8, dot_y + 8], fill=CHOSEN)
        if i < n - 1:
            d.rounded_rectangle([cx + 22, dot_y - 2, cx + bar_w + col_gap - 22, dot_y + 2],
                                radius=2, fill=OUTLINE)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, "JPEG", quality=88, optimize=True)
    kb = out.stat().st_size / 1024
    print(f"wrote {out}  {img.size[0]}x{img.size[1]}  {kb:.0f} KB")
    if kb > 2048:
        print("  WARNING: over the 2 MB limit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
