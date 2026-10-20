"""Fetch real road photographs from Wikimedia Commons, with their licences.

The benchmark deliberately uses synthetic images: the degradation is a measured dial
rather than an accident, which is what makes the hard cases reproducible. But nobody
believes a vision agent because it read clip-art, so the demo also needs photographs
that a person took of a real road.

Pulling images off a web search would be the fast way and the wrong one. A hackathon
submission that ships unlicensed images is a submission that can be disqualified, and
"I found it on the internet" is not a licence. So this script talks to the Commons
API, reads each file's licence out of its metadata, and **refuses anything it cannot
attribute**. What it writes alongside the images is the attribution record.

    python scripts/fetch_commons_assets.py
    python scripts/fetch_commons_assets.py --list     # show candidates, download nothing

Every file keeps its own licence -- CC0, CC BY or CC BY-SA, recorded per image in
demo/photos/ATTRIBUTION.md. That is separate from the Apache-2.0 on the code.
"""

from __future__ import annotations

import argparse
import html
import json
import pathlib
import re
import sys
import urllib.parse
import urllib.request

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "RoadSense/0.1 (hackathon demo; https://github.com/Rodrigo2326/roadsense-agent)"}
OUT = pathlib.Path(__file__).resolve().parents[1] / "demo" / "photos"

#: Licences we will ship. Anything else is skipped rather than guessed at.
ALLOWED = ("cc0", "public domain", "cc by", "cc-by")
#: Preference order: the fewer obligations a licence carries, the better.
def licence_rank(name: str) -> int:
    low = name.lower()
    if "cc0" in low or "public domain" in low:
        return 0
    if "sa" in low:
        return 2
    return 1


#: Curated by hand rather than searched blind. Searching gives you whatever the index
#: ranks today, which is neither reproducible nor safe: these were each opened and
#: checked for readable number plates and identifiable faces before being listed.
#: Licences lean CC BY and CC0 on purpose -- share-alike is honoured where it appears,
#: but fewer obligations is a better default for an asset someone else may reuse.
#: Files whose text must be blurred before use. A street photograph of real traffic
#: carries real number plates, and "I did not notice them" is not a privacy policy --
#: Rekognition read MP 7866 and GF 4205 off this one on the first try.
REDACT_TEXT = {"street-traffic"}

WANTED = [
    ("File:Red octagonal stop sign.jpg",
     "stop-sign", "a real stop sign"),
    ("File:10 US Highway speed limit sign 55 MPH - Garfield County, Colorado.jpg",
     "speed-limit", "a real speed limit sign"),
    ("File:Intersection of 119th Street and Jasper Road at Canfield, unincorporated Boulder County, Colorado, USA - 2025.jpg",
     "intersection", "a real rural intersection"),
    ("File:Street scene Causeway Bay HK. (16065463789).jpg",
     "street-traffic", "a real street with vehicles"),
]


def api(**params) -> dict:
    params.update(format="json", formatversion="2")
    url = API + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=45) as r:
        return json.loads(r.read())


def strip_html(value: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", value or "")).strip()


def say(line: str) -> None:
    """Print without dying on a name the terminal's codepage cannot encode.

    Author names are whatever the photographer typed; the file keeps the real one.
    """
    enc = sys.stdout.encoding or "utf-8"
    print(line.encode(enc, errors="replace").decode(enc))


def describe(titles: list[str], width: int) -> list[dict]:
    """Licence, author and a scaled URL for each file. Unattributable files drop out."""
    if not titles:
        return []
    r = api(action="query", titles="|".join(titles), prop="imageinfo",
            iiprop="url|extmetadata|size|mime", iiurlwidth=str(width))
    out = []
    for page in r.get("query", {}).get("pages", []):
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata", {})
        licence = strip_html(meta.get("LicenseShortName", {}).get("value", ""))
        artist = strip_html(meta.get("Artist", {}).get("value", ""))
        if not licence or not artist:
            continue  # cannot attribute it, so we do not take it
        if not any(tag in licence.lower() for tag in ALLOWED):
            continue
        if info.get("mime") not in ("image/jpeg", "image/png"):
            continue
        out.append({
            "title": page["title"],
            "licence": licence,
            "artist": artist,
            "source": "https://commons.wikimedia.org/wiki/" + urllib.parse.quote(page["title"].replace(" ", "_")),
            "url": info.get("thumburl") or info.get("url"),
            "width": info.get("thumbwidth") or info.get("width"),
        })
    out.sort(key=lambda d: (licence_rank(d["licence"]), -(d["width"] or 0)))
    return out


def redact_text(path: pathlib.Path, region: str) -> int:
    """Blur every text region Rekognition can find, and verify none survives.

    Blurring only what *looks* like a plate means trusting a guess about which
    rectangle is sensitive. Blurring every detected word and then re-reading the
    file turns the claim into something checkable: if DetectText comes back empty,
    the photograph no longer carries legible text of any kind.
    """
    import boto3
    from PIL import Image, ImageFilter

    client = boto3.client("rekognition", region_name=region)
    image = Image.open(path).convert("RGB")
    width, height = image.size
    found = client.detect_text(Image={"Bytes": path.read_bytes()})
    words = [d for d in found["TextDetections"] if d["Type"] == "WORD"]

    for word in words:
        box = word["Geometry"]["BoundingBox"]
        pad = 0.004
        x0 = max(0, int((box["Left"] - pad) * width))
        y0 = max(0, int((box["Top"] - pad) * height))
        x1 = min(width, int((box["Left"] + box["Width"] + pad) * width))
        y1 = min(height, int((box["Top"] + box["Height"] + pad) * height))
        if x1 <= x0 or y1 <= y0:
            continue
        patch = image.crop((x0, y0, x1, y1))
        image.paste(patch.filter(ImageFilter.GaussianBlur(max(3, (x1 - x0) // 4))), (x0, y0))

    image.save(path, "JPEG", quality=88)
    left = [d for d in client.detect_text(Image={"Bytes": path.read_bytes()})["TextDetections"]
            if d["Type"] == "WORD"]
    if left:
        raise SystemExit(f"redaction failed on {path.name}: still readable {[d['DetectedText'] for d in left]}")
    return len(words)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--list", action="store_true", help="show what would be fetched")
    ap.add_argument("--region", default="us-east-1", help="region for the redaction check")
    args = ap.parse_args()

    found = describe([title for title, _, _ in WANTED], args.width)
    by_title = {d["title"]: d for d in found}
    chosen: list[dict] = []
    for title, slug, role in WANTED:
        item = by_title.get(title)
        if item is None:
            say(f"  SKIPPED (no readable licence): {title[:62]}")
            continue
        item["slug"], item["role"] = slug, role
        chosen.append(item)
        say(f"  {item['licence']:16} {slug:15} {item['artist'][:34]:34} {item['title'][:44]}")

    if args.list or not chosen:
        print(f"\n{len(chosen)} candidates (nothing downloaded)")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Photograph attribution",
        "",
        "These photographs are **not** ours. Each is used under its own licence, listed",
        "below, and that licence is separate from the Apache-2.0 covering this repository's",
        "code. Fetched by `scripts/fetch_commons_assets.py`, which refuses any file whose",
        "licence and author it cannot read from the Wikimedia Commons API.",
        "",
        "The benchmark does not use these: it runs on the synthetic set, whose degradation",
        "is a measured dial rather than an accident. These are here so the demo shows the",
        "agent working on photographs a person actually took.",
        "",
        "| File | Role | Licence | Author | Source |",
        "|---|---|---|---|---|",
    ]
    for item in chosen:
        ext = ".jpg" if item["url"].lower().split("?")[0].endswith((".jpg", ".jpeg")) else ".png"
        name = f"{item['slug']}{ext}"
        req = urllib.request.Request(item["url"], headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            (OUT / name).write_bytes(r.read())
        note = ""
        if item["slug"] in REDACT_TEXT:
            blurred = redact_text(OUT / name, args.region)
            note = f" (text redacted: {blurred} regions blurred)"
        kb = (OUT / name).stat().st_size / 1024
        say(f"  wrote {name} ({kb:.0f} KB){note}")
        lines.append(
            f"| `{name}` | {item['role']}{note} | {item['licence']} | {item['artist']} | "
            f"[Commons]({item['source']}) |"
        )

    lines += [
        "",
        "## What was changed, and what was checked",
        "",
        "`street-traffic.jpg` is a photograph of real traffic, so it carried real number",
        "plates: Rekognition read `MP 7866` and `GF 4205` off it on the first try. Every",
        "text region it could find has been blurred, and the script re-reads the file",
        "afterwards and fails if any word is still detectable. The copy published here is",
        "therefore strictly less identifying than the original, which remains available",
        "under its own licence at the source link.",
        "",
        "The other three carry no plates and no people. Faces are not resolvable in any of",
        "these at the published resolution.",
    ]
    (OUT / "ATTRIBUTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT / 'ATTRIBUTION.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
