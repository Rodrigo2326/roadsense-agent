"""Generate the benchmark image set, deterministically.

Synthetic rather than photographed, for three reasons: the licence is unambiguous
(we drew them), nobody's face or number plate is in them, and the degradation is a
dial rather than an accident -- which is what makes the hard cases reproducible.

The degradation levels are not guesses. They come from a sweep against the live
Rekognition DetectText API:

    effective width 95 px -> ['STOP', 'AHEAD'] at 0.986
                    88 px -> ['STOP']          at 0.985   <- partial: AHEAD is lost
                    72 px -> ['STOP', 'I']     at 0.973
                    65 px -> ['I']             at 0.854   <- garbled: confidently wrong
                    45 px -> no text at all                <- illegible

Worth recording what that sweep did *not* find: with high-contrast block text there
is no level that yields a genuinely low-confidence *correct* reading. Rekognition's
confidence here is bimodal -- it reads the word at ~0.98, or it reads something else
at ~0.85, or it returns nothing. A confidence threshold alone therefore cannot
separate "right" from "wrong" on this font; the garbled case sails past a 0.80 floor.
That is a real limitation of confidence-based filtering and the benchmark reports it
rather than hiding it behind a tuned constant.

Run:  python scripts/make_assets.py
"""

from __future__ import annotations

import pathlib

from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = pathlib.Path(__file__).resolve().parents[1] / "demo" / "images"

SKY = (135, 180, 220)
ROAD = (90, 90, 95)
VERGE = (110, 140, 90)


def _font(size: int):
    for name in ("arialbd.ttf", "DejaVuSans-Bold.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _base(night: bool = False) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    sky = (25, 30, 55) if night else SKY
    img = Image.new("RGB", (900, 600), sky)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 400, 900, 600], fill=(45, 45, 50) if night else ROAD)
    d.rectangle([0, 360, 900, 400], fill=(45, 60, 40) if night else VERGE)
    for x in range(40, 900, 160):
        d.rectangle([x, 492, x + 80, 504], fill=(200, 200, 200) if night else (240, 240, 240))
    return img, d


def _sign(d: ImageDraw.ImageDraw, lines: list[tuple[str, int, tuple[int, int, int]]],
          box=(520, 120, 700, 255)) -> None:
    x0, y0, x1, y1 = box
    d.rectangle([(x0 + x1) // 2 - 6, y1 - 5, (x0 + x1) // 2 + 6, 420], fill=(120, 120, 120))
    d.rectangle([x0, y0, x1, y1], fill=(255, 255, 255), outline=(20, 20, 20), width=6)
    y = y0 + 22
    for text, size, colour in lines:
        d.text((x0 + 22, y), text, fill=colour, font=_font(size))
        y += size + 6


def _vehicle(d: ImageDraw.ImageDraw, x: int, y: int, w: int, colour) -> None:
    """A crude car silhouette. Rekognition may or may not label it; the benchmark's
    ground truth is driven by the question asked, not by what is drawn."""
    h = int(w * 0.42)
    d.rounded_rectangle([x, y, x + w, y + h], radius=8, fill=colour)
    d.rounded_rectangle([x + w * 0.22, y - h * 0.5, x + w * 0.74, y + 4], radius=6, fill=colour)
    for cx in (x + w * 0.24, x + w * 0.76):
        d.ellipse([cx - 13, y + h - 11, cx + 13, y + h + 15], fill=(25, 25, 25))


def degrade(img: Image.Image, width_px: int, blur: float, quality: int) -> Image.Image:
    h = int(img.height * width_px / img.width)
    small = img.resize((width_px, h), Image.BILINEAR).resize(img.size, Image.BILINEAR)
    out = small.filter(ImageFilter.GaussianBlur(blur))
    buf = OUT / "_tmp.jpg"
    out.save(buf, "JPEG", quality=quality)
    result = Image.open(buf).copy()
    buf.unlink()
    return result


def build() -> list[str]:
    OUT.mkdir(parents=True, exist_ok=True)
    made: list[str] = []

    def save(img: Image.Image, name: str, quality: int = 92) -> None:
        img.save(OUT / name, "JPEG", quality=quality)
        made.append(name)

    # 1. Clear STOP AHEAD sign, no vehicles.
    img, d = _base()
    _sign(d, [("STOP", 54, (190, 20, 20)), ("AHEAD", 34, (20, 20, 20))])
    clear = img.copy()
    save(img, "synthetic_sign_clear.jpg")

    # 2. Same sign plus vehicles.
    img, d = _base()
    _sign(d, [("STOP", 54, (190, 20, 20)), ("AHEAD", 34, (20, 20, 20))])
    _vehicle(d, 90, 455, 190, (40, 90, 170))
    _vehicle(d, 320, 470, 150, (190, 60, 50))
    save(img, "synthetic_sign_and_vehicles.jpg")

    # 3. Vehicles, no sign at all.
    img, d = _base()
    _vehicle(d, 120, 450, 210, (60, 140, 90))
    _vehicle(d, 420, 468, 160, (200, 170, 60))
    save(img, "synthetic_vehicles_no_sign.jpg")

    # 4. Empty road: nothing to find.
    img, _ = _base()
    save(img, "synthetic_empty_road.jpg")

    # 5. Speed limit sign.
    img, d = _base()
    _sign(d, [("SPEED", 34, (20, 20, 20)), ("LIMIT 50", 40, (20, 20, 20))])
    save(img, "synthetic_speed_limit.jpg")

    # 6. Roadworks sign plus a vehicle.
    img, d = _base()
    _sign(d, [("ROAD", 44, (200, 110, 10)), ("WORK", 44, (200, 110, 10))])
    _vehicle(d, 150, 460, 180, (230, 150, 40))
    save(img, "synthetic_roadwork.jpg")

    # 7. Sign half hidden behind foliage.
    img, d = _base()
    _sign(d, [("STOP", 54, (190, 20, 20)), ("AHEAD", 34, (20, 20, 20))])
    d.rectangle([520, 170, 700, 262], fill=(60, 110, 55))
    save(img, "synthetic_sign_occluded.jpg")

    # 8. Night scene.
    img, d = _base(night=True)
    _sign(d, [("STOP", 54, (150, 30, 30)), ("AHEAD", 34, (90, 90, 90))])
    save(img, "synthetic_sign_night.jpg")

    # 9-11. The three calibrated hard cases, from the measured sweep.
    save(degrade(clear, 88, 3.5, 22), "synthetic_sign_partial.jpg", quality=22)
    save(degrade(clear, 65, 4.5, 18), "synthetic_sign_garbled.jpg", quality=18)
    save(degrade(clear, 45, 2, 20), "synthetic_sign_illegible.jpg", quality=20)

    # 11. A .txt file, for the invalid-input cases.
    notes = OUT / "not_an_image.txt"
    notes.write_text("This is not an image. It exists so a test can hand it to the agent.\n")
    made.append(notes.name)

    # 12. A zero-byte .jpg.
    (OUT / "empty_file.jpg").write_bytes(b"")
    made.append("empty_file.jpg")

    return made


if __name__ == "__main__":
    for name in build():
        print("wrote", name)
