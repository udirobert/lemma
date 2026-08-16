"""Generate brand raster assets: apple-touch-icon (180) and OG image (1200x630).

The mark is the spectrum glass-xylophone bars on deep navy — same glyph as
favicon.svg. Run from repo root:
    .venv/bin/python scripts/gen_brand_assets.py
Outputs land in site/public/ (committed).
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "site" / "public"

BG = (11, 16, 24)
BARS = [
    ((79, 142, 247), 1.0),  # blue
    ((212, 69, 242), 1.0),  # magenta
    ((30, 194, 181), 1.0),  # teal
    ((250, 181, 51), 1.0),  # gold
    ((120, 102, 237), 0.55),  # violet (dim = honest inconclusive)
]


def draw_bars(
    d: ImageDraw.ImageDraw,
    x0: float,
    y_base: float,
    bar_w: float,
    gap: float,
    h_unit: float,
    radius: float,
) -> None:
    heights = [5, 7, 9, 11, 8]
    x = x0
    for (color, alpha), h in zip(BARS, heights, strict=True):
        top = y_base - h * h_unit
        d.rounded_rectangle(
            [x, top, x + bar_w, y_base],
            radius=radius,
            fill=(*color, int(255 * alpha)),
        )
        # glass sheen at the top of each bar
        sheen_h = max(2.0, h * h_unit * 0.28)
        d.rounded_rectangle(
            [x + bar_w * 0.16, top + bar_w * 0.1, x + bar_w * 0.84, top + sheen_h],
            radius=radius * 0.6,
            fill=(255, 255, 255, 70),
        )
        x += bar_w + gap


def load_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for c in candidates:
        if Path(c).exists():
            try:
                return ImageFont.truetype(c, size)
            except OSError:
                continue
    return ImageFont.load_default()


def apple_touch_icon() -> None:
    s = 180
    img = Image.new("RGBA", (s, s), (*BG, 255))
    d = ImageDraw.Draw(img)
    draw_bars(d, 30, 132, 20, 6, 10.5, 6)
    img.convert("RGB").save(OUT / "apple-touch-icon.png")
    print("wrote apple-touch-icon.png (180x180)")


def og_image() -> None:
    w, h = 1200, 630
    img = Image.new("RGBA", (w, h), (*BG, 255))

    # radial glows (brand colors, blurred)
    glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    gd.ellipse([-150, -260, 620, 240], fill=(79, 142, 247, 46))
    gd.ellipse([620, -220, 1400, 260], fill=(212, 69, 242, 34))
    gd.ellipse([300, 380, 1100, 900], fill=(30, 194, 181, 24))
    glow = glow.filter(ImageFilter.GaussianBlur(90))
    img.alpha_composite(glow)

    d = ImageDraw.Draw(img)

    # right-hand xylophone motif
    draw_bars(d, 760, 500, 62, 26, 30, 16)

    # kicker
    mono = load_font(22, bold=False)
    d.text((72, 118), "EVIDENCE OVER ASSERTION", font=mono, fill=(30, 194, 181, 235))

    # wordmark
    disp = load_font(128)
    d.text((66, 148), "Lemma", font=disp, fill=(241, 243, 246, 255))

    # tagline
    body = load_font(34, bold=False)
    d.text(
        (72, 306),
        "the AI scientist that distrusts itself",
        font=body,
        fill=(152, 162, 179, 255),
    )

    # scoreboard strip
    stats = [
        ("2", "papers", (79, 142, 247)),
        ("12", "claims", (212, 69, 242)),
        ("8", "reproduced", (30, 194, 181)),
        ("15", "failures preserved", (250, 181, 51)),
    ]
    x = 72
    num = load_font(46)
    lab = load_font(17, bold=False)
    for value, label, color in stats:
        d.text((x, 398), value, font=num, fill=(*color, 255))
        d.text((x, 456), label.upper(), font=lab, fill=(100, 112, 138, 255))
        x += 200

    # bottom spectrum strip
    seg = w / 5
    colors = [
        (79, 142, 247),
        (212, 69, 242),
        (30, 194, 181),
        (250, 181, 51),
        (120, 102, 237),
    ]
    for i, c in enumerate(colors):
        d.rectangle([i * seg, h - 8, (i + 1) * seg, h], fill=(*c, 255))

    img.convert("RGB").save(OUT / "og-image.png", quality=92)
    print("wrote og-image.png (1200x630)")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    apple_touch_icon()
    og_image()
