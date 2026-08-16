"""Generate deck artwork with Runware (FLUX) for the submission deck.

Usage:  .venv/bin/python scripts/gen_deck_art.py [--variants N]

Discovers a FLUX model from the free catalog API, generates the brand
images (16:9, matched to the deck palette), downloads them to
tmp/deck_art/ and writes tmp/deck_art/manifest.json with the public URLs
for the Slides generator.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO / ".env", override=True)

from runware import Runware, RunwareError  # noqa: E402

ART_DIR = REPO / "tmp" / "deck_art"

# Shared style suffix keeps the set visually consistent
STYLE = (
    ", iridescent refractive 3D glass material, glowing colored rim light, "
    "vibrant spectrum of colors across the bars - electric blue, magenta, "
    "amber, teal and violet - each bar a different hue, colorful light leaks "
    "and caustics on a very dark navy backdrop hex #0B1018, energetic premium "
    "tech keynote art, elegant, ultra clean composition, no text, no words, "
    "no logos, no watermark"
)

# Art direction: Codrops glass-xylophone / DNA-ladder aesthetic — floating
# translucent glass bars arranged like xylophone keys evoking a DNA strand,
# with the color energy of a real xylophone (spectrum bars).
PROMPTS = {
    "hero": (
        "a sweeping horizontal row of floating glass bars of graduated lengths, "
        "spaced and angled like xylophone keys, evoking the rungs of a DNA "
        "double helix ladder, bars glowing in a vivid rainbow spectrum, two "
        "luminous strands of light weaving through them, dramatic depth, "
        "energetic and musical" + STYLE
    ),
    "arc": (
        "an ascending crescendo of floating glass bars rising from lower left "
        "to upper right, each bar bigger, brighter and more saturated than the "
        "last, colors intensifying from cool blue to hot magenta and gold at "
        "the peak, suggesting a dramatic phase transition breakthrough, a thin "
        "glowing baseline beneath" + STYLE
    ),
    "artifacts": (
        "receding rows of small glass bars arranged like archived specimens in "
        "a dark museum, each row a different spectral color, soft depth of "
        "field with the nearest row vivid and glowing, orderly yet lively, "
        "gentle colored light pooling on a dark reflective floor" + STYLE
    ),
}

W, H = 1344, 768  # 16:9, native-ish FLUX resolution


async def discover_model(client: Runware) -> tuple[str, str]:
    """Find a FLUX image model AIR from the free catalog. Returns (air, name)."""
    try:
        models = await client.content.list_models(
            {"category": "image", "search": "flux"}
        )
    except RunwareError:
        models = await client.model_search({"search": "flux", "category": "image"})
    # normalize possible envelope shapes
    if isinstance(models, dict):
        models = models.get("models") or models.get("results") or []
    best = None
    for m in models:
        name = str(m.get("name") or m.get("headline") or "").lower()
        air = m.get("air") or m.get("model") or ""
        # prefer dev over schnell if both present; first match otherwise
        if (
            "flux" in name
            and air
            and (best is None or ("dev" in name and "dev" not in best[1].lower()))
        ):
            best = (str(air), name)
    if best is None:
        raise SystemExit("no FLUX model found in catalog")
    return best


async def gen(client: Runware, air: str, name: str, prompt: str, variant: int) -> dict:
    images = await client.run(
        {
            "taskType": "imageInference",
            "model": air,
            "positivePrompt": prompt,
            "negativePrompt": "text, words, letters, logo, watermark, busy, cluttered, bright background, white background, constellation, network graph, dots and lines diagram, particles",
            "width": W,
            "height": H,
            "deliveryMethod": "sync",
        }
    )
    url = images[0]["imageURL"]
    local = ART_DIR / f"{name}_v{variant}.jpg"
    urllib.request.urlretrieve(url, local)
    return {"name": name, "variant": variant, "url": url, "local": str(local)}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", type=int, default=2)
    args = ap.parse_args()

    ART_DIR.mkdir(parents=True, exist_ok=True)
    async with Runware(transport="rest") as client:
        air, model_name = await discover_model(client)
        print(f"Using model: {air} ({model_name})")
        results = []
        for name, prompt in PROMPTS.items():
            for v in range(1, args.variants + 1):
                try:
                    r = await gen(client, air, name, prompt, v)
                    print(f"  {name} v{v}: {r['url'][:80]}...")
                    results.append(r)
                except RunwareError as exc:
                    print(f"  {name} v{v} FAILED: {exc}")
        manifest = {"model": air, "images": results}
        (ART_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"manifest: {ART_DIR / 'manifest.json'} ({len(results)} images)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
