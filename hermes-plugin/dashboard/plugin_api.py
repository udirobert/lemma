"""Backend routes for the lemma Audit Room desktop pane.

Mounted at /api/plugins/lemma/ inside the gateway process. Serves the
same data the lemma_status tool reads — plugin workdirs under
plugin_data_dir("lemma")/papers — plus claim detail and figures
(data-URI inlined so the renderer needs no file access).
"""

import base64
import json
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools import PAPERS_DIR, _verdicts

router = APIRouter()

MAX_FIGURES, MAX_FIG_BYTES = 9, 400_000


def _dirs():
    if not PAPERS_DIR.is_dir():
        return []
    return sorted(
        d for d in PAPERS_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")
    )


@router.get("/papers")
async def papers():
    out = []
    for d in _dirs():
        entry = {"workdir": d.name}
        entry.update(_verdicts(d))
        out.append(entry)
    return {"papers": out}


@router.get("/papers/{name}")
async def paper_detail(name: str):
    if "/" in name or ".." in name:
        raise HTTPException(400, "bad name")
    d = PAPERS_DIR / name
    if not d.is_dir():
        raise HTTPException(404, "no such workdir")
    out = {"workdir": name}
    out.update(_verdicts(d))

    claims_p = d / "claims.json"
    if claims_p.is_file():
        out["claim_texts"] = {
            c.get("id"): (c.get("claim") or c.get("text") or "")[:200]
            for c in json.loads(claims_p.read_text())
        }
    figures = []
    for png in sorted(d.glob("results/*/*.png"))[:MAX_FIGURES]:
        if png.stat().st_size > MAX_FIG_BYTES:
            continue
        figures.append(
            {
                "name": png.name,
                "claim": png.parent.name,
                "data_uri": "data:image/png;base64,"
                + base64.b64encode(png.read_bytes()).decode(),
            }
        )
    out["figures"] = figures
    return out
