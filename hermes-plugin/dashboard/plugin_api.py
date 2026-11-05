"""Backend routes for the lemma Audit Room desktop pane.

Mounted at /api/plugins/lemma/ inside the gateway process. Serves the
same data the lemma_status tool reads — plugin workdirs under
plugin_data_dir("lemma")/papers — plus claim detail and figures
(data-URI inlined so the renderer needs no file access).
"""

import base64
import contextlib
import importlib.util
import json
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_spec = importlib.util.spec_from_file_location(
    "lemma_plugin_tools", Path(__file__).resolve().parent.parent / "tools.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
PAPERS_DIR, _verdicts = _mod.PAPERS_DIR, _mod._verdicts

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


@router.get("/runs")
async def runs():
    """In-flight audits: workdir had file activity < 5 min ago and no
    judge verdict yet. Includes the last trace lines — the pane shows
    the audit *working*, not just its results."""
    import time

    now = time.time()
    out = []
    for d in _dirs():
        newest = 0.0
        for p in d.rglob("*"):
            with contextlib.suppress(OSError):
                newest = max(newest, p.stat().st_mtime)
        if now - newest > 300 or (d / "judge_report.json").is_file():
            continue
        tail = []
        trace = d / "trace.jsonl"
        if trace.is_file():
            with contextlib.suppress(Exception):
                tail = [
                    json.loads(ln).get("event") or ln[:80]
                    for ln in trace.read_text().splitlines()[-3:]
                ]
        out.append(
            {
                "workdir": d.name,
                "last_activity_s": int(now - newest),
                "trace_tail": tail,
            }
        )
    return {"runs": out}


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
