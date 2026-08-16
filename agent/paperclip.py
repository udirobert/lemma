"""Paperclip (GXL) integration — literature evidence cross-check stage.

Talks to the hosted MCP endpoint (https://paperclip.gxl.ai/mcp) directly
over HTTP with an X-API-Key header — no SDK dependency (the gxl_paperclip
package is not on PyPI; the MCP route is the scripted-friendly path).
Key: PAPERCLIP_API_KEY in .env (hackathon keys work; verified 2026-08-16).

What Lemma uses it for:
  1. discovery    — `lemma search` across the Paperclip corpus
                    (arXiv/PMC/bioRxiv/medRxiv/FDA/trials/proteins)
  2. cross_check  — after audits, surface related work per claim as a
                    "Literature context" logbook cell (context only —
                    never alters audit verdicts)

Degrades gracefully: every call returns [] / a note when unconfigured.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from dotenv import load_dotenv

from agent.traces import Trace

MCP_URL = "https://paperclip.gxl.ai/mcp"
TIMEOUT_S = 60
DEFAULT_SOURCE = "arxiv"


def available() -> tuple[bool, str]:
    load_dotenv(override=True)
    key = os.environ.get("PAPERCLIP_API_KEY", "").strip()
    if not key:
        return False, "PAPERCLIP_API_KEY not set"
    return True, "ok"


def run_command(
    command: str, trace: Trace | None = None, timeout: float = TIMEOUT_S
) -> dict:
    """Run one Paperclip CLI command via the hosted MCP endpoint."""
    ok, reason = available()
    if not ok:
        return {"ok": False, "text": "", "error": reason}
    key = os.environ["PAPERCLIP_API_KEY"].strip()
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {"name": "paperclip", "arguments": {"command": command}},
    }
    req = urllib.request.Request(
        MCP_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-API-Key": key},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        text = "".join(
            c.get("text", "") for c in data.get("result", {}).get("content", [])
        )
        if trace:
            trace.tool_run("paperclip", command[:60], 0, 0.0, len(text))
        return {"ok": True, "text": text, "error": None}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:300]
        if trace:
            trace.note("paperclip", f"command failed HTTP {e.code}: {body[:120]}")
        return {"ok": False, "text": "", "error": f"HTTP {e.code}: {body}"}
    except OSError as e:
        if trace:
            trace.note("paperclip", f"command failed: {e}")
        return {"ok": False, "text": "", "error": str(e)}


def search(
    query: str, limit: int = 5, source: str = DEFAULT_SOURCE, trace: Trace | None = None
) -> list[dict]:
    """Corpus search; returns [{url,title,description}] like firecrawl."""
    # double-quote escaping for the command string
    q = query.replace('"', "'")
    res = run_command(f'search -s {source} "{q}" -n {limit}', trace)
    if not res["ok"]:
        return []
    return _parse_search_output(res["text"])


def _parse_search_output(output: str) -> list[dict]:
    """Parse CLI-style search output into hit dicts (best effort)."""
    import re

    hits: list[dict] = []
    # numbered blocks: "1. Title\n   Authors\n   id · source · year\n..."
    blocks = re.split(r"\n\s*\d+\.\s+", "\n" + output)
    for block in blocks[1:]:
        lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
        if not lines:
            continue
        title = lines[0]
        meta = " | ".join(lines[1:4])
        url = ""
        m = re.search(r"\b((?:arx|bio|med|pmc)_[0-9a-zA-Z]+)\b", block)
        doi = re.search(r"(https://doi\.org/\S+)", block)
        if m:
            url = f"paperclip:/papers/{m.group(1)}"
        if doi:
            url = doi.group(1).rstrip(".")
        hits.append({"url": url, "title": title, "description": meta[:300]})
    return hits


def cross_check(claim: dict, max_papers: int = 3, trace: Trace | None = None) -> dict:
    """Literature context for one audited claim.

    Returns {"query": ..., "hits": [...], "note": ...} — the evidence stage
    renders it as a markdown cell on the claim page. Never changes the audit
    verdict: prior literature is context, not ground truth.
    """
    ok, reason = available()
    if not ok:
        return {"query": "", "hits": [], "note": f"Paperclip unavailable ({reason})"}

    query = f"{claim['title']} {claim['statement'][:120]}"
    hits = search(query, limit=max_papers, trace=trace)
    note = (
        f"{len(hits)} related works surfaced via Paperclip (GXL) corpus search "
        f"({DEFAULT_SOURCE} source); listed as context only — audit verdicts "
        "rest on the numerical evidence above."
    )
    if trace:
        trace.log("paperclip", "cross_check", claim_id=claim["id"], n_hits=len(hits))
    return {"query": query, "hits": hits, "note": note}


def render_markdown(result: dict) -> str:
    if not result.get("hits"):
        return f"*Literature context: {result.get('note', 'none found')}*"
    lines = ["**Literature context (Paperclip / GXL corpus).**", ""]
    for h in result["hits"]:
        lines.append(f"- {h['title']}")
        if h.get("description"):
            lines.append(f"  {h['description'][:220]}")
        if h.get("url"):
            lines.append(f"  {h['url']}")
    lines += ["", f"_{result['note']}_"]
    return "\n".join(lines)
