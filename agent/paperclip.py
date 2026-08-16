"""Paperclip (GXL) integration — literature evidence cross-check stage.

Paperclip is the re:AGENT host tool for literature access: 11M+ papers,
FDA docs, clinical trials, UniProt/PDB/ChEMBL via CLI, Python SDK
(gxl_paperclip), or MCP. Free for hackathon participants (API key from
the venue / dashboard; PAPERCLIP_API_KEY in .env).

What Lemma uses it for:
  1. discovery   — replace/supplement Firecrawl in `lemma search`
                   (richer corpus: full text + figure QA + SQL)
  2. cross_check — after audits, search the corpus for prior work on each
                   claim; the returned hits become a "literature context"
                   cell on the claim page (supporting or contradicting)

Degrades gracefully: every call returns {"ok": False, ...} when the SDK or
key is missing, and callers skip the stage with a trace note.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from agent.traces import Trace


def available() -> tuple[bool, str]:
    load_dotenv(override=True)
    if not os.environ.get("PAPERCLIP_API_KEY", "").strip():
        return False, "PAPERCLIP_API_KEY not set"
    try:
        import gxl_paperclip  # noqa: F401

        return True, "ok"
    except ImportError:
        return False, "gxl_paperclip not installed (pip install gxl-paperclip)"


def _client():
    from gxl_paperclip import PaperclipClient

    return PaperclipClient.from_env()


def search(query: str, limit: int = 5, trace: Trace | None = None) -> list[dict]:
    """Paperclip discovery; returns [{url,title,description}] like firecrawl."""
    ok, reason = available()
    if not ok:
        if trace:
            trace.note("paperclip", f"search skipped: {reason}")
        return []
    try:
        res = _client().search(query, limit=limit)
        if trace:
            trace.tool_run(
                "paperclip",
                f"search {query[:40]!r}",
                0,
                (res.elapsed_ms or 0) / 1000,
                len(res.output or ""),
            )
        return _parse_search_output(res.output or "", res.result_id)
    except Exception as e:
        if trace:
            trace.note("paperclip", f"search failed: {type(e).__name__}: {e!s:.120}")
        return []


def _parse_search_output(output: str, result_id: str | None) -> list[dict]:
    """Parse the CLI-style search output into hit dicts (best effort)."""
    import re

    hits: list[dict] = []
    blocks = re.split(r"\n\s*\d+\.\s+", "\n" + output)
    for block in blocks[1:]:
        lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
        if not lines:
            continue
        title = lines[0]
        meta = " ".join(lines[1:4])
        doc_id = ""
        m = re.search(
            r"\b((?:bio|med|pmc|arxiv)_[0-9a-f]+|PMC\d+|\d{4}\.\d{4,5})\b", block
        )
        if m:
            doc_id = m.group(1)
        hits.append(
            {
                "url": f"https://paperclip.gxl.ai/papers/{doc_id}" if doc_id else "",
                "title": title,
                "description": meta[:300],
                "result_id": result_id or "",
            }
        )
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

    query = f"{claim['title']} {claim['statement'][:140]}"
    hits = search(query, limit=max_papers, trace=trace)
    note = (
        f"{len(hits)} related works surfaced via Paperclip (GXL) corpus search; "
        "listed as context only — audit verdicts rest on the numerical evidence above."
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
            lines.append(f"  {h['description'][:200]}")
    lines += ["", f"_{result['note']}_"]
    return "\n".join(lines)
