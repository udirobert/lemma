"""Firecrawl integration: literature search + PDF parsing for the agent.

Keyless tier (works now):
  POST /v2/search  {"query": ..., "sources": ["web"], "limit": N}
Keyed tier (set FIRECRAWL_API_KEY in .env):
  sources may include {"research": {...}} -> 41M+ life-science paper index.

Graceful degradation: every call returns a dict with "ok"; callers fall back
to arXiv API when Firecrawl is unavailable.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

from agent.traces import Trace

BASE = "https://api.firecrawl.dev"
TIMEOUT_S = 45


def _headers() -> dict:
    load_dotenv(override=True)
    key = os.environ.get("FIRECRAWL_API_KEY", "").strip()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _post(endpoint: str, payload: dict) -> dict:
    req = urllib.request.Request(
        BASE + endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=_headers(),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        return {"success": False, "error": f"HTTP {e.code}: {body}"}
    except OSError as e:
        return {"success": False, "error": str(e)}


def web_search(query: str, limit: int = 5, trace: Trace | None = None) -> dict:
    data = _post("/v2/search", {"query": query, "sources": ["web"], "limit": limit})
    if trace:
        trace.tool_run(
            "search",
            f"firecrawl web_search {query[:40]!r}",
            0 if data.get("success") else 1,
            0.0,
            len(json.dumps(data)),
        )
    return data


def research_search(query: str, limit: int = 5, trace: Trace | None = None) -> dict:
    """Life-science index (41M+ papers). Requires FIRECRAWL_API_KEY."""
    payload = {"query": query, "sources": [{"research": {}}], "limit": limit}
    data = _post("/v2/search", payload)
    if not data.get("success") and trace:
        trace.note(
            "search",
            f"research index unavailable ({str(data.get('error', ''))[:120]}); "
            "will fall back to web/arxiv",
        )
    # schema may be web-only on this deployment: degrade to web results if present
    if data.get("success"):
        d = data.get("data", {})
        if "research" not in d and "web" in d:
            d["research"] = d.pop("web")
    if trace:
        trace.tool_run(
            "search",
            f"firecrawl research_search {query[:40]!r}",
            0 if data.get("success") else 1,
            0.0,
            len(json.dumps(data)),
        )
    return data


def arxiv_fallback(query: str, limit: int = 5, trace: Trace | None = None) -> dict:
    """Direct arXiv API search when Firecrawl is down or keyless."""
    import re

    url = (
        "https://export.arxiv.org/api/query?search_query=all:"
        + urllib.parse.quote(query)
        + f"&max_results={limit}"
    )
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_S) as resp:
            xml = resp.read().decode("utf-8", errors="replace")
        entries = []
        for block in re.findall(r"<entry>[\s\S]*?</entry>", xml):
            title = re.search(r"<title>([\s\S]*?)</title>", block)
            link = re.search(r"<id>([\s\S]*?)</id>", block)
            summary = re.search(r"<summary>([\s\S]*?)</summary>", block)
            entries.append(
                {
                    "url": link.group(1).strip() if link else "",
                    "title": re.sub(r"\s+", " ", title.group(1)).strip()
                    if title
                    else "",
                    "description": (summary.group(1).strip()[:300] if summary else ""),
                }
            )
        data = {"success": True, "data": {"web": entries}, "backend": "arxiv"}
    except OSError as e:
        data = {"success": False, "error": str(e)}
    if trace:
        trace.tool_run(
            "search",
            f"arxiv_fallback {query[:40]!r}",
            0 if data.get("success") else 1,
            0.0,
            len(json.dumps(data)),
        )
    return data


def search(
    query: str, limit: int = 5, *, research: bool = False, trace: Trace | None = None
) -> list[dict]:
    """Search with graceful degradation; returns [{url,title,description}]."""
    results = research_search(query, limit, trace) if research else None
    if not results or not results.get("success"):
        results = web_search(query, limit, trace)
    if not results or not results.get("success"):
        results = arxiv_fallback(query, limit, trace)
    if not results.get("success"):
        return []
    d = results.get("data", {})
    hits = d.get("research") or d.get("web") or []
    return [
        {
            "url": h.get("url", ""),
            "title": h.get("title", ""),
            "description": h.get("description", ""),
        }
        for h in hits
    ]


def markdown_from_pdf(pdf_path: Path, trace: Trace | None = None) -> str | None:
    """Parse a local PDF to clean markdown via Firecrawl (keyed tier)."""
    if not os.environ.get("FIRECRAWL_API_KEY"):
        return None
    import base64

    payload = {
        "data": base64.b64encode(pdf_path.read_bytes()).decode("ascii"),
    }
    req = urllib.request.Request(
        BASE + "/v2/extract",
        data=json.dumps(payload).encode("utf-8"),
        headers=_headers(),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        if trace:
            trace.tool_run(
                "extract", "firecrawl markdown_from_pdf", 0, 0.0, len(json.dumps(data))
            )
        if data.get("success"):
            return data.get("data", {}).get("markdown")
    except (urllib.error.HTTPError, OSError):
        pass
    return None
