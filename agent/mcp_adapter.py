"""MCP adapter stub for lemma `extract_claims`.

The working free surface is the HTTP extract server:

  ./lemma extract-server --port 8787
  GET  /tools
  POST /extract
  POST /mcp   (thin JSON-RPC: initialize | tools/list | tools/call)

This module documents how to wrap that endpoint as a full MCP server once the
official Streamable HTTP / stdio SDK path is wired for ChatGPT directory
review. It also exposes a tiny in-process helper so other agents can call
extract without standing up HTTP.

Remaining for production ChatGPT listing:
- Official MCP Streamable HTTP transport (SSE sessions)
- OAuth / demo account for review
- Privacy policy URL + support contact
- Point directory MCP URL at a deployed extract-server (or gateway)
"""

from __future__ import annotations

from typing import Any

from agent.extract_http import DEMO_CLAIMS, PRODUCT_URL, extract_claims

TOOL_NAME = "extract_claims"

TOOL_DESCRIPTOR: dict[str, Any] = {
    "name": TOOL_NAME,
    "description": (
        "Use when the user asks to audit a paper or extract checkable claims "
        "from an arXiv / OpenReview link. Free Stage-1 extract only."
    ),
    "annotations": {
        "readOnlyHint": True,
        "destructiveHint": False,
        "openWorldHint": True,
    },
    "inputSchema": {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "arXiv id, arXiv URL, OpenReview id, or OpenReview URL.",
            },
            "demo": {
                "type": "boolean",
                "description": "Return labelled sample claims without live extract.",
            },
        },
        "required": ["source"],
    },
}


def call_extract_claims(arguments: dict[str, Any]) -> dict[str, Any]:
    """In-process tool handler used by adapters / tests."""
    if arguments.get("demo") is True:
        return {
            "ok": True,
            "tool": "lemma.extract-claims",
            "demo": True,
            "claims": DEMO_CLAIMS,
            "summary": f"Demo claims. Full audit at {PRODUCT_URL}.",
            "upgrade": {"url": PRODUCT_URL},
        }
    source = (arguments.get("source") or "").strip()
    if not source:
        raise ValueError("source is required")
    return extract_claims(source)


def mcp_stdio_instructions() -> str:
    """Human-readable wiring notes (no SDK dependency yet)."""
    return (
        "Wire ChatGPT / Codex to POST http://<host>:8787/mcp with JSON-RPC "
        "tools/call name=extract_claims, or call POST /extract directly. "
        "Full MCP SDK stdio/SSE server is not bundled in this pass — see "
        "docs/CONNECT.md Remaining."
    )
