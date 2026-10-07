"""Minimal HTTP claim-extract endpoint (free ChatGPT discovery wedge).

Stdlib only — same family as agent/room_server.py. Runs claim extract for an
arXiv id/URL or OpenReview id without the full audit.

  ./lemma extract-server --port 8787
  curl -s -X POST http://127.0.0.1:8787/extract \\
    -H 'content-type: application/json' \\
    -d '{"source":"https://arxiv.org/abs/2510.10981"}'
"""

from __future__ import annotations

import json
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = REPO_ROOT / "papers"
DEFAULT_PORT = 8787
MAX_BODY = 32_768
PRODUCT_URL = "https://lemmabio.netlify.app"
PLANS_URL = "https://lemmabio.netlify.app"  # informational; no in-plugin checkout


def _tools_payload() -> dict:
    return {
        "ok": True,
        "service": "lemma",
        "protocol": "mcp-http",
        "triggerPhrase": "audit this paper",
        "tools": [
            {
                "name": "extract_claims",
                "title": "Extract checkable claims from a paper",
                "description": (
                    "Use when the user asks to audit a paper, check if claims are "
                    "trustworthy, or extract testable claims from an arXiv / "
                    "OpenReview link. Free. Returns structured claims only — does "
                    "not run the full numerical audit. Do not use for submitting "
                    "payments or starting a paid audit checkout inside ChatGPT."
                ),
                "annotations": {
                    "readOnlyHint": True,
                    "destructiveHint": False,
                    "openWorldHint": True,
                },
                "endpoint": "POST /extract",
                "price": "free",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": (
                                "arXiv id (e.g. 2510.10981), arXiv URL, OpenReview "
                                "id, or OpenReview URL."
                            ),
                        },
                        "demo": {
                            "type": "boolean",
                            "description": (
                                "If true, return a labelled sample claims payload "
                                "without calling the LLM or downloading a PDF."
                            ),
                        },
                    },
                    "required": ["source"],
                },
            }
        ],
        "paidFollowUp": {
            "actions": ["full claim audit", "evidence trail", "judge"],
            "where": PRODUCT_URL,
            "note": (
                "Full audit via existing account / product queue / informational "
                "plans page — not in-plugin digital checkout."
            ),
        },
    }


DEMO_CLAIMS = [
    {
        "id": "C1",
        "title": "Risk identity (demo)",
        "statement": "Demo claim: excess risk equals Bayes gap plus estimation error.",
        "kind": "theorem",
        "testable": True,
        "test_plan": "Simulate toy posterior; compare risk decomposition numerically.",
        "evidence_in_paper": "Prop. 3.1 (demo fixture)",
        "compute": "cpu-fast",
        "success_criterion": "Relative difference of both sides < 1%.",
    },
    {
        "id": "C2",
        "title": "Coupling rate (demo)",
        "statement": "Demo claim: error scales as m/(pN).",
        "kind": "empirical",
        "testable": True,
        "test_plan": "Log-log regression of measured error vs m/(pN).",
        "evidence_in_paper": "Fig. 2 (demo fixture)",
        "compute": "cpu-fast",
        "success_criterion": "Slope within 0.2 of -1 and r² > 0.9.",
    },
]


def extract_claims(source: str, *, workdir: Path | None = None) -> dict:
    """Run Stage-1 extract only. Returns agent-friendly payload."""
    from agent.extract import extract
    from agent.papers import resolve
    from agent.traces import Trace

    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
    staging = workdir or (PAPERS_DIR / "_extract_http" / run_id)
    staging.mkdir(parents=True, exist_ok=True)

    paper = resolve(source, workdir=staging)
    paper_workdir = PAPERS_DIR / paper["paper_id"]
    paper_workdir.mkdir(parents=True, exist_ok=True)

    pdf = Path(paper["pdf_path"])
    if pdf.parent != paper_workdir:
        moved = paper_workdir / pdf.name
        if not moved.exists():
            moved.write_bytes(pdf.read_bytes())
        paper["pdf_path"] = str(moved)

    # Prefer fresh extract for API calls: use a temp claims path by clearing
    # reuse only when caller wants cache — keep CLI semantics in paper_workdir.
    trace = Trace(f"extract-http-{run_id}", paper_workdir / "trace.extract-http.jsonl")
    claims = extract(
        paper["paper_id"], paper["title_hint"], paper["text"], paper_workdir, trace
    )
    n_testable = sum(1 for c in claims if c.get("testable"))
    summary = (
        f"{paper['title_hint']}: extracted {len(claims)} claims "
        f"({n_testable} testable). Free wedge only — full audit (scripts + "
        f"evidence + judge) runs on the product / existing account at {PRODUCT_URL}."
    )
    return {
        "ok": True,
        "tool": "lemma.extract-claims",
        "paper_id": paper["paper_id"],
        "title": paper["title_hint"],
        "source_kind": paper["source_kind"],
        "summary": summary,
        "claims": claims,
        "n_claims": len(claims),
        "n_testable": n_testable,
        "upgrade": {
            "message": (
                "Run the full audit (numerical scripts, evidence trail, judge) "
                "via an existing account or the product site — not in-plugin checkout."
            ),
            "url": PRODUCT_URL,
            "plansUrl": PLANS_URL,
            "cli": f"./lemma audit {paper['paper_id'].removeprefix('arxiv-')}",
        },
    }


class ExtractHandler(BaseHTTPRequestHandler):
    server_version = "lemma-extract/0.1"

    def log_message(self, fmt: str, *args) -> None:  # quieter default
        print(f"[extract-http] {self.address_string()} {fmt % args}")

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > MAX_BODY:
            raise ValueError("request body too large")
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path in ("/", "/health"):
            self._send(
                200,
                {
                    "ok": True,
                    "service": "lemma-extract",
                    "tool": "extract_claims",
                    "usage": 'POST /extract {"source":"<arxiv id|url>"} or {"demo":true}',
                },
            )
            return
        if path == "/tools":
            self._send(200, _tools_payload())
            return
        self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path.rstrip("/") or "/"
        try:
            body = self._read_json()
        except Exception as exc:
            self._send(400, {"ok": False, "error": f"invalid JSON: {exc}"})
            return

        if path == "/extract":
            if body.get("demo") is True:
                self._send(
                    200,
                    {
                        "ok": True,
                        "tool": "lemma.extract-claims",
                        "demo": True,
                        "paper_id": "demo-arxiv-0000.00000",
                        "title": "Demo paper (labelled sample)",
                        "summary": (
                            "Demo: 2 sample claims. Live extract skipped. "
                            f"Full audit lives at {PRODUCT_URL}."
                        ),
                        "claims": DEMO_CLAIMS,
                        "n_claims": len(DEMO_CLAIMS),
                        "n_testable": sum(1 for c in DEMO_CLAIMS if c["testable"]),
                        "upgrade": {
                            "message": "Full audit on product site / existing account.",
                            "url": PRODUCT_URL,
                            "plansUrl": PLANS_URL,
                        },
                    },
                )
                return

            source = (body.get("source") or "").strip()
            if not source:
                self._send(400, {"ok": False, "error": "source is required"})
                return
            try:
                result = extract_claims(source)
                self._send(200, result)
            except ValueError as exc:
                self._send(422, {"ok": False, "error": str(exc)})
            except Exception as exc:
                traceback.print_exc()
                self._send(500, {"ok": False, "error": str(exc)})
            return

        if path == "/mcp":
            # Thin JSON-RPC stub — same shape as nuncio/databard adapters.
            rpc_id = body.get("id")
            method = body.get("method")
            if method == "initialize":
                self._send(
                    200,
                    {
                        "jsonrpc": "2.0",
                        "id": rpc_id,
                        "result": {
                            "protocolVersion": "2025-03-26",
                            "serverInfo": {"name": "lemma", "version": "0.1.0"},
                            "capabilities": {"tools": {}},
                            "instructions": (
                                "Free tool extract_claims pulls checkable claims "
                                "from an arXiv/OpenReview paper. Full audit requires "
                                "the product site / existing account — never checkout "
                                "in ChatGPT."
                            ),
                        },
                    },
                )
                return
            if method in ("tools/list", "tools/listChanged"):
                tools = _tools_payload()["tools"]
                self._send(
                    200,
                    {
                        "jsonrpc": "2.0",
                        "id": rpc_id,
                        "result": {
                            "tools": [
                                {
                                    "name": t["name"],
                                    "description": t["description"],
                                    "inputSchema": t["inputSchema"],
                                    "annotations": t["annotations"],
                                }
                                for t in tools
                            ]
                        },
                    },
                )
                return
            if method == "tools/call":
                params = body.get("params") or {}
                name = params.get("name")
                args = params.get("arguments") or {}
                if name != "extract_claims":
                    self._send(
                        200,
                        {
                            "jsonrpc": "2.0",
                            "id": rpc_id,
                            "error": {
                                "code": -32601,
                                "message": f"Unknown tool: {name}",
                            },
                        },
                    )
                    return
                # Re-enter extract via internal call
                if args.get("demo") is True:
                    payload = {
                        "ok": True,
                        "demo": True,
                        "claims": DEMO_CLAIMS,
                        "summary": "Demo claims (labelled sample).",
                    }
                    self._send(
                        200,
                        {
                            "jsonrpc": "2.0",
                            "id": rpc_id,
                            "result": {
                                "content": [
                                    {"type": "text", "text": json.dumps(payload)}
                                ],
                                "structuredContent": payload,
                            },
                        },
                    )
                    return
                source = (args.get("source") or "").strip()
                if not source:
                    self._send(
                        200,
                        {
                            "jsonrpc": "2.0",
                            "id": rpc_id,
                            "result": {
                                "isError": True,
                                "content": [
                                    {
                                        "type": "text",
                                        "text": '{"error":"source is required"}',
                                    }
                                ],
                            },
                        },
                    )
                    return
                try:
                    payload = extract_claims(source)
                    self._send(
                        200,
                        {
                            "jsonrpc": "2.0",
                            "id": rpc_id,
                            "result": {
                                "content": [
                                    {"type": "text", "text": json.dumps(payload)}
                                ],
                                "structuredContent": payload,
                            },
                        },
                    )
                except Exception as exc:
                    self._send(
                        200,
                        {
                            "jsonrpc": "2.0",
                            "id": rpc_id,
                            "result": {
                                "isError": True,
                                "content": [
                                    {
                                        "type": "text",
                                        "text": json.dumps({"error": str(exc)}),
                                    }
                                ],
                            },
                        },
                    )
                return
            self._send(
                200,
                {
                    "jsonrpc": "2.0",
                    "id": rpc_id,
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}",
                    },
                },
            )
            return

        self._send(404, {"ok": False, "error": "not found"})


def serve(host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> int:
    httpd = ThreadingHTTPServer((host, port), ExtractHandler)
    print(f"[lemma] extract-server http://{host}:{port}  (POST /extract, GET /tools)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[lemma] extract-server stopped")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ns = ap.parse_args()
    raise SystemExit(serve(ns.host, ns.port))
