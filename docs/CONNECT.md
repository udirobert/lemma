# Connect — ChatGPT / MCP (lemma)

Trigger phrase: **"audit this paper"**

## Free HTTP extract server

```bash
# from repo root (uses .venv when present)
./lemma extract-server --port 8787
```

| Surface | Method | Path |
| --- | --- | --- |
| Health | `GET` | `/` or `/health` |
| Tool discovery | `GET` | `/tools` |
| Free wedge | `POST` | `/extract` |
| Thin MCP stub | `POST` | `/mcp` (JSON-RPC `initialize` / `tools/list` / `tools/call`) |

```bash
# Demo (no LLM / PDF)
curl -s -X POST http://127.0.0.1:8787/extract \
  -H 'content-type: application/json' \
  -d '{"demo":true,"source":"2510.10981"}'

# Live extract (needs ANTHROPIC_API_KEY or OPENAI_API_KEY in .env)
curl -s -X POST http://127.0.0.1:8787/extract \
  -H 'content-type: application/json' \
  -d '{"source":"https://arxiv.org/abs/2510.10981"}'
```

In-process helper (no HTTP): `agent.mcp_adapter.call_extract_claims({"source": "2510.10981"})`.

Equivalent CLI Stage-1 only:

```bash
./lemma audit 2510.10981 --stages extract
```

## Paid / deep follow-up (not in-plugin)

Full audit (scripts → evidence → judge) via **existing account**, product queue, or informational plans page — **not** in-plugin checkout.

- Product / demo: https://lemmabio.netlify.app
- CLI: `./lemma audit <source>`

## Adapter stub

`agent/mcp_adapter.py` documents wiring `extract_claims` for ChatGPT. Full MCP Streamable HTTP (SSE sessions) + OAuth for directory review is **remaining** — HTTP `/extract` + `/mcp` stub is the shippable wedge this pass.

## ChatGPT directory checklist

See [CHATGPT_PLUGIN_PLAYBOOK.md](./CHATGPT_PLUGIN_PLAYBOOK.md). Listing needs privacy policy, support contact, [STARTER_PROMPTS.md](./STARTER_PROMPTS.md), and intent QA ([EVAL.md](./EVAL.md)).
