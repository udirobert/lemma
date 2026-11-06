# lemma — the agent that audits what agents read

Lemma audits scientific claims end to end: paste a paper (arXiv id,
OpenReview id, or a local PDF) and it extracts testable claims, writes
and runs numerical audit scripts with a mandatory positive control, and
judges the resulting evidence trail — supported / falsified /
inconclusive, never a bare paragraph. Every artifact is preserved under
`plugin_data_dir("lemma")/papers/<slug>/` (claims.json,
audit_report.json, judge_report.json, trace.jsonl).

## Setup

```
hermes plugins install lemma        # catalog install, pinned commit
```

At install you are prompted for `ANTHROPIC_API_KEY` (declared
`requires_env`, stored through Hermes's credential route — the plugin
reads it via `agent.secret_scope.get_secret`, never from `.env` or
files). Python deps install into the Hermes environment from
`python_dependencies` in `plugin.yaml`.

The audit pipeline ships inside the pin: catalog installs clone the
whole repo at the reviewed SHA, so `agent/` sits next to
`hermes-plugin/` and the code that runs is the code that was reviewed.

## Tools / hooks / commands

| Surface | What it does |
|---|---|
| `lemma_quick_audit` | extract → audit first N claims (≤5) → judge; bounded ~10 min |
| `lemma_audit` | full pipeline in a detached background run; poll `lemma_status` |
| `lemma_status` | per-workdir verdicts, judge result, trace/evidence paths |
| `lemma_judge` | re-run the evidence-trustworthiness judge on a workdir |
| `lemma_improve` | feedback loop → re-audit → before/after record (advisory) |
| `/audit <paper>` | slash command → `lemma_quick_audit` |
| `pre_llm_call` hook | notices paper links in user turns and the agent's own citations; injects an offer, never runs unasked |
| Audit Room pane | Desktop view backed by `dashboard/plugin_api.py` at `/api/plugins/lemma/*` |

## Network endpoints

- `api.anthropic.com` — auditor/extractor/improve LLM calls (your key)
- `export.arxiv.org`, `arxiv.org`, `openreview.net` — paper fetches
- `api.firecrawl.dev` — only when `FIRECRAWL_API_KEY` is set in env
  (PDF/page parsing; lemma falls back to PyMuPDF without it)
- `paperclip.gxl.ai` — only for `lemma search` (not wired to a tool)
- `*.wandb.ai` — only when `LEMMA_WEAVE=1` is explicitly set (off by
  default; `WANDB_API_KEY` alone never enables tracing)
- pypi — plugin dependency install, at enable time

## Generated code — how it executes

The audit stage runs scripts **written by an LLM from untrusted paper
text** — a prompt-injected paper can smuggle code. Policy
(`agent/script_sandbox.py`):

1. macOS: `sandbox-exec` deny-by-default profile — no network, no exec
   outside the interpreter, writes only under the paper workdir, home
   directories unreadable, rlimits + minimal env.
2. Linux: `bwrap` — private namespaces (no network), read-only system
   binds, workdir is the only writable mount, same env/rlimits.
3. Elsewhere: generated code **does not run** — attempts record
   `sandbox_unavailable` and the claim is left inconclusive.
   `LEMMA_AUDIT_HOST_EXEC=1` is an explicit operator opt-in for
   unsandboxed platforms (still a minimal env — no inherited secrets —
   but no isolation; the plugin never sets it).

Pipeline children themselves spawn with a minimal env (`PATH`, `HOME`,
`ANTHROPIC_API_KEY`, lemma markers) — never the whole Hermes process
environment.

## Detached background runs

`lemma_audit` / `lemma_improve` return immediately and run detached
(`start_new_session`) — they keep working after the turn ends, logging
to `lemma_audit.log` / `lemma_improve.log` inside the workdir. Poll
with `lemma_status` or the Audit Room pane. `lemma_quick_audit` and
`lemma_judge` run in-process with timeouts.

## File access

Reads/writes only under `plugin_data_dir("lemma")` plus the pinned
pipeline source. Local paper sources are restricted to `.pdf` files
with a `%PDF` header inside the plugin papers dir (or a
`LEMMA_LOCAL_PDF_DIRS`-listed dir). Workdir `paper` parameters accept
names only — paths and `..` are rejected.
