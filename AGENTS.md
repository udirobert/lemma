# AGENTS.md — lemma

Guidance for AI agents (and humans) working in this repository.

## What this is

The **re:AGENT Track A build**: an AI-scientist that audits scientific claims.
Given an unseen paper (arxiv id, openreview id, or local PDF), the agent
pipeline extracts testable claims, writes and runs honest numerical audit
scripts, assembles an inspectable evidence trail (a Trackio logbook), and runs
an automated judge that scores whether the trail is trustworthy — PASS /
CONDITIONAL PASS / FAIL. Every LLM call and tool result is appended to a JSONL
trace so the reasoning is inspectable.

Event: re:AGENT — End to End Agentic Science, Founders Inc SF, Aug 15–16 2026.
Short planning doc: `HACKATHON.md`.

The ICML 2026 Agent Reproductions challenge (July 15 – Aug 2, 2026) is
**completed prior art**: `papers/5nNNVY8NW4-grokking/` passed that challenge's
judge end to end and now serves as the regression fixture
(`./lemma judge --regression`). Do not treat the ICML challenge as an active
target; treat it as the validation fixture.

## Core workflow

```
lemma audit <arxiv-id | openreview-id | paper.pdf> [--stages extract,audit,evidence,judge]
  1. extract   paper → claims.json                 (agent/extract.py)
  2. audit     claim → script → run → iterate       (agent/auditor.py)
  3. evidence  results → Trackio logbook            (agent/evidence.py)
  4. judge     logbook → verdict + rubric           (agent/judge.py)

lemma eval [workdirs...]      score audit outcomes (Weave Evaluation when configured)
lemma improve <workdir>       self-improvement loop: feedback.auto.md → re-audit → before/after
```

The `lemma` wrapper runs `python -m agent.cli`. Each stage is idempotent:
re-running a stage that already has outputs under `papers/<id>/` reuses them
(claims.json, results/audit_report.json) rather than redoing work.

## Module map (`agent/`)

| Module | Role |
|--------|------|
| `cli.py` | `lemma` entry point: `audit`, `judge`, `search`; resolves `--workdir` to an absolute path (relative paths double up inside subprocess `cwd`) |
| `extract.py` | Stage 1 — claims.json; `success_criterion` from the claim's own statement only |
| `auditor.py` | Stage 2 — writes/runs self-contained numpy scripts, iterates ≤3×; mandatory positive control; verdict-vs-metrics validator; feedback + reviewer-reference escalation (below) |
| `evidence.py` | Stage 3 — builds the Trackio logbook from results + trace; refuses to build if an ancestor dir holds a `.trackio` logbook; resolves the `trackio` binary next to `sys.executable` |
| `judge.py` | Stage 4 — automated evidence-trustworthiness judge (structure, evidence, integrity, cost, trace) |
| `papers.py` | Resolve arxiv / openreview / local PDF → text; Firecrawl markdown with PyMuPDF fallback |
| `firecrawl.py` | Evidence search (web → research index → arxiv fallback) + PDF parsing |
| `paperclip.py` | Paperclip (GXL) corpus: `lemma search` discovery + per-claim literature-context cells (X-API-Key auth) |
| `llm.py` | Multi-provider LLM wrapper: ordered `LEMMA_ENDPOINTS` + anthropic + openai; explicit 240 s HTTP timeout, no SDK retries (we own the loop); 429 Retry-After backoff; 60s error window → fallback; `LEMMA_<NAME>_MAX_TOKENS` floor for reasoning models. W&B Inference plugs in as a named endpoint (`LEMMA_WANDB_*`) |
| `traces.py` | Append-only JSONL trace logger (`llm_call`, `tool_run`, `note`) |
| `weave_ops.py` | Optional Weave shim (CoreWeave Hacks): `init_weave()` when `WANDB_API_KEY` is set; `@op` wraps stages so extract/audit/script-runs/judge + auto-patched LLM calls land in the Weave UI. Pass-through otherwise — `LEMMA_WEAVE=off` forces it off |
| `evals.py` | `lemma eval` — scorer suite (verdict-vs-reference, self-consistency, control honesty, evidence completeness, decisiveness) run as a Weave Evaluation or a local table (`--no-weave`) |
| `meta.py` | `lemma improve` — self-improvement loop: reviewer-persona LLM drafts `results/<cid>/feedback.auto.md` from a claim's failure context (latest script, summary, persisted failure tails, trace events), then re-audits. Advisory only: human `feedback.md` outranks it and `reviewer_reference.py` still wins — the loop can't edit either. `improve_<ts>.json` records before/after plus `verify_gain`: any flip *to* `supported` post-feedback is flagged for eyeballing (that's the reward-hacking direction) |
| `records.py` | Immutable per-execution attempt snapshots (`results/<cid>/attempts/<uuid>/`): input.json (claim snapshot, feedback, hashes), script, stdout/stderr, outcome.json, changed figures. `finish_attempt` refuses to overwrite; numbering advances above all existing legacy artifacts |
| `audit_room.py` | Exporter for `/room/` — builds the typed bundle (claims, attempts, checks, figures, feedback, improvements) from paper workdirs; used by `scripts/build_audit_room.py` and the local runner |
| `room_manifest.py` | Allowlist for local reruns: exact (paper, claim, attempt) rows + script SHA-256s, plus `RUN_LIMITS` (wall/cpu/output caps) |
| `room_server.py` | `lemma room` — serves `site/dist` + a token-gated rerun API on 127.0.0.1; manifest/digest verified at startup, one active job, results land under `runs/audit-room/` (never touches recorded evidence) |
| `room_sandbox.py` | macOS `sandbox-exec` profile for reruns: deny-by-default, no network, no exec outside the venv Python, reads denied under `/Users` except work/snapshot/venv, rlimits + sanitized env. `readiness_probe()` fail-closed — if isolation can't be verified the server serves recorded-only mode |

### Human-in-the-loop audit contract (`auditor.py`)

- `results/<cid>/feedback.md` — authoritative reviewer corrections, loaded at
  the start of each (re-)audit and injected into the prompt above the test plan.
- `results/<cid>/feedback.auto.md` — generated reviewer notes written by
  `lemma improve` (agent/meta.py). Injected as a separate advisory section,
  below human feedback; the loop never touches feedback.md or
  reviewer_reference.py.
- `results/<cid>/run_attempt*.failed.json` — persisted failure tails
  (crashed/rejected attempts: exit code, problems, stdout+stderr tail).
  run_attempt*.json only exists for accepted runs; the failed files are what
  `_failure_context` feeds the improve loop's reviewer draft.
- `results/<cid>/reviewer_reference.py` — hand-verified reference
  implementation (same `SUMMARY_JSON=` contract). Symmetric check: after a
  round's LLM attempts finish, the reference executes whatever the generated
  outcome was; a valid reference verdict becomes the final outcome, while a
  failed reference never overwrites a valid candidate. The report records
  `candidate_summary`, `reference_summary`, `reference_checked`, and
  `reference_disagreement` explicitly. Trace records
  `reviewer_reference_executed` plus the reason. Verified examples live in
  `scripts/ref_c*.py` and `scripts/test_escalation_gate.py`.
- `results/<cid>/attempts/<uuid>/` — per-execution provenance snapshot
  (`agent/records.py`): `input.json` (claim snapshot, feedback strings,
  run_id, attempt number, source), `script.py`, `outcome.json`,
  `stdout.txt`/`stderr.txt`, and any figures the run created or changed.
  Snapshots are never overwritten; numbering always advances above existing
  `audit_attempt*.py`/`run_attempt*.json` files.
- `_summary_problems()` validator rejects self-contradictory verdicts (failed
  OR MISSING control with a claimed verdict, NaN/None primary metrics,
  `n_measurable_points=0`, control residuals contradicting `control_pass`).
  The positive control is mandatory: `supported`/`falsified` requires an
  explicitly passing control — dropping the control is the cheap way
  to game a criterion. Rejected summaries become `inconclusive` attempts,
  logged as `summary_rejected`.
- Subset re-audits: `lemma audit <source> --stages audit --claims C2,C4,C6`
  merges into the existing `audit_report.json`; unlisted claims keep their
  prior verdicts. Attempt numbering continues across rounds.
- Concurrency: `--jobs N` audits N claims in parallel threads (claims are
  independent; each writes only under `results/<cid>/`).
- Per-stage provider pin: `LEMMA_<STAGE>_PROVIDER=<name>` (EXTRACT, AUDIT,
  IMPROVE) makes one stage prefer an endpoint without disturbing fallback
  order — e.g. keep IMPROVE on the strongest model, ride a cheaper one for
  AUDIT.

`.env` drives provider selection: `LEMMA_ENDPOINTS` (+ per-endpoint
`LEMMA_<NAME>_API_KEY/_BASE_URL/_MODEL`), `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
plus optional `FIRECRAWL_API_KEY`. Values in `.env` override the shell
environment (`load_dotenv(override=True)`).

## Working with paper directories

Agent-generated audits land in `papers/<new-id>/`. A complete workdir contains:

- `meta.json` — curated paper metadata (slug, title, source links, blurb, xylo labels); required for the registry/site pipeline
- `claims.json` — extracted claims
- `results/audit_report.json` — per-claim outcomes
- `results/c<k>/` — per-claim audit script `*.py`, run summaries `*.json`, figures `*.png`, `audit_summary.json`, optional `feedback.md` and `reviewer_reference.py`
- `trace.jsonl` — the run trace
- `judge_report.json` — generated by the judge stage
- `reproduce/` — paper-specific reproduction/training code (self-contained, with its own README); repo-wide tooling stays in `scripts/`
- `.trackio/logbook/` — generated logbook state (gitignored); publish with `trackio logbook publish <user/space>` from the paper dir

Untestable claims are recorded as `not_audited` with a declared reason at
extraction time — never silently skipped.

## Paper registry & site data flow

The website is fully data-driven from the paper workdirs; adding a paper
should require no hardcoded site edits:

1. Author `papers/<id>/meta.json` (slug, title, links, blurb, xylo labels).
2. `python scripts/build_paper_index.py` → `papers/_index.json` (merges meta with claims, audit report, judge verdict, trace stats, figure inventory).
3. `python scripts/build_site_data.py` → `site/src/data/papers.json` + copies figures to `site/public/papers/<slug>/figures/`.
4. `python scripts/build_trace_data.py` → `site/public/traces/<slug>.json` for the trace player (keeps `ca.json`/`icl.json` legacy aliases).
5. `python scripts/build_audit_room.py` → `site/src/data/audit-room.json` + hashed figure assets for the `/room/` Audit Room page (also runs automatically after `build_site_data.py`).
6. Commit the generated JSON/figures so Netlify builds need no Python.

Landing counters, xylophone bars, artifact links, `/papers/` index and
`/papers/<slug>/` detail pages all render from `papers.json`.

## Compute & overnight strategy

- Audits are designed to be **CPU-fast** (<~20 min by the auditor timeout) and
  numpy-first. Keep them tractable; a claim that needs a GPU says so in its
  summary rather than hanging.
- For heavier or longer jobs, HF Jobs or a VPS is appropriate. The repo ships
  VPS runbooks:
  - `VPS_HOST=user@host ./scripts/deploy_vps.sh` — rsyncs the repo **including
    `.env`** over encrypted SSH and bootstraps the venv.
  - `./scripts/run_overnight.sh <paper-source> [run-name]` — launches an audit
    inside tmux that survives logout; logs to `runs/<name>-<ts>.log`.
- A `RUNNING` state is not proof of progress and a detached submit's exit 0 is
  not proof of completion — poll the log (`tail -f`) until you see real output
  before recording a run as done.

## Secrets and credentials

- All secrets live in `.env` (gitignored) — copy from `.env.example`. It has
  `600` permissions; pre-commit hooks scan for leaked secrets on every commit.
- `deploy_vps.sh` refuses to run if `.env` has no Anthropic/OpenAI key.

## Documentation discipline

The trail is the deliverable. Document:

- What you tried (even if it failed)
- Why you tried it (your hypothesis)
- What actually happened (the result)
- What you'd do differently next time

Record these in `notes.md` / traces as you work, not at the end.

## Anti-patterns

- ❌ Cherry-pick results — report ALL runs, including failures
- ❌ Patch a failed audit into a pass — a false result is worse than "inconclusive"
- ❌ Trust the paper's hyperparameters blindly — the auditor re-derives them
- ❌ Skip the positive control — a buggy statistic must read "inconclusive", never "falsified"
- ❌ Hide compute costs — log wall time and attempt counts (the judge checks cost disclosure)
- ❌ Commit weights, large data, `.env`, or traces — all gitignored
- ❌ Copy paper code verbatim without understanding — explain what each piece does
- ❌ Rebuild a logbook over a partial state — if the evidence stage crashed mid-build, quarantine `.trackio/` (e.g. into `runs/quarantine/`) before re-running; a retry appends duplicate cells
- ❌ Hand-edit site data (artifacts, counters, xylo bars, trace tabs) — everything renders from `site/src/data/papers.json`; update the paper workdir (`meta.json`, claims, report, judge) and re-run the registry pipeline (`build_paper_index.py` → `build_site_data.py` → `build_trace_data.py`)
- ❌ Relative `--workdir` assumptions in new code — script paths must be absolute before `subprocess.run(cwd=...)` or the path doubles up
- ❌ `scipy.special.hyp2f1` for the ball-model cap integral near h=1 — catastrophic cancellation for D≥5; use the θ-quadrature form (see `scripts/ref_c2_ball_exponent.py`)
