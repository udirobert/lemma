# re:AGENT build — Lemma: an end-to-end claim-audit agent

**Event:** re:AGENT – End to End Agentic Science, Founders Inc. SF, Aug 15–16 2026.
**Track:** A — Build an AI Scientist. **Team:** solo (Udi / Papajams).

## The pitch in one paragraph

Lemma is an **AI scientist that audits scientific claims and distrusts itself**.
Give it an unseen paper and it extracts testable claims, writes and runs
numerical audits with a mandatory positive control, assembles an inspectable
evidence trail, and runs its own judge that decides whether that trail is
trustworthy — all before a human looks at it. Every LLM call and tool result is
appended to a trace, so nothing is hidden: supported, falsified, and
inconclusive are all reported as results. This repo already contains one
hand-driven reproduction that **passed an independent auto-judge end to end**
(`papers/5nNNVY8NW4-grokking`, 5 claims audited, a falsification experiment,
published logbook, poster, agent traces); the build generalizes that workflow
to any paper.

How this lands on Track A's criteria:

| Track A asks | How Lemma answers it |
|--------------|----------------------|
| Gather evidence, use tools/databases | arxiv / openreview / PDF resolution + Firecrawl search (web → research index → arxiv fallback); optional Paperclip as an evidence source |
| Generate & test hypotheses | extract claims → auditor writes & runs scripts (≤3 attempts, positive control mandatory) |
| Produce structured output | claims.json + Trackio logbook + judge_report.json |
| Make reasoning easy to inspect | append-only JSONL trace attached to the logbook; judge scores evidence completeness |

## Pipeline

```
lemma audit <arxiv-id | openreview-id | paper.pdf>
  1. extract   paper → claims.json           (agent/extract.py)
  2. audit     claim → script → run → iterate (agent/auditor.py)
  3. evidence  results → Trackio logbook      (agent/evidence.py)
  4. judge     logbook → verdict + rubric     (agent/judge.py)
```

Traces: `agent/traces/<run-id>.jsonl` (append-only, attached to the logbook).

## Demo plan (by Sun 10:45) — in this exact order, boring-safe

1. **Regression (offline, deterministic, no API):** `./lemma judge --regression`
   on the grokking fixture → **PASS**. This is the safety net; run it first.
2. **Fresh run (centerpiece):** overnight end-to-end audit of one unseen,
   CPU-tractable paper (VPS: `./scripts/run_overnight.sh <source>`) → show
   claims, audit scripts/figures, judge verdict, trace.
3. **Live (if compute cooperates):** `lemma audit <fresh-paper>` on a tiny
   claim with the trace on screen; watch the judge produce a verdict live;
   show a published logbook URL.
4. **Fallback:** if step 2/3 flake, land on the pre-recorded artifacts.

**Artifacts a demo must have ready:** (a) the regression PASS output, (b) the
overnight paper's `claims.json`, `results/`, `judge_report.json`, `trace.jsonl`,
(c) one live tiny audit logged to a trace, (d) a published logbook URL.

## Repo layout

| Path | Role |
|------|------|
| `agent/` | The build — pipeline package (see AGENTS.md for the module map) |
| `papers/5nNNVY8NW4-grokking/` | Prior-art validation + regression fixture |
| `papers/<new-id>/` | Fresh paper audits (agent-generated) |
| `scripts/` | Deploy + overnight + regression-fixture helpers |
| `README.md` | Public face (re:AGENT pitch + quickstart) |
| `AGENTS.md` | Guidance for coding agents working in-repo |
| `HACKATHON.md` | This file |

## Sponsor usage (creative-but-honest, grounded in their actual docs)

| Sponsor | Role in Lemma | Status |
|---|---|---|
| **Paperclip (GXL)** co-host | Literature stage: `lemma search` discovery (11M papers + FDA + trials + UniProt/PDB/ChEMBL) and per-claim `cross_check()` "Literature context" logbook cells. Context only — never alters audit verdicts. | Code shipped (`agent/paperclip.py`, commit `26639a5`); needs `PAPERCLIP_API_KEY` from venue (free for participants) or their install.sh / MCP server. Degrades to Firecrawl/arXiv until then. |
| **Anthropic** co-host | Claude as the audit-script generator for hard claims (stronger codegen than 27B Qwen). `LEMMA_PROVIDER=anthropic` already wired in `agent/llm.py`. | Code ready; needs venue Claude credits in `.env`. Best used for a targeted round-3 on C1/C2 if 27B keeps failing the closed-form dynamics. |
| **Modal** sponsor | GPU compute backend for `gpu-small` claims (C5 Rule-30) if CPU audits cannot reproduce it; also a demo of "the agent knows when a claim needs real compute". | `MODAL_*` creds already in `.env`. Only if C5 reaches round 3 — not forced. |
| **BenchFlow** co-host | Their runtime is "tasks, harnesses, agents, verifiers" — the conceptual twin of our judge. Pitch language: *the judge is a BenchFlow-style verifier over the agent's evidence trail.* Full task packaging only if submission slack exists. | Research done; no code integration before 10:45. |
| **Benchling / LatchBio / Boltz / Strand** sponsors | Not applicable to a theory-paper audit. Benchling's "auditable provenance: inputs, outputs, model versions, timestamps" mirrors our trace+judge story → one-line demo name-drop. | Name-drop only. |
| **future.bio / Arc / Biohub** co-hosts | Audience/context alignment (agentic science framing); no tool integration fits this build. | None. |

**Principle:** every sponsor touch must be load-bearing in the pipeline, not a
logo slide. Paperclip = evidence gathering, Anthropic = better auditor brain,
Modal = honest compute escalation, BenchFlow = verifier framing.

## Options we may fold in before judging (only if time allows)

- **One falsification headline** in the overnight paper (judges reward honest
  negative results).

## Rules of engagement (carried over from the ICML work)

- Report ALL runs, including failures. A falsified claim is a result.
- Patch nothing into a pass; a failed control is "inconclusive", never "falsified".
- Log wall-clock and compute for every stage; the judge checks cost disclosure.
- Never trust paper hyperparameters blindly; the auditor re-derives them.
- The trail is the deliverable.

## Milestone log (checkpoint so this reopens cleanly)

### Day 1 → Day 2 overnight (2026-08-16 00:40 BST / 16:40 PST)

**Status: overnight audit run in progress on the VPS; all local work committed.**

Built, tested, and shipped (commits `041ca5b` → `8e856a7`):

1. `agent/` pipeline: `traces.py` (append-only JSONL), `llm.py` (multi-endpoint
   stack with 429/Retry-After + 60s error-window fallback), `papers.py`
   (arxiv/openreview/local-PDF resolution), `extract.py`, `auditor.py`
   (≤3 attempts/claim, positive control mandatory, figures to
   `results/<slug>/`, numpy-safe JSON), `evidence.py` (Trackio logbook
   assembly), `judge.py` (5-point rubric: structure / evidence / integrity /
   cost disclosure / trace), `cli.py` + `./lemma` wrapper.
2. Firecrawl integration (`agent/firecrawl.py`): keyless web search working,
   research index + PDF-markdown when `FIRECRAWL_API_KEY` set, arXiv fallback.
   CLI: `./lemma search "<query>" [--research]`.
3. Regression fixture: `papers/5nNNVY8NW4-grokking/claims.json` +
   `results/<cid>/audit_summary.json` adapted from the July hand-driven work
   (`scripts/build_regression_fixture.py`); `./lemma judge --regression` →
   **PASS 5/5** locally AND on the VPS bootstrap. Negative test (empty trace,
   missing figures) correctly → **FAIL**.
4. Provider stack live-tested: primary HF public Qwen3.8-27B endpoint
   (~4.6s, thinking OFF via `LEMMA_HF_EXTRA_BODY={"reasoning_effort":"none"}`
   — without it content comes back empty), fallback OrcaRouter
   `qwen/qwen3.8-27b-free` (~15–60s, `reasoning:false`). Stale shell-env
   gateway (out of credits) neutralized by emptying `OPENAI_*` in `.env`
   (dotenv override=True).
5. VPS (`nuncio-vultr`, shared box: Coolify + Traefik on 80/443/8000, ~30
   containers, 13 pm2 services — we touch none of them): deployed to `~/lemma`
   via `scripts/deploy_vps.sh` (rsync incl. `.env` over SSH, `--delete`
   scoped to that dir only); runner `scripts/run_overnight.sh` uses tmux +
   `nice -n 19` + tee'd log.

**Overnight run (launched 23:15 UTC on VPS):**
`./scripts/run_overnight.sh papers/_staging/jmlr-22-1228-ca-grokking.pdf
ca-grokking` → tmux session `lemma-ca-grokking`,
log `~/lemma/runs/ca-grokking-20260815-231542.log`,
workdir `~/lemma/papers/jmlr-22-1228-ca-grokking/`.
Paper: "Grokking phase transitions in learning local rules with gradient
descent" (Žunkovič, Ilievski; JMLR 25 (2024) 1–52).
Extraction produced 6 claims, all testable (5× cpu-fast, 1× gpu-small Rule-30
CA): C1 critical exponent (1D exponential model), C2 critical exponent
(D-dim uniform ball), C3 L1↑→grokking-probability↑, C4 D↑→probability↓,
C5 Rule-30 CA grokking, C6 grokking-time bimodality.
Progress at 00:40: C1 audited (falsified @112s), C2 on attempt 3,
0 provider fallbacks, 2 script failures preserved in trace.

**Known issue:** the VPS sshd intermittently stops answering banner exchange
(box stays alive — Traefik/Coolify respond on 80/443/8000). The tmux run is
unaffected; artifacts persist to disk. Recovery sentinel on laptop: PID 61303,
log `runs/ssh-sentinel.log` (probes every 15 min; note: it exited on the
first successful probe — recheck `runs/ssh-sentinel.log` first thing).

### Round 1 results (completed ~23:55 UTC, 39 min total)

All four stages ran unattended: 6 claims → 3 falsified / 3 inconclusive /
0 supported; judge **PASS 5/5** (71 trace events, 10 LLM calls, 39 tool runs,
3 failed attempts preserved). Results synced to laptop and committed (`9af2842`).

Triage (human review of metrics vs verdicts):
- **C1** falsified: degenerate exponent fit (ν≈1e-14) — wrong dynamics/fit
  window in the generated script, not physics.
- **C2** inconclusive: fits returned N/A; simulation never produced curves.
- **C3** inconclusive: control failed (P_grok=0 for both settings) — broken
  training procedure, correctly flagged by the positive-control rule.
- **C4** falsified but metrics SUPPORT the claim (P=1.0 at D=1, P=0 for D≥5,
  monotone) — statistical-resolution issue, verdict logic needs fixing.
- **C5** inconclusive: Rule-30 setup not per Sec 4.2 (train error never ~0).
- **C6** falsified but metrics show clear bimodality (clusters 0.60 vs 2.83,
  analytic 3.23) — detector threshold was wrong.

This is the human-in-the-loop moment the judges want: two of three
"falsified" verdicts were the agent disagreeing with its own data, caught on
review, documented in `results/<cid>/feedback.md`, not patched silently.

### Round 2 (in progress)

`agent/auditor.py` now ingests `results/<cid>/feedback.md` as authoritative
reviewer corrections (commit `9af2842`): attempt numbering continues,
re-audited outcomes merge into `audit_report.json`. Six feedback files written
with paper-grounded corrections (Eq. 5–6 closed-form dynamics for C1, ball-model
gradient flow for C3/C4, Sec 4.2 perceptron map for C5, verdict-logic fixes for
C4/C6). Re-audit runs on the VPS:
`./lemma audit papers/_staging/jmlr-22-1228-ca-grokking.pdf --stages audit --claims C1,C2,C3,C4,C5,C6`

### Reopen checklist (Sunday morning)

1. `cat runs/ssh-sentinel.log` → when did SSH recover? `ssh nuncio-vultr`
   may need patience if sshd is again in banner timeout.
2. On VPS: `cd ~/lemma && tail -50 runs/ca-grokking-20260815-231542.log`,
   inspect `papers/jmlr-22-1228-ca-grokking/{trace.jsonl,results/audit_report.json,judge_report.json}`.
3. Sync results back: from laptop `rsync -az nuncio-vultr:lemma/papers/ papers/`
   (mind `--delete` — do NOT use it here; or commit on VPS and `git fetch`).
4. Triage: rerun failed/inconclusive claims (`./lemma audit <pdf> --stages
   audit` reuses `claims.json`); a falsified claim is a headline, keep it.
5. Publish: assemble logbook (`--stages evidence`) then
   `trackio logbook publish <HF_USERNAME>/repro-ca-grokking-local-rules`;
   validate with `./scripts/validate_logbook.sh` if the ICML shape is kept,
   else the built-in judge report stands as the evidence gate.
6. Demo artifacts (boring-safe order): regression PASS → overnight claims +
   figures + judge verdict + trace → live tiny audit if time → logbook URL.
7. Stretch (only if 1–6 land): Adaption AutoScientist as a delegated training
   tool for a finetuning-style claim; Benchling name-drop (their Model Hub
   "auditable provenance" language mirrors our judge + trace story).

**Credentials state:** all in `.env` (gitignored): `LEMMA_ENDPOINTS=HF,ORCA`
+ per-endpoint key/URL/model/extra-body, `LEMMA_HF_EXTRA_BODY` thinking-off,
`OPENAI_*` emptied to kill the stale gateway. HF/Modal/Trackio creds as
before. `.env` also rsynced to `~/lemma/.env` on the VPS.

**Venue grab-list (check-in, Day 2 morning):**
1. **Paperclip API key** (free for participants) → `PAPERCLIP_API_KEY=pk_...`
   in `.env`; redeploy to VPS; re-run evidence stage so every claim page gets
   a Literature-context cell before publishing.
2. **Anthropic / Claude API credits** → `ANTHROPIC_API_KEY` in `.env`
   (already first in the provider order if `LEMMA_PROVIDER=anthropic`, or
   just set the key and it joins the stack).
3. Optionally: Firecrawl key (`FIRECRAWL_API_KEY`) for the 41M-paper research
   index + cleaner PDF parsing.

**Local env note (2026-08-16):** the laptop's old `.browser-use-env` was
deleted; pipeline deps now live in the repo `.venv` (Python 3.12.13, built
from `requirements-agent.txt`); `./lemma` prefers it automatically. The VPS
has its own `.venv` from `scripts/bootstrap_vps.sh`.
