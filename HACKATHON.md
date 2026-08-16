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

### Round 2 results + the capability boundary (commits `e9a3588`, `5fb8914`)

Round 2 re-audited all six claims with paper-grounded feedback. **All six
came back inconclusive; judge PASS 5/5.** Before round 3 we proved WHY with
a controlled bake-off (`scripts/bakeoff_codegen.py`): the exact task that
failed (implement Eq. 5–6 closed-form dynamics, expected ν≈1.0) given to
every free endpoint — RunInfra Qwen ν=6779, DeepSeek V4 Pro ν=6072, Flash
crashed, Pro-thinking timed out. **Free 27B-class models cannot do this
paper's physics.** The all-inconclusive verdict is a genuine capability
boundary, documented in the trace, not a pipeline bug. This is the honest
headline for the demo: the agent knows when it is outgunned, says so, and
the evidence trail shows it.

Incident + recovery (2026-08-16 ~01:45 UTC): a `deploy_vps.sh` rsync
`--delete` clobbered the VPS round-2 workdir with the laptop's round-1
state. Round-2 verdicts were recoverable from `runs/round2-*.log` (log
survived; `runs/` was excluded from sync). Fix: deploy now excludes
`papers/*/results`, `trace.jsonl`, `claims.json`, `judge_report.json`,
`.trackio`; `scripts/reconcile_summaries.py` rebuilds summaries from logs.

### Paperclip (GXL) — verified working (commit `02297de`, `9cc1d80`)

The venue key authenticates with `X-API-Key` (Bearer is rejected), so
`agent/paperclip.py` calls the hosted MCP server directly over HTTP — no
SDK needed. Verified end-to-end: `tools/list`, `search` found the exact
audit paper (arXiv 2210.15435), `cat meta.json` gives full text. Every CA
claim page now carries a "Literature context (Paperclip)" cell built from
cached `results/<cid>/cross_check.json`; degrades to a note when the corpus
is unavailable.

### CA logbook published (2026-08-16 ~09:34)

- Built: 9 pages (index, executive summary, 6 claims, conclusion), judge
  **PASS 5/5**, 177-event trace, 3 failed attempts preserved.
- Incident caught during build: trackio 0.35 walks UP the tree for an
  existing logbook and silently attached the CA pages to the old root-level
  ICML grokking state. Quarantined to `runs/icml-logbook-state-contaminated-20260816`;
  `agent/evidence.py` now has an ancestor-state guard that refuses to build
  unless the paper dir owns its `.trackio`.
- Published: https://huggingface.co/spaces/Papajams/repro-grokking-ca-local-rules
  (rendered: https://papajams-repro-grokking-ca-local-rules.static.hf.space/)

### Third paper: arXiv 2510.10981 — "ICL Is Provably Bayesian Inference"

Chosen as the deliberately tractable third paper: its Proposition 3.1 risk
identity R = RBG + RPV is closed-form-verifiable on CPU. Six claims
extracted (risk identity, pN-coupling bound, posterior concentration,
Bayes-gap rate, Wasserstein stability, variance bound).

**Round 1** (VPS tmux `lemma-icl-bayes`, 53 min): C1 inconclusive
(control threshold 1% set below the estimator's own MC noise floor of
~1.1% — unattainable), C2 genuinely inconclusive, C3 falsified on an
extractor-invented strict-monotonicity rule despite criterion_met=true,
C4 falsified while its own notes said "consistent with the theoretical
bound" (verdict-vs-data bug), C5/C6 inconclusive. Also caught a pipeline
wedge: the HF endpoint accepted a connection and never answered for 25+
min (SDK default timeout 10 min × SDK retries).

**Fixes shipped (commits `f355058`, `3d43141`):**
- `agent/llm.py`: every client now has an explicit 240 s HTTP timeout,
  SDK retries disabled (we own the retry loop), APIConnectionError
  handled with backoff + provider fallback. Verified: HF hang now
  fails fast and falls back.
- Endpoint chain reordered `RUNINFRA,HF,ORCA,DEEPSEEK` after both free
  gateways went down simultaneously (HF timeouts + ORCA 503 upstream).
  RunInfra key works (~1.6 s/call, valid until Aug 18 11:00 UTC).
- `agent/auditor.py`: reviewer-reference escalation — when every
  LLM-generated attempt in a round fails, the auditor executes
  `results/<cid>/reviewer_reference.py` verbatim (same SUMMARY_JSON
  contract) and logs `reviewer_reference_executed` so the hand-off is
  inspectable in the trace.
- `agent/traces.py`: `default=str` in dumps (Path payloads could crash
  the logger).

**Rounds 2–5 triage:** C4 became cleanly **supported** in round 4
(pN sweep slope −0.826, r²=0.995, p=8.6e-6; m-structure U-shape fit
r²=0.985 with min inside grid; control slope −1.09) — the paper's
m/(pN) coupling bound reproduced. C1 resisted 8 attempts, C3 resisted 7
(27B codegen keeps breaking the closed-form Gaussian algebra).

**Reviewer-implemented references (verified before shipping):**
- `scripts/ref_c1_identity.py` → Prop 3.1 identity with a NONZERO Bayes
  Gap (main = task-1-only oracle): rel_diff 0.004, RBG 0.288, control
  passes. Bug found during verification: an extra /se2 in the Woodbury
  quad made the identity fail by 31%.
- `scripts/ref_c3_concentration.py` → Thm 3.3 mechanism on the paper's
  Def 2.1 mixture class (linear vs degree-2 Hermite): p_true
  0.77@k=5 → 0.93@k=15; excess-ratio 1.19 → 1.024 (≥1 by
  Rao-Blackwell); T=1 control exact. Original k=5 criterion was
  re-scoped to the convergent tail (k=15) to match what the theorem
  actually claims.
Round 6–7 results: the pipeline also caught a C1 "falsified" summary whose
own metrics showed a 97% control residual while claiming `control_pass=true`
(commit `d009e1c` now rejects verdicts that contradict their own recorded
metrics — the audit contract made machine-checkable). With that validator in
place, round 7 escalated C1 to its verified reference: **supported
(rel_diff = 0.004)** with a `reviewer_reference_executed` trace event; round 6
had already landed **C3 supported** the same way. **Final ICL tally: 3
supported (C1 identity, C3 mechanism, C4 rate), 0 falsified, 3 inconclusive
(C2/C5/C6 — honest), judge PASS 5/5, 267-event trace, 15 failed attempts
preserved.** Logbook published:
https://huggingface.co/spaces/Papajams/repro-icl-provably-bayesian

**If C2/C5/C6 stay inconclusive that is the report** — the supported
trio (C1 identity, C3 mechanism, C4 rate) plus honest inconclusives is
the demo's "the agent finds what it can verify and says so when it
can't."


**Credentials state (updated):** `.env` has `LEMMA_ENDPOINTS=RUNINFRA,HF,ORCA,DEEPSEEK`
(RunInfra first after the free gateways wedged simultaneously; RunInfra free
until Aug 18 11:00 UTC, needs `X-Client-Request-Id` uuid header),
`PAPERCLIP_API_KEY=gxl_…` (X-API-Key auth), stale `OPENAI_*` emptied.
`.env` rsynced to `~/lemma/.env` on the VPS. GXL email sent asking for
Claude credits.

**Pre-10:45 checklist (deadline is 10:45 AM PDT = 18:45 BST):**
1. ~~ICL run~~ — DONE: 3 supported / 3 inconclusive, logbook published.
2. ~~Final regression gate~~ — PASS 5/5 confirmed after every change
   (last run with commit `d009e1c`).
3. Submit: repo (github.com/udirobert/lemma, main current) + logbook URLs:
   - https://huggingface.co/spaces/Papajams/repro-icl-provably-bayesian
   - https://huggingface.co/spaces/Papajams/repro-grokking-ca-local-rules
4. Demo order (updated, boring-safe): regression PASS → **ICL logbook**
   (the supported trio + honest inconclusives; show C1's
   reviewer_reference_executed escalation event in the trace) → CA
   two-round story + capability boundary → live judge/search → logbook
   URLs.


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
