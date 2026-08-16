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

### Kimi K3 endpoint + CA rounds 3–5 (2026-08-16 ~11:45–14:45)

User-provided Kimi K3 on a Modal serverless proxy
(`papaandthejimjams--ep-kimi-k3-server.us-west.modal.direct`, model
`moonshotai/Kimi-K3`, proxy bearer auth). Integration notes (commit
`e85c9ff`):
- Thinking mode counts reasoning tokens against max_tokens — one
  completion burned 11.7k reasoning + 2.5k content. `LEMMA_<NAME>_MAX_TOKENS`
  now acts as a floor; KIMI configured `reasoning_effort: low` + 16384.
- The Modal proxy intermittently rejects with "Duplicate request ID";
  `LEMMA_KIMI_HEADERS` sends a fresh `X-Request-Id` uuid per call.
- Endpoint hit a usage cap mid-test (user added budget); chain is now
  `KIMI,RUNINFRA,HF,ORCA,DEEPSEEK`.

**Bake-off gate PASSED**: on the exact task every free 27B failed
(CA closed-form 1D dynamics, expect ν=1.0), Kimi K3 produced
**ν = 0.9911** in one generation (92s). Qwen/DeepSeek had given ν≈6000–6700.

**CA round 3** (Kimi primary, ~33 min): C1 now **SUPPORTED** —
ν = 0.9983 vs paper's 1.0, control ν = 1.0000 exact, 745 fit points.
The headline claim of the JMLR paper is reproduced. Others: C2
"falsified" but its own control reproduced the (D+1)/2 series EXACTLY
(main setup bug, not physics); C3 inconclusive (regime saturation
L1=1.000 vs L2=0.980); C4 "falsified" with zero usable observations
(P=0 at all D, NaN slope — verdict void); C5 honest negative
(perceptron proxy trivializes Rule-30's 8-triplet lookup; tensor-network
claim is gpu-small, out of scope); C6 "falsified" despite its metrics
showing the predicted bimodality (detector thresholds miscalibrated).

**Validator v2** (commit `9816015`): verdicts built on NaN primary
metrics or n_measurable_points=0 are now rejected as contradictions
(C4's failure mode), added after round 3 exposed it.

**Round 4** (completed, ~20 min): C3 now **SUPPORTED** —
P_grok(L1)=0.205 vs P_grok(L2)=0.000 in the hard regime (D=10, eps=1.01).
C2 inconclusive again: control exact (ν = 1.5/3.0/5.5) but main ν=0.5000
at all D with r²=0.9999 — a perfectly fitted power law with the WRONG
exponent means the simulated dynamics followed (1−h)^{1/2}, i.e. the LLM
kept re-deriving the 1D ball instead of the D-dimensional one. C5
resolved as scope-limited inconclusive (proxy cannot exhibit grokking by
construction; tensor-network claim out of CPU scope). C4/C6 falsified by
broken implementations again (eps saturation / miscalibrated detector),
which is what motivated the reviewer-reference escalation below.

**Reviewer references + escalation gate v2** (commits `5ad6c04`,
`b5667e4`, `6b0c527`): three hand-written closed-form references, each
verified locally AND on the VPS before deployment:
- C2 `ref_c2_ball_exponent.py` — Eq 23 via the paper's exact large-N
  gradient flow (Eqs 20–21); Eq 22's cap integral evaluated by
  θ-quadrature because `scipy.hyp2f1` cancels catastrophically near h=1
  for D≥5 (this was the hidden bug behind rounds 3–4's garbage
  exponents). Exponent fitted in h-space, where the asymptotic law is
  exact: **ν = 1.500 / 2.9998 / 5.4993** vs targets 1.5 / 3.0 / 5.5.
- C4 `ref_c4_pgrok.py` — Eq 86 (Appendix B.1, λ₁=0) evaluated in
  closed form with the Eq 82 coefficients and Eq 84–85 Gaussian laws;
  at common eps=1.05, P_grok(D=2,5,10,20) = [0.857, 0.311, 0.017, 5e-6],
  strictly decreasing; 20k-draw MC cross-check max err 0.042.
- C6 `ref_c6_bimodality.py` — Monte Carlo over the paper's zeroth-order
  grokking-time PDF (Eq 42 fast branch + Eq 47 Dirac peak, Fig 9
  params D=5, eps=2, λ₂=0.01); all four bimodality tests and the D/λ₂
  dependence pass.
- The escalation gate in `agent/auditor.py` now fires whenever the round
  did NOT end "supported" (previously only on "inconclusive") — a
  falsified verdict from a broken LLM implementation gets the same
  authoritative-reference re-examination. Offline smoke test covers both
  branches (`scripts/test_escalation_gate.py`).

**Round 5** (completed, ~9 min, tmux `lemma-ca-round5b`): all three
remaining testable claims **SUPPORTED**.
- C2 via reviewer reference (ν = 1.5000 / 2.9998 / 5.4993 vs targets
  1.5 / 3.0 / 5.5; LLM attempts still produced the wrong dynamics shape).
- C4 via the LLM's OWN attempt 6 — the round-5 feedback (Eqs 82/86 from
  Appendix B.1) was sufficient guidance: P_grok(D=2,5,10,20) =
  [0.7295, 0.176, 0.032, 0.0125] strictly decreasing at common eps=1.005.
- C6 via reviewer reference after LLM attempts failed again (all four
  bimodality tests pass; slow-time rel err 1e-16).
  Also surfaced and fixed en route: a relative `--workdir` path doubled
  up inside subprocess invocations (every script run exited 2);
  `agent/cli.py` now resolves it to absolute (commit `fc3ae36`).

**Final CA tally: 5 supported / 0 falsified / 1 inconclusive** (C5,
scope-limited by compute: the auditable proxy trivializes Rule-30 and
the paper's tensor-network regime is gpu-small). Logbook re-published:
https://huggingface.co/spaces/Papajams/repro-grokking-ca-local-rules
Judge: PASS 5/5 (353 trace events after the clean rebuild, 9 failed
attempts preserved in full).

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
2. ~~CA run~~ — DONE: 5 supported / 0 falsified / 1 inconclusive (C5,
   compute scope); final logbook published (judge PASS 5/5 on the
   clean rebuild, 353-event trace).
3. ~~Final regression gate~~ — PASS 5/5 confirmed after every change
   (last run with commit `60b90ac`).
4. Submit: repo (github.com/udirobert/lemma, main current) + logbook URLs:
   - https://huggingface.co/spaces/Papajams/repro-icl-provably-bayesian
   - https://huggingface.co/spaces/Papajams/repro-grokking-ca-local-rules
5. Demo order (updated, boring-safe): regression PASS → **ICL logbook**
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

**Published artifacts (final, 2026-08-16 ~16:40 BST):**
- Logbook (ICL): https://huggingface.co/spaces/Papajams/repro-icl-provably-bayesian
- Logbook (CA): https://huggingface.co/spaces/Papajams/repro-grokking-ca-local-rules
- Evidence dataset (ICL): https://huggingface.co/datasets/Papajams/repro-evidence-icl-provably-bayesian
- Evidence dataset (CA): https://huggingface.co/datasets/Papajams/repro-evidence-grokking-ca-local-rules
  (scripts, summaries, figures, feedback, references, trace; via
  `scripts/publish_evidence_dataset.py papers/<id>`)
- Kaggle: deliberately skipped (no CLI/API keypair available; audit trails
  fit HF datasets, not Kaggle notebooks).

## Post-submission roadmap

### Frontend (updated 2026-08-16 evening)

Public surface: the Trackio logbooks (claim pages, figures, trace), the HF
evidence datasets, and the landing page. The landing is the entry point and
grows a tier at a time — each tier gated on measured interest, not guesses.

**Tier 1 — live at lemmabio.netlify.app.**
Astro static site, the product framed as its own demo:

- playable claim xylophone (12 bars = 12 audited claims, Web Audio)
- scroll journey: story beats → pinned append-only trace reveal → artifacts
- **trace player** (`/#replay`): replays the real `trace.jsonl` of either
  paper line by line — model calls, script writes, failed attempts,
  escalations, verdict flips across rounds — with play/pause + scrubber.
  Data regenerated by `scripts/build_trace_data.py` (evidence-stage
  bookkeeping aggregated; audit events verbatim, failures preserved).

**Tier 2 — "try it yourself": paste-an-arXiv-id extract demo.**
Users run step 1 of the pipeline on their own paper. Shape:

- frontend form (arXiv id) → Modal web function (we already have an
  authenticated Modal account) → extract stage only in a sandboxed
  container → returns claims + success criteria in ~1-3 min
- full single-claim audits as a metered/queued follow-up (real audits are
  20-40 min; async job with trace streaming, per-session temp workdirs)
- hard constraints: per-IP rate limit, container resource caps + timeout,
  cost ceiling per run

**Tier 3 — the product: audit-as-a-service.**
Claim ledger across papers, per-user job ledger, trace explorer with
round-over-round verdict diffs, BenchFlow-style verifier API over the
evidence trail, billing for metered audits. This is the "control plane"
story — weeks of work, gated on Tier 2 signals.

### Interest indicators (gate each tier on these)

Instrument now (all free/zero-backend):

| Indicator | What it measures | How |
|---|---|---|
| **Waitlist signups** | concrete demand, contactable | `?waitlist` form on the landing → email capture (Formspree/Google Form) |
| **"Request an audit" clicks** | purest intent signal | button on the landing → same form; count clicks in analytics |
| **Trace player engagement** | do people actually watch? | Plausible custom events: play started, scrub used, run switched, finished a run |
| **GitHub stars** | public social proof | star CTA on the landing + repo; API count |
| **HF dataset/space views** | research-community reach | HF stats pages |
| **Logbook revisits** | do people come back? | Space view counts over time |

Gate decisions (review weekly):

- **Build Tier 2** when: ≥ 100 waitlist signups, or ≥ 200 "request an
  audit" clicks, or ≥ 50 GitHub stars. (Any one of the three.)
- **Build Tier 3** when: ≥ 500 waitlist signups AND ≥ 5 concrete
  conversations with teams who would pay; stars become secondary.
- If Tier 1 engagement is flat after two weeks of the X/LinkedIn/Discord
  push, the problem is distribution, not features — don't build Tier 2.

### Publishing cadence (agreed)

Every completed audit publishes three artifacts: the logbook Space
(human-readable), the evidence Dataset (machine-reusable, failures
included), and a row in the trace-player catalog. Kaggle is not a target.

**Local env note (2026-08-16):** the laptop's old `.browser-use-env` was
deleted; pipeline deps now live in the repo `.venv` (Python 3.12.13, built
from `requirements-agent.txt`); `./lemma` prefers it automatically. The VPS
has its own `.venv` from `scripts/bootstrap_vps.sh`.
