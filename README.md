# lemma

> *An AI-scientist that audits scientific claims and distrusts itself.*

lemma is an end-to-end claim-audit agent: give it an unseen paper and it
**extracts** its checkable claims, **writes and runs honest numerical audits**,
**assembles an inspectable evidence trail**, and finally **runs a judge** that
decides — before any human looks at it — whether the trail is trustworthy.
Every LLM call and tool result lands in an append-only trace, so the reasoning
is fully auditable. A paper becomes a *self-judged, preserved-in-amber evidence
trail*: supported, falsified, or inconclusive, with the receipts attached.

Built for **re:AGENT – End to End Agentic Science** (Founders Inc, San
Francisco, Aug 15–16 2026), **Track A: Build an AI Scientist**. The repo's
first hand-driven reproduction already **passed an independent auto-judge end
to end** — see [Prior art](#prior-art--validation) below — and this build is
the agent that generalizes that workflow to any paper.

## Quick start

```bash
# 1. Environment + hooks
cp .env.example .env          # set at least ANTHROPIC_API_KEY (or OPENAI_API_KEY)
pip install -r requirements-agent.txt
pip install pre-commit detect-secrets ruff && pre-commit install

# 2. Run the safe, offline regression check (no API, deterministic)
./lemma judge --regression    # PASS expected on the grokking fixture

# 3. Audit a paper end to end
./lemma audit <arxiv-id|openreview-id|paper.pdf>

# 4. Run any stage in isolation (handy for demos)
./lemma audit <source> --stages extract,judge
./lemma search "interpretability grokking topology" --limit 6
```

## Pipeline

```
lemma audit <arxiv-id | openreview-id | paper.pdf>
  1. extract   paper → claims.json                 (agent/extract.py)
  2. audit     claim → script → run → iterate       (agent/auditor.py)
  3. evidence  results → Trackio logbook            (agent/evidence.py)
  4. judge     logbook → verdict + rubric           (agent/judge.py)
```

- **extract** — pulls every testable claim (`claims.json`, max 6, ordered by
  importance). `success_criterion` may only be derived from the claim's own
  statement — no added thresholds.
- **audit** — for each testable claim, the auditor proposes a self-contained
  numpy/scipy/matplotlib script, runs it, and iterates on failures. A
  **mandatory positive control** runs the same statistic on a case with a known
  answer, so a buggy statistic reads "inconclusive", never "falsified".
- **evidence** — assembles a Trackio logbook (executive summary → one page per
  claim → conclusion) and attaches the trace.
- **judge** — an automated evidence-trustworthiness checker (PASS /
  CONDITIONAL PASS / FAIL) scoring structure, per-claim evidence, integrity
  (failures preserved, untestable claims declared), cost disclosure, and trace
  completeness.

Traces (append-only JSONL) live in `agent/traces/<run-id>.jsonl` and inside
each paper's workdir for the judge to verify.

## Submission results (Aug 16, 2026)

Two unseen papers audited end to end overnight/overnight-adjacent on CPU,
both with self-judged evidence trails:

| Paper | Verdict | Logbook |
|-------|---------|---------|
| [arXiv 2510.10981](https://arxiv.org/abs/2510.10981) — *In-Context Learning Is Provably Bayesian Inference* | **3 supported / 0 falsified / 3 inconclusive** — the Prop. 3.1 risk identity (rel. diff 0.4%, nonzero Bayes Gap), the Thm 3.3 posterior-concentration mechanism, and the m/(pN) coupling rate (slope −0.83, r²=0.995) | https://huggingface.co/spaces/Papajams/repro-icl-provably-bayesian |
| [JMLR 22-1228](https://www.jmlr.org/papers/v22/22-1228.html) — *Grokking phase transitions in learning local rules with gradient descent* | **5 supported / 0 falsified / 1 inconclusive** — critical exponents ν=1 (1D) and ν=(D+1)/2 (D-ball), P_grok(L1)>P_grok(L2), P_grok decreasing with D (Eq 86), and bimodal grokking-time PDF; C5 inconclusive by compute scope (the CPU-auditable proxy trivializes Rule-30) | https://huggingface.co/spaces/Papajams/repro-grokking-ca-local-rules |

Both workdirs carry the full trace (267 and 305 events respectively, failed
attempts preserved) and judge verdicts of **PASS 5/5**. Honest outcomes over
green checkmarks: where the pipeline's own verdict contradicted its data, the
trace shows the correction (human-in-the-loop feedback → re-audit;
reviewer-reference escalation when LLM implementations stayed broken). The
JMLR paper's early rounds documented a genuine free-model capability
boundary (ν≈6000 vs expected 1.0 in a controlled bake-off); a later
capability-tier swap (Kimi K3, gated by passing the same bake-off) plus
hand-verified closed-form references closed the remaining claims.

## Prior art & validation

`papers/5nNNVY8NW4-grokking/` is a hand-driven reproduction that **passed the
ICML 2026 Agent Reproductions logbook judge end to end**: 5 claims audited,
including a falsification experiment, a published poster, and agent traces.
It doubles as the regression fixture (`./lemma judge --regression`) that proves
the pipeline produces judge-passing, honest trails. That completed challenge
(ICML 2026 Agent Reproductions, July 15 – Aug 2, 2026) is history; its value
now is as the validation case and demo anchor.

## Project structure

```
lemma/
├── agent/               # The AI-scientist pipeline (the build)
│   ├── cli.py           #   lemma entry point (audit / judge / search)
│   ├── extract.py       #   stage 1 — claim extraction
│   ├── auditor.py       #   stage 2 — write + run numerical audits
│   ├── evidence.py      #   stage 3 — Trackio logbook assembly
│   ├── judge.py         #   stage 4 — automated evidence judge
│   ├── papers.py        #   arxiv / openreview / PDF resolution
│   ├── firecrawl.py     #   evidence search + PDF parsing (arxiv fallback)
│   ├── paperclip.py     #   GXL corpus: discovery + literature cross-check cells
│   ├── llm.py           #   multi-endpoint LLM stack (timeouts, 429s, fallback)
│   ├── traces.py        #   append-only JSONL trace logger
│   └── traces/          #   run traces (gitignored)
├── papers/              # per-paper workdirs
│   ├── 5nNNVY8NW4-grokking/   # prior-art validation + regression fixture
│   └── <new-id>/        # agent-generated audits
├── scripts/             # deploy / overnight / regression-fixture helpers
├── data/                # synthetic / small data only (large data is gitignored)
├── .env.example         # template — real secrets live in .env (gitignored)
├── .pre-commit-config.yaml
├── HACKATHON.md         # re:AGENT build + demo plan
├── AGENTS.md            # guidance for AI agents working in this repo
└── README.md
```

## Expected outputs per paper

Each audited paper workdir contains: `claims.json`, `results/audit_report.json`,
one `results/c<k>/*.py` audit script (fully self-contained) plus `*.png`
figures and `*_summary.json` per claim, `trace.jsonl`, and a generated
`judge_report.json`. All compute is **idempotent**: re-running produces the same
structure from scratch.

## Secrets & hygiene

- Secrets live in `.env` (gitignored) — copy from `.env.example`; never commit
  the real file.
- Pre-commit hooks (`detect-secrets`, `detect-private-key`, ruff, hygiene) run
  on every commit. `.env` has `600` permissions; `deploy_vps.sh` syncs it over
  encrypted SSH for overnight runs.
