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

## Options we may fold in before judging (only if time allows)

- **Paperclip as an evidence source** in the gather stage — a first-party host
  tool, strengthens the "tools/databases" criterion.
- **One falsification headline** in the overnight paper (judges reward honest
  negative results).

## Rules of engagement (carried over from the ICML work)

- Report ALL runs, including failures. A falsified claim is a result.
- Patch nothing into a pass; a failed control is "inconclusive", never "falsified".
- Log wall-clock and compute for every stage; the judge checks cost disclosure.
- Never trust paper hyperparameters blindly; the auditor re-derives them.
- The trail is the deliverable.
