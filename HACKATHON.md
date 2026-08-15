# re:AGENT build — Lemma: an end-to-end claim-audit agent

**Event:** re:AGENT – End to End Agentic Science, Founders Inc. SF, Aug 15–16 2026.
**Track:** A — Build an AI Scientist. **Team:** solo (Udi / Papajams).

## The pitch in one paragraph

This repo already contains one hand-driven reproduction that passed the ICML 2026
logbook judge end to end (papers/5nNNVY8NW4-grokking: 5 claims audited, falsification
experiment, published logbook, poster, agent traces). The hackathon build is the
**agent that generalizes that workflow**: give it an unseen paper, and it extracts
claims, writes and runs numerical audits, iterates on failures, assembles an
inspectable evidence trail, and finally runs an automated judge that decides whether
the trail is trustworthy. Every LLM call and tool result is logged to an append-only
trace, so the reasoning is auditable — the system evaluates its own evidence before
a human ever looks at it.

## Pipeline

```
lemma audit <arxiv-id | openreview-id | paper.pdf>
  1. extract   paper → claims.json           (agent/extract.py)
  2. audit     claim → script → run → iterate (agent/auditor.py)
  3. evidence  results → Trackio logbook      (agent/evidence.py)
  4. judge     logbook → verdict + rubric     (agent/judge.py)
```

Traces: `agent/traces/<run-id>.jsonl` (append-only, attached to the logbook).

## Demo plan (by Sun 10:45)

- **Regression:** run stages 1, 3, 4 against the grokking paper's known-good
  outputs; judge must PASS.
- **Fresh run:** overnight end-to-end audit of one unseen, CPU-tractable paper.
- **Live demo:** `lemma audit <fresh-paper>` on a tiny claim with traces on screen;
  judge verdict live; published logbook URL.
- **Fallback:** pre-recorded artifacts of the overnight run if live compute flakes.

## Repo layout for the weekend

| Path | Role |
|------|------|
| `agent/` | The hackathon build (pipeline package) |
| `papers/5nNNVY8NW4-grokking/` | Prior-art evidence + regression fixture |
| `papers/<new-id>/` | Fresh paper audits (agent-generated) |
| `scripts/validate_icml_logbook.py` | ICML-challenge validator (kept; judge.py generalizes it) |
| `HACKATHON.md` | This file |

## Rules of engagement (carried over from ICML work)

- Report ALL runs, including failures. A falsified claim is a result.
- Log wall-clock and compute for every stage; the judge checks cost disclosure.
- Never trust paper hyperparameters blindly; the auditor re-derives them.
- The trail is the deliverable.
