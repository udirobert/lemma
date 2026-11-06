---
name: audit-this-paper
description: >-
  Audit a scientific paper's claims end to end with lemma: extract
  testable claims, run numerical audits with a mandatory positive
  control, and judge the evidence trail. Use when the user pastes an
  arXiv id / OpenReview id / PDF and asks whether its claims hold up.
---

# Audit this paper

Lemma audits what a paper claims, with receipts. It does not summarize
or review — it picks the paper's *testable* claims, writes and runs
numerical audit scripts, and reports honest verdicts:
`supported`, `falsified`, or `inconclusive` (never silent skips).

## Choosing a tool

| Situation | Tool |
|---|---|
| "is this claim real?" / quick check / demo | `lemma_quick_audit` — first ≤5 testable claims, ~minutes, returns verdicts + evidence paths |
| Full-paper audit | `lemma_audit` — background run, poll `lemma_status` |
| "what did the audit find?" / progress check | `lemma_status` — per-claim verdicts + judge verdict + paths |
| "do I trust this trail?" | `lemma_judge` — re-runs the rubric (structure, evidence, integrity, cost, trace) |

## Contract to honor

- **Positive control is mandatory.** A verdict without a passing
  control is `inconclusive`, not `falsified`. Never present an
  inconclusive as a negative result.
- **The trail is the deliverable.** After a run, point the user at the
  workdir (`claims.json`, `results/audit_report.json`,
  `judge_report.json`, `trace.jsonl`) — not just the prose verdict.
- **Report failures.** Crashed or rejected audit attempts are results;
  do not soften them.
- Quick path is bounded (~15 min). If it times out, say so and offer
  `lemma_status`/`lemma_audit` rather than retrying silently.

## Setup the user may need

The plugin runs the pipeline shipped inside its own install — if a
call returns "lemma pipeline not found", the install is incomplete;
reinstall. The auditor's LLM needs `ANTHROPIC_API_KEY` — set it in the
plugin's credential fields (requires_env), not in chat.
