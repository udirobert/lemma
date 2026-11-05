---
title: Beam Audit Grader
emoji: 🎯
colorFrom: yellow
colorTo: green
sdk: static

pinned: false
license: cc-by-4.0
tags:
  - benchmark
  - evaluation
  - reflection-beam
  - agentic-science
  - verifiable-rewards
---

# Beam Audit Grader — interactive verifiable-reward demo

Paste an audit script. The grader runs it in a sandbox, extracts
`SUMMARY_JSON`, and scores it against the hand-verified answer key.

**Reward ladder:**
- 0.00 — crash / no parseable summary / metrics missing
- 0.10 — honest inconclusive (broken control, reported as such)
- 0.25 — self-contradictory verdict or fabrication guard
- 0.50 — reference metrics recovered but control failed/missing
- 0.70 — reference recovered but under-claimed
- 1.00 — reference recovered + control passed + status supported

No LLM in the reward path. Reward-hacking guards cap at 0.25:
metric == answer-key literal in source, or decisive metric with no computation.

Built from [Lemma](https://github.com/udirobert/lemma) for the Reflection × Hugging Face
Build-with-Beam challenge.
