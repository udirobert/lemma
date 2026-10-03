# Lemma Lab — a verification-driven AI lab for scientific claims

**Hack-Nation × Databricks Omnigent · 7th Global AI Hackathon · Agentic Scientific Discovery**

> Agentic labs are about to flood science with claims nobody can verify.
> Lemma is the layer that checks them — turning a paper into testable
> claims, running controlled numerical audits, and preserving an evidence
> trail a judge can independently inspect.

**Demo (2 min):** `lemmabio.netlify.app/demo/` — a mission-control replay
of the full discovery loop below. Every frame maps to a real trace event.

---

## The question

**arXiv 1912.07242 — "More Data Can Hurt for Linear Regression:
Sample-wise Double Descent"** — a famous ML-theory result: for
ridgeless regression, test error peaks at n=d. Perfect for the challenge:
closed-form predictions, CPU-cheap numerical tests, and unexplored
boundary claims the paper never checked.

## What happened (the complete loop)

Question → Evidence → Hypothesis → Experiment → Result → Updated decision.

| Beat | What the lab did | Result |
|---|---|---|
| Question | Extract stage → 6 testable claims, all cpu-fast | `claims.json` |
| Evidence | Auditor audited C1 (peak at n=d) + C2 (risk approximation) | C1 **inconclusive** (3× 20-min timeouts, d=1000 MC too heavy); C2 **falsified** (bias err 24.2%, control passed) |
| Decision | **Human reviewer** filed `feedback.md`: the claims are dimension-agnostic — test at d≤300 | Orchestration-level intervention, preserved in the trail |
| Evidence | Re-audit with feedback loaded | C1 **supported** (peak exactly at n=200=d, control passed, 87 s); C2 **supported** (≤2% err at d=150) — round-1 falsification was MC noise, not a paper flaw |
| Hypothesis | Proposer → director promoted P1 to claim **C7** (`kind: generated-hypothesis`): the risk valley sits at γ\*=0.90 | `proposals.json`, `claims.json` |
| Experiment | Audit of C7 — MC valley vs closed form, control = σ=0.5 (analytic valley at γ\*=0.75) | **supported in 14 s** — valley at γ=0.90, 4.1% err, control 1.7% |
| Result | The lab verified the paper's headline claim *and* produced a new verified result the paper didn't test | `results/c7/` |
| Updated decision | Director's report: audit proposal P2 (independent bias check), extend to C3–C6 — **P2 parked behind human approval gate** | next steps owned by the human |

**Judge: PASS 5/5** — structure, evidence, integrity, cost disclosure,
trace. 156 trace events. 7 failed attempts preserved, including a
1-second `NameError` crash.

## The catch (our favorite moment)

C2's round-2 bias check was **circular** — the script hard-coded
E[β̂]=γβ and verified the formula against itself. The proposer caught it,
the director disclosed it in the final report, and the fix (P2) waits for
human approval rather than running silently. This is the product: an AI
lab that distrusts itself.

## Omnigent orchestration (30%)

`omnigent/lab/` — a YAML agent bundle:

- **lemma-lab director** (`config.yaml`) — plans, triages claims by
  expected learning vs cost, dispatches specialists, promotes proposals
  into first-class claims, recovers from sub-agent death by reading
  workdir state.
- **auditor** (`agents/auditor/`) — dispatches `./lemma audit --claims`
  and reports verdicts.
- **proposer** (`agents/proposer/`) — generates claim variants from
  audited results; output promoted only by the director.

Specialist handoffs are visible live in session
`a475c5592cf44a82b93b42d126b441a8` (and the failed first session
`c62c9b43ce82406bb903af0ff0a605aa` — kept, because failure is evidence).
The deterministic engine (`./lemma audit`) stays the substrate: agents
route *decisions*; the engine produces *evidence*.

**Human gates:** `feedback.md` (authoritative reviewer corrections
loaded at audit start) and the P2 approval hold — consequential next
steps wait for a human.

## Discovery acceleration & learning (20%)

The bottleneck we attacked: **verification is the rate limiter of
agentic science.** Generating hypotheses is cheap; knowing which are
true is not.

Measured on this run:

| Metric | Value |
|---|---|
| Unguided audit round | 65+ min (3 timeouts, 1 falsified) |
| Reviewer-guided re-audit | 4.5 min → 2 verdicts (**~14×**) |
| Generated-hypothesis audit | **14 s** |
| Paper → verdicts + verified new result | ~2h13m wall incl. provider outages |

The 14× is real but modest in scope — it's the right frame: the biggest
acceleration is *knowing what to test cheaply*, and the loop itself
surfaced that lesson (the agents kept choosing d=1000 until a reviewer
said d≤300).

## Scientific rigor (15%)

- **Mandatory positive control** on every verdict — a failed control
  yields `inconclusive`, never `falsified`.
- **Failed attempts preserved** (7 this run), timeouts and crashes
  included.
- **Compute cost disclosed** — wall time + attempt counts, checked by
  the judge.
- **Agent-generated hypotheses labeled** (`kind`, `source` in
  claims.json).
- **Uncertainty preserved** — inconclusive is a first-class verdict;
  C3–C6 declared `not_audited`, never silently skipped.
- **Immutable attempt snapshots** — every script, stdout/stderr, input,
  and outcome under `results/<cid>/attempts/<uuid>/`.

## Creativity & responsibility (10%)

- The demo tells the failure story, not just the win: a provider 402
  killed session 1 mid-run, two sub-agents crashed, and the loop still
  completed — resilience is part of the product.
- The circular-bias-check disclosure happened *inside* the agent report,
  not in a footnote we added later.

## Validation still needed (stated honestly)

- C2's supported verdict rests partly on a circular bias check — P2
  (independent MC derivation of E[β̂]) is filed and gated.
- C1 verifies peak *location* and non-monotonicity; the peak *magnitude*
  diverges at n=d (as in the paper's own Fig. 1).
- C3–C6 are declared, not audited.
- Omnigent 0.16.0 is alpha: sub-agent instability and a client
  subscribe-after-post race were observed and worked around (see
  `notes.md`).

## Where's the evidence

| Artifact | Path |
|---|---|
| Discovery replay | `lemmabio.netlify.app/demo/` (source: `site/src/pages/demo.astro`, data: `site/public/demo/double-descent.json`) |
| Agent bundle | `omnigent/lab/` |
| Full paper audit page | `/papers/double-descent/` |
| Claims + generated hypothesis | `papers/arxiv-1912.07242/claims.json` |
| All verdicts + metrics | `papers/arxiv-1912.07242/results/` |
| Proposals (P1 promoted, P2 gated) | `papers/arxiv-1912.07242/proposals.json` |
| Reviewer interventions | `papers/arxiv-1912.07242/results/c1/feedback.md`, `c2/feedback.md` |
| Judge verdict | `papers/arxiv-1912.07242/judge_report.json` |
| Full event trace | `papers/arxiv-1912.07242/trace.jsonl` |
| Run record incl. failures | `notes.md` (2026-10-03 entry) |

## Reproduce

```bash
# deterministic engine — same run, no agents
./lemma audit 1912.07242 --stages extract,audit,evidence,judge

# agentic lab — Omnigent orchestrates the loop
uv tool install omnigent          # then configure a provider
omni run omnigent/lab -p "Run the discovery loop on arxiv 1912.07242 ..."
```

## Prior art & why this is different

Verification ≠ reproduction ≠ consensus. Lemma asks whether a claim is
*true*, re-deriving the test itself with a control that keeps a broken
measurement from producing a false falsification.

- **Reproduction benchmarks** — PaperBench (OpenAI), CORE-Bench
  (Princeton), RECLAIM (arXiv:2609.28850): can an agent re-run what the
  authors shipped? Fidelity, not validity.
- **Consistency auditing** — ReAgent (arXiv:2609.22111): does an
  agent-written paper match its repo? Consistency, not truth.
- **The verification gap** — "Autonomous Research Agents: A Survey of
  AI Scientists and the Verification Gap" (arXiv:2608.05179): of 24
  runnable AI-scientist systems, none demonstrates an externally
  validated in-loop oracle. That missing piece is what lemma's
  control-gated auditor + evidence judge are.

## Next experiment

1. Un-gate P2 (independent MC bias check) — one approval click.
2. Audit C3–C6 (underparameterized regime + the two lemmas).
3. Take the loop to a corpus: `lemma search` → planner ranks claims by
   testability × impact → parallel audits under a compute budget.
4. Cross-model bake-off: two auditor sub-agents on different harnesses,
   disagreements surfaced to the human gate.
