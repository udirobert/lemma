# Sealed but not certain: tamper-evident evidence trails and a test-retest study of an autonomous claim-auditing agent

**Udi Ngethe** — Independent

**With** Apart Research

**Abstract**

AI agents are beginning to produce scientific audit verdicts at scale, but the logs of what they did are written by the agents' own operators and can be silently rewritten — and even an untampered trail does not tell you whether the verdict in it is right. We address both problems on a live claim-auditing agent (Lemma), which reads a paper, extracts testable claims, writes and runs numerical audit scripts under a mandatory positive control, and emits a verdict (supported / falsified / inconclusive) per claim. **New work:** (1) a tamper-evident evidence trail — each trace record is hash-chained and every artifact is hashed into a manifest, collapsing the whole trail to a single 256-bit "evidence head"; (2) a red-team evaluation of that trail under four tamper scenarios, including a well-formed forged record appended with a valid chain and a full re-seal, which delineates exactly where an external anchor is required; and (3) a test-retest calibration study over 48 audited claims in 9 paper workdirs, including two same-paper, same-claims pairs audited by different model backends, where claim-level verdict agreement was only 6/13 (46%). A worked example traces one flip to the audit script's fit window, not the paper. Integrity and correctness are separable problems for agentic science; our layer provably delivers the first, and our measurements show how far the second still has to go.

*Track 2 (Trust Infrastructure), also relevant to Track 1 (Model Epistemics).*

## 1. Introduction

Two failure modes sit underneath any workflow in which AI agents audit or produce scientific claims. First, **provenance**: the record of what the agent did — prompts, tool calls, outputs, intermediate files — lives on the operator's infrastructure, under the operator's write access. A "supported" verdict is worth little if the log behind it can be edited after the fact with no trace. Second, **calibration**: even a perfectly preserved trail does not tell you whether the verdict it contains is right. Audit verdicts are produced by stochastic models writing one-off experimental code; the same claim can be upheld or falsified depending on implementation choices nobody reviewed.

We built and measured both halves on Lemma, an existing autonomous claim-auditing agent. Lemma ingests a paper (arXiv id, OpenReview id, or PDF), extracts testable claims, writes and executes self-contained numerical audit scripts in a sandbox with a mandatory positive control, assembles an inspectable evidence trail, and runs an automated judge over that trail. It was built for an earlier event; the pipeline itself is **preexisting work**, disclosed as such in the README and here.

Our main contributions are:

1. **A tamper-evident evidence trail** for agentic audit workdirs: per-record hash chaining of the append-only trace, plus a sealed artifact manifest and a sealed-prefix hash covering legacy traces. The whole trail collapses to one 256-bit evidence head (`lemma seal` / `lemma verify`).
2. **A red-team characterization of what sealing catches.** Four escalating tamper scenarios show the layer detects every post-hoc edit — including a forged, correctly-chained record appended after sealing — while a full re-seal is detectable only against an externally published head. We state this boundary explicitly rather than overselling tamper-*proofness*.
3. **A test-retest calibration study** of the agent's own verdicts: across 48 claims in 9 paper workdirs, and critically on two natural control pairs — the same paper with byte-identical claim sets audited independently by different model backends — where verdict agreement was 46%, isolating audit-implementation variance as a first-class risk to reported verdicts.

## 2. Related Work

**Tamper-evident logging.** Certificate Transparency (RFC 6962) established the pattern: append-only hash-chained records whose heads are publicly anchored, so misbehavior is provable rather than deniable. C2PA applies signed provenance manifests to media. The sprint's own design sketches call for a "tamper-evident action recorder" over OpenTelemetry GenAI conventions. Our work instantiates that idea where it is currently missing: the evidence trail of an agentic science pipeline, and — unusually — measures rather than assumes what the layer catches.

**Model self-knowledge and honesty evals.** MASK, DarkBench, SycEval and ELEPHANT measure whether models knowingly misrepresent, manipulate, or capitulate; Kalai et al. (2025) show binary-graded benchmarks reward confident guessing over honest abstention. We study the same question one level up, on an *agentic* system: does the audit agent abstain (inconclusive) rather than oversell, and are its decisive verdicts reproducible?

**Agentic science and reproducibility.** Automated reproduction and claim-verification efforts (the ICML agent reproductions challenge, AVeriTeC-style claim verification) evaluate outputs but rarely interrogate the audit trail's integrity or the auditor's verdict stability across runs. Our test-retest protocol is, to our knowledge, the first published measurement of claim-level verdict agreement for an autonomous paper-auditing agent.

## 3. Methods

**Preexisting system (disclosed).** Lemma's four-stage pipeline: `extract` (paper → typed claims with test plans), `audit` (LLM writes a self-contained numpy script per claim; script runs in a network-less, workdir-scoped sandbox; ≤3 attempts; a positive control must pass before `supported`/`falsified` is accepted; a validator rejects self-contradictory summaries), `evidence` (results → Trackio logbook), `judge` (evidence-trustworthiness rubric → PASS / CONDITIONAL PASS / FAIL). Every LLM call and tool run appends to a JSONL trace. All of this predates the sprint.

**New: hash-chained trace.** Each trace record now carries `seq` and `chain = SHA-256(chain_{i-1} ‖ canonical(record))`, with the genesis bound to the run id. Editing, deleting, or reordering any record invalidates the chain from that point; the final chain value is a commitment to every record.

**New: sealed workdir.** `lemma seal <workdir>` writes `provenance.json`: (i) SHA-256 of every artifact (claims, scripts, summaries, figures, reports); (ii) a *sealed-prefix* hash of the trace's first N lines — append-only growth is legal, edits inside the sealed region are not; this also covers legacy unchained traces; (iii) `manifest_root`, a single digest over the manifest. The seal event itself is logged into the trace, so the trace head commits to the manifest. `lemma verify` recomputes all three layers and reports per-file ok / modified / missing / added plus chain health.

**Threat model.** The operator controls the writer, so nothing prevents wholesale regeneration; sealing guarantees *detection of post-hoc divergence from an anchored head*, CT-style. `provenance.json` is a convenience anchor; the security anchor is the head published externally at seal time. We evaluate exactly this boundary rather than claiming more.

**Red-team scenarios.** On a copy of a sealed workdir: (A) edit a verdict in `audit_report.json`; (B) flip one byte inside a sealed trace record; (C) append a *correctly-chained* forged verdict record (the head is in the file — an attacker can extend the chain legally); (D) re-seal the tampered workdir so `provenance.json` is self-consistent.

**Test-retest study.** Nine workdirs, 48 claims. Two pairs share a paper *and byte-identical claim statements* but were audited by disjoint model stacks: a free-endpoint fallback chain (Kimi-K3, Qwen3-8-27B, Kimi-K2.7-Code) vs a single 501B-parameter model (Beam-1.0-501B-A23B). Any verdict disagreement on identical claims isolates audit-stage variance — the test the agent happened to write — rather than extraction or claim framing.

## 4. Results

### 4.1 Sealing detects every post-hoc edit we tried; the honest boundary is the anchor

| Scenario | Mechanism expected to catch it | Result |
|---|---|---|
| A. Verdict edited in `audit_report.json` | file manifest | detected; modified file named |
| B. Byte flipped inside sealed trace record | sealed-prefix hash (and chain on new traces) | detected |
| C. Forged record appended with a valid chain | trace head vs sealed head | detected — *only* vs anchored head; chain itself stays valid |
| D. Full re-seal of tampered workdir | externally published `manifest_root` | self-consistent under convenience anchor; detectable only because the published root differs |

Scenario C is the instructive one: appending is *legal* trace growth, and the forger's chain is cryptographically valid — the seal's stored head is the only thing that distinguishes it. Scenario D is the limit case the report does not hide: if the operator re-seals and re-publishes, integrity checks pass; the defense is anchoring heads in a place the operator does not control (a public log, a timestamped commit — the same trust model CT settled on years ago).

### 4.2 The audit agent abstains a meaningful fraction of the time

Across 48 claims: 29 supported (60%), 9 falsified (19%), 10 inconclusive (21%) — a 79% decisive rate with a genuine abstention option that was exercised, including verdicts recorded as `falsified` against the paper where the control passed. 6 claims recorded control failures; 54 failed run attempts were persisted rather than hidden. All 7 completed judge runs returned PASS on the preexisting rubric.

### 4.3 Test-retest: same claims, different backends, 46% verdict agreement

| Pair | Shared claims | Agree | Flips |
|---|---|---|---|
| arXiv:2510.10981, fallback-chain vs Beam-501B | 7 | 2 (29%) | 5 |
| JMLR grokking, fallback-chain vs Beam-501B | 6 | 4 (67%) | 2 |
| **Total** | **13** | **6 (46%)** | **7** |

(Figure 3 in the appendix shows the fit-window sweep described below.)

*Worked example — and a causal check.* JMLR C2 claims test error scales as $E_D(t) \propto (t_\epsilon-t)^{(D+1)/2}$ near the transition. Run A fitted the exponent inside the grokking window → ν̂ = 1.4999, 3.000, 5.499 ≈ (D+1)/2 → **supported**. Run B fitted every point with $t_\epsilon-t \in (10^{-3},5)$ → ν̂ = 2.00, 3.86, 6.78 → **falsified**. Both controls green, opposite verdicts. We then reran run B's own audit script changing *only* the fit-window cutoff δ: ν̂(D=2) moved 2.00 → 1.75 → 1.60 → 1.56 → 1.53 as δ went 5 → 0.5 → 0.2 → 0.1 → 0.05, converging on the claimed 1.5 (D=5 and D=10 converge likewise). The claim, which is an asymptotic statement about the critical regime, holds; the "falsified" verdict was a fit-window artifact — invisible to the reader, expensive for the paper's reputation, and exactly the kind of failure only re-audits or recorded-trace inspection can catch.

### 4.4 Fresh end-to-end run during the sprint

A full pipeline run on arXiv:1712.09203 (algorithmic regularization in over-parameterized matrix sensing; unseen, 6 extracted claims) ran end-to-end during the sprint with all stages pinned to a single frontier model. Its trace is hash-chained from record 1 and its workdir was sealed at completion (heads in Appendix A). Outcome: **3 supported, 3 inconclusive, 0 falsified, judge PASS** over a 145-record chained trace.

| Claim | Verdict | Why |
|---|---|---|
| C1 | inconclusive | generated script never ran cleanly in 6 tries — abstained rather than guess |
| C2 | **supported** | train–test gap shrinks monotonically with α (control Spearman 1.0) |
| C3 | inconclusive | **its own positive control failed** — refused to issue a verdict on a suspect statistic |
| C4 | **supported** | test error decreases through 1e5 iterations, no divergence |
| C5 | **supported** | rank-1 recovery across the full m sweep |
| C6 | inconclusive | SGD short of the 1e-6 threshold inside the reduced compute budget |

Table 3. Fresh audit outcomes on arXiv:1712.09203.

The abstentions are the instructive part: C3 is control-honesty operating on unseen work — the agent's own positive control failed its RIP check, so the pipeline declined to report a verdict rather than launder a broken statistic into a confident one — and C6 reports its compute budget explicitly instead of quietly falsifying. The fresh trail demonstrates the report's two halves at once: a sealed, inspectable record, and abstention over false confidence.

## 5. Discussion

Integrity and correctness are separable, and confusing them is the actual risk. A sealed trail proves *what the agent did*; it says nothing about *whether the audit was any good*. Our test-retest result is the sobering half of the story: on identical claims, verdict agreement across model backends was under half, and the mechanism is mundane — the agent writes a different experiment each run, and fit windows, seeds, and tolerances move the answer. Sealing makes such variance permanently inspectable rather than silently lost, which is precisely why the two halves belong together.

Practical implications: (i) audit verdicts should ship with their evidence head *and* a reproducibility caveat — "supported" means "the test this run wrote passed", not a theorem; (ii) for consequential verdicts, consensus across independent runs (or a reviewer-reference implementation, which Lemma supports) looks necessary rather than decorative; (iii) abstention (`inconclusive`) is a feature worth protecting — it fired in 21% of claims and in borderline cases was often the honest answer.

**Future work.** Anchor heads in a public transparency log or timestamped commit; add a verdict-confidence field calibrated against reviewer references; run test-retest at scale (k-way independent audits per claim) to map the flip surface.

## 6. Limitations and Dual-Use / Ethical Considerations

**Limitations.** The test-retest sample is 13 paired claims on 2 papers — enough to demonstrate the phenomenon, not to bound its rate; backend, prompt and sampling variance are confounded (deliberately — that is the deployed configuration). We did not hand-adjudicate every verdict, so "which run was right" is argued case-by-case, not scored. Sealing detects divergence from an anchor; it does not prove the anchored content was ever true, and scenario D shows a malicious operator can always re-anchor — the defense is publication of heads, not mathematics. The audit scripts are LLM-written one-offs; a false verdict with a passing control and a sealed trail is indistinguishable from a true one except by independent re-audit.

**Dual-use.** Verification infrastructure cuts both ways: a seal could lend unearned legitimacy if readers treat "verified trail" as "verified conclusion" — we deliberately surface this in the report and CLI output. Conversely, the same tooling helps detect fabricated agent logs in peer review, forensics, and governance. No personal data was used; all artifacts are public papers or generated data. The tamper demo fabricates forged records strictly inside temp copies of our own workdirs.

## References

1. Laurie, B., Langley, A., Kasper, E. *Certificate Transparency.* RFC 6962, 2013.
2. C2PA Technical Specification v2.4, 2025.
3. Ren, R. et al. *The MASK Benchmark.* CAIS, 2025.
4. Kran, E. et al. *DarkBench.* ICLR 2025.
5. Fanous, A. et al. *SycEval.* AIES 2025. / Cheng, M. et al. *ELEPHANT*, 2025.
6. Kalai, A.T. et al. *Why Language Models Hallucinate.* OpenAI, 2025.
7. Cotton-Barratt, O., Vaintrob, A., Sourbut, E., Hadshar, R. *Design Sketches for a More Sensible World.* Forethought, 2026.
8. Schlichtkrull, M. et al. *AVeriTeC.* arXiv:2305.13117.
9. Li, Y., Ma, T., Zhang, H. *Algorithmic Regularization in Over-parameterized Matrix Sensing.* COLT 2018, arXiv:1712.09203.
10. Hastie, T. et al. *Surprises in High-Dimensional Ridgeless Least Squares.* arXiv:1903.08560. Bartlett, P. et al. *Benign Overfitting in Linear Regression.* arXiv:1906.11300.

## Appendix A. Evidence heads and artifacts

- Tamper demo: `scripts/tamper_demo.py` — prints the four-scenario result verbatim.
- Calibration analysis: `sprints/epistemics-2026/analysis.json`; figures in `figures/`.
- Sealed corpus workdir `papers/arxiv-1912.07242` (9 claims, judge PASS, legacy trace):
  `manifest_root = ec712f4244518e40dedc03eeb7c28f341e76cc886f9f78c60a7d2ba35e0077f6`;
  `trace_head (post-seal) = 6a524c5c9876fca0bf3054bdd258176327025cce64480ba11ada0de9ec8d932c`.
  Re-verify with `./lemma verify papers/arxiv-1912.07242`.
- Seal of the fresh `papers/arxiv-1712.09203` audit (chained trace, 145 records, judge PASS): `manifest_root = 2234edcf6785bc2d656f544f078f199f086e1b700cf0d27b71b9b36d0d595ab7`; `trace_head (post-seal) = 21f922af1e0b8b5126e5f5c30d211f87016f1da20149ecd04d7f7d1c34248481`. Re-verify with `./lemma verify papers/arxiv-1712.09203`.

## Appendix B. Preexisting vs new work

Preexisting (built before the sprint, disclosed): the Lemma pipeline (`extract/audit/evidence/judge`, sandbox, evals, reviewer-reference mechanism) and all `papers/*` audit workdirs except the fresh run. New during the sprint: hash-chained `agent/traces.py`, `agent/provenance.py`, `seal`/`verify` CLI, `scripts/tamper_demo.py`, `scripts/sprint_calibration.py`, the test-retest study, the arXiv:1712.09203 audit, and this report.
