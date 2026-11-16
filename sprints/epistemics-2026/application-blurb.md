# Canonical application blurb — Lemma

Reuse/adapt per funder. Repo: https://github.com/udirobert/lemma · Report: sprints/epistemics-2026/report.pdf · Applicant: Udi Ngethe (solo), papaandthejimjams@gmail.com

## One-liner

Lemma is an open-source agent that audits scientific claims end-to-end — and the first to make its own evidence trail tamper-evident while measuring how much you should trust its verdicts.

## Short pitch (~100 words)

Agents are starting to produce scientific audit verdicts at scale, but the logs of what they did are written by their own operators and can be silently rewritten — and an untampered log still doesn't tell you the verdict is right. Lemma is an open pipeline that reads a paper, extracts testable claims, writes and runs numerical audit scripts under a mandatory positive control, and emits supported/falsified/inconclusive verdicts. New this month: per-record hash-chained traces and sealed artifact manifests that collapse a whole audit to one 256-bit "evidence head," plus the first published test-retest measurement of an audit agent's verdict stability — only 46% agreement on identical claims across model backends.

## Longer description (~300 words)

Lemma audits scientific claims the way a careful replication researcher would: extract the paper's testable claims, write a self-contained numerical audit script for each, run it under a sandbox with a mandatory positive control, and refuse a verdict when its own control fails. Every LLM call and tool run is appended to a JSONL trace.

This month we added a tamper-evident layer to that trail: each trace record is hash-chained (SHA-256 over the canonical record plus the previous link, genesis bound to the run), and `lemma seal` hashes every artifact into a manifest whose root — together with the post-seal trace head — is a single 256-bit commitment to the entire evidence trail. `lemma verify` recomputes all of it. We red-teamed it under four tamper scenarios; it catches every post-hoc edit, and we document honestly where it stops: a full re-seal is detectable only against an externally anchored head — the same trust model Certificate Transparency settled on.

Then we measured the harder half. Across 48 audited claims in 9 paper workdirs, and crucially on two same-paper, byte-identical-claims pairs run through different model backends, claim-level verdict agreement was only 6/13 (46%). A worked example causally isolates the mechanism: rerunning a falsifying audit's own script with only its fit window changed recovers the claimed value exactly — the "falsified" verdict was an implementation artifact. Integrity and correctness are separable problems for agentic science; we ship the first and quantify how far the second has to go.

## The ask (~use of funds)

- Compute for scaled test-retest studies: k-way independent audits per claim across backends, to map where verdicts are unstable (the consensus-audit feature this motivates).
- Engineering: `lemma seal/verify` hardening, external anchoring (public transparency log or timestamped commits), consensus mode (`--consensus N`), verdict-confidence calibrated against reviewer references.
- Dissemination: arXiv preprint + expanded workshop submission (eval/agent-reliability venues).

Suggested ask sizes: $5k–15k (compute + part-time eng, Manifund/EV scale); $25k+ for a full reliability program (Lightcone/Coefficient scale).

## Impact

Agent-produced scientific verdicts are entering peer review, replication pipelines, and policy-relevant literature checks. If those verdicts' provenance is forgeable and their outputs unstable, the trust infrastructure fails silently. Lemma provides both halves: evidence trails that are verifiably untampered given a public anchor, and empirical evidence of where verdict variance comes from — a prerequisite for any deployed "AI auditor" to be trustworthy.

## Track record / credibility

- Working open-source pipeline with 10 completed sealed paper audits, automated judge, sandboxed execution, reviewer-reference mechanism (github.com/udirobert/lemma)
- Shipped a vendored Hermes agent plugin (live in the Hermes plugin catalog)
- Prior artifacts: ICML agent-reproductions challenge fixture passed end-to-end; protein-design audit work (Boltz-2 contact-level audit on Modal)
- Sprint report + red-team + calibration study completed in one weekend

## Budget lines (if a form asks)

| Item | Est. |
|---|---|
| LLM/API compute for consensus + test-retest sweeps | $3–8k |
| Engineering time (sealing hardening, consensus mode, anchoring) | $2–10k |
| Dissemination (travel/fees) | $0–3k |
