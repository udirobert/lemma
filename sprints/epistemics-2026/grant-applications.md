# Grant application drafts — Lemma

Ready-to-paste text per funder, built from `application-blurb.md`.
Applicant: Udi Ngethe · papaandthejimjams@gmail.com · https://github.com/udirobert/lemma

---

## 1. Manifund — public project proposal

Fields: title, description, funding goal, deadline, links. Manifund proposals are public — write for donors *and* regrantors.

**Title:** Lemma — tamper-evident evidence trails for AI scientific auditing

**Funding goal:** $10,000 · **Deadline:** 45 days

**Description:**

AI agents are starting to produce scientific audit verdicts — "this paper's central claim replicates" — at scale. Two things make those verdicts fragile: the log of what the agent did lives on its operator's infrastructure and can be silently rewritten, and even an honest log doesn't tell you the verdict was right.

Lemma is my open-source pipeline for end-to-end claim auditing (extract claims → generate and sandbox numerical audit scripts → assemble evidence → judge). Every audit already produces an inspectable trail. This month I made that trail tamper-evident: hash-chained trace records plus a sealed artifact manifest, so an entire audit collapses to a single 256-bit "evidence head" that anyone can recompute with `lemma verify`. I red-teamed it under four tamper scenarios and documented exactly where the guarantee ends — an honest threat model, not security theater.

I also ran the first published test-retest measurement of an audit agent's verdict stability: on identical claims audited by different model backends, verdict agreement was only 46%. In one case I causally isolated the flip — the falsified verdict came from the audit script's fit window, not the paper. Agent verdicts are unstable *and* forgeable; both problems need fixing before this infrastructure can be trusted.

**What funding buys:** (a) scaled test-retest studies — k-way independent audits per claim to map the instability surface; (b) a consensus-audit mode (`--consensus N`) + verdict-confidence field, motivated directly by the measurement; (c) external anchoring of evidence heads (public transparency log / timestamped commits); (d) a preprint + expanded workshop submission.

**Why me:** I built the whole pipeline and have shipped it through two prior events (ICML reproduction challenge fixture; a Hermes plugin now live in the catalog). The work is open source and reproducible: github.com/udirobert/lemma, report at sprints/epistemics-2026/report.pdf.

---

## 2. Lightcone Commons — lightconecommons.com/apply (4 sections)

**1. What are you working on?**

Verification infrastructure for agentic science. Lemma is an open-source agent that audits scientific papers claim-by-claim — extracting testable claims, writing and sandboxing numerical audit scripts under mandatory positive controls, and issuing supported/falsified/inconclusive verdicts with a fully inspectable evidence trail. The new work: (a) the trail is now tamper-evident — hash-chained trace records plus a sealed artifact manifest collapsing each audit to a 256-bit evidence head anyone can verify; (b) the first empirical test-retest study of an audit agent's verdict reliability — identical claims audited by different model backends agreed only 46% of the time, with one flip causally traced to the audit script's fit window rather than the paper.

**2. What would you do with funding?**

Three workstreams over ~3 months: (1) Consensus auditing — N independent audits per claim with an agreement metric and a verdict-confidence output, turning the observed instability into a signal instead of noise. (2) Anchoring — publish evidence heads to an external, operator-independent log (transparency-log style or timestamped public commits), completing the trust model so verification doesn't rely on the operator's honesty. (3) Scale the measurement — extend the test-retest study across more papers and backends, and open the audit corpus as a reproducibility benchmark for the community. $20–30k covers API compute for the sweeps, part-time engineering, and dissemination.

**3. Who is involved?**

Solo: Udi Ngethe, independent researcher/engineer, builder of Lemma end-to-end (pipeline, Hermes plugin — live in the Hermes catalog, sealed audit corpus, sprint report). Advisors/collaborators: none formally; open to mentorship.

**4. Logistical details**

Individual (not an org). Based in the UK/remote. Funding would go to API compute (~60%), engineering time (~30%), dissemination (~10%). Open-source (repo public), artifacts reproducible, work disclosable in full.

---

## 3. Emergent Ventures — ≤1,500-word proposal

**Personal story:** I'm an independent builder who keeps running into the same gap: AI systems produce confident scientific verdicts with no verifiable record of how they got there. After building a claim-auditing agent end-to-end for earlier events, I spent a weekend making its evidence trail tamper-evident and measuring whether its verdicts were even stable — and found they weren't (46% agreement on identical claims). That combination — provable integrity plus honest unreliability — is what I want to keep building on.

**A mainstream view I agree with:** most AI-evals discourse treats "the model said X" as evidence. I agree with the emerging view that evaluation claims need provenance the way media needs C2PA — and that unreliability is itself measurable.

**The idea:** Lemma turns scientific audit trails into verifiable artifacts — hash-chained traces and sealed manifests collapse each audit to one 256-bit evidence head, externally anchorable, CT-style. The novelty is doing this on a *live audit agent* and measuring, not assuming, what sealing buys — plus the first claim-level test-retest reliability measurement for an autonomous auditor. The problem it solves: agent-produced verdicts are already entering review pipelines and replication workflows with forgeable logs and unstable outputs; nobody can currently check either property.

**Amount requested:** $15,000 — ~60% API compute for consensus-audit sweeps and scaled test-retest, ~30% engineering (external anchoring, `--consensus N`, verdict-confidence), ~10% dissemination.

---

## 4. Anthropic — External Researcher Access Program (primary) + AI for Science

**ERAP blurb (AI safety/alignment fit):**

Independent research on verification infrastructure for agentic scientific auditing. Lemma audits papers claim-by-claim under mandatory positive controls; this month added hash-chained tamper-evident evidence trails + sealed workdir manifests, and measured claim-level verdict stability across model backends (46% agreement on identical claims). Requesting API credits to run consensus-audit sweeps (k-way independent audits per claim) and scaled test-retest studies — measuring where frontier-model audit verdicts are stable vs. implementation-dependent, and building the abstention/consensus machinery that makes agentic audit verdicts deployable. All outputs open source.

**AI for Science variant:** same, leading with the paper-auditing use case — a tool that makes scientific claims checkable, used on real arXiv papers with reproducible artifacts.

---

## 5. ARM Fund (airiskfund.com/apply) — reuse Lightcone §1–2

Lead with the catastrophic-risk angle: agent-produced verdicts with forgeable provenance scale epistemic failure; sealed trails + reliability measurement are prerequisites for trusting AI auditors in safety-relevant workflows.

---

## Status

| Funder | Status |
|---|---|
| Manifund | draft ready — create proposal at manifund.org |
| Lightcone Commons | draft ready — lightconecommons.com/apply |
| Emergent Ventures | draft ready — Mercatus "Apply Now" |
| Anthropic ERAP / AI for Science | draft ready — support.claude.com program pages |
| ARM Fund | reuse LC draft — airiskfund.com/apply |
| Coefficient Giving TAIS | evaluate — bigger lift; the consensus-audit program could fit "control evaluations" |
