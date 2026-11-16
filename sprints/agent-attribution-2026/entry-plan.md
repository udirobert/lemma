# Agent Attribution Hackathon entry plan (Dec 11–13, 2026)

$5,000 prizes ($2,500 1st). Submissions close Sun Dec 13, 11:59 PM AoE.
Sign up + submit at https://apartresearch.com/sprints/agent-attribution-2026

## Tracks (from the brief)

1. **Detect** — attribution classifiers over labeled activity logs (5 actor types, held-out scoring; extra for model/scaffold/location ID)
2. **Simulate** — generate traces incl. evasion (human mimicking agent timing, agent adding human-like delays)
3. **Discover** — find rogue agents in the wild

## The angle: attribution needs intact logs

Every attribution classifier assumes the activity log is real. Our sprint work makes that assumption checkable — a forged or post-hoc edited trace breaks the hash chain / sealed prefix — *and* produces the labeled data the classifier needs: lemma runs **are** structured agent activity logs (timestamped `llm_call`/`tool_run`/`attempt` records), and its `feedback.md` human-in-the-loop contract is a natural steering label.

## Proposed build (Track 1 primary, Track 2 secondary)

**T1 — steering-signal classifier over traces.**
- Data: lemma corpus traces labeled by supervision level — fully autonomous runs vs runs with human feedback injections vs post-hoc edited traces. (We can also synthesize steering variants mechanically.)
- Features: inter-event latency distributions, feedback-injection signatures (feedback.md presence + subsequent behavior deltas), provider/model switches mid-run, tool-call cadence, chain-verification failures.
- Target: predict presence and density of human steering — maps onto their actor-type spectrum.

**T2 — evasion suite (cheap to build, we already know the attacks).**
- Forged well-chained appended records (red-team scenario C machinery)
- Human-latency-gap injection into autonomous traces (timing padding)
- Measure whether our verifier + T1 classifier catch each — an integrity-vs-attribution stress test, which is itself a contribution.

## Deliverables

- `lemma trace classify <workdir>` — steering-signal report per trace
- `scripts/trace_evasion.py` — the evasion generator + detection results table
- Report (same template rules): integrity precondition for attribution + classifier results + evasion findings

## Pre-sprint prep (do before Dec 11)

- [ ] Collect/label supervision levels across the existing `papers/*` traces (which runs used human feedback.md)
- [ ] Stub the classifier feature extractor (latencies, feedback events, chain status)
- [ ] Confirm sprint sign-up + read released datasets/security rules when posted
- [ ] Fresh-angle disclosure: seal/verify pipeline is preexisting (disclose); classifier + evasion suite are new sprint work
