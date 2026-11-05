# Demo fixture — a verdict we know cold

The ≤90 s demo must not flake. Choreography:

1. `lemma_status` — shows prior audit workdirs (empty on a fresh
   install; fine, it's honest).
2. `lemma_quick_audit` on the fixture source — arXiv `1903.08560`
   (ICLR in-context-learning paper; lemma's recorded audit of it lives
   in `papers/arxiv-1903.08560/` upstream). Expected shape: claims
   extracted, first 2 audited, per-claim verdicts + a workdir path.
3. `lemma_status` again — the new workdir listed with verdicts, judge
   result, and the trace path. The money shot is the *evidence folder*,
   not a paragraph.

## Why this fixture

- Claims are CPU-tractable numpy audits (no GPU hang risk on camera).
- The upstream repo carries its recorded outcome, so expected verdicts
  are known before filming.
- If live LLM/network flakes mid-recording, `lemma_status` on the
  pre-seeded workdir still demonstrates the full contract.

## Known flake windows (declare, don't hide)

- `extract` needs paper resolution (arxiv/firecrawl) — on a flaky
  network it can stall; the quick path's 15-min cap returns a clean
  error with the workdir path for `lemma_status` pickup.
- Audit stage = LLM calls + generated script runs. Expect single-digit
  minutes per claim, not seconds — set the audience's clock honestly.
