# PROTEIN.md — Lemma designs what it can falsify

**Status:** capability brief, not yet built. Sibling to `HACKATHON.md`.
**Last updated:** 2026-11-04.

## Why this, and why it belongs in lemma

Lemma audits what *other* scientists claim. The next muscle is the flip: make
its own risky claims — a designed protein is a claim about the world — and
apply the same distrust to itself. Protein binder design is the cleanest venue,
because the discipline transfers exactly:

| Lemma's audit loop | Lemma's design loop (new) |
|--------------------|---------------------------|
| extract claims from a paper | propose design hypotheses (what surface, what mechanism, what KD range) |
| write & run honest numerical audits | run structure & binding models, honestly (incl. models' own error bars) |
| **positive control is mandatory** | **anti-objectives are mandatory**: the design must fail to bind the decoys, checked in-silico before any wet lab sees it |
| append-only evidence trail | append-only design trail: every score, every seed, every rejection |
| judge scores the trail | design judge binates: submit / hold / reject, with reasons |

The Anthropic × Adaptyv competition hands us a target family whose judged
objectives already read like a falsification audit (see below): *bind the
mutant, fail to bind the wild-type, fail to bind the naked MHC* — three
ready-made negative controls. A design pipeline that ships decoy-failures
as first-class results is exactly lemma's voice. That is the method-novelty
story, and it is honest.

## Venue 1 — Challenge 05: KRAS G12D pMHC (open track, now)

Full brief lives in `targets/kras-g12d-pmhc/`. Skeleton:

- **Target:** bind the KRAS G12D peptide (VVVGADGVGK) presented by HLA-A*11:01,
  do **not** bind the wild-type peptide (VVVGAGGVGK, one position differs), do
  **not** bind peptide-free HLA-A*11:01. Structure reference PDB 7OW6 (TCR removed).
- **Deadline:** Sunday 2026-11-08, 23:59 AoE.
- **Constraints (open/Tool-3 track):** up to 20 designs, de novo and zero-shot
  only, single chain, nanobody / scFv / Fab or general protein, 10–250 aa,
  novel sequences (no natural-binder derivation). Submit via Proteinbase with a
  method note and declared target surface; designs screened ~50/25/25 across
  tracks (we ride in the un-favoured 25% — sets expectations).
- **Honest odds:** most submissions come from groups with standing pipelines.
  We compete on method transparency and a clean evidence trail, low odds of a
  measured hit.

## Venue 2 — Protein Design Hackathon Basel (Adaptyv track)

Nov 28–29 at D-BSSE ETH, with Adaptyv running a binder-design track (winners
get lab tests). The pipeline built for Venue 1 is the CTO needed to compete
here properly; Basel is where a genuinely judged "pipeline + pedagogy" build
has room to shine. Application deadline noted on their site (from October 5,
apply if we commit).

## The pipeline we will build — `lemma design <target>`

Mirrors `lemma audit` stage-for-stage. New top-level command in `agent/cli.py`,
new stages, same workdir + registry + trace conventions:

```
lemma design targets/kras-g12d-pmhc       # or --target kras-g12d-pmhc

  1. propose   target.json + rated literature → design_hypotheses.json
               (surface choice, mechanism, why it discriminates the mutant)
  2. design    hypotheses → candidate sequences (multi-approach; each design
               carries a "why" card, its constraint residuals and its novelty
               attestation — no derived-from-natural-binder)
  3. audit     every candidate through the anti-objectives IN SILICO:
               complex scoring (wt + peptide-free decoys), interface stability,
               solubility/aggregation heuristics — failures are recorded, not
               discarded silently ("designed, tried, and falsified" is data)
  4. evidence  Trackio logbook per design; JSONL trace per run (same as audit)
  5. judge     design judge ranks candidates: submit ≤20 with reasons, links
               to evidence; prepare Proteinbase submission bundle
```

Workdir layout mirrors `papers/`:

```
targets/
  _index.json               registry (same shape as papers/_index.json)
  _template/                (README.md + design.py stub — mirrors papers/_template)
  kras-g12d-pmhc/           Challenge 05 workdir (see its README for data)
```

## Models and compute

Track 3's own-compute rules; all open:

- **Structure + affinity scoring:** Boltz-2 and/or Chai-1 community weights for
  complex prediction; KD deltas as DECISIONS, not promises — models disagree and
  that disagreement is itself audited.
- **Embeddings/filters:** ESM-2-style embedding similarity for novelty
  attestation vs natural protein DBs (uniRef50 needle distance, coarse is OK
  if we say so).
- **Generation:** structure-conditioned inverse folding (ProteinMPNN line) and
  small guided generative passes; every pass logged. If a pass needs more than
  this Mac can give, log the shortfall in the runnotes and degrade gracefully —
  no silent GPU leaps.
- No wet-lab promises: Adaptyv testing for Open/Track-3 pool is contest
  allocation, not a right.

## What we are NOT doing

- Not PDB-mining for existing anti-KRAS antibodies and passing them off as
  designs (violates de novo / zero-shot; also it is lying).
- Not promising KD numbers the models cannot support; scoring outputs are
  hypotheses, and the method note will say so.
- Not silently discarding failed designs: the logbook records them, judge
  explains why they died.

## Launch checklist

- [ ] scaffold `targets/` (registry + template + kras-g12d-pmhc README) — done in this PR
- [x] pip env: `.venv-bio` (boltz 2.2.1) + `.venv-chai` (chai_lab 0.6.1, pins torch<2.7 — separate venv), documented in `requirements-bio.txt`. arm64-only caveat recorded.
- [x] `lemma design` CLI stage skeleton wired to traces — `agent/design.py`, all 5 stages dispatch + trace; judge already emits real `judge.json` (HOLD). No-op first run verified.
- [x] Boltz/Chai smoke test — moved to Modal L4 (`scripts/modal_boltz_smoke.py`): boltz 110 s, chai-1 ~150 s steady-state per 390-res complex; local 8 GB M3 CPU DNF ≥80 min. Numbers: `runs/protein-smoke-2026-11-04.md`
- [x] design stage wired — `agent/design.py` calls `modal run scripts/modal_rfd.py`, writes `designs/d<k>/` (backbone.pdb, seqs.fa, candidate.json, why.md). First 5-design round done: 87–116 aa binders, 8 MPNN seqs each
- [x] audit stage — boltz scoring of each candidate seq vs all 3 complexes (pmhc_g12d / pmhc_wt / hla_apo); the anti-objective gate
- [x] first 5-design round: honest anti-objective audit + judge — verdict: REJECT (10/10 decoy_leak; binders predicted to dock HLA scaffold indiscriminately). Round 2: generation-side specificity pressure needed
- [ ] evidence stage — Trackio logbook per design (not yet built)
- [ ] round 2+: specificity-pressured generation (peptide-anchored potentials or score-filtered MPNN), re-audit, then submission bundle
- [x] Challenge 05 hardware check — resolved: Apple Silicon is dev-only; Modal L4 is the scoring backend (~2 h for the full 20×3 screen)
