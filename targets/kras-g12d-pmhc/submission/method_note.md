# Method note — KRAS G12D pMHC mutation-specific binder (Challenge 05)

Submitting researcher review: [ ] reviewed before submission (required)
Track: open / Track 3 · max 20 designs · de novo, zero-shot, single chain

## Pipeline (all open, all runs logged)

1. **Generation** — RFdiffusion (Complex_base checkpoint) against 7OW6
   pMHC (TCR removed): contig `A1-276/0 B1-99/0 C1-10/0 <len>`, binder
   diffused as a new terminal chain. Hotspots on peptide residues
   C4–C8 (round 1) then tightened to C5–C7 centered on the
   discriminating p6-Asp (round 2). ProteinMPNN designs the binder
   chain only (receptor fixed), 8 seqs/backbone, temp 0.1.
2. **Screen** — every candidate is predicted with Boltz-2 against all
   THREE challenge targets: `pmhc_g12d` (objective), `pmhc_wt`
   (mutation-specificity decoy), `hla_apo` (peptide-dependence decoy).
   ipTM margin over the worst decoy is the ranking; binder→peptide
   pair ipTM is reported as a locality proxy.
3. **Positive control** — the 7OW6 TCR (α+β), a binder known to
   discriminate G12D from WT, scored through the identical screen
   (rules explicitly permit known binders for metric calibration).
   Result: complex ipTM 0.956 / 0.961 / 0.907 on g12d / wt / apo —
   margin −0.005, i.e. **the control also reads decoy-parity**. The
   ipTM-margin gate cannot resolve the p6 single-atom difference; our
   in-silico decoy_leak verdicts are uninformative about specificity
   and we submit ranked by margin + binder→peptide pair ipTM with this
   caveat declared, not hidden. Kept sub-signal: TCR→HLA pair ipTM
   drops ~0.10 in the peptide-free complex (0.84 vs 0.93–0.94).
4. **Novelty** — de novo RFdiffusion backbones; no natural-binder
   derivation. Sequence-similarity attestation: `<fill: status>`.

## What we claim, and what we do not

- Boltz-2 ipTM is a docking-confidence proxy, **not a KD**. Margins are
  screen-level signals, not affinity measurements.
- A single-atom p6 Asp→Gly difference on a 10-mer buried in a 390-res
  complex sits below this screen's resolution — demonstrated, not
  hypothesised: the known-discriminative 7OW6 TCR scores decoy-parity
  (−0.005 margin). Our decoy_leak screen verdicts therefore cannot
  convict designs on specificity; the wet-lab decoy measurements are
  the real gate, exactly as the challenge intends.
- Round 1 verdict (5 backbones × top-2 MPNN seqs × 3 complexes = 30
  predictions): 10/10 decoy_leak — binders predicted to dock the HLA
  scaffold indiscriminately. Round 2 (12 backbones, p6-anchored
  hotspots) reran under the same screen; results and per-design
  metrics are in the linked evidence trail.

## Declared target surface

Peptide + groove mouth, centered on p6-Asp — the surface a TCR reads.
Binding the HLA scaffold, β2m, or the C-terminal Twin-Strep tag does
not count (off-limits per challenge rules).

## Evidence

- Repo trail (audit scripts, per-prediction confidence outputs, judge
  verdicts, run notes): `<link to GitHub / shared drive>`
- designs.csv is generated from `audit/audit_report.json` by
  `scripts/build_submission.py`; ranking = g12d ipTM minus worst-decoy
  ipTM, descending.
- Compute: all inference on Modal L4/A10G; per-prediction ~110 s;
  counts and timings in run notes.
