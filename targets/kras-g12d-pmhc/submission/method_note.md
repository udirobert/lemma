# Method note — KRAS G12D pMHC mutation-specific binder (Challenge 05)

Submitting researcher review: [x] reviewed before submission (required)
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
   derivation. Coarse attestation (`scripts/novelty_check.py`):
   17–21% NW identity vs the 7OW6 TCR (background) and 18–69%
   verbatim 8-mer coverage vs SwissProt — the overlap is a
   low-complexity artifact (Ala-rich MPNN collapse), declared as such.
   Sequences are novel but low-diversity; see per-design metrics in
   `novelty_attestation.json`.

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
  scaffold indiscriminately.
- Round 2 verdict (22 backbones, p6-anchored hotspots, 44 ranked
  candidates): **hold** — 39 decoy_leak, 5 borderline, and the positive
  control itself fails the screen (7OW6 TCR margin −0.0045), so no
  candidate can be convicted on specificity by this screen. designs.csv
  ships the borderline tail ranked by margin (best 0.1281); the wet-lab
  decoy measurements are the real gate, exactly as the challenge
  intends.

## Contact-level audit (post-submission re-analysis, round-2b)

After the hold verdict we re-audited every prediction at the resolution
the screen cannot reach: per-residue binder→peptide contact maps and
interface PAE parsed from the stored Boltz-2 outputs (script:
`scripts/modal_contact_audit.py`, results: `audit/contact_audit.json`,
control anchored the same way).

- **Control anchor:** the 7OW6 TCR engages 6/10 peptide residues with
  12 heavy-atom contacts (<5 A) to p6 and interface ipSAE 0.993 — in
  BOTH g12d and wt, exactly the "reads the peptide position" signature
  expected of a TCR. Contact metrics resolve what ipTM margins cannot.
- **Ranking rule changed.** Margin ranking convicted nothing; contact
  ranking uses (peptide engagement, p6 differential =
  g12d p6 contacts − wt p6 contacts, g12d p6 contacts, ipSAE, peptide
  coverage). designs.csv is regenerated under this rule — margin kept
  as a column, no longer the sort key.
- **7 candidates are predicted mutation-preferring (Δ > 0):** top hit
  `lemma-d8-d8_2_s2` makes 20 contacts to p6-Asp at 2.9 A in G12D and
  **zero** contacts (28 A away) in WT — engages the discriminating
  residue only in the mutant complex. d16_3_s1 (+13), d21_8_s4 (+7),
  d15_2_s1 (+6), d14_11_s1 (+3) show the same differential geometry;
  d10_4_s4 (+2, ipSAE 0.989) and d12_1_s5 (+1) weaker.
- **The margin screen's top picks were misleading:** its rank-1
  (d12_1_s6) contacts p6 0× in G12D but 27× in WT — predicted
  WT-*preferring*, the anti-target. Its rank-2 (d13_10_s7) sits 33 A
  from the peptide entirely. Both are excluded from this submission.
- Caveats stand: contact maps are still predictions on the same model
  family that failed the margin gate; a predicted contact to the p6
  carboxylate is a specificity *handle*, not proof. Round-1 units
  (d1–d5) are excluded from contact ranking — their prediction files
  were not retained; they remain in audit_report.json for the record.
  The wet-lab decoy measurements remain the real gate.

## Declared target surface

Peptide + groove mouth, centered on p6-Asp — the surface a TCR reads.
Binding the HLA scaffold, β2m, or the C-terminal Twin-Strep tag does
not count (off-limits per challenge rules).

## Evidence

- Repo trail (audit scripts, per-prediction confidence outputs, judge
  verdicts, run notes): https://github.com/udirobert/lemma/tree/main/targets/kras-g12d-pmhc
  (run notes: https://github.com/udirobert/lemma/tree/main/runs)
- designs.csv is generated from `audit/audit_report.json` +
  `audit/contact_audit.json` by `scripts/build_submission.py`;
  ranking = peptide engagement → p6 contact differential
  (g12d − wt) → g12d p6 contacts → interface ipSAE → coverage,
  descending. ipTM margin retained as an evidence column.
- Compute: all inference on Modal L4/A10G; per-prediction ~110 s;
  counts and timings in run notes.
