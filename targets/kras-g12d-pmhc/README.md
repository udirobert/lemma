# Target: KRAS G12D pMHC (Challenge 05, Anthropic × Adaptyv 2026)

**Design a mutation-specific pMHC binder.** Cells carrying the KRAS G12D
mutation display the ten-residue mutant peptide VVVGADGVGK in the groove of
HLA-A*11:01. The wild-type peptide, VVVGAGGVGK, differs at one position of ten
(glycine 12 → aspartate, peptide position 6).

**Deadline: Sunday 2026-11-08, 23:59 AoE. Open track (Track 3).**
Competition page: proteinbase.com/competitions/anthropic-adaptyv-2026/challenges/pmhc

## The three objectives (ranked in this order by the wet lab)

1. **Mutation specificity** — measurable KD against the G12D complex AND no
   detectable binding to the wild-type complex. (Designs binding both rank
   after these, ordered by KD-wt / KD-G12D ratio, largest first.)
2. **Peptide-dependent binding** — no detectable binding to peptide-free
   HLA-A*11:01 + β2-microglobulin. (A binder that reads the HLA or β2m binds
   all three targets and ranks at the bottom.)
3. **Affinity** — lowest KD against the G12D complex. Target the peptide and
   the groove around it — the surface a T-cell receptor reads, the only part
   that differs between the complexes.

Binding = fittable sensorgram curve with KD, or association shift >300% over
negative control. Every tested design is measured against all three targets
under the same conditions, same assays, G12D + wild-type + peptide-free.

## Chains (design against the whole complex; recommend: peptide + groove)

| Chain | Reference | Range | Length |
|-------|-----------|-------|--------|
| HLA-A*11:01 heavy chain | UniProt Q5S3G3 | 25–305 | 281 aa |
| β2-microglobulin | UniProt P61769 | 21–119 | 99 aa |
| KRAS G12D peptide | UniProt P01116 | 7–16 | 10 aa |

Allele: A*11:01:01:01 (IPD-IMGT/HLA HLA00043). Structure: **PDB 7OW6**,
chains A–C, TCR removed.

Peptide sequence (G12D): `VVVGADGVGK`
Peptide sequence (wild-type): `VVVGAGGVGK`

## Submission constraints (Track 3 / open)

- Up to **20 designs**, all **de novo and zero-shot** — no derivation from
  known binders of this or similar targets.
- **Single chain**: general protein, nanobody, scFv, or Fab.
- Length **10–250 aa**.
- Unique designs (no duplicates), sequences only.
- Include: method note, declared target surface, links to evidence.
- Screening is contest allocation (~375 of ~1500 designs to Track 3), not a right.

## Copy-ready data

HLA-A*11:01 heavy chain:
`GSHSMRYFYTSVSRPGRGEPRFIAVGYVDDTQFVRFDSDAASQRMEPRAPWIEQEGPEYWDQETRNVKAQSQTDRVDLGTLRGYYNQSEDGSHTIQIMYGCDVGPDGRFLRGYRQDAYDGKDYIALNEDLRSWTAADMAAQITKRKWEAAHAAEQQRAYLEGRCVEWLRRYLENGKETLQRTDPPKTHMTHHPISDHEATLRCWALGFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVVPSGEEQRYTCHVQHEGLPKPLTLRWELSSQPT`

β2-microglobulin:
`IQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYYTEFTPTEKDEYACRVNHVTLSQPKIVKWDRDM`

## Lemma-workdir notes

This workdir mirrors `papers/<id>/`. Planned contents after the first design
round: `target.json` (machine-readable chains + decoys), `designs/` (each with
its why-card + scores), `audit/` (anti-objective results), `logbook/`
(Trackio), `judge.json` (submit/hold/reject with reasons). Trace:
`agent/traces/design-kras-g12d-pmhc-<runid>.jsonl`.

Status: **round 1 audited — 10/10 decoy_leak, judge = reject.** All five
stages real end-to-end: `designs/d1`–`d5` scored vs all three complexes
under `audit/`, verdicts in `audit/audit_report.json`, `judge.json` =
reject (boltz predicts indiscriminate HLA-scaffold binding — the apo-decoy
failure mode). Round 2 needs generation-side specificity pressure.
See `PROTEIN.md` at repo root.
