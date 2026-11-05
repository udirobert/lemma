#!/usr/bin/env python3
"""Coarse novelty attestation for design candidates.

    python3 scripts/novelty_check.py [workdir]

Two checks, both declared coarse in the method note:

1. **Natural-binder derivation** (the actual rule): global-alignment
   %identity of each candidate vs the 7OW6 TCR alpha/beta chains —
   the known binder for this target. A de novo design should sit at
   sequence-background identity (~10-20%).

2. **Verbatim k-mer coverage** vs SwissProt: fraction of a candidate's
   8-mers (stride 2) found verbatim anywhere in the reviewed-protein
   database. Single pass over the flat DB, set-intersect per record —
   pure stdlib, ~1-2 min for ~570k seqs. Low coverage -> no exact
   natural ancestor; low-complexity kmers (<=3 distinct residues, e.g.
   Ala runs) are reported separately since they exist in nature and
   are not derivation evidence.

   Requires: data/cache/uniprot_sprot.fasta.gz
   (curl the canonical uniprot_sprot.fasta.gz once; gitignored).

Declared limits: k-mer coverage is not %identity — a design can be
related to a natural protein without any verbatim 8-mer. Upgrade path:
`mmseqs easy-search` for real %identity when the toolchain allows.
"""

import gzip
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SPROT = REPO / "data/cache/uniprot_sprot.fasta.gz"
K, STRIDE = 8, 2


def nw_identity(a: str, b: str) -> float:
    """Global %identity (Needleman-Wunsch, match +1 / mismatch -1 /
    gap -2). Pure python — seqs are <=250 aa."""
    la, lb = len(a), len(b)
    grid = [[-2 * j for j in range(lb + 1)]]
    for i in range(1, la + 1):
        row = [-2 * i] + [0] * lb
        ai = a[i - 1]
        for j in range(1, lb + 1):
            row[j] = max(
                grid[i - 1][j - 1] + (1 if ai == b[j - 1] else -1),
                grid[i - 1][j] - 2,
                row[j - 1] - 2,
            )
        grid.append(row)
    i, j, matches, aligned = la, lb, 0, 0
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and (grid[i][j] == grid[i - 1][j - 1] + (1 if a[i - 1] == b[j - 1] else -1))
        ):
            aligned += 1
            matches += a[i - 1] == b[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and grid[i][j] == grid[i - 1][j] - 2:
            aligned += 1
            i -= 1
        else:
            aligned += 1
            j -= 1
    return matches / aligned if aligned else 0.0


def kmers(seq: str) -> set[str]:
    return {seq[i : i + K] for i in range(0, len(seq) - K + 1, STRIDE)}


def low_complexity(k: str) -> bool:
    return len(set(k)) <= 3


def main() -> int:
    wd = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "targets/kras-g12d-pmhc"
    tcr = {
        "tcr_alpha": (wd / "inputs/tcr_d.fasta").read_text().splitlines()[-1],
        "tcr_beta": (wd / "inputs/tcr_e.fasta").read_text().splitlines()[-1],
    }

    if not SPROT.is_file():
        sys.exit(f"missing {SPROT} — curl uniprot_sprot.fasta.gz first")

    # unwrapped one-seq-per-line db for rg (fasta lines are 60-char wrapped)
    db_flat = REPO / "data/cache/uniprot_sprot.flat"
    if not db_flat.is_file():
        with gzip.open(SPROT, "rt") as f, db_flat.open("w") as o:
            seq = ""
            for ln in f:
                if ln.startswith(">"):
                    if seq:
                        o.write(seq + "\n")
                    seq = ""
                else:
                    seq += ln.strip()
            if seq:
                o.write(seq + "\n")

    # Collect candidates first, then one DB pass attributes every hit.
    units = []  # (label, seq, kmer set)
    kmer_to_units: dict[str, list[int]] = {}
    for ddir in sorted((wd / "designs").iterdir()):
        if not ddir.is_dir() or ddir.name.startswith("_"):
            continue
        cand = json.loads((ddir / "candidate.json").read_text())
        top2 = sorted(cand["sequences"], key=lambda s: s["mpnn_global_score"])[:2]
        for s in top2:
            ui = len(units)
            ks = kmers(s["aa"])
            units.append((f"{cand['id']}-{s['seq_id']}", s["aa"], ks))
            for k in ks:
                kmer_to_units.setdefault(k, []).append(ui)

    hits = [set() for _ in units]  # kmer hits per candidate
    with db_flat.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            seen = {
                line[p : p + K]
                for p in range(0, len(line) - K + 1)
                if line[p : p + K] in kmer_to_units
            }
            for k in seen:
                for ui in kmer_to_units[k]:
                    hits[ui].add(k)
            if i % 100000 == 0:
                print(f"  ... {i} db records", file=sys.stderr, flush=True)

    results = []
    for ui, (label, aa, ks) in enumerate(units):
        tcr_id = max(nw_identity(aa, t) for t in tcr.values())
        hit = hits[ui]
        frac_all = len(hit & ks) / len(ks)
        k_hc = {k for k in ks if not low_complexity(k)}
        frac_hc = len(hit & k_hc) / len(k_hc) if k_hc else 0.0
        results.append(
            {
                "design": label,
                "len": len(aa),
                "tcr_max_identity": round(tcr_id, 4),
                "kmer8_verbatim_frac": round(frac_all, 4),
                "kmer8_verbatim_frac_nolowcomp": round(frac_hc, 4),
            }
        )
        print(
            f"{label}: tcr_id={tcr_id:.1%} "
            f"kmer8={frac_all:.1%} (nolowcomp {frac_hc:.1%})",
            flush=True,
        )

    out_p = wd / "submission" / "novelty_attestation.json"
    out_p.parent.mkdir(exist_ok=True)
    out_p.write_text(
        json.dumps(
            {
                "method": (
                    "NW %identity vs 7OW6 TCR a/b + verbatim 8-mer "
                    "coverage vs SwissProt (rg -F); coarse, declared"
                ),
                "results": results,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"\nwrote {out_p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
