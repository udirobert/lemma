"""Contact-level audit of Boltz-2 predictions — runs ON Modal against the
protein-boltz-results volume so the 12 GB of structures never downloads.

    modal run scripts/modal_contact_audit.py

For every audited unit (manifest yamls carry the chain ids) it computes:

- per-peptide-residue binder contact profile (heavy atoms < 5 A)
- p6 engagement: min binder distance to peptide residue 6 — the residue
  that differs between G12D (Asp) and WT (Gly). A binder touching the p6
  sidechain is the only geometry that can possibly read the mutation.
- contact budget split: binder<->peptide vs binder<->HLA vs binder<->b2m
  residue-pair contacts (is the binder reading the peptide or the groove?)
- interface PAE: mean predicted aligned error over binder<->peptide
  residue pairs, and an ipSAE-style score 1/(1+(pae/d0)^2), d0=10 A,
  restricted to pairs in predicted contact (<10 A CA).

Output: audit/contact_audit.json (small). The same metrics run on the
control TCR — the known discriminator anchors what "peptide-engaged"
actually looks like in this screen.
"""

import json
from pathlib import Path

import modal

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = "/boltz-results"

results_volume = modal.Volume.from_name("protein-boltz-results")
image = modal.Image.debian_slim(python_version="3.12").pip_install("gemmi", "numpy")
app = modal.App("lemma-contact-audit")


def classify(unit_yaml: str) -> dict:
    """chain id -> role, by sequence identity (b2m ~99 aa collides with
    binder lengths, so classify on sequence, not length)."""
    import re

    B2M_PREFIX = "IQRTPKIQ"
    roles: dict[str, list[str]] = {}
    pep = None
    for cid, seq in re.findall(r"id:\s*(\w+)\s*\n\s*sequence:\s*(\w+)", unit_yaml):
        if len(seq) == 10 and seq.startswith("VVVG"):
            role, pep = "peptide", seq
        elif seq.startswith("GSHSMR"):
            role = "hla"
        elif seq.startswith(B2M_PREFIX):
            role = "b2m"
        else:
            role = "binder"
        roles.setdefault(role, []).append(cid)
    return {"chains": roles, "peptide_seq": pep}


@app.function(image=image, volumes={RESULTS_DIR: results_volume}, timeout=3600)
def contact_audit(jobs: list[dict]) -> dict:

    import gemmi
    import numpy as np

    root = Path(RESULTS_DIR)
    # Index every prediction dir by unit name (concurrent jobs wrote into
    # shared dirs, so locate by glob rather than assuming <name>/ layout).
    index: dict[str, Path] = {}
    for cif in root.glob("*/boltz_results_*/predictions/*/*_model_0.cif"):
        index[cif.stem.replace("_model_0", "")] = cif
    for cif in root.glob("*/*/predictions/*/*_model_0.cif"):
        index.setdefault(cif.stem.replace("_model_0", ""), cif)

    out = {"units": {}, "n_indexed": len(index)}

    for job in jobs:
        name = job["name"]
        cif_p = index.get(name)
        if cif_p is None:
            out["units"][name] = {"error": "no cif on volume"}
            continue
        pred_dir = cif_p.parent
        st = gemmi.read_structure(str(cif_p))[0]

        roles = job["chains"]
        pep_ids = roles.get("peptide", [])
        binder_ids = roles.get("binder", [])
        receptor_ids = roles.get("hla", []) + roles.get("b2m", [])

        # per-chain residue -> [(atom_name, xyz)]
        chains: dict[str, list[dict]] = {}
        for ch in st:
            residues = []
            for res in ch:
                atoms = [
                    (a.name, np.array([a.pos.x, a.pos.y, a.pos.z]))
                    for a in res
                    if a.element.name != "H"
                ]
                ca = next((xyz for an, xyz in atoms if an in ("CA", "P")), None)
                residues.append(
                    {
                        "atoms": atoms,
                        "ca": ca,
                        "resname": res.name,
                        "seqid": str(res.seqid),
                    }
                )
            chains[ch.name] = residues

        def atoms_of(ids, chs=chains):
            return [
                (cid, ri, an, xyz)
                for cid in ids
                for ri, r in enumerate(chs.get(cid, []))
                for an, xyz in r["atoms"]
            ]

        binder_atoms = atoms_of(binder_ids)
        receptor_atoms = atoms_of(receptor_ids)
        if not binder_atoms:
            out["units"][name] = {
                "error": "no binder atoms",
                "chains_seen": list(chains),
            }
            continue
        b_xyz = np.array([a[3] for a in binder_atoms])

        res: dict = {"chains": dict(roles.items())}

        # ---- receptor engagement (works for apo too) ----
        r_xyz = np.array([a[3] for a in receptor_atoms])
        d_bR = np.sqrt(((b_xyz[:, None] - r_xyz[None]) ** 2).sum(-1))
        res["binder_receptor_min_dist"] = round(float(d_bR.min()), 2)

        pep_metrics: dict = {}
        if pep_ids and pep_ids[0] in chains:
            pep_res = chains[pep_ids[0]]
            pep_atoms = atoms_of(pep_ids)
            p_xyz = np.array([a[3] for a in pep_atoms])
            d_bP = np.sqrt(
                ((b_xyz[:, None] - p_xyz[None]) ** 2).sum(-1)
            )  # binder x pep-atom

            # per-peptide-residue profile
            atom_cursor = 0
            profile = []
            for ri, r in enumerate(pep_res):
                n = len(r["atoms"])
                dcol = d_bP[:, atom_cursor : atom_cursor + n]
                atom_cursor += n
                profile.append(
                    {
                        "pos": ri + 1,
                        "resname": r["resname"],
                        "min_dist": round(float(dcol.min()), 2),
                        "n_contacts_5A": int((dcol < 5.0).sum()),
                    }
                )
            pep_metrics["peptide_profile"] = profile
            pep_metrics["peptide_residues_contacted"] = sum(
                1 for p in profile if p["n_contacts_5A"] > 0
            )
            pep_metrics["peptide_contact_fraction"] = round(
                pep_metrics["peptide_residues_contacted"] / max(len(profile), 1), 3
            )
            pep_metrics["binder_peptide_min_dist"] = round(float(d_bP.min()), 2)

            # p6 = peptide residue index 6 (the discriminating position)
            p6 = next((p for p in profile if p["pos"] == 6), None)
            if p6:
                pep_metrics["p6"] = p6
                pep_metrics["p6_contacted"] = p6["n_contacts_5A"] > 0

            # contact budget: residue-pair counts <5A
            pep_pair_contacts = int(sum(p["n_contacts_5A"] for p in profile))
            pep_metrics["binder_peptide_atom_contacts_5A"] = pep_pair_contacts

        # receptor contacts
        res["binder_receptor_atom_contacts_5A"] = int((d_bR < 5.0).sum())
        if pep_metrics:
            res["peptide"] = pep_metrics
            res["peptide_vs_receptor_contacts"] = round(
                pep_metrics["binder_peptide_atom_contacts_5A"]
                / max(res["binder_receptor_atom_contacts_5A"], 1),
                3,
            )

        # ---- interface PAE ----
        pae_files = list(pred_dir.glob(f"pae_{name}_model_0.npz"))
        if pae_files:
            npz = np.load(pae_files[0])
            pae = npz[npz.files[0]]
            # residue index ranges by chain order in the yaml (== cif order)
            order = [
                cid
                for role in ("hla", "b2m", "peptide", "binder")
                for cid in roles.get(role, [])
            ]
            order = [c for c in order if c in chains]
            if pae.shape[0] == sum(len(chains[c]) for c in order):
                idx = {}
                off = 0
                for c in order:
                    idx[c] = range(off, off + len(chains[c]))
                    off += len(chains[c])
                b_idx = [i for cid in binder_ids if cid in idx for i in idx[cid]]
                if pep_ids and pep_ids[0] in idx:
                    p_idx = list(idx[pep_ids[0]])
                    sub = pae[np.ix_(b_idx, p_idx)]
                    pep_metrics["interface_pae_mean"] = round(float(sub.mean()), 2)
                    pep_metrics["interface_pae_min"] = round(float(sub.min()), 2)
                    # CA-distance-restricted ipSAE-style score
                    ca_b = np.array(
                        [
                            r["ca"]
                            for cid in binder_ids
                            for r in chains.get(cid, [])
                            if r["ca"] is not None
                        ]
                    )
                    ca_p = np.array([r["ca"] for r in pep_res if r["ca"] is not None])
                    if len(ca_b) and len(ca_p):
                        dca = np.sqrt(((ca_b[:, None] - ca_p[None]) ** 2).sum(-1))
                        mask = dca < 10.0
                        vals = sub[: len(ca_b), : len(ca_p)][mask]
                        if len(vals):
                            pep_metrics["ipsae_d0_10"] = round(
                                float((1 / (1 + (vals / 10.0) ** 2)).mean()), 3
                            )
                            pep_metrics["interface_pae_contacts"] = round(
                                float(vals.mean()), 2
                            )
            else:
                pep_metrics["pae_shape_mismatch"] = list(pae.shape)
        if pep_metrics and "peptide" not in res:
            res["peptide"] = pep_metrics
        out["units"][name] = res

    return out


@app.local_entrypoint()
def main() -> None:
    wd = REPO / "targets/kras-g12d-pmhc"
    jobs = []
    for manifest in [
        wd / "audit/manifest.json",
        wd / "audit/control_tcr/manifest.json",
    ]:
        if manifest.is_file():
            for unit in json.loads(manifest.read_text()):
                meta = classify(unit["yaml"])
                jobs.append({"name": unit["name"], **meta})
    print(f"{len(jobs)} units to audit at contact level")
    result = contact_audit.remote(jobs)
    out_p = wd / "audit" / "contact_audit.json"
    out_p.write_text(json.dumps(result, indent=1))
    errs = {n: u for n, u in result["units"].items() if "error" in u}
    print(
        f"wrote {out_p} — {len(result['units'])} units, "
        f"{len(errs)} errors, {result['n_indexed']} cifs indexed"
    )
    if errs:
        for n, u in list(errs.items())[:10]:
            print(f"  {n}: {u}")
