"""Modal-hosted RFdiffusion + ProteinMPNN binder design for PROTEIN.md.

Run:
  modal run scripts/modal_rfd.py --doctor          # image sanity check
  modal run scripts/modal_rfd.py --n 5             # generate 5 backbones
                                                 # + ProteinMPNN sequences

RFdiffusion needs DGL/SE3-Transformer — a conda-era dependency stack that
pip can still satisfy: torch 2.4.x (bundled cu121) + dgl from the dgl team
wheel index. Model weights persist in the modal.Volume "rfd-cache".

Design contract (PROTEIN.md): every backbone/sequence produced here is a
hypothesis, not a claim — the audit stage scores it against the decoys.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import modal

app = modal.App("lemma-protein-rfd")

CACHE_DIR = "/rfd-cache"
cache_volume = modal.Volume.from_name("rfd-cache", create_if_missing=True)

RFD_MODELS = {
    "Base_ckpt.pt": "http://files.ipd.uw.edu/pub/RFdiffusion/"
    "6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt",
    "Complex_base_ckpt.pt": "http://files.ipd.uw.edu/pub/RFdiffusion/"
    "e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt",
}

rfd_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "build-essential")
    .pip_install(
        "torch==2.4.1",
        "numpy<2",
        "scipy",
        "pandas",
        "hydra-core",
        "omegaconf",
        "icecream",
        "pyrsistent",
        "decorator",
        "psutil",
        "pynvml",
        "e3nn",
        "opt_einsum",
    )
    .run_commands(
        "pip install 'dgl==2.4.0' "
        "-f https://data.dgl.ai/wheels/torch-2.4/cu121/repo.html",
        "git clone --depth 1 https://github.com/RosettaCommons/RFdiffusion "
        "/opt/RFdiffusion",
        "pip install /opt/RFdiffusion/env/SE3Transformer",
        "git clone --depth 1 https://github.com/dauparas/ProteinMPNN /opt/ProteinMPNN",
    )
)


@app.function(image=rfd_image, gpu="a10g", timeout=1800)
def doctor() -> dict:
    """Verify the fiddly import chain inside the RFD image."""
    import subprocess

    out = {}

    def check(mod, attr=None):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, attr) if attr else getattr(m, "__version__", "ok")
        except Exception as e:
            out[mod] = f"FAIL {type(e).__name__}: {e}"

    check("torch", "__version__")
    import torch

    out["cuda_available"] = torch.cuda.is_available()
    check("dgl", "__version__")
    check("e3nn", "__version__")
    check("se3_transformer")
    check("rfdiffusion")
    out["run_inference.py"] = subprocess.run(
        ["ls", "/opt/RFdiffusion/scripts/run_inference.py"],
        capture_output=True,
        text=True,
    ).returncode
    out["mpnn"] = subprocess.run(
        ["ls", "/opt/ProteinMPNN/protein_mpnn_run.py"],
        capture_output=True,
        text=True,
    ).returncode
    import json

    # str, not dict: the client side has no torch and any pickled value
    # that references it fails to deserialize locally.
    return json.dumps({k: str(v) for k, v in out.items()})


def _ensure_models() -> None:
    import subprocess
    from pathlib import Path as P

    models = P(CACHE_DIR) / "models"
    models.mkdir(parents=True, exist_ok=True)
    for name, url in RFD_MODELS.items():
        dest = models / name
        if not dest.exists():
            subprocess.run(["wget", "-q", "-O", str(dest), url], check=True)
    cache_volume.commit()


@app.function(
    image=rfd_image,
    gpu="a10g",
    timeout=7200,
    memory=32768,
    volumes={CACHE_DIR: cache_volume},
)
def rfd_design(
    pdb_text: str,
    hotspots: list[str],
    binder_len: str = "80-120",
    n: int = 5,
    contig: str | None = None,
    seed: int | None = None,
) -> dict:
    """Diffuse `n` binder backbones onto the target PDB + run ProteinMPNN.

    Returns timings, logs, and output files (backbone pdbs + mpnn fasta).
    """
    import subprocess
    import time
    from pathlib import Path as P

    _ensure_models()
    work = P("/tmp/rfd")
    (work / "out").mkdir(parents=True, exist_ok=True)
    inp = work / "target.pdb"
    inp.write_text(pdb_text)

    if contig is None:
        # 7OW6 pMHC chains: A=HLA 276, B=b2m 99, C=peptide 10 — all fixed,
        # then a 0-gap diffused binder segment.
        # Each receptor chain needs its own /0 chain break — '/' alone reads
        # them as fragments of ONE chain and RFD asserts.
        contig = f"A1-276/0 B1-99/0 C1-10/0 {binder_len}"

    cmd = [
        "python",
        "/opt/RFdiffusion/scripts/run_inference.py",
        "inference.output_prefix=/tmp/rfd/out/rfd",
        f"inference.input_pdb={inp}",
        f"contigmap.contigs=[{contig}]",
        f"ppi.hotspot_res={hotspots}",
        f"inference.num_designs={n}",
        f"inference.model_directory_path={CACHE_DIR}/models",
        f"inference.ckpt_override_path={CACHE_DIR}/models/Complex_base_ckpt.pt",
    ]
    if seed is not None:
        cmd.append(f"inference.seed={seed}")

    env = {"PYTHONPATH": "/opt/RFdiffusion", "PATH": "/usr/local/bin:/usr/bin:/bin"}
    t0 = time.monotonic()
    rfd = subprocess.run(
        cmd,
        cwd="/opt/RFdiffusion",
        env=env,
        capture_output=True,
        text=True,
        timeout=6000,
    )
    rfd_wall = time.monotonic() - t0

    backbones = sorted((work / "out").glob("*.pdb"))
    t0 = time.monotonic()
    fastas, mpnn_logs, mpnn_rc = _run_mpnn(backbones, work)
    mpnn_wall = time.monotonic() - t0

    files = {}
    for f in list(backbones) + list(fastas):
        if f.stat().st_size < 20 * 1024 * 1024:
            files[
                f"{f.parent.parent.name}/{f.parent.name}/{f.name}"
                if f.parent.name != "out"
                else f.name
            ] = f.read_bytes()

    return {
        "rfd_wall_s": round(rfd_wall, 1),
        "mpnn_wall_s": round(mpnn_wall, 1),
        "rfd_rc": rfd.returncode,
        "rfd_log_tail": (rfd.stdout + rfd.stderr)[-2500:],
        "n_backbones": len(backbones),
        "backbones": [b.name for b in backbones],
        "mpnn_rc": mpnn_rc,
        "mpnn_log_tail": (mpnn_logs or [""])[-1][-1500:],
        "files": files,
    }


def _run_mpnn(backbones, work) -> tuple:
    """Canonical ProteinMPNN 3-step over all backbones at once:
    parse -> assign (design binder chain, others visible) -> run.

    Returns (fasta_paths, logs, worst_rc)."""
    import subprocess
    from pathlib import Path as P

    PMPNN = P("/opt/ProteinMPNN")
    pdbs = work / "pdbs"
    pdbs.mkdir(exist_ok=True)
    binder = "D"
    for bb in backbones:
        dest = pdbs / bb.name
        if not dest.exists():
            dest.write_bytes(bb.read_bytes())
        chains = sorted(
            {ln[21] for ln in bb.read_text().splitlines() if ln.startswith("ATOM")}
        )
        binder = chains[-1]  # RFD emits the designed chain last (D here)

    out_dir = work / "mpnn"
    out_dir.mkdir(parents=True, exist_ok=True)
    parsed = out_dir / "parsed.jsonl"
    assigned = out_dir / "assigned.jsonl"
    logs, worst_rc = [], 0
    for cmd in [
        [
            "python",
            str(PMPNN / "helper_scripts/parse_multiple_chains.py"),
            "--input_path",
            str(pdbs),
            "--output_path",
            str(parsed),
        ],
        [
            "python",
            str(PMPNN / "helper_scripts/assign_fixed_chains.py"),
            "--input_path",
            str(parsed),
            "--output_path",
            str(assigned),
            "--chain_list",
            binder,
        ],
        [
            "python",
            str(PMPNN / "protein_mpnn_run.py"),
            "--jsonl_path",
            str(parsed),
            "--chain_id_jsonl",
            str(assigned),
            "--out_folder",
            str(out_dir),
            "--num_seq_per_target",
            "8",
            "--sampling_temp",
            "0.1",
            "--batch_size",
            "8",
            "--seed",
            "37",
        ],
    ]:
        m = subprocess.run(
            cmd,
            cwd=str(PMPNN),
            capture_output=True,
            text=True,
            timeout=1800,
        )
        logs.append(m.stderr[-600:] + "\n" + m.stdout[-600:])
        if m.returncode != 0:
            worst_rc = m.returncode
            logs.append(f"STEP FAILED rc={m.returncode}: {' '.join(cmd)}")
            break
    return list(out_dir.rglob("*.fa")), logs, worst_rc


@app.local_entrypoint()
def main(
    n: int = 5,
    check: bool = False,
    out: str = "",
    binder_len: str = "80-120",
    seed: int = 0,
    hotspots: str = "C4,C5,C6,C7,C8",
):
    repo = Path(__file__).resolve().parent.parent
    if check:
        print(doctor.remote())
        return

    pdb = repo / "targets/kras-g12d-pmhc/inputs/7ow6_pmhc.pdb"
    hotspot_list = [h.strip() for h in hotspots.split(",") if h.strip()]
    print(f"submitting rfd design: n={n} len={binder_len} hotspots={hotspot_list}")
    result = rfd_design.remote(
        pdb.read_text(),
        hotspot_list,
        binder_len=binder_len,
        n=n,
        seed=seed or None,
    )
    print(
        f"rfd {result['rfd_wall_s']}s rc={result['rfd_rc']} "
        f"backbones={result['n_backbones']} | mpnn {result['mpnn_wall_s']}s "
        f"rc={result.get('mpnn_rc')}"
    )
    if result["rfd_rc"] != 0:
        print(result["rfd_log_tail"])
    if result.get("mpnn_rc"):
        print(result["mpnn_log_tail"])

    dest = Path(out) if out else (repo / "targets/kras-g12d-pmhc/designs/_rfd_round1")
    for rel, data in result["files"].items():
        p = dest / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    print(f"saved {len(result['files'])} files under {dest}")
