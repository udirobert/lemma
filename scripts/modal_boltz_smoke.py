"""Modal-hosted Boltz-2 smoke test for the kras-g12d-pmhc target.

Run:  modal run scripts/modal_boltz_smoke.py [--input pmhc_g12d]

Runs `boltz predict` on a real GPU (L4) inside Modal instead of the local
8 GB Mac. Model weights + CCD cache persist in the modal.Volume
"boltz-cache" across runs; prediction outputs come back as file bytes and
are saved under targets/kras-g12d-pmhc/smoke/modal_<name>/.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import modal

app = modal.App("lemma-protein-smoke")

CACHE_DIR = "/boltz-cache"
cache_volume = modal.Volume.from_name("boltz-cache", create_if_missing=True)

# Every finished prediction is also written here inside the remote
# function, so a client disconnect cannot lose completed work —
# recover with `modal volume get protein-boltz-results <name> <dir>`.
RESULTS_DIR = "/boltz-results"
results_volume = modal.Volume.from_name("protein-boltz-results", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.12").pip_install("boltz")

chai_image = modal.Image.debian_slim(python_version="3.12").pip_install("chai_lab")

CHAI_CACHE = "/chai-cache"
chai_volume = modal.Volume.from_name("chai-cache", create_if_missing=True)


@app.function(
    image=chai_image,
    gpu="L4",
    timeout=3600,
    memory=32768,
    volumes={CHAI_CACHE: chai_volume},
)
def chai_predict(fasta_name: str, fasta_text: str) -> dict:
    import os
    import time
    from pathlib import Path as P

    os.environ["CHAI_DOWNLOADS_DIR"] = CHAI_CACHE

    import torch
    from chai_lab.chai1 import run_inference

    work = P("/tmp/work")
    work.mkdir(exist_ok=True)
    fasta = work / f"{fasta_name}.fasta"
    fasta.write_text(fasta_text)
    out = work / "out"
    out.mkdir(exist_ok=True)

    t0 = time.monotonic()
    try:
        candidates = run_inference(
            fasta_file=fasta,
            output_dir=out,
            num_trunk_recycles=3,
            num_diffn_timesteps=200,
            num_diffn_samples=1,
            seed=42,
            device=torch.device("cuda:0"),
            use_esm_embeddings=True,
        )
        err = None
    except Exception as exc:  # honest failure, recorded not raised
        candidates = None
        err = f"{type(exc).__name__}: {exc}"
    wall_s = time.monotonic() - t0
    chai_volume.commit()

    files = {}
    for f in out.rglob("*"):
        if f.is_file() and f.stat().st_size < 20 * 1024 * 1024:
            files[str(f.relative_to(out))] = f.read_bytes()

    return {
        "wall_s": wall_s,
        "error": err,
        "n_candidates": None if candidates is None else len(candidates.cif_paths),
        "files": files,
    }


@app.function(
    image=image,
    gpu="a10g",  # L4 starved 2026-11-05 (queued undispatched); a10g dispatches
    timeout=3600,
    memory=32768,
    volumes={CACHE_DIR: cache_volume, RESULTS_DIR: results_volume},
)
def predict(
    yaml_name: str,
    yaml_text: str,
    use_msa_server: bool = True,
    no_kernels: bool = True,
) -> dict:
    import subprocess
    import time
    from pathlib import Path as P

    work = P("/tmp/work")
    work.mkdir(exist_ok=True)
    inp = work / f"{yaml_name}.yaml"
    inp.write_text(yaml_text)
    out = work / "out"

    cmd = [
        "boltz",
        "predict",
        str(inp),
        "--out_dir",
        str(out),
        "--accelerator",
        "gpu",
        "--cache",
        CACHE_DIR,
    ]
    if use_msa_server:
        cmd.append("--use_msa_server")
    if no_kernels:
        # cuequivariance_torch is not packaged for the image; without this
        # flag boltz crashes inside pairformer triangular_mult on CUDA.
        cmd.append("--no_kernels")

    t0 = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3300)
    wall_s = time.monotonic() - t0
    cache_volume.commit()

    files = {}
    if out.exists():
        for f in out.rglob("*"):
            if f.is_file() and f.stat().st_size < 20 * 1024 * 1024:
                files[str(f.relative_to(out))] = f.read_bytes()

    import json

    res_root = P(RESULTS_DIR) / yaml_name
    for rel, data in files.items():
        p = res_root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    (res_root / "_done.json").write_text(
        json.dumps({"exit_code": proc.returncode, "wall_s": round(wall_s, 1)})
    )
    results_volume.commit()

    return {
        "wall_s": wall_s,
        "exit_code": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "files": files,
    }


@app.local_entrypoint()
def main(input: str = "pmhc_g12d", model: str = "boltz", manifest: str = ""):
    if manifest:
        # Batch mode: manifest.json = [{"name":..., "out_dir":..., "yaml":...}]
        import json

        jobs = json.loads(Path(manifest).read_text())
        print(
            f"submitting {len(jobs)} boltz predictions (spawn, gpu=a10g; "
            "with --detach survives disconnect; results in "
            "protein-boltz-results)"
        )
        # spawn(), not starmap(): caller isn't blocked on delivery.
        # NOTE: spawn alone does NOT survive client disconnect (the
        # ephemeral app stops and kills the calls) — invoke with
        # `modal run --detach`; finished predictions then still land
        # in the protein-boltz-results volume for `modal volume get`.
        calls = [predict.spawn(j["name"], j["yaml"], True, True) for j in jobs]
        results = [c.get() for c in calls]
        for j, result in zip(jobs, results, strict=True):
            dest = Path(j["out_dir"])
            for rel, data in result["files"].items():
                p = dest / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(data)
            print(
                f"{j['name']}: wall_s={result['wall_s']:.1f} "
                f"exit={result['exit_code']} files={len(result['files'])}"
            )
            if result["exit_code"] != 0:
                print(result["stderr_tail"][-800:])
        return
    repo = Path(__file__).resolve().parent.parent
    inputs_dir = repo / "targets" / "kras-g12d-pmhc" / "inputs"
    dest = repo / "targets" / "kras-g12d-pmhc" / "smoke" / f"modal_{model}_{input}"

    if model == "boltz":
        src = inputs_dir / f"{input}.yaml"
        if not src.exists():
            print(f"no such input: {src}")
            sys.exit(1)
        print(f"submitting {input} to Modal boltz (gpu=L4) ...")
        result = predict.remote(input, src.read_text(), True)
        print(f"wall_s={result['wall_s']:.1f} exit={result['exit_code']}")
        if result["stderr_tail"]:
            print("--- stderr tail ---")
            print(result["stderr_tail"][-2000:])
        files = result["files"]
    elif model == "chai":
        src = inputs_dir / f"{input}.fasta"
        if not src.exists():
            print(f"no such input: {src}")
            sys.exit(1)
        print(f"submitting {input} to Modal chai-1 (gpu=L4) ...")
        result = chai_predict.remote(input, src.read_text())
        print(
            f"wall_s={result['wall_s']:.1f} error={result['error']} "
            f"candidates={result['n_candidates']}"
        )
        files = result["files"]
    else:
        print(f"unknown model: {model}")
        sys.exit(1)

    for rel, data in files.items():
        p = dest / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    print(f"saved {len(files)} files under {dest}")
