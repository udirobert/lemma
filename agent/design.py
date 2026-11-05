"""`lemma design` — protein-design pipeline (PROTEIN.md).

Mirrors `lemma audit` stage-for-stage:

  1. propose   target.json + rated literature -> design_hypotheses.json
  2. design    hypotheses -> designs/<did>/: sequence + why-card + novelty
               attestation (no derived-from-natural-binder)
  3. audit     every candidate through the anti-objectives IN SILICO:
               complex scoring on the primary target AND on each decoy
               complex; decoy-failures are first-class results
  4. evidence  Trackio logbook per design + JSONL trace (as in audit)
  5. judge     design judge: submit <=max_designs / hold / reject, with
               reasons; prepares the submission bundle

Status: stage skeleton. Each stage validates its declared inputs and logs
what it would do; stages without implementations report "pending" rather
than fabricating outputs. The judge stage emits a real judge.json (a HOLD
verdict is the honest state while no candidates exist).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from agent.llm import complete, extract_json, model_name
from agent.traces import Trace
from agent.weave_ops import op

STAGES = ("propose", "design", "audit", "evidence", "judge")

TARGET_JSON = "target.json"
HYPOTHESES_JSON = "design_hypotheses.json"
JUDGE_JSON = "judge.json"

SYSTEM = """You are the hypothesis stage of Lemma's protein-design pipeline. \
Lemma designs only what it can falsify: a design hypothesis is a claim, and \
it must state up front HOW it could be wrong. You think like a structural \
biologist who has read too many overpromised binder papers.

Rules:
- Every hypothesis must name its discrimination MECHANISM against the \
decoys — "high affinity to target" without a reason the decoys fail is \
worthless here.
- Reason from chemistry/geometry, not vibes: which atoms of the binder \
touch which atoms of the peptide/groove, and why that contact pattern \
depends on the mutant side chain or the peptide's presence.
- Prefer mechanisms that give affinity *and* specificity from the same \
contact set (a salt bridge to the mutant residue is worth more than an \
extra hydrophobic patch that also grabs the wild-type).
- State what would kill each hypothesis (failure_modes) honestly.
- Never propose deriving from a known natural binder — de novo only.
- Return ONLY a JSON array."""

USER_TMPL = """Propose design hypotheses for this target.

Target spec:
{target_json}

Wet-lab objective ranking (the decoys ARE the ranking):
1. mutation specificity — bind {primary_id}, no detectable binding to the \
single-mutation decoy
2. peptide-dependence — no binding to the peptide-free complex
3. affinity — lowest KD against {primary_id}

Constraints: single chain, 10-250 aa, <=20 designs total, de novo \
zero-shot (no natural-binder derivation), declared surface: {surface}.

{literature_block}
Return a JSON array of 3-5 hypothesis objects with keys:
id (H1, H2, ...), title, binder_format (nanobody|scfv|fab|general), \
scaffold_rationale, target_surface (concrete: which peptide positions / \
groove helices / atoms), mechanism (how it binds), discrimination_hypothesis \
(why the mutant decoy fails), peptide_dependence_hypothesis (why the apo \
decoy fails), generation_plan (which model/toolchain could produce \
sequences for this), failure_modes (array), risk (low|medium|high + why)."""


def load_target(workdir: Path) -> dict | None:
    p = workdir / TARGET_JSON
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _literature_context(target: dict, trace: Trace) -> list[dict]:
    """Best-effort paperclip hits for the propose prompt. Empty when the
    corpus is unavailable — propose does not depend on it."""
    try:
        from agent import paperclip

        ok, _why = paperclip.available()
        if not ok:
            trace.note("propose", f"paperclip unavailable ({_why}); no literature")
            return []
        query = f"{target.get('name', '')} binder"
        hits = paperclip.search(query, 4)
        trace.log("propose", "literature", query=query, hits=len(hits))
        return hits
    except Exception as e:  # honest degrade, never block the stage
        trace.note("propose", f"literature lookup failed ({e}); continuing")
        return []


def _validate_hypotheses(raw: object) -> list[dict]:
    if isinstance(raw, dict) and isinstance(raw.get("hypotheses"), list):
        raw = raw["hypotheses"]
    if not isinstance(raw, list) or not raw:
        raise ValueError("propose produced no hypothesis list")
    required = {
        "title",
        "binder_format",
        "target_surface",
        "discrimination_hypothesis",
        "generation_plan",
        "failure_modes",
    }
    out = []
    for i, h in enumerate(raw, 1):
        if not isinstance(h, dict):
            continue
        missing = required - h.keys()
        if missing:
            raise ValueError(f"hypothesis {i} missing keys: {sorted(missing)}")
        h = dict(h)
        h.setdefault("id", f"H{i}")
        if not isinstance(h.get("failure_modes"), list):
            h["failure_modes"] = [str(h["failure_modes"])]
        out.append(h)
    if not out:
        raise ValueError("propose produced only malformed hypotheses")
    return out


def _pending(trace: Trace, stage: str, reason: str) -> dict:
    trace.log(stage, "stage_pending", reason=reason)
    return {"stage": stage, "status": "pending", "reason": reason}


@op
def propose(workdir: Path, trace: Trace) -> dict:
    """target.json + literature -> design_hypotheses.json (LLM, real)."""
    target = load_target(workdir)
    if target is None:
        trace.log("propose", "stage_blocked", reason="missing target.json")
        return {
            "stage": "propose",
            "status": "blocked",
            "reason": "target.json missing",
        }

    out = workdir / HYPOTHESES_JSON
    if out.is_file():
        existing = json.loads(out.read_text(encoding="utf-8"))
        n = len(
            existing.get("hypotheses", existing if isinstance(existing, list) else [])
        )
        trace.log("propose", "reuse", path=str(out), n_hypotheses=n)
        return {"stage": "propose", "status": "reused", "n_hypotheses": n}

    literature = _literature_context(target, trace)
    lit_block = (
        "Literature context (unverified hits — treat as pointers, not ground "
        "truth):\n" + json.dumps(literature, indent=1)[:3000]
        if literature
        else "No literature context available; reason from the target spec."
    )
    primary = next(
        (c for c in target.get("complexes", []) if c.get("role") == "primary"),
        {"id": "primary"},
    )
    user = USER_TMPL.format(
        target_json=json.dumps(target, indent=1),
        primary_id=primary["id"],
        surface=target.get("declared_target_surface", "peptide + groove"),
        literature_block=lit_block,
    )
    raw_text = complete(trace, "propose", SYSTEM, user, temperature=0.4)
    # Provenance: the serving provider:model is whatever the last llm_call
    # event recorded — model_name() alone can't see endpoint fallback.
    llm_events = [
        r
        for r in trace.read()
        if r.get("event") == "llm_call" and r.get("stage") == "propose"
    ]
    served_model = llm_events[-1]["model"] if llm_events else model_name()
    try:
        hypotheses = _validate_hypotheses(extract_json(raw_text))
    except (ValueError, TypeError, KeyError) as e:
        trace.log(
            "propose",
            "llm_output_rejected",
            reason=str(e),
            response_tail=raw_text[-400:],
        )
        return {
            "stage": "propose",
            "status": "failed",
            "reason": f"LLM output did not validate: {e}",
        }

    payload = {
        "target_id": target.get("id"),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": served_model,
        "target_sha256": hashlib.sha256(
            json.dumps(target, sort_keys=True).encode()
        ).hexdigest()[:16],
        "literature_hits": len(literature),
        "hypotheses": hypotheses,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    trace.log(
        "propose",
        "generated",
        path=str(out),
        n_hypotheses=len(hypotheses),
        model=payload["model"],
    )
    return {
        "stage": "propose",
        "status": "done",
        "n_hypotheses": len(hypotheses),
        "reason": f"{len(hypotheses)} hypotheses -> {out.name}",
    }


def _parse_mpnn_fa(fa_path: Path) -> list[dict]:
    """MPNN fasta: first record is the native (all-Gly) placeholder, then
    >T=.., sample=N, score=.., global_score=.., seq_recovery=.."""
    seqs = []
    header = None
    for line in fa_path.read_text().splitlines():
        if line.startswith(">"):
            header = line[1:]
            continue
        if header is None:
            continue
        meta = dict(kv.strip().split("=", 1) for kv in header.split(",") if "=" in kv)
        if "sample" not in meta:  # native placeholder record
            header = None
            continue
        seqs.append(
            {
                "seq_id": f"{fa_path.stem}_s{meta['sample']}",
                "aa": line.strip(),
                "mpnn_score": float(meta.get("score", "nan")),
                "mpnn_global_score": float(meta.get("global_score", "nan")),
            }
        )
        header = None
    return seqs


@op
def design(workdir: Path, trace: Trace) -> dict:
    """hypotheses -> designs/<did>/ via Modal RFdiffusion + ProteinMPNN."""
    import subprocess

    hyp_path = workdir / HYPOTHESES_JSON
    if not hyp_path.is_file():
        trace.log("design", "stage_blocked", reason=f"missing {HYPOTHESES_JSON}")
        return {
            "stage": "design",
            "status": "blocked",
            "reason": f"run propose first ({HYPOTHESES_JSON} missing)",
        }
    payload = json.loads(hyp_path.read_text(encoding="utf-8"))
    hypotheses = payload.get("hypotheses", [])

    designs_dir = workdir / "designs"
    designs_dir.mkdir(exist_ok=True)
    existing = sorted(
        d for d in designs_dir.iterdir() if d.is_dir() and not d.name.startswith("_")
    )

    # organize any raw _rfd_round* dirs that haven't been turned into
    # designs/d* yet (a dropped-in batch counts as new work, not reuse)
    raw_dirs = sorted(
        d
        for d in designs_dir.iterdir()
        if d.is_dir()
        and d.name.startswith("_rfd")
        and not (d / ".organized").exists()
        and list(d.glob("rfd_*.pdb"))
    )
    if not raw_dirs and not existing:
        raw_dir = designs_dir / "_rfd_round1"
        n_backbones = 5
        repo = workdir.parent.parent
        cmd = [
            "modal",
            "run",
            str(repo / "scripts/modal_rfd.py"),
            "--n",
            str(n_backbones),
            "--out",
            str(raw_dir),
        ]
        t0 = time.time()
        proc = subprocess.run(
            cmd, cwd=repo, capture_output=True, text=True, timeout=7200
        )
        trace.tool_run(
            "design",
            " ".join(cmd),
            proc.returncode,
            time.time() - t0,
            len(proc.stdout) + len(proc.stderr),
        )
        if proc.returncode != 0 or not list(raw_dir.glob("rfd_*.pdb")):
            trace.log(
                "design",
                "generation_failed",
                rc=proc.returncode,
                tail=(proc.stdout + proc.stderr)[-600:],
            )
            return {
                "stage": "design",
                "status": "failed",
                "reason": "rfd generation produced no backbones",
            }
        raw_dirs = [raw_dir]

    hyp_ids = [h.get("id") for h in hypotheses]
    made = []
    next_i = (
        max(int(d.name[1:]) for d in existing if d.name[1:].isdigit()) + 1
        if existing
        else 1
    )
    for raw_dir in raw_dirs:
        for bb in sorted(raw_dir.glob("rfd_*.pdb")):
            did = f"d{next_i}"
            next_i += 1
            ddir = designs_dir / did
            ddir.mkdir(exist_ok=True)
            (ddir / "backbone.pdb").write_bytes(bb.read_bytes())
            fa = raw_dir / "mpnn" / "seqs" / f"{bb.stem}.fa"
            seqs = _parse_mpnn_fa(fa) if fa.is_file() else []
            if fa.is_file():
                (ddir / "seqs.fa").write_bytes(fa.read_bytes())
            for s in seqs:
                s["seq_id"] = f"{did}_{s['seq_id'].split('_', 1)[-1]}"
            candidate = {
                "id": did,
                "backbone": bb.name,
                "batch": raw_dir.name,
                "binder_len_aa": len(seqs[0]["aa"]) if seqs else None,
                "sequences": seqs,
                "hypotheses": hyp_ids,
                "generation": {
                    "method": "rfdiffusion Complex_base + proteinmpnn v_48_020",
                    "batch_dir": raw_dir.name,
                    "mpnn_sampling_temp": 0.1,
                    "seed": 37,
                    "compute": "modal a10g",
                },
                "novelty_attestation": "de novo RFD backbone; not derived "
                "from any natural binder. Sequence novelty NOT yet checked "
                "vs reference DBs — flagged for the audit stage.",
            }
            (ddir / "candidate.json").write_text(
                json.dumps(candidate, indent=2) + "\n", encoding="utf-8"
            )
            why = "\n".join(
                [
                    f"# {did} — why this design exists",
                    "",
                    f"- Batch: `{raw_dir.name}` (contig + hotspots recorded "
                    "in the batch command).",
                    f"- Intent (propose stage): {', '.join(hyp_ids)} — a "
                    "binder that reads the p6-Asp / peptide region and "
                    "cannot be satisfied by WT p6-Gly or the apo groove.",
                    "- Mechanism claimed: RFD placed this backbone over "
                    "peptide hotspots; MPNN proposed sequences for chain D.",
                    "- What audit must falsify: does the designed sequence "
                    "actually contact p6-Asp, and does it fail on pmhc_wt + "
                    "hla_apo? A backbone near the peptide is not proof.",
                    "",
                ]
            )
            (ddir / "why.md").write_text(why, encoding="utf-8")
            made.append(did)
        (raw_dir / ".organized").write_text(
            f"{len(made)} designs organized\n", encoding="utf-8"
        )

    trace.log("design", "generated", designs=made, n_seqs=None)
    return {
        "stage": "design",
        "status": "done",
        "n_designs": len(made),
        "reason": f"{len(made)} candidates under designs/",
    }


def _complex_yaml(target: dict, complex_id: str, binder_seq: str) -> str:
    """Boltz YAML for one complex + binder as chain D."""
    comp = next(c for c in target["complexes"] if c["id"] == complex_id)
    chains = target["chains"]
    parts = ["version: 1", "sequences:"]
    letter = "A"
    for key in comp["chains"]:
        parts.append(
            f"  - protein:\n      id: {letter}\n"
            f"      sequence: {chains[key]['sequence']}"
        )
        letter = chr(ord(letter) + 1)
    parts.append(f"  - protein:\n      id: {letter}\n      sequence: {binder_seq}")
    return "\n".join(parts) + "\n"


@op
def audit(workdir: Path, trace: Trace) -> dict:
    """Score top MPNN seqs per candidate on primary + both decoy complexes.

    Every prediction is kept — decoy failures are first-class results.
    Metric per complex: boltz complex iptm + binder->peptide pair iptm.
    """
    import subprocess

    target = load_target(workdir)
    if not target:
        return {"stage": "audit", "status": "blocked", "reason": "target.json missing"}
    designs_dir = workdir / "designs"
    candidates = (
        sorted(
            d
            for d in designs_dir.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        )
        if designs_dir.is_dir()
        else []
    )
    if not candidates:
        trace.log("audit", "nothing_to_do", reason="no candidates in designs/")
        return {"stage": "audit", "status": "skipped", "reason": "no candidates"}

    audit_dir = workdir / "audit"
    audit_dir.mkdir(exist_ok=True)
    complexes = [c["id"] for c in target["complexes"]]
    chain_keys = {c["id"]: c["chains"] for c in target["complexes"]}

    jobs = []
    units = []  # (did, seq_id, complex_id, out_dir)
    for ddir in candidates:
        cand = json.loads((ddir / "candidate.json").read_text())
        seqs = sorted(cand["sequences"], key=lambda s: s["mpnn_global_score"])[
            :2
        ]  # top-2 MPNN seqs — bounds compute, recorded not hidden
        for s in seqs:
            for cx in complexes:
                name = f"{cand['id']}_{s['seq_id']}_{cx}"
                out_dir = audit_dir / cand["id"] / s["seq_id"] / cx
                if next(out_dir.rglob("confidence_*_model_0.json"), None):
                    continue  # idempotent — prior predictions kept
                jobs.append(
                    {
                        "name": name,
                        "out_dir": str(out_dir),
                        "yaml": _complex_yaml(target, cx, s["aa"]),
                    }
                )
                units.append((cand["id"], s["seq_id"], cx))

    if jobs:
        manifest = audit_dir / "manifest.json"
        manifest.write_text(json.dumps(jobs, indent=1), encoding="utf-8")
        repo = workdir.parent.parent
        cmd = [
            "modal",
            "run",
            str(repo / "scripts/modal_boltz_smoke.py"),
            "--manifest",
            str(manifest),
        ]
        t0 = time.time()
        proc = subprocess.run(
            cmd, cwd=repo, capture_output=True, text=True, timeout=7200
        )
        trace.tool_run(
            "audit",
            " ".join(cmd),
            proc.returncode,
            time.time() - t0,
            len(proc.stdout) + len(proc.stderr),
        )
        if proc.returncode != 0:
            trace.log(
                "audit", "scoring_failed", tail=(proc.stdout + proc.stderr)[-600:]
            )
            return {
                "stage": "audit",
                "status": "failed",
                "reason": "boltz batch failed; see trace",
            }
        trace.log("audit", "scored", n_predictions=len(jobs))
    else:
        trace.log("audit", "reuse", reason="all predictions present")

    # Collect confidence metrics -> per-seq anti-objective verdicts.
    report = {
        "units": [],
        "note": "boltz iptm is a docking-confidence "
        "proxy, not a KD — verdicts are screen-level calls",
    }
    for ddir in candidates:
        cand = json.loads((ddir / "candidate.json").read_text())
        for s in sorted(cand["sequences"], key=lambda s: s["mpnn_global_score"])[:2]:
            unit = {
                "did": cand["id"],
                "seq_id": s["seq_id"],
                "aa": s["aa"],
                "complexes": {},
            }
            for cx in complexes:
                cdir = audit_dir / cand["id"] / s["seq_id"] / cx
                conf = next(cdir.rglob("confidence_*_model_0.json"), None)
                if conf is None:
                    unit["complexes"][cx] = {"error": "no confidence output"}
                    continue
                c = json.loads(conf.read_text())
                pair = c.get("pair_chains_iptm", {})
                binder = str(len(chain_keys[cx]))  # last chain index
                peptide_idx = None
                for i, k in enumerate(chain_keys[cx]):
                    if "peptide" in k:
                        peptide_idx = str(i)
                unit["complexes"][cx] = {
                    "iptm": c.get("iptm"),
                    "binder_peptide_iptm": (
                        pair.get(binder, {}).get(peptide_idx)
                        if peptide_idx is not None
                        else None
                    ),
                    "complex_plddt": c.get("complex_plddt"),
                }
            g = unit["complexes"].get("pmhc_g12d", {})
            w = unit["complexes"].get("pmhc_wt", {})
            a = unit["complexes"].get("hla_apo", {})
            gi, wi, ai = g.get("iptm"), w.get("iptm"), a.get("iptm")
            if None in (gi, wi, ai):
                verdict = "unscored"
            elif wi is not None and ai is not None:
                if gi >= 0.6 and gi - wi >= 0.15 and gi - ai >= 0.15:
                    verdict = "passes_screen"
                elif wi >= gi - 0.05 or ai >= gi - 0.05:
                    verdict = "decoy_leak"
                elif gi < 0.5:
                    verdict = "no_signal"
                else:
                    verdict = "borderline"
            unit["verdict"] = verdict
            report["units"].append(unit)

    n_pass = sum(1 for u in report["units"] if u["verdict"] == "passes_screen")
    (audit_dir / "audit_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    trace.log(
        "audit",
        "verdicts",
        counts={
            v: sum(1 for u in report["units"] if u["verdict"] == v)
            for v in {u["verdict"] for u in report["units"]}
        },
    )
    return {
        "stage": "audit",
        "status": "done",
        "reason": f"{n_pass}/{len(report['units'])} seqs pass the "
        "anti-objective screen",
    }


@op
def evidence(workdir: Path, trace: Trace) -> dict:
    """Trackio logbook per design + trace attach (not yet built)."""
    audit_dir = workdir / "audit"
    if not audit_dir.is_dir():
        trace.log("evidence", "stage_blocked", reason="no audit/ results yet")
        return {
            "stage": "evidence",
            "status": "blocked",
            "reason": "no audit results to assemble",
        }
    return _pending(trace, "evidence", "logbook assembly not implemented")


@op
def judge(workdir: Path, trace: Trace) -> dict:
    """Design judge: submit / hold / reject with reasons -> judge.json.

    Unlike the upstream stages this one is already honest output: with no
    candidates the only defensible verdict is HOLD.
    """
    target = load_target(workdir) or {}
    designs_dir = workdir / "designs"
    candidates = (
        sorted(
            d.name
            for d in designs_dir.iterdir()
            if d.is_dir() and not d.name.startswith("_")
        )
        if designs_dir.is_dir()
        else []
    )
    audited_dir = workdir / "audit"
    n_audited = (
        sum(1 for d in audited_dir.iterdir() if d.is_dir())
        if audited_dir.is_dir()
        else 0
    )
    max_designs = target.get("constraints", {}).get("max_designs", 20)

    if not candidates:
        verdict, reasons = (
            "hold",
            [
                "no candidate designs exist yet",
                "propose/design stages not implemented (skeleton run)",
            ],
        )
    elif n_audited < len(candidates):
        verdict, reasons = (
            "hold",
            [
                f"{len(candidates) - n_audited} candidate(s) not yet through "
                "anti-objective audit",
            ],
        )
    else:
        audit_report_path = audited_dir / "audit_report.json"
        units = []
        if audit_report_path.is_file():
            units = json.loads(audit_report_path.read_text()).get("units", [])
        passing = [u for u in units if u.get("verdict") == "passes_screen"]
        if not passing:
            counts = {}
            for u in units:
                counts[u["verdict"]] = counts.get(u["verdict"], 0) + 1
            verdict = "reject"
            reasons = [
                f"no sequences pass the anti-objective screen: {counts}",
                "do NOT submit — the decoys are the ranking; a decoy_leak "
                "design ranks at the bottom",
            ]
        else:
            verdict = "submit"
            reasons = [
                f"{len(passing)} sequence(s) pass in-silico anti-objectives: "
                f"{[u['seq_id'] for u in passing]}",
                "boltz iptm is a docking proxy, not a measured KD — say so "
                "in the method note",
            ]
    report = {
        "target": target.get("id"),
        "verdict": verdict,
        "reasons": reasons,
        "candidates": candidates,
        "n_audited": n_audited,
        "max_designs": max_designs,
    }
    out = workdir / JUDGE_JSON
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    trace.log("judge", "verdict", verdict=verdict, reasons=reasons, path=str(out))
    return {"stage": "judge", "status": "done", "verdict": verdict}


STAGE_FNS = {
    "propose": propose,
    "design": design,
    "audit": audit,
    "evidence": evidence,
    "judge": judge,
}


def run(workdir: Path, stages: list[str], trace: Trace) -> int:
    """Run the requested design stages; returns a process exit code."""
    rc = 0
    for name in stages:
        fn = STAGE_FNS.get(name)
        if fn is None:
            trace.log(name, "unknown_stage")
            continue
        result = fn(workdir, trace)
        status = result.get("status", "?")
        detail = result.get("verdict") or result.get("reason") or ""
        print(f"[lemma]   {name}: {status} {detail}".rstrip())
        if status in ("blocked", "failed"):
            rc = 1
    return rc
