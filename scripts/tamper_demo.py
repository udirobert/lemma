#!/usr/bin/env python3
"""Tamper-evidence demonstration for the Apart AI x Epistemics sprint.

Copies a sealed-capable paper workdir to a temp dir, seals it, then runs four
attack scenarios and reports what ``lemma verify`` catches:

  A. Edit a verdict inside results/audit_report.json        (artifact tamper)
  B. Edit a record inside trace.jsonl                       (chain break)
  C. Append a properly-chained forged record post-seal       (needs head anchor)
  D. Re-seal the whole tampered workdir (resign everything)  (needs external anchor)

Usage: scripts/tamper_demo.py <workdir> [--keep]
Prints a JSON summary; exits 0 when every scenario behaves as designed.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LEMMA = REPO / "lemma"
sys.path.insert(0, str(REPO))

from agent.traces import Trace  # noqa: E402


def lemma(*args: str) -> tuple[int, str]:
    p = subprocess.run([str(LEMMA), *args], capture_output=True, text=True, cwd=REPO)
    return p.returncode, p.stdout + p.stderr


def verify(d: Path) -> dict:
    rc, out = lemma("verify", str(d))
    return {"rc": rc, **json.loads((out.splitlines()[0] and out) or "{}")}


def seal(d: Path) -> dict:
    rc, out = lemma("seal", str(d))
    assert rc == 0, out
    return json.loads((d / "provenance.json").read_text())


def flip_verdict(report: Path) -> None:
    data = json.loads(report.read_text())
    data["claims"][0]["status"] = (
        "falsified" if data["claims"][0]["status"] != "falsified" else "supported"
    )
    report.write_text(json.dumps(data, indent=1))


def main() -> int:
    src = Path(sys.argv[1]).resolve()
    keep = "--keep" in sys.argv
    tmp = Path(tempfile.mkdtemp(prefix="lemma-tamper-"))
    work = tmp / src.name
    ignore = shutil.ignore_patterns(".trackio", "__pycache__", "provenance.json")
    shutil.copytree(src, work, ignore=ignore)
    results: dict = {"workdir": str(work), "scenarios": {}}

    m = seal(work)
    results["seal"] = {
        "manifest_root": m["manifest_root"],
        "trace_head": m.get("trace_head_after_seal"),
        "n_files": m["n_files"],
    }
    base = verify(work)
    results["scenarios"]["baseline_clean"] = base["status"] == "ok"
    assert base["status"] == "ok", json.dumps(base, indent=2)

    report_path = work / "results" / "audit_report.json"
    orig = report_path.read_bytes()

    # A: tamper an artifact
    flip_verdict(report_path)
    r = verify(work)
    results["scenarios"]["A_edit_artifact"] = {
        "detected": r["status"] == "tampered"
        and "results/audit_report.json" in r["files"]["modified"],
        "modified": r["files"]["modified"],
    }
    report_path.write_bytes(orig)

    # B: tamper a trace record in place — flip one character in a sealed line
    tp = work / "trace.jsonl"
    raw = tp.read_text()
    lines = raw.splitlines()
    mid = len(lines) // 2
    target = lines[mid]
    for pos, ch in enumerate(target):
        if ch.isdigit():
            target = target[:pos] + ("0" if ch != "0" else "1") + target[pos + 1 :]
            break
    else:
        target = target[:-2] + "X" + target[-1]
    lines[mid] = target
    tp.write_text("\n".join(lines) + "\n")
    r = verify(work)
    results["scenarios"]["B_edit_trace_record"] = {
        "detected": r["status"] == "tampered"
        and (
            r["trace"].get("chain_problems")
            or any("prefix" in p for p in r["problems"])
        ),
        "problems": (r["problems"] + r["trace"].get("chain_problems", []))[:4],
    }
    tp.write_text(raw)

    # C: append a correctly-chained forged record (attacker knows the head —
    # it is in the file). Caught only because the head no longer matches the
    # seal recorded in provenance.json.
    run_id = json.loads(tp.read_text().splitlines()[0])["run_id"]
    t = Trace(run_id=run_id, path=tp)
    t.log("audit", "verdict", claim_id="C1", status="supported", forged=True)
    r = verify(work)
    results["scenarios"]["C_append_chained_forgery"] = {
        "detected": r["trace"].get("head_matches_seal") is False,
        "chain_still_valid": not r["trace"].get("chain_problems"),
        "note": "chain intact; caught only vs sealed head",
    }
    tp.write_text(raw)

    # D: attacker re-seals everything — provenance.json now self-consistent,
    # so only the EXTERNALLY published manifest_root detects it.
    flip_verdict(report_path)
    m2 = seal(work)
    r = verify(work)
    results["scenarios"]["D_full_reseal"] = {
        "verify_status": r["status"],
        "root_changed": m2["manifest_root"] != m["manifest_root"],
        "note": "self-consistent after reseal; only external anchor catches it",
    }

    results["all_ok"] = (
        results["scenarios"]["baseline_clean"] is True
        and results["scenarios"]["A_edit_artifact"]["detected"]
        and results["scenarios"]["B_edit_trace_record"]["detected"]
        and results["scenarios"]["C_append_chained_forgery"]["detected"]
        and results["scenarios"]["D_full_reseal"]["root_changed"]
    )
    print(json.dumps(results, indent=2))
    if not keep:
        shutil.rmtree(tmp, ignore_errors=True)
    return 0 if results["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
