"""Smoke test: the reviewer reference runs after every generated outcome.

Verifies the symmetric reference check in agent/auditor.py:
  1. LLM attempts end 'falsified' + reviewer_reference.py present
     -> reference executes and its valid verdict wins.
  2. LLM attempts end 'supported' + reviewer_reference.py present
     -> reference still executes (a win is checked, not skipped) and its
        valid verdict remains authoritative.
Run: .venv/bin/python scripts/test_escalation_gate.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent.auditor as auditor
from agent.traces import Trace

FAKE_SCRIPT_FALSIFIED = """
import json
print("SUMMARY_JSON=" + json.dumps({
  "claim_id": "CX", "status": "falsified",
  "metrics": {"x": 1.0, "control_pass": True}, "notes": "fake falsified"}))
"""

FAKE_SCRIPT_SUPPORTED = """
import json
print("SUMMARY_JSON=" + json.dumps({
  "claim_id": "CX", "status": "supported",
  "metrics": {"x": 1.0, "control_pass": True}, "notes": "fake supported"}))
"""

REFERENCE_SCRIPT = """
import json
print("SUMMARY_JSON=" + json.dumps({
  "claim_id": "CX", "status": "supported",
  "metrics": {"ref": 2.0, "control_pass": True}, "notes": "reviewer reference"}))
"""

CLAIM = {
    "id": "CX",
    "title": "test claim",
    "statement": "test",
    "evidence_in_paper": "test",
    "test_plan": "test",
    "success_criterion": "test",
}


def fake_complete_factory(script_src: str):
    def fake_complete(trace, stage, system, user_prompt, **kw):
        return json.dumps({"script": script_src})

    return fake_complete


def run_case(name: str, llm_script: str) -> bool:
    tmp = Path(tempfile.mkdtemp(prefix=f"esc_{name}_"))
    try:
        workdir = tmp / "paper"
        results = workdir / "results"
        claim_dir = results / "cx"
        claim_dir.mkdir(parents=True)
        (claim_dir / "reviewer_reference.py").write_text(REFERENCE_SCRIPT)
        trace = Trace("escalation_test", workdir / "trace.jsonl")

        auditor.complete = fake_complete_factory(llm_script)
        out = auditor.audit_one(CLAIM, "paper text", workdir, results, trace)

        events = [
            json.loads(line)
            for line in (workdir / "trace.jsonl").read_text().splitlines()
        ]
        fired = any(e.get("event") == "reviewer_reference_executed" for e in events)
        final = out["status"]
        ok = fired and final == "supported" and out["source"] == "reviewer_reference"
        print(
            f"{'PASS' if ok else 'FAIL'} [{name}] ref_fired={fired} "
            f"(expect True) final={final} source={out['source']}"
        )
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ok1 = run_case("falsified_fires", FAKE_SCRIPT_FALSIFIED)
    ok2 = run_case("supported_also_checked", FAKE_SCRIPT_SUPPORTED)
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    sys.exit(main())
