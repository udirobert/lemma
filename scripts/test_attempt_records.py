"""Attempt-provenance tests: every executed attempt gets an immutable
snapshot under results/<cid>/attempts/<uuid>/ and the reviewer reference
runs symmetrically after the generated attempts.

Stubs auditor.complete (no LLM) and auditor._run_script (no execution).
Run: .venv/bin/python scripts/test_attempt_records.py
"""

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent.auditor as auditor
import agent.meta as meta
from agent import records
from agent.traces import Trace

CLAIM_C1 = {
    "id": "C1",
    "title": "claim one",
    "statement": "s",
    "evidence_in_paper": "e",
    "test_plan": "p",
    "success_criterion": "c",
    "testable": True,
}
CLAIM_C2 = {**CLAIM_C1, "id": "C2"}
CLAIM_C3 = {**CLAIM_C1, "id": "C3"}


def summary_line(cid: str, status: str, metrics: dict | None = None) -> str:
    payload = {
        "claim_id": cid,
        "status": status,
        "metrics": metrics or {"x": 1.0, "control_pass": True},
        "notes": "n",
    }
    return "SUMMARY_JSON=" + json.dumps(payload)


def fake_complete(trace, stage, system, user_prompt, **kw):
    return json.dumps({"script": "print('stub')\n"})


def run_script_factory(gen: tuple, ref: tuple | None):
    def fake(script_path, workdir, timeout_s=0):
        if script_path.name == "reviewer_reference.py":
            return (*ref, 0.01)
        return (*gen, 0.01)

    return fake


class AttemptRecordsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="attempt_records_"))
        self.workdir = self.tmp / "paper"
        self.results = self.workdir / "results"
        self.results.mkdir(parents=True)
        self.trace = Trace("attempt_records_test", self.workdir / "trace.jsonl")
        self._saved = (auditor.complete, auditor._run_script)
        auditor.complete = fake_complete

    def tearDown(self):
        auditor.complete, auditor._run_script = self._saved
        shutil.rmtree(self.tmp, ignore_errors=True)

    def claim_dir(self, cid: str) -> Path:
        d = self.results / cid.lower()
        d.mkdir(parents=True, exist_ok=True)
        return d

    def attempt_dirs(self, cid: str) -> list[Path]:
        return sorted((self.results / cid.lower() / "attempts").iterdir())

    def audit(self, claim, gen, ref=None, reference_script="print('ref')\n"):
        claim_dir = self.claim_dir(claim["id"])
        if ref is not None:
            (claim_dir / "reviewer_reference.py").write_text(reference_script)
        auditor._run_script = run_script_factory(gen, ref)
        return auditor.audit_one(
            claim, "paper text", self.workdir, self.results, self.trace
        )

    def test_supported_candidate_falsified_reference(self):
        entry = self.audit(
            CLAIM_C1,
            gen=(0, summary_line("C1", "supported"), ""),
            ref=(0, summary_line("C1", "falsified"), ""),
        )
        self.assertEqual(entry["status"], "falsified")
        self.assertEqual(entry["source"], "reviewer_reference")
        self.assertTrue(entry["reference_checked"])
        self.assertTrue(entry["reference_disagreement"])
        self.assertEqual(entry["candidate_summary"]["status"], "supported")
        self.assertEqual(entry["reference_summary"]["status"], "falsified")
        self.assertEqual(entry["attempts"], 2)
        self.assertEqual(len(entry["attempt_ids"]), 2)
        dirs = self.attempt_dirs("C1")
        self.assertEqual(len(dirs), 2)
        inputs = [json.loads((d / "input.json").read_text()) for d in dirs]
        self.assertEqual(
            sorted(i["source"] for i in inputs),
            ["agent_generated", "reviewer_reference"],
        )
        by_source = {i["source"]: i for i in inputs}
        self.assertEqual(by_source["agent_generated"]["number"], 1)
        self.assertEqual(by_source["reviewer_reference"]["number"], 2)
        ref_dir = next(
            d for d in dirs if d.name == by_source["reviewer_reference"]["attempt_id"]
        )
        for d in dirs:
            self.assertTrue((d / "input.json").is_file())
            self.assertTrue((d / "script.py").is_file())
            self.assertTrue((d / "outcome.json").is_file())
            self.assertTrue((d / "stdout.txt").is_file())
            self.assertTrue((d / "stderr.txt").is_file())
            with self.assertRaises(FileExistsError):
                records.finish_attempt(
                    d, self.results / "c1", {}, 0, "", "", 0.0, None, [], self.trace
                )
        run_json = json.loads((self.results / "c1" / "run_attempt2.json").read_text())
        self.assertEqual(run_json["source"], "reviewer_reference")
        self.assertEqual(run_json["attempt_id"], ref_dir.name)

    def test_same_statuses_no_disagreement(self):
        entry = self.audit(
            CLAIM_C1,
            gen=(0, summary_line("C1", "supported"), ""),
            ref=(0, summary_line("C1", "supported"), ""),
        )
        self.assertEqual(entry["status"], "supported")
        self.assertEqual(entry["source"], "reviewer_reference")
        self.assertTrue(entry["reference_checked"])
        self.assertFalse(entry["reference_disagreement"])

    def test_failed_reference_preserves_candidate(self):
        entry = self.audit(
            CLAIM_C1,
            gen=(0, summary_line("C1", "supported"), ""),
            ref=(1, "", "boom on line 3"),
        )
        self.assertEqual(entry["status"], "supported")
        self.assertEqual(entry["source"], "agent_generated")
        self.assertFalse(entry["reference_checked"])
        self.assertFalse(entry["reference_disagreement"])
        self.assertIsNone(entry["reference_summary"])
        failed = json.loads(
            (self.results / "c1" / "run_attempt2.failed.json").read_text()
        )
        self.assertEqual(failed["kind"], "reference_failed")
        self.assertEqual(failed["source"], "reviewer_reference")
        self.assertIn("boom on line 3", failed["tail"])
        self.assertTrue(failed["problems"])

    def test_numbering_advances_above_legacy_files(self):
        claim_dir = self.claim_dir("C1")
        legacy = claim_dir / "run_attempt9.json"
        legacy.write_text('{"attempt": 9, "legacy": true}\n')
        entry = self.audit(CLAIM_C1, gen=(0, summary_line("C1", "supported"), ""))
        self.assertEqual(entry["attempts"], 1)
        self.assertTrue((claim_dir / "audit_attempt10.py").is_file())
        self.assertTrue((claim_dir / "run_attempt10.json").is_file())
        self.assertEqual(legacy.read_text(), '{"attempt": 9, "legacy": true}\n')
        run_json = json.loads((claim_dir / "run_attempt10.json").read_text())
        self.assertEqual(run_json["attempt"], 10)
        self.assertIn("attempt_id", run_json)
        self.assertIn("run_id", run_json)

    def test_crash_snapshot_records_problems(self):
        entry = self.audit(CLAIM_C1, gen=(1, "partial output, no summary", "err"))
        self.assertEqual(entry["status"], "inconclusive")
        dirs = self.attempt_dirs("C1")
        self.assertEqual(len(dirs), auditor.MAX_ATTEMPTS)
        outcome = json.loads((dirs[0] / "outcome.json").read_text())
        self.assertFalse(outcome["accepted"])
        self.assertEqual(
            outcome["problems"], ["script produced no parseable SUMMARY_JSON"]
        )
        self.assertIsNone(outcome["summary"])
        failed = json.loads(
            (self.results / "c1" / "run_attempt1.failed.json").read_text()
        )
        self.assertEqual(failed["kind"], "crashed")

    def test_snapshot_stores_generated_feedback_at_prompt_bound(self):
        claim_dir = self.claim_dir("C1")
        notes = "z" * 5000
        (claim_dir / "feedback.auto.md").write_text(notes)
        self.audit(CLAIM_C1, gen=(0, summary_line("C1", "supported"), ""))
        dirs = self.attempt_dirs("C1")
        inputs = json.loads((dirs[0] / "input.json").read_text())
        self.assertEqual(inputs["auto_feedback"], notes[:4000])
        self.assertEqual(len(inputs["auto_feedback"]), 4000)
        self.assertEqual((claim_dir / "feedback.auto.md").read_text(), notes)

    def test_independent_claims_get_separate_snapshots(self):
        self.audit(CLAIM_C1, gen=(0, summary_line("C1", "supported"), ""))
        self.audit(CLAIM_C2, gen=(0, summary_line("C2", "supported"), ""))
        d1 = self.attempt_dirs("C1")
        d2 = self.attempt_dirs("C2")
        self.assertEqual(len(d1), 1)
        self.assertEqual(len(d2), 1)
        self.assertNotEqual(d1[0], d2[0])
        i1 = json.loads((d1[0] / "input.json").read_text())
        i2 = json.loads((d2[0] / "input.json").read_text())
        self.assertEqual(i1["claim"]["id"], "C1")
        self.assertEqual(i2["claim"]["id"], "C2")


class ImproveArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="improve_records_"))
        self.workdir = self.tmp / "paper"
        self.results = self.workdir / "results"
        self.results.mkdir(parents=True)
        self.trace = Trace("improve_records_test", self.workdir / "trace.jsonl")
        self._saved = (meta.complete, auditor.audit_all)
        meta.complete = lambda *a, **kw: "reviewer notes"
        auditor.audit_all = self.fake_audit_all

    def tearDown(self):
        meta.complete, auditor.audit_all = self._saved
        shutil.rmtree(self.tmp, ignore_errors=True)

    def fake_audit_all(self, claims, paper_text, workdir, trace, only=None, jobs=1):
        report = json.loads((workdir / "results" / "audit_report.json").read_text())
        for c in report["claims"]:
            c["attempt_ids"] = ["after-" + c["id"]]
        return report

    def write_state(self):
        claims = [
            {**CLAIM_C1, "id": "C1"},
            {**CLAIM_C2, "id": "C2"},
            {**CLAIM_C3, "id": "C3"},
        ]
        (self.workdir / "claims.json").write_text(json.dumps(claims))
        report = {
            "claims": [
                {
                    "id": "C1",
                    "status": "falsified",
                    "summary": {
                        "status": "falsified",
                        "metrics": {"x": 1.0, "control_pass": True},
                    },
                    "attempt_ids": ["before-C1"],
                },
                {
                    "id": "C2",
                    "status": "supported",
                    "summary": {"status": "supported", "metrics": {"x": 1.0}},
                    "attempt_ids": ["before-C2"],
                },
                {
                    "id": "C3",
                    "status": "inconclusive",
                    "summary": {"status": "inconclusive", "metrics": {"x": 1.0}},
                    "attempt_ids": ["before-C3"],
                },
            ]
        }
        (self.results / "audit_report.json").write_text(json.dumps(report))

    def test_default_targets_skip_valid_falsified(self):
        self.write_state()
        record = meta.improve(self.workdir, self.trace)
        self.assertEqual(sorted(record["targets"]), ["C2", "C3"])
        self.assertIn("round_id", record)
        archive = self.results / "improvements" / record["round_id"]
        self.assertTrue((archive / "input.json").is_file())
        self.assertTrue((archive / "output.json").is_file())
        inputs = json.loads((archive / "input.json").read_text())
        self.assertEqual(sorted(c["id"] for c in inputs["claims"]), ["C2", "C3"])
        self.assertEqual(inputs["before_report"]["claims"][0]["id"], "C1")
        self.assertEqual(record["before_attempt_ids"]["C2"], ["before-C2"])
        self.assertEqual(record["after_attempt_ids"]["C2"], ["after-C2"])

    def test_explicit_only_can_target_falsified(self):
        self.write_state()
        record = meta.improve(self.workdir, self.trace, only={"C1"})
        self.assertEqual(record["targets"], ["C1"])
        self.assertEqual(record["before_attempt_ids"]["C1"], ["before-C1"])

    def test_tail_bounded_keeps_newest_within_limit(self):
        newest = "n" * 6997 + "END"
        kept = meta._tail_bounded(["old section", newest], 6000)
        joined = "\n".join(kept)
        self.assertLessEqual(len(joined), 6000)
        self.assertTrue(joined.endswith("END"))
        kept = meta._tail_bounded(["a" * 100, "b" * 100, newest], 6000)
        self.assertEqual(len(kept), 1)

    def test_round_archive_preserves_drafted_notes(self):
        self.write_state()
        record = meta.improve(self.workdir, self.trace)
        archive = self.results / "improvements" / record["round_id"]
        for cid in record["targets"]:
            archived = archive / f"{cid}.feedback.auto.md"
            self.assertTrue(archived.is_file())
            self.assertEqual(archived.read_text(), "reviewer notes")


if __name__ == "__main__":
    unittest.main()
