import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.audit_room import build_bundle, checks, claim_record, json_safe, safe_url
from agent.records import begin_attempt, finish_attempt
from agent.room_manifest import RERUN_MANIFEST
from agent.traces import Trace


class AuditRoomDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.paper = Path(self.temp.name) / "paper"
        self.claim_dir = self.paper / "results" / "c1"
        self.claim_dir.mkdir(parents=True)
        self.claim = {"id": "C1", "title": "Claim", "success_criterion": "x equals 1"}
        self.summary = {
            "claim_id": "C1",
            "status": "supported",
            "metrics": {"x": 1, "control_pass": True},
            "notes": "measured",
        }

    def test_legacy_attempt_order_and_missing_outputs_are_honest(self):
        for number in (1, 2, 10):
            (self.claim_dir / f"audit_attempt{number}.py").write_text("print('test')")
        (self.claim_dir / "run_attempt1.json").write_text(
            json.dumps({"attempt": 1, "summary": self.summary, "exit_code": 0})
        )
        (self.claim_dir / "feedback.auto.md").write_text("current note")
        record = claim_record(self.paper, self.claim, {"summary": self.summary}, None)
        self.assertEqual([a["number"] for a in record["attempts"]], [1, 2, 10])
        self.assertEqual(record["attempts"][1]["execution"], "output_unavailable")
        self.assertIsNone(record["attempts"][1]["summary"])
        self.assertIsNone(record["attempts"][0]["criterion_sha256"])
        self.assertEqual(record["attempts"][0]["feedback"]["binding"], "unavailable")
        self.assertEqual(record["current_feedback"]["binding"], "current_unversioned")
        self.assertEqual(record["source"], "unknown")
        self.assertEqual(record["attempts"][0]["figures"], [])

    def test_failed_attempt_tail_does_not_become_a_verdict(self):
        (self.claim_dir / "run_attempt3.failed.json").write_text(
            json.dumps({"attempt": 3, "tail": "NameError", "exit_code": 1})
        )
        attempt = claim_record(self.paper, self.claim, {}, None)["attempts"][0]
        self.assertEqual(attempt["checks"]["state"], "failed")
        self.assertEqual(attempt["stderr"]["text"], "NameError")
        self.assertIsNone(attempt["summary"])

    def test_checks_distinguish_inconclusive_from_a_passing_control(self):
        summary = {
            **self.summary,
            "status": "inconclusive",
            "metrics": {"control_pass": False},
        }
        self.assertEqual(
            checks(summary, "C1"),
            {"state": "inconclusive", "control": "failed", "problems": []},
        )
        summary = {**self.summary, "metrics": {"control_pass": "False"}}
        self.assertEqual(checks(summary, "C1")["state"], "failed")
        self.assertEqual(checks(None, "C1")["state"], "not_evaluated")
        self.assertEqual(checks(self.summary, "C2")["state"], "failed")

    def test_snapshot_has_no_fabricated_trace_links(self):
        (self.paper / "meta.json").write_text(
            json.dumps({"slug": "example", "title": "Example"})
        )
        (self.paper / "claims.json").write_text(json.dumps([self.claim]))
        (self.paper / "results" / "audit_report.json").write_text(
            json.dumps({"claims": [{"id": "C1", "summary": self.summary}]})
        )
        bundle = build_bundle(self.paper.parent)
        self.assertEqual(bundle["mode"], "recorded")
        self.assertEqual(bundle["papers"][0]["claims"][0]["status"], "supported")
        self.assertEqual(bundle["papers"][0]["improvements"], [])
        self.assertIsNone(safe_url("javascript:alert(1)"))
        self.assertIsNone(safe_url("https://secret@example.com/test"))
        self.assertEqual(json_safe({"x": float("inf")}), {"x": "inf"})

    def test_versioned_records_are_pinned_and_tampering_is_visible(self):
        trace = Trace("test-room", self.paper / "trace.jsonl")
        root = begin_attempt(
            self.claim_dir,
            self.claim,
            trace,
            1,
            "agent_generated",
            "print('original')",
            "human",
            "generated",
        )
        finish_attempt(
            root, self.claim_dir, {}, 0, "output", "", 0.1, self.summary, [], trace
        )
        (self.claim_dir / "audit_attempt1.py").write_text("legacy mirror")
        record = claim_record(self.paper, self.claim, {"summary": self.summary}, None)
        self.assertEqual(len(record["attempts"]), 1)
        attempt = record["attempts"][0]
        self.assertEqual(attempt["record_type"], "versioned")
        self.assertEqual(attempt["feedback"]["binding"], "attempt_input")
        self.assertEqual(attempt["criterion"], "x equals 1")
        self.assertEqual(attempt["checks"]["state"], "passed")
        (root / "script.py").write_text("changed")
        record = claim_record(self.paper, self.claim, {"summary": self.summary}, None)
        self.assertEqual(record["attempts"][0]["checks"]["state"], "failed")
        self.assertIn(
            "Script does not match its recorded digest.",
            record["attempts"][0]["checks"]["problems"],
        )

    def test_rerun_manifest_has_explicit_unique_hashes(self):
        self.assertEqual(len(RERUN_MANIFEST), 2)
        self.assertEqual({r["claim_id"] for r in RERUN_MANIFEST}, {"C6"})
        for item in RERUN_MANIFEST:
            self.assertRegex(item["script_sha256"], r"^[0-9a-f]{64}$")

    def test_unsafe_claim_id_is_rejected(self):
        with self.assertRaises(ValueError):
            claim_record(self.paper, {"id": "../other"}, {}, None)


if __name__ == "__main__":
    unittest.main()
