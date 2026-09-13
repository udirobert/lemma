import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.auditor import _parse_summary, _summary_problems, control_passed
from agent.evals import (
    _run_local_eval,
    control_honest,
    render_table,
    verdict_matches_reference,
)


class AuditContractTests(unittest.TestCase):
    def test_controls_have_one_explicit_contract(self):
        for value in (True, 1, 1.0, "True"):
            with self.subTest(value=value):
                output = {"status": "supported", "metrics": {"control_pass": value}}
                self.assertTrue(control_passed(value))
                self.assertEqual(_summary_problems(output), [])
                self.assertTrue(control_honest(output))
        for value in (False, None, 0, "False", "false", "yes", [], {}, 2, "1"):
            with self.subTest(value=value):
                output = {"status": "falsified", "metrics": {"control_pass": value}}
                self.assertFalse(control_passed(value))
                self.assertTrue(_summary_problems(output))
                self.assertFalse(control_honest(output))

    def test_invalid_results_and_nonfinite_measurements(self):
        for output in (
            {"status": "banana", "metrics": {"x": 1}},
            {"status": "supported", "metrics": []},
            {"status": "inconclusive", "metrics": {}},
            {
                "status": "supported",
                "metrics": {"control_pass": True, "x": float("inf")},
            },
            {
                "status": "falsified",
                "metrics": {"control_pass": True, "x": float("nan")},
            },
        ):
            self.assertTrue(_summary_problems(output))
        self.assertEqual(
            _summary_problems(
                {"status": "inconclusive", "metrics": {"reason": "no data"}}
            ),
            [],
        )

    def test_exactly_one_structured_summary(self):
        summary = {
            "claim_id": "C1",
            "status": "supported",
            "metrics": {"control_pass": True},
            "notes": "measured",
        }
        line = "SUMMARY_JSON=" + json.dumps(summary)
        self.assertEqual(_parse_summary("diagnostic\n" + line), summary)
        self.assertIsNone(_parse_summary(line + "\n" + line))
        self.assertIsNone(_parse_summary('SUMMARY_JSON={"status":"supported"}'))
        self.assertIsNone(_parse_summary("SUMMARY_JSON=[]"))
        self.assertIsNone(_parse_summary("SUMMARY_JSON=broken"))

    def test_reference_override_does_not_hide_candidate_disagreement(self):
        output = {
            "source": "reviewer_reference",
            "status": "supported",
            "candidate_source": "agent_generated",
            "candidate_status": "falsified",
        }
        self.assertFalse(verdict_matches_reference(output, "supported"))
        self.assertTrue(verdict_matches_reference(output, "falsified"))
        self.assertIsNone(verdict_matches_reference(output, None))
        self.assertIsNone(
            verdict_matches_reference(
                {**output, "candidate_source": "unknown"}, "supported"
            )
        )

    def test_coverage_excludes_missing_and_circular_references(self):
        rows = []
        for cid, status, reference, source in (
            ("C1", "supported", "supported", "agent_generated"),
            ("C2", "falsified", "supported", "agent_generated"),
            ("C3", "supported", None, "agent_generated"),
            ("C4", "supported", "supported", "reviewer_reference"),
            ("C5", "inconclusive", None, "agent_generated"),
            ("C6", "supported", "supported", "unknown"),
        ):
            rows.append(
                {
                    "paper_id": "fixture",
                    "claim_id": cid,
                    "reference_status": reference,
                    "outcome": {
                        "status": status,
                        "source": source,
                        "metrics": {"control_pass": True},
                        "n_figures": 1,
                    },
                }
            )
        result = _run_local_eval(rows)
        self.assertEqual(result["means"]["verdict_matches_reference"], 0.5)
        self.assertEqual(
            result["coverage"]["verdict_matches_reference"],
            {"evaluated": 2, "total": 6},
        )
        self.assertEqual(
            result["coverage"]["control_honest"], {"evaluated": 5, "total": 6}
        )
        self.assertIsNone(result["rows"][2]["scores"]["verdict_matches_reference"])
        self.assertIn("(2/6 evaluated)", render_table(result))
        self.assertIn("n/a", render_table(result))
        missing = _run_local_eval([rows[2]])
        self.assertIsNone(missing["means"]["verdict_matches_reference"])


if __name__ == "__main__":
    unittest.main()
