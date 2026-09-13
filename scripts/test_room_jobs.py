from __future__ import annotations

import base64
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import room_jobs
from agent.room_manifest import RERUN_MANIFEST

REPO = Path(__file__).resolve().parent.parent
BUNDLE_PATH = REPO / "site" / "src" / "data" / "audit-room.json"
BUNDLE = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))

# minimal valid PNG (1x1 transparent)
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


def c6_attempt(attempt_id: str) -> dict:
    claim = next(
        c
        for p in BUNDLE["papers"]
        if p["slug"] == "icl-bayesian"
        for c in p["claims"]
        if c["id"] == "C6"
    )
    return next(a for a in claim["attempts"] if a["id"] == attempt_id)


class AllowedEntriesTest(unittest.TestCase):
    def test_matches_manifest_rows(self):
        allowed, skipped = room_jobs.allowed_entries(BUNDLE)
        self.assertEqual(skipped, {})
        self.assertEqual(len(allowed), 2)
        for row in RERUN_MANIFEST:
            key = (row["paper_slug"], row["claim_id"], row["attempt_id"])
            self.assertIn(key, allowed)
            entry = allowed[key]
            self.assertEqual(entry["script_sha256"], row["script_sha256"])
            self.assertTrue(entry["script_text"])
            self.assertEqual(entry["claim"]["id"], row["claim_id"])

    def test_digest_mismatch_skips_row(self):
        bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
        paper = next(p for p in bundle["papers"] if p["slug"] == "icl-bayesian")
        claim = next(c for c in paper["claims"] if c["id"] == "C6")
        attempt = next(a for a in claim["attempts"] if a["id"] == "legacy-4")
        attempt["script"]["text"] += "\n# tampered"
        allowed, skipped = room_jobs.allowed_entries(bundle)
        self.assertNotIn(("icl-bayesian", "C6", "legacy-4"), allowed)
        self.assertIn("icl-bayesian/C6/legacy-4", skipped)
        self.assertIn(("icl-bayesian", "C6", "legacy-1"), allowed)

    def test_missing_attempt_skips_row(self):
        bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
        paper = next(p for p in bundle["papers"] if p["slug"] == "icl-bayesian")
        claim = next(c for c in paper["claims"] if c["id"] == "C6")
        claim["attempts"] = [a for a in claim["attempts"] if a["id"] != "legacy-1"]
        allowed, skipped = room_jobs.allowed_entries(bundle)
        self.assertNotIn(("icl-bayesian", "C6", "legacy-1"), allowed)
        self.assertIn("icl-bayesian/C6/legacy-1", skipped)


class BuildAttemptTest(unittest.TestCase):
    def setUp(self):
        self.allowed, _ = room_jobs.allowed_entries(BUNDLE)
        self.sel = self.allowed[("icl-bayesian", "C6", "legacy-4")]
        self.summary = c6_attempt("legacy-4")["summary"]

    def build(self, **kw):
        args = {
            "summary": self.summary,
            "exit_code": 0,
            "wall_s": 1.5,
            "stdout": b"noise\nSUMMARY_JSON={}\n",
            "stderr": b"",
        }
        args.update(kw)
        return room_jobs.build_attempt(self.sel, **args)

    def test_attempt_shape(self):
        a = self.build()
        self.assertEqual(len(a["id"]), 32)
        self.assertEqual(a["number"], self.sel["number"] + 1)
        self.assertEqual(a["source"], "live_rerun")
        self.assertEqual(a["record_type"], "versioned")
        self.assertEqual(a["execution"], "completed")
        self.assertEqual(a["execution_mode"], "live_rerun")
        self.assertEqual(a["replay_of"], "legacy-4")
        self.assertIsNone(a["criterion"])
        self.assertIsNone(a["criterion_sha256"])
        self.assertIsNone(a["run_id"])
        self.assertIsNone(a["weave_url"])
        self.assertEqual(
            a["feedback"],
            {"human": "", "generated": "", "binding": "unavailable"},
        )
        self.assertEqual(a["script"]["sha256"], self.sel["script_sha256"])
        self.assertEqual(a["script"]["text"], self.sel["script_text"])
        self.assertFalse(a["script"]["truncated"])
        self.assertIs(a["summary"], self.summary)
        self.assertEqual(a["exit_code"], 0)
        self.assertEqual(a["wall_s"], 1.5)
        self.assertEqual(a["stdout"]["name"], "stdout.txt")
        self.assertFalse(a["stdout"]["truncated"])
        self.assertIsNotNone(a["created_at"])

    def test_real_summary_failed_control(self):
        # the recorded C6 legacy-4 summary is inconclusive with a failed
        # positive control — the honest outcome must survive verbatim
        a = self.build()
        self.assertEqual(a["summary"]["status"], "inconclusive")
        self.assertEqual(a["checks"]["control"], "failed")
        self.assertEqual(a["checks"]["state"], "inconclusive")

    def test_problems_flip_state_failed(self):
        a = self.build(
            summary=None, problems=["script produced no parseable SUMMARY_JSON"]
        )
        self.assertIsNone(a["summary"])
        self.assertEqual(a["checks"]["state"], "failed")
        self.assertIn(
            "script produced no parseable SUMMARY_JSON",
            a["checks"]["problems"],
        )

    def test_truncated_output_flags(self):
        big = b"x" * (room_jobs.MAX_TAIL_CHARS + 10)
        a = self.build(stdout=big, stderr=b"e" * 50, stderr_truncated=True)
        self.assertTrue(a["stdout"]["truncated"])
        self.assertTrue(a["stderr"]["truncated"])
        self.assertEqual(len(a["stdout"]["text"]), room_jobs.MAX_TAIL_CHARS)
        self.assertTrue(a["stdout"]["text"].endswith("x" * 10))

    def test_figures_data_uri(self):
        figs = [
            {"name": "fig.png", "url": "data:image/png;base64,AA==", "sha256": "0" * 64}
        ]
        a = self.build(figures=figs)
        self.assertEqual(a["figures"], figs)


class FigureDataUrisTest(unittest.TestCase):
    def test_encodes_pngs_and_skips_oversize(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "fig.png").write_bytes(PNG_BYTES)
            (d / "big.png").write_bytes(
                b"\x89PNG\r\n\x1a\n" + b"0" * (room_jobs.MAX_FIGURE_BYTES)
            )
            (d / "notpng.png").write_bytes(b"this is not a png")
            figs, notes = room_jobs.figure_data_uris(d)
        self.assertEqual([f["name"] for f in figs], ["fig.png"])
        self.assertTrue(figs[0]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(base64.b64decode(figs[0]["url"].split(",", 1)[1]), PNG_BYTES)
        import hashlib

        self.assertEqual(figs[0]["sha256"], hashlib.sha256(PNG_BYTES).hexdigest())
        self.assertEqual(len(notes), 2)
        self.assertTrue(any("big.png" in n for n in notes))
        self.assertTrue(any("notpng.png" in n for n in notes))

    def test_missing_dir(self):
        figs, notes = room_jobs.figure_data_uris(Path("/nonexistent-dir"))
        self.assertEqual(figs, [])
        self.assertEqual(notes, [])


class RateOkTest(unittest.TestCase):
    def test_window_prunes_and_limits(self):
        counters = {"ip": [100.0, 200.0, 300.0]}
        self.assertTrue(room_jobs.rate_ok(counters, "ip", 10, window_s=1000, now=350.0))
        self.assertFalse(room_jobs.rate_ok(counters, "ip", 3, window_s=1000, now=350.0))
        # everything outside the window is pruned in place
        self.assertTrue(room_jobs.rate_ok(counters, "ip", 1, window_s=100, now=1000.0))
        self.assertEqual(counters["ip"], [])

    def test_missing_key_ok(self):
        counters = {}
        self.assertTrue(room_jobs.rate_ok(counters, "new", 1, now=0.0))
        self.assertEqual(counters["new"], [])


class SynthesizeEventsTest(unittest.TestCase):
    def test_shape(self):
        events = room_jobs.synthesize_events(
            ("accepted", "job accepted"), ("running", "replaying")
        )
        self.assertEqual(len(events), 2)
        for e, (stage, msg) in zip(
            events,
            [("accepted", "job accepted"), ("running", "replaying")],
            strict=True,
        ):
            self.assertEqual(e["stage"], stage)
            self.assertEqual(e["message"], msg)
            self.assertIn("ts", e)


if __name__ == "__main__":
    unittest.main(verbosity=2)
