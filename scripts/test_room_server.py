from __future__ import annotations

import http.client
import json
import shutil
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import room_sandbox, room_server

REPO = Path(__file__).resolve().parent.parent
BUNDLE_PATH = REPO / "site" / "src" / "data" / "audit-room.json"


def make_repo(tmp: Path) -> Path:
    (tmp / "site" / "src" / "data").mkdir(parents=True)
    (tmp / "site" / "dist" / "room").mkdir(parents=True)
    shutil.copy2(BUNDLE_PATH, tmp / "site" / "src" / "data" / "audit-room.json")
    (tmp / "site" / "dist" / "room" / "index.html").write_text(
        "<html>room</html>", encoding="utf-8"
    )
    (tmp / "papers").mkdir()
    return tmp


class RoomServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="lemma-room-test-"))
        repo = make_repo(cls.tmp)
        cls.state = room_server.RoomState(repo)
        cls.state.probe = {"available": True, "reason": None, "checks": {}}
        cls.server = ThreadingHTTPServer(
            ("127.0.0.1", 0), room_server.make_handler(cls.state)
        )
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        hdrs = {"Host": f"127.0.0.1:{self.port}"}
        hdrs.update(headers or {})
        conn.request(method, path, body=body, headers=hdrs)
        res = conn.getresponse()
        data = res.read()
        conn.close()
        return res.status, json.loads(data) if data else {}

    def post(self, payload, **headers):
        return self.request(
            "POST",
            "/api/reruns",
            body=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", **headers},
        )

    def test_session_shape(self):
        status, body = self.request("GET", "/api/session")
        self.assertEqual(status, 200)
        self.assertIn("token", body)
        self.assertEqual(body["limits"], {"wall_s": 90, "cpu_s": 60})
        self.assertEqual(len(body["allowed"]), 2)

    def test_host_validation_before_execution(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request("GET", "/api/session", headers={"Host": "evil.example.com"})
        res = conn.getresponse()
        res.read()
        conn.close()
        self.assertEqual(res.status, 403)

    def test_origin_rejected_before_token(self):
        status, _ = self.post(
            {"paper_slug": "x", "claim_id": "y", "attempt_id": "z"},
            Origin="https://evil.example",
        )
        self.assertEqual(status, 403)

    def test_cross_site_fetch_rejected(self):
        status, _ = self.post(
            {"paper_slug": "x", "claim_id": "y", "attempt_id": "z"},
            **{"Sec-Fetch-Site": "cross-site"},
        )
        self.assertEqual(status, 403)

    def test_token_required(self):
        status, _ = self.post(
            {"paper_slug": "icl-bayesian", "claim_id": "C6", "attempt_id": "legacy-1"},
            **{"X-Lemma-Token": "wrong"},
        )
        self.assertEqual(status, 403)

    def test_unknown_manifest_selection(self):
        status, _ = self.post(
            {"paper_slug": "icl-bayesian", "claim_id": "C6", "attempt_id": "legacy-2"},
            **{"X-Lemma-Token": self.state.token},
        )
        self.assertEqual(status, 400)

    def test_extra_body_keys_rejected(self):
        status, _ = self.post(
            {
                "paper_slug": "icl-bayesian",
                "claim_id": "C6",
                "attempt_id": "legacy-4",
                "evil": "x",
            },
            **{"X-Lemma-Token": self.state.token},
        )
        self.assertEqual(status, 400)

    def test_wrong_content_type(self):
        status, _ = self.request(
            "POST",
            "/api/reruns",
            body=b"{}",
            headers={
                "Content-Type": "text/plain",
                "X-Lemma-Token": self.state.token,
            },
        )
        self.assertEqual(status, 415)

    def test_oversized_body(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        conn.request(
            "POST",
            "/api/reruns",
            body=b"x" * 3000,
            headers={
                "Host": f"127.0.0.1:{self.port}",
                "Content-Type": "application/json",
                "X-Lemma-Token": self.state.token,
            },
        )
        res = conn.getresponse()
        res.read()
        conn.close()
        self.assertEqual(res.status, 413)

    def test_put_method_rejected(self):
        status, _ = self.request("PUT", "/api/reruns")
        self.assertEqual(status, 405)

    def test_busy_409(self):
        self.state.active = True
        try:
            status, _ = self.post(
                {
                    "paper_slug": "icl-bayesian",
                    "claim_id": "C6",
                    "attempt_id": "legacy-4",
                },
                **{"X-Lemma-Token": self.state.token},
            )
            self.assertEqual(status, 409)
        finally:
            self.state.active = False

    def test_quota_429(self):
        for i in range(20):
            self.state.jobs[f"{i:032x}"] = room_server.Job(f"{i:032x}", {})
        try:
            status, _ = self.post(
                {
                    "paper_slug": "icl-bayesian",
                    "claim_id": "C6",
                    "attempt_id": "legacy-4",
                },
                **{"X-Lemma-Token": self.state.token},
            )
            self.assertEqual(status, 429)
        finally:
            self.state.jobs.clear()

    def test_isolation_unavailable_503(self):
        real = self.state.probe
        self.state.probe = {"available": False, "reason": "no sandbox"}
        try:
            status, body = self.post(
                {
                    "paper_slug": "icl-bayesian",
                    "claim_id": "C6",
                    "attempt_id": "legacy-4",
                },
                **{"X-Lemma-Token": self.state.token},
            )
            self.assertEqual(status, 503)
            self.assertIn("no sandbox", body["error"])
        finally:
            self.state.probe = real

    def test_digest_mismatch_skips_manifest(self):
        bundle = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
        paper = next(p for p in bundle["papers"] if p["slug"] == "icl-bayesian")
        claim = next(c for c in paper["claims"] if c["id"] == "C6")
        attempt = next(a for a in claim["attempts"] if a["id"] == "legacy-4")
        attempt["script"]["text"] += "\n# tampered"
        allowed, skipped = room_server.load_allowed(bundle)
        self.assertNotIn(("icl-bayesian", "C6", "legacy-4"), allowed)
        self.assertIn("icl-bayesian/C6/legacy-4", skipped)

    def test_env_sanitized_no_parent_secret(self):
        import os

        os.environ["LEMMA_PROBE_SECRET"] = "should-not-leak"
        try:
            env = room_sandbox.sandbox_env(self.tmp / "w")
            self.assertNotIn("LEMMA_PROBE_SECRET", env)
            self.assertNotIn("ANTHROPIC_API_KEY", env)
            self.assertNotIn("OPENAI_API_KEY", env)
        finally:
            del os.environ["LEMMA_PROBE_SECRET"]

    def test_completed_job_inconclusive(self):
        fake = room_sandbox.SandboxResult()
        fake.exit_code = 0
        fake.wall_s = 1.5
        summary = {
            "claim_id": "C6",
            "status": "inconclusive",
            "metrics": {"control_pass": False},
            "notes": "positive control failed",
        }
        fake.stdout = ("noise\nSUMMARY_JSON=" + json.dumps(summary) + "\n").encode()
        fake.stderr = b""
        sel = self.state.allowed[("icl-bayesian", "C6", "legacy-4")]
        job = room_server.Job("0" * 32, sel)
        papers_before = {
            str(p.relative_to(REPO / "papers")) for p in (REPO / "papers").rglob("*")
        }
        with mock.patch.object(room_sandbox, "run_script", return_value=fake):
            self.state.run_job(job)
        papers_after = {
            str(p.relative_to(REPO / "papers")) for p in (REPO / "papers").rglob("*")
        }
        self.assertEqual(job.state, "completed")
        self.assertEqual(job.attempt["summary"]["status"], "inconclusive")
        self.assertEqual(job.attempt["checks"]["control"], "failed")
        self.assertEqual(job.attempt["record_type"], "versioned")
        self.assertEqual(papers_before, papers_after)

    def test_failed_job(self):
        fake = room_sandbox.SandboxResult()
        fake.error = "isolation unavailable: probe"
        sel = self.state.allowed[("icl-bayesian", "C6", "legacy-4")]
        job = room_server.Job("1" * 32, sel)
        with mock.patch.object(room_sandbox, "run_script", return_value=fake):
            self.state.run_job(job)
        self.assertEqual(job.state, "failed")
        self.assertIn("isolation unavailable", job.error)

    def test_readiness_probe_reports_reason(self):
        probe = room_sandbox.readiness_probe()
        self.assertIn("available", probe)
        if not probe["available"]:
            self.assertTrue(probe["reason"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
