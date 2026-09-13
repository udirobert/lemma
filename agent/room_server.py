from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import re
import secrets
import shutil
import threading
import uuid
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from agent import audit_room, room_sandbox
from agent.auditor import _parse_summary, _summary_problems
from agent.records import begin_attempt, figure_hashes, finish_attempt
from agent.room_manifest import RERUN_MANIFEST, RUN_LIMITS
from agent.traces import Trace

MAX_BODY = 2048
JOB_ID = re.compile(r"[0-9a-f]{32}\Z")
RUNS_DIRNAME = Path("runs") / "audit-room"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_allowed(bundle: dict) -> tuple[dict, dict[str, str]]:
    allowed: dict[tuple[str, str, str], dict] = {}
    skipped: dict[str, str] = {}
    for row in RERUN_MANIFEST:
        key = (row["paper_slug"], row["claim_id"], row["attempt_id"])
        paper = next(
            (p for p in bundle.get("papers", []) if p["slug"] == row["paper_slug"]),
            None,
        )
        claim = next(
            (c for c in (paper or {}).get("claims", []) if c["id"] == row["claim_id"]),
            None,
        )
        attempt = next(
            (
                a
                for a in (claim or {}).get("attempts", [])
                if a["id"] == row["attempt_id"]
            ),
            None,
        )
        script = (attempt or {}).get("script") or {}
        digest = hashlib.sha256(script.get("text", "").encode()).hexdigest()
        if not (paper and claim and attempt and script.get("text")):
            skipped["/".join(key)] = (
                "manifest row has no matching recorded attempt/script"
            )
        elif digest != row["script_sha256"]:
            skipped["/".join(key)] = "script text digest does not match manifest"
        elif script.get("sha256") != row["script_sha256"]:
            skipped["/".join(key)] = "exported script digest does not match manifest"
        else:
            allowed[key] = {
                **row,
                "script_text": script["text"],
                "claim": claim,
                "number": attempt.get("number"),
            }
    return allowed, skipped


class Job:
    def __init__(self, job_id: str, selection: dict) -> None:
        self.job_id = job_id
        self.selection = selection
        self.state = "queued"
        self.events: list[dict] = []
        self.attempt: dict | None = None
        self.error: str | None = None

    def event(self, stage: str, message: str) -> None:
        self.events.append({"stage": stage, "message": message, "ts": _utcnow()})

    def payload(self) -> dict:
        return {
            "job_id": self.job_id,
            "state": self.state,
            "events": self.events,
            "attempt": self.attempt,
            "error": self.error,
        }


class RoomState:
    def __init__(self, repo: Path) -> None:
        self.repo = repo
        self.dist = repo / "site" / "dist"
        bundle_path = repo / "site" / "src" / "data" / "audit-room.json"
        self.bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        self.allowed, self.skipped = load_allowed(self.bundle)
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.Lock()
        self.jobs: dict[str, Job] = {}
        self.active = False
        self.probe = room_sandbox.readiness_probe()
        self.assets = self.dist / "audit-room" / "assets"

    def session(self) -> dict:
        reason = self.probe.get("reason")
        if self.probe.get("available") and self.skipped:
            reason = f"skipped manifest rows: {'; '.join(self.skipped.values())}"
        return {
            "available": bool(self.probe.get("available")) and bool(self.allowed),
            "reason": reason or (None if self.allowed else "no manifest rows matched"),
            "token": self.token,
            "allowed": [
                {
                    "paper_slug": e["paper_slug"],
                    "claim_id": e["claim_id"],
                    "attempt_id": e["attempt_id"],
                    "script_sha256": e["script_sha256"],
                }
                for e in self.allowed.values()
            ],
            "limits": {
                "wall_s": RUN_LIMITS["wall_s"],
                "cpu_s": RUN_LIMITS["cpu_s"],
            },
        }

    def run_job(self, job: Job, weave: bool = False) -> None:
        sel = job.selection
        cid = sel["claim_id"]
        job.event("queued", "job accepted; waiting for worker")
        job.state = "running"
        job.event(
            "running",
            f"replaying recorded attempt {sel['attempt_id']} (#{sel.get('number')})",
        )
        job_dir = self.repo / RUNS_DIRNAME / uuid.uuid4().hex
        work = job_dir / "work"
        claim_dir = job_dir / "records" / "results" / cid.lower()
        work.mkdir(parents=True)
        claim_dir.mkdir(parents=True)
        try:
            trace = Trace(f"room-{job.job_id[:8]}", job_dir / "trace.jsonl")
            claim_input = {
                "id": cid,
                "title": sel["claim"].get("title", cid),
                "statement": sel["claim"].get("statement", ""),
                "success_criterion": sel["claim"].get("criterion", ""),
                "test_plan": sel["claim"].get("test_plan", ""),
                "live_replay_of": sel["attempt_id"],
                "legacy_source": True,
                "criterion_note": "criterion is the current snapshot, not pinned historically",
            }
            number = (sel.get("number") or 0) + 1
            root = begin_attempt(
                claim_dir,
                claim_input,
                trace,
                number,
                "live_rerun",
                sel["script_text"],
                "",
                "",
            )
            before = figure_hashes(claim_dir)
            job.event("executing", f"sandboxed run of {root / 'script.py'}")
            if weave:
                from agent.weave_ops import op as weave_op

                runner = weave_op(room_sandbox.run_script)
            else:
                runner = room_sandbox.run_script
            result = runner(root / "script.py", work, root)
            if result.error:
                job.state = "failed"
                job.error = result.error
                job.event("failed", result.error)
                return
            (root / "stdout.txt").write_bytes(result.stdout)
            (root / "stderr.txt").write_bytes(result.stderr)
            if result.stdout_truncated:
                result.stderr += b"\n[lemma room] stdout truncated at limit"
            job.event("validating", "parsing SUMMARY_JSON output")
            stdout_text = result.stdout.decode("utf-8", "replace")
            stderr_text = result.stderr.decode("utf-8", "replace")
            if result.timed_out:
                stderr_text += f"\nTIMEOUT after {RUN_LIMITS['wall_s']}s"
            summary = _parse_summary(stdout_text)
            if summary is not None and summary.get("claim_id") != cid:
                summary = None
                problems = ["summary claim_id does not match selected claim"]
            elif summary is not None:
                problems = _summary_problems(summary)
            else:
                problems = ["script produced no parseable SUMMARY_JSON"]
            for name, _ in figure_hashes(work / "results" / cid.lower()).items():
                shutil.copy2(work / "results" / cid.lower() / name, claim_dir / name)
            finish_attempt(
                root,
                claim_dir,
                before,
                result.exit_code if result.exit_code is not None else -1,
                stdout_text,
                stderr_text,
                result.wall_s,
                summary,
                problems,
                trace,
            )
            attempt = audit_room.versioned_attempt(root, cid, self.assets)
            attempt["execution_mode"] = "live_rerun"
            attempt["replay_of"] = sel["attempt_id"]
            job.attempt = attempt
            job.state = "completed"
            job.event("completed", f"attempt {root.name} recorded ({job_dir})")
        except Exception as exc:
            job.state = "failed"
            job.error = f"internal error: {exc}"
            job.event("failed", job.error)
        finally:
            with self.lock:
                self.active = False


def make_handler(state: RoomState) -> type[SimpleHTTPRequestHandler]:

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(state.dist), **kw)

        def log_message(self, fmt, *args):
            pass

        def _api_headers(self) -> None:
            self.send_header("Cache-Control", "no-store")

        def end_headers(self) -> None:
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            super().end_headers()

        def list_directory(self, path):
            self.send_error(404, "Not found")
            return None

        def _host_ok(self) -> bool:
            port = self.server.server_address[1]
            return self.headers.get("Host", "") in {
                f"localhost:{port}",
                f"127.0.0.1:{port}",
            }

        def _origin_ok(self) -> bool:
            port = self.server.server_address[1]
            allowed = {f"http://localhost:{port}", f"http://127.0.0.1:{port}"}
            origin = self.headers.get("Origin")
            if origin is not None and origin not in allowed:
                return False
            fetch_site = self.headers.get("Sec-Fetch-Site")
            return fetch_site in (None, "same-origin", "same-site", "none")

        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self._api_headers()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _deny(self, status: int, reason: str) -> None:
            self._json(status, {"error": reason})

        def do_GET(self) -> None:
            if self.path == "/api/session":
                if not self._host_ok():
                    return self._deny(403, "host not allowed")
                return self._json(200, state.session())
            match = JOB_ID.fullmatch(self.path.removeprefix("/api/reruns/"))
            if match:
                if not self._host_ok():
                    return self._deny(403, "host not allowed")
                with state.lock:
                    job = state.jobs.get(match[0])
                if job is None:
                    return self._deny(404, "unknown job")
                return self._json(200, job.payload())
            if self.path.startswith("/api/"):
                return self._deny(404, "unknown endpoint")
            super().do_GET()

        def do_POST(self) -> None:
            if self.path != "/api/reruns":
                return self._deny(404, "unknown endpoint")
            if not self._host_ok():
                return self._deny(403, "host not allowed")
            if not self._origin_ok():
                return self._deny(403, "origin not allowed")
            token = self.headers.get("X-Lemma-Token", "")
            if not hmac.compare_digest(token, state.token):
                return self._deny(403, "invalid token")
            ctype = self.headers.get("Content-Type", "")
            if ctype.split(";")[0].strip() != "application/json":
                return self._deny(415, "content type must be application/json")
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._deny(411, "invalid content length")
            if length > MAX_BODY or length <= 0:
                return self._deny(413, "request body too large or empty")
            try:
                body = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return self._deny(400, "invalid JSON body")
            if not isinstance(body, dict) or set(body) != {
                "paper_slug",
                "claim_id",
                "attempt_id",
            }:
                return self._deny(
                    400, "body must be {paper_slug, claim_id, attempt_id}"
                )
            if not all(isinstance(v, str) for v in body.values()):
                return self._deny(400, "fields must be strings")
            if not state.probe.get("available"):
                return self._deny(
                    503, f"isolation unavailable: {state.probe.get('reason')}"
                )
            key = (body["paper_slug"], body["claim_id"], body["attempt_id"])
            if key not in state.allowed:
                return self._deny(400, "selection not in rerun manifest")
            with state.lock:
                if len(state.jobs) >= RUN_LIMITS["max_jobs"]:
                    return self._deny(429, "job quota reached")
                if state.active:
                    return self._deny(409, "a rerun is already in progress")
                state.active = True
                job = Job(uuid.uuid4().hex, state.allowed[key])
                state.jobs[job.job_id] = job
            thread = threading.Thread(
                target=state.run_job, args=(job, self.server.weave), daemon=True
            )
            thread.start()
            return self._json(202, {"job_id": job.job_id, "state": "queued"})

        def do_PUT(self):
            self._deny(405, "method not allowed")

        def do_DELETE(self):
            self._deny(405, "method not allowed")

    return Handler


def serve(repo: Path, port: int = 8765, weave: bool = False) -> int:
    state = RoomState(repo)
    room_page = state.dist / "room" / "index.html"
    if not room_page.is_file():
        print(
            "[lemma room] site/dist/room/index.html missing; build first: "
            "npm --prefix site run build"
        )
        return 1
    if weave:
        from agent.weave_ops import init_weave

        init_weave()
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))
    server.weave = weave
    status = "available" if state.probe.get("available") else "recorded-only"
    reason = state.probe.get("reason") or ""
    print(f"[lemma room] http://127.0.0.1:{port}/room/ · isolation {status} {reason}")
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()
    return 0
