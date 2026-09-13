"""Modal-hosted backend for the Audit Room's bounded live reruns.

Deploy: `modal deploy agent/room_modal.py` from the repo root.

Two functions on one app:

- run_attempt: single-use, network-blocked, modal-access-restricted
  container that replays one manifest-allowlisted script under rlimits
  (the container itself is the sandbox — no sandbox-exec needed on
  Linux). Always returns {events, attempt, error}; never raises raw.
- api: one FastAPI ASGI app at a stable URL serving the same contract
  as agent/room_server.py: GET /api/session, POST /api/reruns,
  GET /api/reruns/{job_id}. Job ids are Modal FunctionCall object_ids.

State (session token, per-IP daily counters, inflight jobs) lives in a
modal.Dict named lemma-room-state. The audit-room bundle is frozen into
the image at deploy time — same semantics as the local runner reading
site/src/data/audit-room.json.
"""

import sys

# `modal deploy agent/room_modal.py` imports this file as a top-level
# module; the frozen agent package lives at /app/agent in the image.
for _p in ("", "/app"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import modal  # noqa: E402 — sys.path must be set before any agent/modal import

app = modal.App("lemma-room")

image = (
    modal.Image.debian_slim(python_version="3.12")
    # numpy/matplotlib run the audit scripts; python-dotenv satisfies the
    # agent.auditor import chain (agent.llm imports dotenv at top level;
    # anthropic/openai are only lazy imports inside llm.py and are never
    # executed here, so they are intentionally not installed).
    .pip_install("numpy", "matplotlib", "python-dotenv", "fastapi[standard]")
    .add_local_dir(
        "agent",
        "/app/agent",
        ignore=lambda p: p.name == "__pycache__" or (p.is_dir() and p.name == "traces"),
    )
    .add_local_file("site/src/data/audit-room.json", "/app/audit-room.json")
)

BUNDLE_PATH = "/app/audit-room.json"
DICT_NAME = "lemma-room-state"
SITE_ORIGIN = "https://lemmabio.netlify.app"
MAX_BODY = 2048
PER_IP_PER_DAY = 10
MAX_CONCURRENT = 4
WORKER_TIMEOUT_S = 240


def _load_bundle() -> dict:
    import json
    from pathlib import Path

    return json.loads(Path(BUNDLE_PATH).read_text(encoding="utf-8"))


@app.function(
    image=image,
    single_use_containers=True,
    block_network=True,
    restrict_modal_access=True,
    timeout=WORKER_TIMEOUT_S,
    cpu=1.0,
    memory=1024,
)
def run_attempt(payload: dict) -> dict:
    """Replay one allowlisted script in an isolated container."""
    import contextlib
    import os
    import signal
    import subprocess
    import sys as _sys
    import tempfile
    import time
    from pathlib import Path

    from agent.auditor import _parse_summary, _summary_problems
    from agent.room_jobs import (
        allowed_entries,
        build_attempt,
        figure_data_uris,
        synthesize_events,
    )
    from agent.room_manifest import RUN_LIMITS
    from agent.room_sandbox import _LAUNCHER, SandboxResult, _read_bounded

    events: list[dict] = []

    def ev(stage: str, message: str) -> None:
        events.extend(synthesize_events((stage, message)))

    try:
        allowed, _ = allowed_entries(_load_bundle())
        key = (
            payload.get("paper_slug"),
            payload.get("claim_id"),
            payload.get("attempt_id"),
        )
        sel = allowed.get(key)
        if sel is None:
            ev("failed", "selection not in rerun manifest")
            return {
                "events": events,
                "attempt": None,
                "error": "selection not in rerun manifest",
            }
        cid = sel["claim_id"]

        base = Path(tempfile.mkdtemp(prefix="lemma-room-", dir="/tmp"))
        work = base / "work"
        for d in ("home", "tmp", "mplconfig", "results/" + cid.lower()):
            (work / d).mkdir(parents=True, exist_ok=True)
        script_path = work / "script.py"
        script_path.write_text(sel["script_text"], encoding="utf-8")

        env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": str(work / "home"),
            "TMPDIR": str(work / "tmp"),
            "MPLCONFIGDIR": str(work / "mplconfig"),
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONNOUSERSITE": "1",
        }

        ev(
            "executing",
            f"isolated container run of recorded attempt {sel['attempt_id']} "
            f"(script sha256 {sel['script_sha256'][:12]}…)",
        )
        result = SandboxResult()
        out_path = base / "stdout.txt"
        err_path = base / "stderr.txt"
        start = time.monotonic()
        with open(out_path, "wb") as stdout_f, open(err_path, "wb") as stderr_f:
            proc = subprocess.Popen(
                [_sys.executable, "-I", "-c", _LAUNCHER, str(script_path)],
                cwd=work,
                env=env,
                stdout=stdout_f,
                stderr=stderr_f,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
            try:
                result.exit_code = proc.wait(timeout=RUN_LIMITS["wall_s"])
            except subprocess.TimeoutExpired:
                result.timed_out = True
                ev(
                    "executing",
                    f"wall limit {RUN_LIMITS['wall_s']}s reached; process group killed",
                )
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                result.exit_code = proc.wait()
        result.wall_s = time.monotonic() - start
        result.stdout, result.stdout_truncated = _read_bounded(out_path)
        result.stderr, result.stderr_truncated = _read_bounded(err_path)

        if result.stdout_truncated:
            result.stderr += b"\n[lemma room] stdout truncated at limit"
        stdout_text = result.stdout.decode("utf-8", "replace")
        stderr_text = result.stderr.decode("utf-8", "replace")
        if result.timed_out:
            stderr_text += f"\nTIMEOUT after {RUN_LIMITS['wall_s']}s"

        ev("validating", "parsing SUMMARY_JSON output")
        summary = _parse_summary(stdout_text)
        if summary is not None and summary.get("claim_id") != cid:
            summary = None
            problems = ["summary claim_id does not match selected claim"]
        elif summary is not None:
            problems = _summary_problems(summary)
        else:
            problems = ["script produced no parseable SUMMARY_JSON"]

        figures, fig_notes = figure_data_uris(work / "results" / cid.lower())
        for note in fig_notes:
            ev("figures", note)

        attempt = build_attempt(
            sel,
            summary=summary,
            exit_code=result.exit_code,
            wall_s=result.wall_s,
            stdout=result.stdout,
            stderr=stderr_text.encode("utf-8", "replace"),
            stdout_truncated=result.stdout_truncated,
            stderr_truncated=result.stderr_truncated,
            figures=figures,
            problems=problems,
        )
        verdict = (summary or {}).get("status", "no summary")
        ev(
            "completed",
            f"rerun finished: exit {result.exit_code} in {result.wall_s:.1f}s"
            f" · verdict {verdict}"
            + (f" · {len(problems)} validation problem(s)" if problems else ""),
        )
        return {"events": events, "attempt": attempt, "error": None}
    except Exception as exc:  # honest infra failure — never raise raw
        ev("failed", f"internal error: {exc}")
        return {"events": events, "attempt": None, "error": f"internal error: {exc}"}


@app.function(image=image)
@modal.asgi_app()
def api():
    """FastAPI app serving the room-server contract at a stable URL."""
    import hmac
    import json
    import re
    import secrets
    import time

    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    from agent.room_jobs import allowed_entries, rate_ok, synthesize_events
    from agent.room_manifest import RUN_LIMITS

    web = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    state = modal.Dict.from_name(DICT_NAME, create_if_missing=True)
    allowed, skipped = allowed_entries(_load_bundle())

    job_id_re = re.compile(r"[A-Za-z0-9_-]{4,80}\Z")
    self_origin = "https://thepapajams--lemma-room-api.modal.run"
    allowed_origins = {SITE_ORIGIN, self_origin}

    def _json(payload: dict, status: int = 200) -> JSONResponse:
        return JSONResponse(
            payload,
            status_code=status,
            headers={"Cache-Control": "no-store"},
        )

    def _deny(status: int, reason: str) -> JSONResponse:
        return _json({"error": reason}, status)

    async def _token() -> str:
        tok = await state.get.aio("token")
        if not tok:
            tok = secrets.token_urlsafe(32)
            await state.put.aio("token", tok)
        return tok

    def _client_ip(request: Request) -> str:
        fwd = request.headers.get("x-forwarded-for", "")
        if fwd:
            return fwd.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def _inflight() -> dict:
        data = await state.get.aio("inflight") or {}
        cutoff = time.time() - WORKER_TIMEOUT_S
        return {k: v for k, v in data.items() if v.get("ts", 0) > cutoff}

    @web.get("/api/session")
    async def session():
        reason = (
            "skipped manifest rows: " + "; ".join(skipped.values())
            if skipped
            else (None if allowed else "no manifest rows matched")
        )
        return _json(
            {
                "available": bool(allowed),
                "reason": reason,
                "token": await _token(),
                "allowed": [
                    {
                        "paper_slug": e["paper_slug"],
                        "claim_id": e["claim_id"],
                        "attempt_id": e["attempt_id"],
                        "script_sha256": e["script_sha256"],
                    }
                    for e in allowed.values()
                ],
                "limits": {
                    "wall_s": RUN_LIMITS["wall_s"],
                    "cpu_s": RUN_LIMITS["cpu_s"],
                },
            }
        )

    @web.post("/api/reruns")
    async def create_rerun(request: Request):
        origin = request.headers.get("origin")
        if origin is not None and origin not in allowed_origins:
            return _deny(403, "origin not allowed")
        fetch_site = request.headers.get("sec-fetch-site")
        if fetch_site not in (None, "same-origin", "same-site", "none"):
            return _deny(403, "cross-site fetch not allowed")
        if not hmac.compare_digest(
            request.headers.get("x-lemma-token", ""), await _token()
        ):
            return _deny(403, "invalid token")
        ctype = request.headers.get("content-type", "")
        if ctype.split(";")[0].strip() != "application/json":
            return _deny(415, "content type must be application/json")
        try:
            length = int(request.headers.get("content-length", "0"))
        except ValueError:
            return _deny(411, "invalid content length")
        if length > MAX_BODY or length <= 0:
            return _deny(413, "request body too large or empty")
        body_raw = await request.body()
        if len(body_raw) > MAX_BODY:
            return _deny(413, "request body too large or empty")
        try:
            body = json.loads(body_raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _deny(400, "invalid JSON body")
        if not isinstance(body, dict) or set(body) != {
            "paper_slug",
            "claim_id",
            "attempt_id",
        }:
            return _deny(400, "body must be {paper_slug, claim_id, attempt_id}")
        if not all(isinstance(v, str) for v in body.values()):
            return _deny(400, "fields must be strings")
        key = (body["paper_slug"], body["claim_id"], body["attempt_id"])
        if key not in allowed:
            return _deny(400, "selection not in rerun manifest")

        now = time.time()
        ip = _client_ip(request)
        counters = await state.get.aio("ip_counters") or {}
        if not rate_ok(counters, ip, PER_IP_PER_DAY, window_s=86_400, now=now):
            return _deny(429, "per-IP daily job quota reached")
        inflight = await _inflight()
        if len(inflight) >= MAX_CONCURRENT:
            return _deny(429, "too many reruns in progress")

        call = await run_attempt.spawn.aio(body)
        job_id = call.object_id
        counters[ip] = [*counters.get(ip, []), now]
        await state.put.aio("ip_counters", counters)
        inflight[job_id] = {"ts": now, "selection": dict(body)}
        await state.put.aio("inflight", inflight)
        return _json({"job_id": job_id, "state": "queued"}, status=202)

    @web.get("/api/reruns/{job_id}")
    async def poll_rerun(job_id: str):
        if not job_id_re.fullmatch(job_id):
            return _deny(404, "unknown job")
        try:
            call = modal.FunctionCall.from_id(job_id)
            result = await call.get.aio(timeout=0)
        except modal.exception.TimeoutError:
            inflight = await _inflight()
            meta = inflight.get(job_id) or {}
            sel = meta.get("selection") or {}
            events = synthesize_events(
                ("accepted", "job accepted; waiting for worker"),
                (
                    "running",
                    "replaying recorded attempt "
                    + (sel.get("attempt_id") or "unknown"),
                ),
            )
            return _json(
                {
                    "job_id": job_id,
                    "state": "running",
                    "events": events,
                    "attempt": None,
                    "error": None,
                }
            )
        except modal.exception.NotFoundError:
            return _deny(404, "unknown job")
        except modal.exception.OutputExpiredError:
            return _json(
                {
                    "job_id": job_id,
                    "state": "failed",
                    "events": synthesize_events(
                        ("failed", "job result expired before it was read")
                    ),
                    "attempt": None,
                    "error": "job result expired",
                }
            )
        except Exception as exc:
            return _json(
                {
                    "job_id": job_id,
                    "state": "failed",
                    "events": synthesize_events(("failed", f"runner error: {exc}")),
                    "attempt": None,
                    "error": f"runner error: {exc}",
                }
            )
        # Result arrived — drop it from the inflight ledger (best effort).
        inflight = await _inflight()
        if job_id in inflight:
            inflight.pop(job_id)
            await state.put.aio("inflight", inflight)
        result = result if isinstance(result, dict) else {}
        done_state = "failed" if result.get("error") else "completed"
        return _json(
            {
                "job_id": job_id,
                "state": done_state,
                "events": result.get("events") or [],
                "attempt": result.get("attempt"),
                "error": result.get("error"),
            }
        )

    @web.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def unknown_api(path: str):
        return _deny(404, "unknown endpoint")

    return web
