"""Tool handlers — thin subprocess wrappers over `python -m lemma_pipeline.cli`.

Handlers never raise: every path returns a JSON string. The plugin only
orchestrates — all audit state lives under plugin_data_dir("lemma"),
never in the install tree or the repo.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from plugins.plugin_storage import plugin_data_dir

    DATA_ROOT = plugin_data_dir("lemma")
except Exception:  # doctor/dev context without Hermes plugin storage
    DATA_ROOT = Path(os.environ.get("LEMMA_PLUGIN_DATA", "/tmp/lemma-plugin"))

PAPERS_DIR = DATA_ROOT / "papers"
QUICK_TIMEOUT = 900  # seconds; the quick path must stay video-length


def _pipeline_root(ctx=None) -> Path | None:
    """The installed plugin dir: catalog installs check out only
    hermes-plugin/ (subdir install), so the pipeline lives in the vendored
    lemma_pipeline/ package inside it — generated from agent/ by
    scripts/sync_hermes_plugin.py. The package is deliberately not named
    `agent`: that would collide with hermes-agent's own agent package on
    sys.path. Never an arbitrary checkout — the code that runs is the code
    that was reviewed."""
    root = Path(__file__).resolve().parent
    return root if (root / "lemma_pipeline" / "cli.py").is_file() else None


def _plugin_flag(ctx, key: str, env_name: str) -> bool:
    """Behaviour toggles are plugin config (plugins.entries.lemma.settings,
    declared in plugin.yaml config_schema — surfaced in the Desktop Plugins
    settings form), not .env entries. The env var remains as a dev/direct
    fallback when no plugin context is available."""
    try:
        if ctx is not None:
            val = ctx.get_config(key)
            if val is not None:
                if isinstance(val, str):
                    return val.strip().lower() in ("1", "true", "yes", "on")
                return bool(val)
    except Exception:
        pass
    return os.environ.get(env_name) == "1"


def _child_env(ctx=None) -> dict[str, str]:
    """Minimal env for pipeline children: no inherited Hermes process
    env (no other profile's secrets, no tool tokens). Credentials enter
    only through agent.secret_scope.get_secret — Hermes's route."""
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "LEMMA_NO_DOTENV": "1",
        "LEMMA_LOCAL_PDF_DIRS": str(PAPERS_DIR),
    }
    # PM activates the dependency environment by putting its site-packages
    # on this process's sys.path; children spawned from sys.executable do
    # not inherit sys.path, so hand them the same entries explicitly —
    # otherwise plugin python_dependencies (fitz, trackio's deps, ...) are
    # invisible to the pipeline. Restricted to site-packages so the
    # hermes-agent checkout root never leaks `agent`/`tools` names in.
    site_dirs = [
        p
        for p in dict.fromkeys(sys.path)
        if p and "site-packages" in p and Path(p).is_dir()
    ]
    if site_dirs:
        env["PYTHONPATH"] = os.pathsep.join(site_dirs)
    if _plugin_flag(ctx, "host_exec", "LEMMA_AUDIT_HOST_EXEC"):
        env["LEMMA_AUDIT_HOST_EXEC"] = "1"
    if _plugin_flag(ctx, "weave", "LEMMA_WEAVE"):
        env["LEMMA_WEAVE"] = "1"
    # Provider parity with agent/llm.py: Anthropic direct, or any
    # OpenAI-compatible endpoint (OPENAI_API_KEY/BASE_URL/MODEL covers
    # Venice, OpenRouter, local servers, ...). Values come only through
    # Hermes's secret route, never inherited process env.
    try:
        from agent.secret_scope import get_secret

        secrets = {
            v: get_secret(v)
            for v in (
                "ANTHROPIC_API_KEY",
                "OPENAI_API_KEY",
                "OPENAI_BASE_URL",
                "OPENAI_API_BASE",
                "OPENAI_MODEL",
                "LEMMA_MODEL",
            )
        }
    except Exception:  # doctor/dev context without hermes-agent
        secrets = {}
    for var, val in secrets.items():
        if val:
            env[var] = val
    return env


def _slug(source: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", source.strip()).strip("-").lower()
    return s[:60] or "paper"


def _run(root: Path, argv: list[str], timeout: int, ctx=None) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "lemma_pipeline.cli", *argv],
        cwd=root,
        env=_child_env(ctx),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout + proc.stderr


def _paper_dir(paper: str) -> Path | None:
    """Resolve a workdir name under the plugin papers dir — names only,
    never paths."""
    if (
        not paper
        or Path(paper).is_absolute()
        or any(s in paper for s in ("/", "\\", ".."))
    ):
        return None
    d = (PAPERS_DIR / paper).resolve()
    return d if d.is_dir() and d.parent == PAPERS_DIR.resolve() else None


def _claim_ids(workdir: Path, max_claims: int) -> list[str]:
    claims = json.loads((workdir / "claims.json").read_text())
    testable = [c["id"] for c in claims if c.get("testable")]
    return testable[:max_claims]


def _verdicts(workdir: Path) -> dict:
    out = {"workdir": str(workdir)}
    report_p = workdir / "results" / "audit_report.json"
    if report_p.is_file():
        report = json.loads(report_p.read_text())
        out["claims"] = [
            {
                "id": c.get("id") or c.get("claim_id"),
                "verdict": c.get("status") or c.get("verdict"),
                "attempts": (
                    len(a) if isinstance((a := c.get("attempts")), list) else a
                ),
            }
            for c in report.get("claims", [])
        ]
    judge_p = workdir / "judge_report.json"
    if judge_p.is_file():
        jr = json.loads(judge_p.read_text())
        out["judge"] = jr.get("verdict") or jr.get("overall")
    flags = []
    for imp in workdir.glob("results/improve_*.json"):
        try:
            if json.loads(imp.read_text()).get("verify_gain"):
                flags.append("verify_gain")
                break
        except Exception:
            continue
    report_p2 = workdir / "results" / "audit_report.json"
    if report_p2.is_file():
        try:
            rep = json.loads(report_p2.read_text())
            if any(c.get("reference_disagreement") for c in rep.get("claims", [])):
                flags.append("reference_disagreement")
        except Exception:
            pass
    if flags:
        out["flags"] = flags
    out["trace"] = str(workdir / "trace.jsonl")
    return out


def _need_pipeline(ctx=None) -> str | None:
    if _pipeline_root(ctx) is None:
        return json.dumps(
            {
                "error": "lemma pipeline not found in the installed plugin "
                "— expected lemma_pipeline/cli.py beside tools.py"
            }
        )
    return None


def lemma_quick_audit(args: dict, ctx=None, **kwargs) -> str:
    """extract -> audit first N testable claims -> judge, one workdir."""
    err = _need_pipeline(ctx)
    if err:
        return err
    source = (args.get("source") or "").strip()
    if not source:
        return json.dumps(
            {"error": "source required (arXiv/OpenReview id or PDF path)"}
        )
    max_claims = max(1, min(int(args.get("max_claims") or 2), 5))
    root = _pipeline_root(ctx)
    workdir = PAPERS_DIR / _slug(source)
    workdir.mkdir(parents=True, exist_ok=True)

    steps = []
    try:
        rc, log = _run(
            root,
            ["audit", source, "--workdir", str(workdir), "--stages", "extract"],
            QUICK_TIMEOUT // 3,
            ctx,
        )
        steps.append({"stage": "extract", "rc": rc, "tail": log[-400:]})
        if rc != 0:
            return json.dumps({"error": "extract failed", "steps": steps})

        ids = _claim_ids(workdir, max_claims)
        if not ids:
            return json.dumps({"error": "no testable claims extracted", "steps": steps})

        rc, log = _run(
            root,
            [
                "audit",
                source,
                "--workdir",
                str(workdir),
                "--stages",
                "audit,judge",
                "--claims",
                ",".join(ids),
            ],
            QUICK_TIMEOUT,
            ctx,
        )
        steps.append({"stage": "audit+judge", "rc": rc, "tail": log[-400:]})
        out = _verdicts(workdir)
        out["audited_claims"] = ids
        out["steps"] = steps
        return json.dumps(out)
    except subprocess.TimeoutExpired:
        return json.dumps(
            {
                "error": f"quick audit exceeded {QUICK_TIMEOUT}s — "
                "check lemma_status for partial results",
                "workdir": str(workdir),
                "steps": steps,
            }
        )
    except Exception as e:  # never raise from a handler
        return json.dumps({"error": f"{type(e).__name__}: {e}", "steps": steps})


def lemma_audit(args: dict, ctx=None, **kwargs) -> str:
    """Full pipeline in the background; lemma_status reports progress."""
    err = _need_pipeline(ctx)
    if err:
        return err
    source = (args.get("source") or "").strip()
    if not source:
        return json.dumps({"error": "source required"})
    root = _pipeline_root(ctx)
    workdir = PAPERS_DIR / _slug(source)
    workdir.mkdir(parents=True, exist_ok=True)
    log_p = workdir / "lemma_audit.log"
    try:
        with log_p.open("ab") as log:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "lemma_pipeline.cli",
                    "audit",
                    source,
                    "--workdir",
                    str(workdir),
                ],
                cwd=root,
                env=_child_env(ctx),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        return json.dumps(
            {
                "started": True,
                "pid": proc.pid,
                "workdir": str(workdir),
                "log": str(log_p),
                "hint": "poll with lemma_status; a full audit takes tens of minutes",
            }
        )
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


def _llm_provider_present() -> bool:
    try:
        from agent.secret_scope import get_secret

        getter = get_secret
    except Exception:
        getter = os.environ.get
    return any(getter(v) for v in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"))


def _probe(root: Path, code: str, ctx=None, timeout: int = 60) -> tuple[int, str]:
    """Run `python -c <code>` inside the plugin dir — the same interpreter
    and cwd the pipeline children get."""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=_child_env(ctx),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def lemma_doctor(args: dict, ctx=None, **kwargs) -> str:
    """Preflight: pipeline resolves + launches, deps import, trackio and
    sandbox reachable, credential route configured. JSON, never raises."""
    try:
        return _doctor(ctx)
    except Exception as e:
        return json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"})


def _doctor(ctx=None) -> str:
    checks: dict = {}
    issues: list[str] = []

    root = _pipeline_root(ctx)
    checks["pipeline"] = {"ok": root is not None}
    if root is None:
        issues.append("lemma_pipeline/cli.py not found beside tools.py")
    else:
        rc, log = _run(root, ["judge", "--help"], 30, ctx)
        checks["cli"] = {"ok": rc == 0}
        if rc != 0:
            issues.append(f"lemma_pipeline.cli failed to launch: {log[-200:]}")

        rc, out = _probe(
            root,
            "import numpy, scipy, matplotlib, anthropic, openai, fitz, "
            "dotenv, requests; print('deps ok')",
            ctx,
        )
        checks["python_deps"] = {"ok": rc == 0, "tail": out[-200:]}
        if rc != 0:
            last = out.strip().splitlines()[-1] if out.strip() else out
            issues.append(f"python dependency missing: {last}")

        rc, out = _probe(
            root,
            "from lemma_pipeline import script_sandbox; "
            "print(script_sandbox.sandbox_kind() or 'none')",
            ctx,
        )
        kind = out.strip().splitlines()[-1] if out.strip() else "unknown"
        checks["sandbox"] = {
            "ok": kind in ("sandbox-exec", "bwrap"),
            "kind": kind,
            "host_exec": _plugin_flag(ctx, "host_exec", "LEMMA_AUDIT_HOST_EXEC"),
        }
        if kind not in ("sandbox-exec", "bwrap") and not checks["sandbox"]["host_exec"]:
            issues.append(
                "no audit-script sandbox (sandbox-exec/bwrap) and host_exec "
                "is off — generated scripts will record sandbox_unavailable"
            )

    beside = Path(sys.executable).parent / "trackio"
    trackio = (
        str(beside) if beside.is_file() else shutil.which("trackio") or "not found"
    )
    checks["trackio"] = {"ok": trackio != "not found", "path": trackio}
    if trackio == "not found":
        issues.append("trackio CLI not found — evidence stage will fail")

    checks["llm_provider"] = {"ok": _llm_provider_present()}
    if not checks["llm_provider"]["ok"]:
        issues.append(
            "no LLM credential resolvable via Hermes "
            "(ANTHROPIC_API_KEY or OPENAI_API_KEY)"
        )

    try:
        PAPERS_DIR.mkdir(parents=True, exist_ok=True)
        writable = os.access(PAPERS_DIR, os.W_OK)
    except Exception:
        writable = False
    checks["data_dir"] = {"ok": writable, "path": str(PAPERS_DIR)}
    if not writable:
        issues.append(f"cannot write plugin data dir {PAPERS_DIR}")

    checks["settings"] = {
        "host_exec": _plugin_flag(ctx, "host_exec", "LEMMA_AUDIT_HOST_EXEC"),
        "weave": _plugin_flag(ctx, "weave", "LEMMA_WEAVE"),
    }

    return json.dumps({"ok": not issues, "checks": checks, "issues": issues})


def lemma_status(args: dict, ctx=None, **kwargs) -> str:
    """Summarize every workdir under the plugin papers dir."""
    paper = (args.get("paper") or "").strip()
    if not PAPERS_DIR.is_dir():
        return json.dumps({"papers": [], "dir": str(PAPERS_DIR)})
    dirs = (
        [PAPERS_DIR / paper]
        if paper and _paper_dir(paper)
        else sorted(
            d for d in PAPERS_DIR.iterdir() if d.is_dir() and not d.name.startswith("_")
        )
    )
    papers = []
    for d in dirs:
        if not d.is_dir():
            continue
        entry = {"workdir": d.name}
        entry.update(_verdicts(d))
        if (d / "lemma_audit.log").is_file() and "judge" not in entry:
            entry["note"] = "audit log present, no judge verdict yet — running?"
        papers.append(entry)
    return json.dumps({"papers": papers, "dir": str(PAPERS_DIR)})


def lemma_improve(args: dict, ctx=None, **kwargs) -> str:
    """The self-distrust loop, in the background: reviewer feedback ->
    re-audit -> before/after recorded (agent/meta.py)."""
    err = _need_pipeline(ctx)
    if err:
        return err
    paper = (args.get("paper") or "").strip()
    workdir = _paper_dir(paper)
    if workdir is None:
        return json.dumps(
            {"error": f"no workdir {paper!r}", "hint": "lemma_status lists them"}
        )
    root = _pipeline_root(ctx)
    log_p = workdir / "lemma_improve.log"
    try:
        with log_p.open("ab") as log:
            proc = subprocess.Popen(
                [sys.executable, "-m", "lemma_pipeline.cli", "improve", str(workdir)],
                cwd=root,
                env=_child_env(ctx),
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        return json.dumps(
            {
                "started": True,
                "pid": proc.pid,
                "workdir": str(workdir),
                "log": str(log_p),
                "hint": "feedback.auto.md is advisory; re-audit before/after "
                "is recorded in improve_<ts>.json — poll lemma_status",
            }
        )
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


_PAPER_RE = re.compile(
    r"(arxiv\.org/(?:abs|pdf)/[\w.]+|openreview\.net/forum\?id=[\w-]+|"
    r"arxiv:[\w.]+)",
    re.I,
)


def paper_link_hint(ctx=None, user_message="", conversation_history=None, **kwargs):
    """pre_llm_call observer: notice a paper link — in the user's turn OR
    in what the agent itself just claimed — and offer the audit.

    Context injection only — never blocks, never audits unasked."""
    if _pipeline_root(ctx) is None:
        return None
    hay = user_message or ""
    # also scan the last assistant message: audit what the agent *says*
    if not _PAPER_RE.search(hay) and conversation_history:
        try:
            for m in conversation_history[-2:]:
                hay += " " + (m.get("content", "") if isinstance(m, dict) else str(m))
        except Exception:
            pass
    m = _PAPER_RE.search(hay)
    if not m:
        return None
    return {
        "context": (
            f"[lemma] paper link detected ({m.group(1)}). "
            "lemma_quick_audit can extract + numerically audit its claims "
            "(supported/falsified/inconclusive with a preserved evidence "
            "trail) — or /audit <id>. Offer it if the user wants the claims "
            "checked; don't run it unprompted."
        )
    }


def lemma_judge(args: dict, ctx=None, **kwargs) -> str:
    """Re-run the judge stage on an existing workdir."""
    err = _need_pipeline(ctx)
    if err:
        return err
    paper = (args.get("paper") or "").strip()
    workdir = _paper_dir(paper)
    if workdir is None:
        return json.dumps(
            {"error": f"no workdir {paper!r}", "hint": "lemma_status lists them"}
        )
    try:
        rc, log = _run(_pipeline_root(ctx), ["judge", str(workdir)], 300, ctx)
        out = _verdicts(workdir)
        out["rc"] = rc
        out["tail"] = log[-400:]
        return json.dumps(out)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})
