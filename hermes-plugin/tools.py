"""Tool handlers — thin subprocess wrappers over `python -m agent.cli`.

Handlers never raise: every path returns a JSON string. The plugin only
orchestrates — all audit state lives under plugin_data_dir("lemma"),
never in the install tree or the repo.
"""

import json
import os
import re
import subprocess
from pathlib import Path

try:
    from plugins.plugin_storage import plugin_data_dir

    DATA_ROOT = plugin_data_dir("lemma")
except Exception:  # doctor/dev context without Hermes plugin storage
    DATA_ROOT = Path(os.environ.get("LEMMA_PLUGIN_DATA", "/tmp/lemma-plugin"))

PAPERS_DIR = DATA_ROOT / "papers"
QUICK_TIMEOUT = 900  # seconds; the quick path must stay video-length


def _repo(ctx=None) -> Path | None:
    """Resolve the lemma repo: config repo_path -> env -> dev-mode parent."""
    for cand in (
        ctx.get_config("repo_path") if ctx else None,
        os.environ.get("LEMMA_REPO"),
        str(Path(__file__).resolve().parent.parent),  # dev: repo/hermes-plugin
    ):
        if cand and (Path(cand) / "agent" / "cli.py").is_file():
            return Path(cand)
    return None


def _python(repo: Path) -> str:
    venv = repo / ".venv" / "bin" / "python"
    return str(venv) if venv.is_file() else "python3"


def _slug(source: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", source.strip()).strip("-").lower()
    return s[:60] or "paper"


def _run(repo: Path, argv: list[str], timeout: int) -> tuple[int, str]:
    proc = subprocess.run(
        [_python(repo), "-m", "agent.cli", *argv],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout + proc.stderr


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


def _need_repo(ctx=None) -> str | None:
    if _repo(ctx) is None:
        return json.dumps(
            {
                "error": "lemma repo not found — set the plugin's "
                "'repo_path' setting (or LEMMA_REPO) to a checkout "
                "containing agent/cli.py"
            }
        )
    return None


def lemma_quick_audit(args: dict, ctx=None, **kwargs) -> str:
    """extract -> audit first N testable claims -> judge, one workdir."""
    err = _need_repo(ctx)
    if err:
        return err
    source = (args.get("source") or "").strip()
    if not source:
        return json.dumps(
            {"error": "source required (arXiv/OpenReview id or PDF path)"}
        )
    max_claims = max(1, min(int(args.get("max_claims") or 2), 5))
    repo = _repo(ctx)
    workdir = PAPERS_DIR / _slug(source)
    workdir.mkdir(parents=True, exist_ok=True)

    steps = []
    try:
        rc, log = _run(
            repo,
            ["audit", source, "--workdir", str(workdir), "--stages", "extract"],
            QUICK_TIMEOUT // 3,
        )
        steps.append({"stage": "extract", "rc": rc, "tail": log[-400:]})
        if rc != 0:
            return json.dumps({"error": "extract failed", "steps": steps})

        ids = _claim_ids(workdir, max_claims)
        if not ids:
            return json.dumps({"error": "no testable claims extracted", "steps": steps})

        rc, log = _run(
            repo,
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
    err = _need_repo(ctx)
    if err:
        return err
    source = (args.get("source") or "").strip()
    if not source:
        return json.dumps({"error": "source required"})
    repo = _repo(ctx)
    workdir = PAPERS_DIR / _slug(source)
    workdir.mkdir(parents=True, exist_ok=True)
    log_p = workdir / "lemma_audit.log"
    try:
        with log_p.open("ab") as log:
            proc = subprocess.Popen(
                [
                    _python(repo),
                    "-m",
                    "agent.cli",
                    "audit",
                    source,
                    "--workdir",
                    str(workdir),
                ],
                cwd=repo,
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


def lemma_status(args: dict, ctx=None, **kwargs) -> str:
    """Summarize every workdir under the plugin papers dir."""
    paper = (args.get("paper") or "").strip()
    if not PAPERS_DIR.is_dir():
        return json.dumps({"papers": [], "dir": str(PAPERS_DIR)})
    dirs = (
        [PAPERS_DIR / paper]
        if paper
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
    err = _need_repo(ctx)
    if err:
        return err
    paper = (args.get("paper") or "").strip()
    workdir = PAPERS_DIR / paper
    if not workdir.is_dir():
        return json.dumps(
            {"error": f"no workdir {workdir}", "hint": "lemma_status lists them"}
        )
    repo = _repo(ctx)
    log_p = workdir / "lemma_improve.log"
    try:
        with log_p.open("ab") as log:
            proc = subprocess.Popen(
                [_python(repo), "-m", "agent.cli", "improve", str(workdir)],
                cwd=repo,
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
    if _repo(ctx) is None:
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
    err = _need_repo(ctx)
    if err:
        return err
    paper = (args.get("paper") or "").strip()
    workdir = PAPERS_DIR / paper
    if not workdir.is_dir():
        return json.dumps(
            {"error": f"no workdir {workdir}", "hint": "lemma_status lists them"}
        )
    try:
        rc, log = _run(_repo(ctx), ["judge", str(workdir)], 300)
        out = _verdicts(workdir)
        out["rc"] = rc
        out["tail"] = log[-400:]
        return json.dumps(out)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}"})
