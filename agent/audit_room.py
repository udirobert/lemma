from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from agent.auditor import _summary_problems, control_passed

SCHEMA_VERSION = 1
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}\Z")
ATTEMPT_NAME = re.compile(
    r"(?:audit_attempt|run_attempt)(\d+)(?:\.failed)?\.(?:py|json)\Z"
)
MAX_TEXT = 100_000


def load_json(path: Path, default=None):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def text_artifact(path: Path, limit: int = MAX_TEXT) -> dict | None:
    if not path.is_file() or path.is_symlink():
        return None
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    return {
        "name": path.name,
        "text": text[:limit],
        "truncated": len(text) > limit,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def safe_url(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    parsed = urlparse(value)
    return (
        value
        if parsed.scheme == "https" and parsed.netloc and not parsed.username
        else None
    )


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def checks(summary: dict | None, cid: str) -> dict:
    if not summary:
        return {
            "state": "not_evaluated",
            "control": "missing",
            "problems": ["No recorded summary available."],
        }
    problems = _summary_problems(summary)
    if summary.get("claim_id") not in (None, cid):
        problems.append("Summary claim ID does not match this claim.")
    metrics = summary.get("metrics")
    value = metrics.get("control_pass") if isinstance(metrics, dict) else None
    control = (
        "passed"
        if control_passed(value)
        else "failed"
        if value in (False, 0, "False", "false") and value is not None
        else "missing"
    )
    state = (
        "failed"
        if problems
        else "inconclusive"
        if summary.get("status") == "inconclusive"
        else "passed"
    )
    return {"state": state, "control": control, "problems": problems}


def image_asset(path: Path, public_dir: Path | None) -> dict | None:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 2 * 1024 * 1024:
        return None
    raw = path.read_bytes()
    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    digest = hashlib.sha256(raw).hexdigest()
    name = digest + ".png"
    if public_dir is not None:
        public_dir.mkdir(parents=True, exist_ok=True)
        target = public_dir / name
        if not target.exists():
            shutil.copyfile(path, target)
    return {"name": path.name, "url": "/audit-room/assets/" + name, "sha256": digest}


def versioned_attempt(root: Path, cid: str, public_dir: Path | None) -> dict:
    inputs = load_json(root / "input.json", {})
    outcome = load_json(root / "outcome.json", {})
    claim = inputs.get("claim") or {}
    script = text_artifact(root / "script.py")
    problems = list(outcome.get("problems") or [])
    integrity = checks(outcome.get("summary"), cid)
    if script is None:
        problems.append("Versioned script is missing.")
    elif script["sha256"] != inputs.get("script_sha256"):
        problems.append("Script does not match its recorded digest.")
    if inputs.get("attempt_id") != root.name:
        problems.append("Attempt identity does not match its snapshot directory.")
    criterion = claim.get("success_criterion", "")
    if hashlib.sha256(criterion.encode()).hexdigest() != inputs.get("criterion_sha256"):
        problems.append("Criterion does not match its recorded digest.")
    if claim.get("id") != cid:
        problems.append("Attempt input claim ID does not match selected claim.")
    figures = []
    for item in outcome.get("figures", []):
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe figure path in attempt record")
        asset = image_asset(root / relative, public_dir)
        if asset:
            if asset["sha256"] != item.get("sha256"):
                problems.append("Figure does not match its recorded digest.")
            figures.append(asset)
    if problems:
        integrity = {
            **integrity,
            "state": "failed",
            "problems": list(dict.fromkeys(integrity["problems"] + problems)),
        }
    return {
        "id": root.name,
        "number": inputs.get("number"),
        "run_id": inputs.get("run_id"),
        "source": inputs.get("source", "unknown"),
        "record_type": "versioned",
        "created_at": inputs.get("created_at"),
        "script": script,
        "criterion": criterion,
        "criterion_sha256": inputs.get("criterion_sha256"),
        "feedback": {
            "human": inputs.get("feedback", ""),
            "generated": inputs.get("auto_feedback", ""),
            "binding": "attempt_input",
        },
        "summary": outcome.get("summary"),
        "checks": integrity,
        "exit_code": outcome.get("exit_code"),
        "wall_s": outcome.get("wall_s"),
        "stdout": text_artifact(root / "stdout.txt"),
        "stderr": text_artifact(root / "stderr.txt"),
        "figures": figures,
        "weave_url": safe_url(outcome.get("weave_url")),
        "execution": "completed" if (root / "outcome.json").is_file() else "unfinished",
    }


def legacy_attempt(claim_dir: Path, number: int, cid: str) -> dict:
    run = load_json(claim_dir / f"run_attempt{number}.json", {})
    failed = load_json(claim_dir / f"run_attempt{number}.failed.json", {})
    record = failed or run
    script = text_artifact(claim_dir / f"audit_attempt{number}.py")
    summary = record.get("summary")
    integrity = checks(summary, cid)
    if failed:
        integrity = {
            **integrity,
            "state": "failed",
            "problems": failed.get("problems")
            or ["Execution failed or summary was rejected."],
        }
    return {
        "id": f"legacy-{number}",
        "number": number,
        "run_id": record.get("run_id"),
        "source": record.get("source", "agent_generated" if script else "unknown"),
        "record_type": "legacy",
        "created_at": None,
        "script": script,
        "criterion": None,
        "criterion_sha256": None,
        "feedback": {"human": "", "generated": "", "binding": "unavailable"},
        "summary": summary,
        "checks": integrity,
        "exit_code": record.get("exit_code"),
        "wall_s": record.get("wall_s"),
        "stdout": None,
        "stderr": {
            "name": "recorded output tail",
            "text": failed["tail"],
            "truncated": True,
        }
        if failed.get("tail")
        else None,
        "figures": [],
        "weave_url": safe_url(record.get("weave_url")),
        "execution": "recorded" if record else "output_unavailable",
    }


def claim_record(
    workdir: Path, claim: dict, report: dict, public_dir: Path | None
) -> dict:
    cid = claim["id"]
    if not SAFE_ID.fullmatch(cid):
        raise ValueError(f"Unsafe claim id: {cid!r}")
    claim_dir = workdir / "results" / cid.lower()
    attempts = []
    versioned_numbers = set()
    for root in sorted((claim_dir / "attempts").glob("*")):
        if root.is_dir() and not root.is_symlink() and (root / "input.json").is_file():
            attempt = versioned_attempt(root, cid, public_dir)
            attempts.append(attempt)
            versioned_numbers.add(attempt["number"])
    numbers = {
        int(match[1])
        for path in claim_dir.glob("*")
        if (match := ATTEMPT_NAME.fullmatch(path.name))
    }
    attempts.extend(
        legacy_attempt(claim_dir, number, cid)
        for number in sorted(numbers - versioned_numbers)
    )
    attempts.sort(key=lambda a: (a["number"] or 0, a["id"]))
    summary = report.get("summary") or load_json(claim_dir / "audit_summary.json")
    if summary is None and report.get("status") not in (None, "not_audited"):
        summary = {
            "status": report["status"],
            "metrics": report.get("metrics", {}),
            "notes": report.get("notes", ""),
        }
    current_figures = [
        asset
        for p in sorted(claim_dir.glob("*.png"))
        if (asset := image_asset(p, public_dir))
    ]
    return {
        "id": cid,
        "title": claim.get("title", claim.get("claim", cid)),
        "statement": claim.get("statement", ""),
        "evidence_in_paper": claim.get("evidence_in_paper", ""),
        "criterion": claim.get("success_criterion", ""),
        "test_plan": claim.get("test_plan", ""),
        "compute": claim.get("compute", "not recorded"),
        "testable": claim.get("testable", True),
        "status": (summary or {}).get("status", report.get("status", "not_audited")),
        "summary": summary,
        "checks": checks(summary, cid),
        "source": report.get("source", "unknown"),
        "candidate_summary": report.get("candidate_summary"),
        "reference_summary": report.get("reference_summary"),
        "reference_checked": report.get("reference_checked"),
        "reference_disagreement": report.get("reference_disagreement"),
        "final_attempt_id": report.get("attempt_id"),
        "attempts": attempts,
        "current_feedback": {
            "human": text_artifact(claim_dir / "feedback.md"),
            "generated": text_artifact(claim_dir / "feedback.auto.md"),
            "binding": "current_unversioned",
        },
        "current_figures": current_figures,
    }


def paper_record(workdir: Path, meta: dict, public_dir: Path | None) -> dict:
    raw_claims = load_json(workdir / "claims.json", [])
    report = load_json(workdir / "results" / "audit_report.json", {})
    outcomes = {c["id"]: c for c in report.get("claims", [])}
    claims = {c["id"]: c for c in raw_claims}
    for cid, outcome in outcomes.items():
        if cid not in claims:
            claims[cid] = {"id": cid, "title": outcome.get("title", cid)}
    records = [
        claim_record(workdir, claim, outcomes.get(cid, {}), public_dir)
        for cid, claim in claims.items()
    ]
    rounds = []
    for path in sorted((workdir / "results").glob("improve_*.json")):
        record = load_json(path, {})
        rounds.append(
            {
                key: record[key]
                for key in (
                    "ts",
                    "round_id",
                    "targets",
                    "before",
                    "after",
                    "changed",
                    "verify_gain",
                    "before_attempt_ids",
                    "after_attempt_ids",
                )
                if key in record
            }
        )
    return {
        "slug": meta.get("slug", workdir.name),
        "paper_id": workdir.name,
        "title": meta.get("title", workdir.name),
        "source_label": meta.get("source_label", "Paper"),
        "source_url": safe_url(meta.get("source_url")),
        "role": meta.get("role", "audit"),
        "links": {k: safe_url(v) for k, v in meta.get("links", {}).items()},
        "claims": records,
        "improvements": rounds,
        "scope_note": "Recorded numerical audits in their stated test setups; not a proof of the paper or a full reproduction.",
    }


def build_bundle(papers_dir: Path, public_dir: Path | None = None) -> dict:
    papers = []
    for workdir in sorted(papers_dir.iterdir()):
        if (
            not workdir.is_dir()
            or workdir.is_symlink()
            or not (workdir / "meta.json").is_file()
        ):
            continue
        meta = load_json(workdir / "meta.json")
        if not SAFE_ID.fullmatch(meta.get("slug", workdir.name)):
            raise ValueError("Unsafe paper slug")
        papers.append(paper_record(workdir, meta, public_dir))
    return json_safe(
        {
            "schema_version": SCHEMA_VERSION,
            "mode": "recorded",
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "papers": papers,
        }
    )


def export_bundle(repo: Path) -> Path:
    bundle = build_bundle(
        repo / "papers", repo / "site" / "public" / "audit-room" / "assets"
    )
    target = repo / "site" / "src" / "data" / "audit-room.json"
    target.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return target
