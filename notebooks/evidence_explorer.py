"""Evidence-trail explorer (CoreWeave Hacks build).

Interactive marimo notebook for inspecting a Lemma audit workdir: claims,
per-claim audit scripts + figures + verdicts, the event trace, and the judge
rubric — the inspectable evidence trail, clickable.

Run locally:   marimo edit notebooks/evidence_explorer.py   (from repo root)
Share/host:    upload to https://molab.marimo.io alongside a workdir
"""

import marimo

__generated_with = "0.16.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo
    return Path, json, mo


@app.cell
def _(Path, mo):
    # discover candidate workdirs: papers/*/ under the repo root, or any path
    # typed into the box (useful on molab where the workdir was uploaded)
    repo_root = Path.cwd()
    papers_dir = repo_root / "papers"
    candidates = sorted(
        d.name
        for d in papers_dir.iterdir()
        if d.is_dir() and (d / "results" / "audit_report.json").is_file()
    ) if papers_dir.is_dir() else []
    picker = mo.ui.dropdown(
        options=candidates or ["<none found>"],
        label="paper workdir (papers/)",
        value=candidates[0] if candidates else None,
    )
    custom = mo.ui.text(
        label="…or explicit workdir path",
        placeholder="/abs/path/to/papers/<id>",
    )
    mo.vstack([mo.md("## Evidence-trail explorer"), picker, custom])
    return candidates, custom, papers_dir, picker


@app.cell
def _(Path, custom, papers_dir, picker):
    workdir = Path(custom.value).resolve() if custom.value.strip() else papers_dir / picker.value
    return workdir,


@app.cell
def _(json, mo, workdir):
    claims_path = workdir / "claims.json"
    report_path = workdir / "results" / "audit_report.json"
    judge_path = workdir / "judge_report.json"
    claims = json.loads(claims_path.read_text()) if claims_path.is_file() else []
    report = (
        json.loads(report_path.read_text()) if report_path.is_file() else {"claims": []}
    )
    judge = json.loads(judge_path.read_text()) if judge_path.is_file() else None
    verdict_md = (
        f"**Judge: {judge['verdict']}** ({judge['score']})" if judge else "*no judge_report.json*"
    )
    mo.vstack(
        [
            mo.md(f"### `{workdir.name}` — {verdict_md}"),
            mo.ui.table(
                [
                    {
                        "claim": c["id"],
                        "status": c.get("status"),
                        "attempts": c.get("attempts"),
                        "title": next(
                            (cl.get("title", "") for cl in claims if cl["id"] == c["id"]),
                            "",
                        )[:80],
                    }
                    for c in report.get("claims", [])
                ],
                label="audit outcomes",
            ),
        ]
    )
    return claims, report


@app.cell
def _(mo, report):
    audited = [c["id"] for c in report.get("claims", []) if c.get("status") != "not_audited"]
    claim_sel = mo.ui.dropdown(
        options=audited or ["<none>"],
        label="inspect claim",
        value=audited[0] if audited else None,
    )
    claim_sel
    return claim_sel,


@app.cell
def _(json, mo, workdir, claim_sel):
    cdir = workdir / "results" / claim_sel.value.lower()
    blocks = [mo.md(f"#### {claim_sel.value}")]
    summary_path = cdir / "audit_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        blocks.append(mo.md(f"**{summary.get('status')}** — {summary.get('notes', '')}"))
        blocks.append(
            mo.md("```json\n" + json.dumps(summary.get("metrics", {}), indent=2) + "\n```")
        )
    feedback_path = cdir / "feedback.md"
    auto_path = cdir / "feedback.auto.md"
    for p, tag in ((feedback_path, "human feedback"), (auto_path, "generated notes")):
        if p.is_file():
            blocks.append(mo.accordion({f"{tag} ({p.name})": mo.md(p.read_text())}))
    figs = sorted(cdir.glob("*.png")) if cdir.is_dir() else []
    if figs:
        blocks.append(mo.hstack([mo.image(src=str(f)) for f in figs], wrap=True))
    mo.vstack(blocks)
    return cdir,


@app.cell
def _(mo, workdir):
    scripts = sorted((workdir / "results" ).glob("c*/audit_attempt*.py"))
    script_sel = mo.ui.dropdown(
        options=[str(s.relative_to(workdir)) for s in scripts] or ["<none>"],
        label="audit script",
    )
    mo.vstack(
        [
            script_sel,
            mo.md("```python\n" + (workdir / script_sel.value).read_text()[:12000] + "\n```")
            if script_sel.value != "<none>"
            else mo.md("*no scripts*"),
        ]
    )
    return


@app.cell
def _(json, mo, workdir):
    trace_path = workdir / "trace.jsonl"
    events = []
    if trace_path.is_file():
        for line in trace_path.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    mo.vstack(
        [
            mo.md(f"#### trace — {len(events)} events"),
            mo.ui.table(
                [
                    {
                        "t": e.get("elapsed_s"),
                        "stage": e.get("stage"),
                        "event": e.get("event"),
                        "claim": e.get("claim_id", ""),
                        "detail": e.get("status", e.get("message", "")),
                    }
                    for e in events
                ],
                label="events",
                pagination=True,
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
