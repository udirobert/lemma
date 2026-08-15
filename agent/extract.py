"""Stage 1 — extract structured, testable claims from paper text.

Output: <workdir>/claims.json
[
  {
    "id": "C1",
    "title": "...",
    "statement": "verbatim-style statement of the claim",
    "kind": "theorem" | "empirical" | "dataset" | "benchmark",
    "testable": true | false,
    "test_plan": "concrete numerical/empirical audit a script could run",
    "evidence_in_paper": "figure/table/theorem refs the claim rests on",
    "compute": "cpu-fast" | "gpu-small" | "gpu-large" | "external-data",
    "success_criterion": "measurable pass condition"
  }, ...
]
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.llm import complete, extract_json
from agent.traces import Trace

SYSTEM = """You are the claim-extraction stage of Lemma, an AI-scientist pipeline \
that audits research papers. Read the paper text and extract every checkable claim.

Rules:
- A claim is testable only if a script can produce numeric evidence for or against \
it within a few hours of CPU/small-GPU time. Mark theorems as testable when a \
numerical audit (independent simulation verifying the bound/trend) is feasible.
- test_plan must be concrete: what simulation/analysis, key hyperparameters, what \
"pass" means numerically.
- Never invent claims the paper does not make. If the paper has N numbered claims, \
theorems, figures, or table results, map to those.
- success_criterion must be derivable ONLY from the claim's own statement — never \
add conditions the paper does not make (no extra monotonicity, thresholds, or \
tighter tolerances than the paper states).
- success_criterion must be measurable (tolerance, trend direction, threshold).
- compute: "cpu-fast" (<10 min CPU), "gpu-small" (<3 h single GPU), \
"gpu-large", or "external-data" (needs datasets/APIs not in the paper).
- Return ONLY a JSON array."""

USER_TMPL = """Extract the checkable claims from this paper.

Paper id: {paper_id}
Title hint: {title}

--- paper text ---
{text}
--- end ---

Return a JSON array of claim objects with keys: id (C1, C2, ...), title, \
statement, kind, testable, test_plan, evidence_in_paper, compute, success_criterion. \
Order claims by importance to the paper's core contribution (max 6)."""

REQUIRED_KEYS = {
    "id",
    "title",
    "statement",
    "kind",
    "testable",
    "test_plan",
    "evidence_in_paper",
    "compute",
    "success_criterion",
}


def extract(
    paper_id: str, title: str, text: str, workdir: Path, trace: Trace
) -> list[dict]:
    out_path = workdir / "claims.json"
    if out_path.is_file():
        trace.note("extract", f"claims.json exists, reusing: {out_path}")
        return json.loads(out_path.read_text(encoding="utf-8"))

    raw = complete(
        trace,
        "extract",
        SYSTEM,
        USER_TMPL.format(paper_id=paper_id, title=title, text=text),
        max_tokens=6000,
    )
    claims = extract_json(raw)
    if not isinstance(claims, list):
        raise ValueError(f"extract: expected JSON array, got {type(claims).__name__}")

    for claim in claims:
        missing = REQUIRED_KEYS - set(claim)
        if missing:
            raise ValueError(f"extract: claim {claim.get('id')} missing keys {missing}")

    out_path.write_text(json.dumps(claims, indent=2), encoding="utf-8")
    trace.log(
        "extract",
        "claims_written",
        path=str(out_path),
        n_claims=len(claims),
        n_testable=sum(1 for c in claims if c.get("testable")),
    )
    return claims
