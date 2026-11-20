#!/usr/bin/env python3
"""MecCog sheet builder — findings JSON -> submission xlsx -> validator report.

Reads every meccog/findings/<H>__*.json, assembles the official template
(paper rows ordered by best relevance, P1 first), fills the Method sheet,
writes meccog/submissions/<H>.xlsx, then runs the vendored
validate_submission.py and saves the report next to it.

Usage:
    python scripts/meccog_build_sheet.py --hyp H1
    python scripts/meccog_build_sheet.py --hyp H1 --team persidian \
        --leader "Udi Ngethe" --email ungethe@gmail.com
"""

import argparse
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "meccog" / "resources"
FINDINGS = ROOT / "meccog" / "findings"
SUBM = ROOT / "meccog" / "submissions"

EST = r"~?\s*"
RE_PERCENT = re.compile(EST + r"\d+(\.\d+)?%$")
RE_FOLD = re.compile(EST + r"\d+(\.\d+)?x$", re.I)
RE_STD = re.compile(EST + r"[a-zA-Z]+\s*=\s*-?\d+(\.\d+)?$")
RE_QUAL = re.compile(r"qualitative$", re.I)
RE_P = re.compile(
    EST + r"(\d+(\.\d+)?([eE][-+]?\d+)?|[<>]=?\s*\d+(\.\d+)?([eE][-+]?\d+)?)$"
)
RE_N = re.compile(EST + r"(\d+|\d+\s*-\s*\d+)$")

X10 = re.compile(r"^~?\s*([<>]=?\s*)?(\d+(\.\d+)?)\s*[×x]\s*10[-−](\d+)$")
PROV_ABSENT = {"not_found_in_source", "full_text_unavailable", "ambiguous_panel"}


def normalize_text(s):
    s = unicodedata.normalize("NFKC", s)
    for a, b in [
        ("−", "-"),
        ("–", "-"),
        ("—", "-"),
        ("’", "'"),
        ("‘", "'"),
        ("“", '"'),
        ("”", '"'),
    ]:
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip().lower()


def value_in_fulltext(val, norm_text):
    """Can we point at this value in the paper text? 'qualitative' is always
    fine — it describes the finding, which the verified quote already anchors.
    Otherwise the numeric core must literally appear."""
    s = norm_dash(str(val or "")).strip().lower()
    if not s or s == "n/a" or s == "qualitative":
        return True
    if norm_text is None:
        return False
    m = re.search(r"\d+(\.\d+)?([eE][-+]?\d+)?", s)
    return bool(m) and m.group(0) in norm_text


def norm_dash(s):
    return s.replace("−", "-").replace("–", "-").replace("—", "-").strip()


def fix_effect(v):
    """Validator effect-size shapes: ~?percent | ~?Nx | name=number |
    qualitative | N/A. Anything else collapses to 'qualitative' — the real
    number lives verbatim in the quote and in the description."""
    s = norm_dash(str(v or "")).strip()
    if not s or s.lower() == "n/a":
        return "N/A", False
    if RE_PERCENT.match(s) or RE_FOLD.match(s) or RE_STD.match(s) or RE_QUAL.match(s):
        return s, True
    return "qualitative", False


def fix_p(v):
    """P value: number or inequality or N/A. Convert 'x × 10−N' to 'x e-N'."""
    s = norm_dash(str(v or "")).strip()
    if not s or s.lower() == "n/a":
        return "N/A", True
    m = X10.match(s)
    if m:
        s = f"{m.group(1) or ''}{m.group(2)}e-{m.group(4)}"
    if RE_P.match(s):
        return s, True
    return "N/A", False


def fix_n(v):
    """Sample size: int or int-int range or N/A. Pull a leading integer."""
    s = str(v or "").strip()
    if not s or s.lower() == "n/a":
        return "N/A", True
    if RE_N.match(s):
        return s, True
    m = re.match(r"~?\s*(\d+)", s)
    if m:
        return m.group(1), False
    return "N/A", False


def find_hyp(hyp_id):
    h = json.loads((RES / "hypotheses.json").read_text())
    for x in h["hypotheses"]:
        if x["id"] == hyp_id:
            return x["statement"]
    raise SystemExit(f"unknown hypothesis {hyp_id}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--team", default="persidian")
    ap.add_argument("--leader", default="Udi Ngethe")
    ap.add_argument("--email", default="ungethe@gmail.com")
    ap.add_argument("--method-url", default="https://github.com/udirobert/lemma")
    ap.add_argument("--sha", default=None, help="defaults to git HEAD")
    ap.add_argument("--who", default="both")
    ap.add_argument(
        "--model",
        default=None,
        help="defaults to the models used in the findings files",
    )
    a = ap.parse_args()

    files = sorted(FINDINGS.glob(f"{a.hyp}__*.json"))
    if not files:
        raise SystemExit(f"no findings for {a.hyp} — run meccog_extract first")
    papers = [json.loads(f.read_text()) for f in files]
    models = sorted({p["model"] for p in papers})
    models_used = a.model or "; ".join(models)

    sha = (
        a.sha
        or subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()[:40]
    )

    # order papers by their best finding relevance (P1 first), skip empty
    def best_rel(p):
        return max((f.get("relevance") or 0 for f in p["findings"]), default=0)

    papers.sort(key=best_rel, reverse=True)

    wb = load_workbook(RES / "submission_template.xlsx")
    ws = wb["Submission"]
    ws["A2"] = find_hyp(a.hyp)

    r, pnum, fixes = 3, 0, []
    for p in papers:
        if not p["findings"]:
            continue
        pnum += 1
        pid = f"P{pnum}"
        meta = p.get("paper") or {}
        ft = ROOT / "meccog" / "papers" / str(p["pmid"]) / "fulltext.txt"
        norm_ft = normalize_text(ft.read_text()) if ft.exists() else None
        ws.cell(r, 2, meta.get("doi") or "N/A")  # B DOI
        ws.cell(r, 3, "PubMed published")  # C source type
        ws.cell(r, 4, p["pmid"])  # D PMID
        ws.cell(r, 5, pid)  # E id
        r += 1
        for k, f in enumerate(
            sorted(p["findings"], key=lambda x: -(x.get("relevance") or 0)), 1
        ):
            fid = f"{pid}.F{k}"
            es, es_ok = fix_effect(f.get("effect_size"))
            pv, pv_ok = fix_p(f.get("p_value"))
            ns, ns_ok = fix_n(f.get("sample_size"))
            provs, vals = {}, {}
            for label, val, ok, prov_key in (
                ("effect_size", es, es_ok, "effect_size_prov"),
                ("p_value", pv, pv_ok, "p_value_prov"),
                ("sample_size", ns, ns_ok, "sample_size_prov"),
            ):
                prov = f.get(prov_key) or "not_found_in_source"
                if val == "N/A" and prov not in PROV_ABSENT:
                    # a provenance that names a source requires a real value;
                    # model claiming supplement_only for a stat it never read
                    # is downgraded to the honest absent code
                    fixes.append(
                        f"{fid}: {prov_key} '{prov}' -> 'not_found_in_source' (value N/A)"
                    )
                    prov = "not_found_in_source"
                if val != "N/A" and prov in PROV_ABSENT:
                    if value_in_fulltext(val, norm_ft):
                        fixes.append(
                            f"{fid}: {prov_key} '{prov}' -> 'printed_in_text' "
                            f"(value verified in text)"
                        )
                        prov = "printed_in_text"
                    else:
                        # unverifiable number: drop it rather than invent a source
                        fixes.append(
                            f"{fid}: {label} '{val}' not found in paper text — "
                            f"value dropped to N/A, REVIEW"
                        )
                        val = "N/A"
                        prov = "not_found_in_source"
                provs[prov_key] = prov
                vals[label] = val
                if not ok:
                    fixes.append(f"{fid}: {label} '{f.get(label)}' -> '{val}'")
            if f.get("verification") != "verbatim":
                fixes.append(
                    f"{fid}: QUOTE NOT VERBATIM ({f.get('verification')}) — review before submit"
                )
            row = {
                5: fid,
                6: f.get("finding_description", ""),
                7: f.get("quote", ""),
                8: f.get("summary", ""),
                9: f.get("relevance", 0),
                10: f.get("experimental_system", ""),
                11: f.get("data_location", ""),
                12: vals["effect_size"],
                13: provs["effect_size_prov"],
                14: vals["p_value"],
                15: provs["p_value_prov"],
                16: vals["sample_size"],
                17: provs["sample_size_prov"],
                18: f.get("directness", "5"),
                19: f.get("strength", "5"),
            }
            for col, val in row.items():
                ws.cell(r, col, val if val not in (None, "") else "N/A")
            r += 1

    ms = wb["Method"]
    for row_i, val in (
        (1, a.team),
        (2, a.leader),
        (3, a.email),
        (4, a.method_url),
        (5, sha),
        (6, a.who),
        (7, models_used),
    ):
        ms.cell(row_i, 2, val)

    SUBM.mkdir(parents=True, exist_ok=True)
    out = SUBM / f"{a.hyp}.xlsx"
    wb.save(out)

    rep = subprocess.run(
        [sys.executable, str(RES / "validate_submission.py"), str(out)],
        capture_output=True,
        text=True,
    )
    (SUBM / f"{a.hyp}.report.txt").write_text(rep.stdout + rep.stderr)
    print(rep.stdout + rep.stderr)
    if fixes:
        print("NORMALIZATIONS / FLAGS:")
        for x in fixes:
            print("  -", x)
    print(f"\nwrote {out.relative_to(ROOT)} — {pnum} papers")


if __name__ == "__main__":
    sys.exit(main())
