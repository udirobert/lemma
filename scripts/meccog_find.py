#!/usr/bin/env python3
"""MecCog paper discovery — find non-seed papers bearing on each hypothesis.

Sources:
  1. OpenAlex citation snowball — works citing / cited by the seed reference.
  2. Europe PMC keyword queries (fielded, per hypothesis).
  3. LLM screen — the hypothesis + title/abstract batches scored 0-1.

Output: meccog/discovery/<H>.json — scored shortlist, one entry per paper.
--fetch N also pulls the top N full texts through meccog_fetch.fetch_one.

Usage:
    python scripts/meccog_find.py --hyp H1
    python scripts/meccog_find.py --hyp H1 --fetch 3
    python scripts/meccog_find.py --all
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import os

from dotenv import load_dotenv
from meccog_extract import call_anthropic, parse_findings_json
from meccog_fetch import fetch_one, get

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "meccog" / "resources"
DISC = ROOT / "meccog" / "discovery"
load_dotenv(ROOT / ".env", override=True)
DEFAULT_MODEL = os.environ.get("LEMMA_MODEL", "claude-opus-5-5")

EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
OA_WORK = "https://api.openalex.org/works"

# fielded Europe PMC queries per hypothesis — deliberately broad, the LLM
# screen does the precision work
QUERIES = {
    "H1": [
        "APOE4 AND (endosome OR endosomal) AND (pH OR acidification) AND (astrocyte OR glia)",
        "apolipoprotein E4 AND endosomal AND pH",
        "APOE4 AND NHE6",
        "APOE AND endolysosomal AND pH AND brain",
    ],
    "H2": [
        "APOE4 AND ARF6 AND (recycling OR trafficking)",
        "apolipoprotein E4 AND receptor AND recycling AND (astrocyte OR glia OR brain)",
        "ApoE4 AND ABCA1 AND trafficking",
        "APOE4 AND transferrin AND recycling",
    ],
    "H3": [
        "APOE4 AND aggregation AND (endolysosome OR lysosome OR endosome)",
        "apolipoprotein E AND aggregate AND (microglia OR astrocyte)",
        "ApoE4 AND (intracellular OR endosomal) AND aggregation",
        "APOE AND aggregation AND endo-lysosomal",
    ],
    "H4": [
        "APOE4 AND receptor AND binding AND (recycling OR endocytosis)",
        'apolipoprotein E4 AND (LRP1 OR LDLR OR "LDL receptor") AND binding',
        "ApoE4 AND recycling AND block",
        "APOE AND isoform AND receptor AND affinity AND endosome",
    ],
    "H5": [
        'APOE4 AND (lipid OR cholesterol) AND (lysosome OR "late endosome" OR endolysosome)',
        "apolipoprotein E4 AND lipid AND trafficking AND (astrocyte OR microglia)",
        "ApoE4 AND lysosomal AND dysfunction",
        "APOE4 AND cholesterol AND endosome AND brain",
    ],
}

SCREEN = """You are screening biomedical papers for the MecCog challenge.

HYPOTHESIS:
{hypothesis}

TASK: For each numbered paper below (title + abstract), score how likely the
paper contains an EXPERIMENTAL FINDING bearing on the hypothesis — for or
against. 1.0 = certainly contains a direct experiment; 0.5 = plausibly
relevant / adjacent mechanism; 0.1 = only background. Note the hypothesis is
about APOE4-vs-APOE3 in non-Alzheimer's, non-aged brain cells — model systems
still count as relevant, Alzheimer's-patient-only findings count less.
Prefer papers likely to have open full text (PMC/preprint) when unsure.

Return ONLY JSON: {{"scores": [{{"i": 1, "relevance": 0.8, "why": "one clause"}}]}}

PAPERS:
{papers}"""


def epmc_search(query, page_size=30):
    url = (
        f"{EPMC_SEARCH}?query={requests_quote(query)}&format=json"
        f"&pageSize={page_size}&resulttype=core"
    )
    try:
        data = json.loads(get(url))
    except Exception as e:
        print(f"  epmc query failed: {e}", file=sys.stderr)
        return []
    out = []
    for r in data.get("resultList", {}).get("result", []):
        out.append(
            {
                "pmid": r.get("pmid") or r.get("id"),
                "doi": r.get("doi"),
                "title": r.get("title"),
                "journal": r.get("journalTitle"),
                "year": r.get("pubYear"),
                "abstract": re.sub(r"<[^>]+>", "", r.get("abstractText") or ""),
                "pmcid": r.get("pmcid"),
                "open_access": r.get("isOpenAccess") == "Y",
                "source": "epmc",
            }
        )
    return out


def requests_quote(q):
    from urllib.parse import quote

    return quote(q)


def openalex_id_for_pmid(pmid):
    try:
        data = json.loads(get(f"{OA_WORK}/pmid:{pmid}"))
        return data.get("id", "").rsplit("/", 1)[-1], data
    except Exception as e:
        print(f"  openalex resolve pmid:{pmid} failed: {e}", file=sys.stderr)
        return None, None


def oa_to_candidate(w, source):
    inv = w.get("abstract_inverted_index") or {}
    abstract = ""
    if inv:
        pos = {}
        for word, idxs in inv.items():
            for i in idxs:
                pos[i] = word
        abstract = " ".join(pos[i] for i in sorted(pos))
    loc = w.get("primary_location") or {}
    src = loc.get("source") or {}
    doi = (w.get("doi") or "").replace("https://doi.org/", "")
    pmid = None
    ids = w.get("ids") or {}
    if ids.get("pmid"):
        pmid = ids["pmid"].rsplit("/", 1)[-1]
    return {
        "pmid": pmid,
        "doi": doi or None,
        "title": w.get("title"),
        "journal": src.get("display_name"),
        "year": str(w.get("publication_year") or ""),
        "abstract": abstract,
        "pmcid": (ids.get("pmcid") or "").rsplit("/", 1)[-1] or None,
        "open_access": (w.get("open_access") or {}).get("is_oa"),
        "source": source,
    }


def snowball(seed_pmid, per_dir=25):
    """Citing + cited works for the seed paper via OpenAlex."""
    wid, seed_work = openalex_id_for_pmid(seed_pmid)
    cands = []
    if wid:
        try:
            url = (
                f"{OA_WORK}?filter=cites:{wid}&per-page={per_dir}"
                f"&select=id,ids,doi,title,publication_year,primary_location,"
                f"abstract_inverted_index,open_access"
            )
            for w in json.loads(get(url)).get("results", []):
                cands.append(oa_to_candidate(w, "openalex:cites"))
        except Exception as e:
            print(f"  openalex cites failed: {e}", file=sys.stderr)
        time.sleep(0.3)
        refs = (seed_work or {}).get("referenced_works") or []
        if refs:
            try:
                ids = "|".join(r.rsplit("/", 1)[-1] for r in refs[:per_dir])
                url = (
                    f"{OA_WORK}?filter=ids.openalex:{ids}&per-page={per_dir}"
                    f"&select=id,ids,doi,title,publication_year,primary_location,"
                    f"abstract_inverted_index,open_access"
                )
                for w in json.loads(get(url)).get("results", []):
                    cands.append(oa_to_candidate(w, "openalex:refs"))
            except Exception as e:
                print(f"  openalex refs failed: {e}", file=sys.stderr)
    return cands


def screen_batch(hyp_statement, batch, model, api_key):
    papers_txt = "\n\n".join(
        f"[{i}] {c.get('title') or '?'} ({c.get('journal') or '?'} {c.get('year') or '?'})\n"
        f"{(c.get('abstract') or 'NO ABSTRACT')[:2200]}"
        for i, c in batch
    )
    user = SCREEN.format(hypothesis=hyp_statement, papers=papers_txt)
    for attempt in range(3):
        raw, sr = call_anthropic(
            model, "You are a precise biomedical literature screener.", user, api_key
        )
        data = parse_findings_json(raw)
        if data and "scores" in data:
            return data["scores"]
        print(
            f"  screen attempt {attempt + 1} failed ({sr}); retrying", file=sys.stderr
        )
        time.sleep(10 * (attempt + 1))
    return []


def find_hypothesis(hyp_id):
    h = json.loads((RES / "hypotheses.json").read_text())
    for x in h["hypotheses"]:
        if x["id"] == hyp_id:
            return x
    raise SystemExit(f"unknown hypothesis {hyp_id}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument(
        "--fetch", type=int, default=0, help="fetch top-N shortlist entries"
    )
    ap.add_argument("--email", default=None, help="NCBI courtesy email for fetches")
    a = ap.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not in .env")

    hyps = (
        [
            x["id"]
            for x in json.loads((RES / "hypotheses.json").read_text())["hypotheses"]
        ]
        if a.all
        else [a.hyp]
    )
    if not hyps or not all(hyps):
        ap.error("give --hyp or --all")

    already = {p.name for p in (ROOT / "meccog" / "papers").glob("*")}
    DISC.mkdir(parents=True, exist_ok=True)

    for hid in hyps:
        hyp = find_hypothesis(hid)
        seed_pmid = hyp["reference"]["pmid"]
        print(
            f"\n=== {hid}: {hyp['short_name']} (seed {seed_pmid}) ===", file=sys.stderr
        )

        cands, seen = [], set()

        def add(c, seen=seen, cands=cands):
            key = c.get("pmid") or c.get("doi") or c.get("title")
            if not key or key in seen:
                return
            seen.add(key)
            cands.append(c)

        for q in QUERIES.get(hid, []):
            print(f"  epmc: {q[:70]}", file=sys.stderr)
            for c in epmc_search(q):
                add(c)
            time.sleep(0.4)

        for c in snowball(seed_pmid):
            add(c)

        cands = [c for c in cands if (c.get("pmid") or "") not in already]

        # drop near-duplicate titles of the seed paper (preprint versions etc.)
        def words(t):
            return set(re.findall(r"[a-z0-9]{4,}", (t or "").lower()))

        seed_meta = ROOT / "meccog" / "papers" / seed_pmid / "meta.json"
        seed_title = (
            json.loads(seed_meta.read_text()).get("title", "")
            if seed_meta.exists()
            else ""
        )
        sw = words(seed_title)
        cands = [
            c
            for c in cands
            if not sw or len(words(c.get("title")) & sw) / max(len(sw), 1) < 0.7
        ]
        print(
            f"  {len(cands)} unique candidates after dedupe/exclusion", file=sys.stderr
        )

        # screen in batches of 10
        for bi in range(0, len(cands), 10):
            batch = list(enumerate(cands[bi : bi + 10], start=bi + 1))
            scores = screen_batch(hyp["statement"], batch, a.model, api_key)
            for s in scores:
                i = s.get("i")
                if isinstance(i, int) and 1 <= i <= len(cands):
                    cands[i - 1]["relevance"] = s.get("relevance")
                    cands[i - 1]["screen_why"] = s.get("why")

        cands.sort(key=lambda c: -(c.get("relevance") or 0))
        out = {
            "hypothesis_id": hid,
            "hypothesis": hyp["statement"],
            "seed_pmid": seed_pmid,
            "candidates": cands,
        }
        dst = DISC / f"{hid}.json"
        dst.write_text(json.dumps(out, indent=2))
        top = [c for c in cands if (c.get("relevance") or 0) >= 0.6]
        print(
            f"  {len(top)} candidates >=0.6 -> {dst.relative_to(ROOT)}", file=sys.stderr
        )
        for c in top[:8]:
            print(
                f"    {c.get('relevance')}: {str(c.get('title'))[:75]} (pmid {c.get('pmid')})",
                file=sys.stderr,
            )

        if a.fetch:
            for c in top[: a.fetch]:
                pmid = c.get("pmid") or ""
                if pmid.isdigit():
                    fetch_one(pmid, a.email)
                else:
                    print(
                        f"    skip fetch: non-pubmed id {pmid} ({str(c.get('title'))[:60]})",
                        file=sys.stderr,
                    )


if __name__ == "__main__":
    sys.exit(main())
