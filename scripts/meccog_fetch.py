#!/usr/bin/env python3
"""MecCog paper fetcher — PMID -> metadata + full text.

Route order (recorded in fetch.json for the Method README §4):
  1. NCBI idconv:  PMID -> PMCID
  2. PMC efetch:   PMCID -> JATS XML (full text incl. figure/table captions)
  3. Europe PMC:   PMCID -> JATS XML (fallback mirror of the same record)
  4. PubMed efetch: always — abstract + DOI, even when full text is unreachable

Usage:
    python scripts/meccog_fetch.py 30375977
    python scripts/meccog_fetch.py --all            # every hypothesis seed PMID
    python scripts/meccog_fetch.py --all --email you@x.org   # NCBI courtesy email

Writes meccog/papers/<pmid>/: meta.json, abstract.xml, fulltext.xml (when
reachable), fulltext.txt (sectioned plain text, captions preserved), fetch.json.
"""

import argparse
import json
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "meccog" / "papers"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
IDCONV = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"

UA = "lemma-meccog/0.1 (research; contact via HF space MecCog/meccog-challenge)"


def get(url, binary=False):
    # curl via subprocess: Python's urllib hits CERTIFICATE_VERIFY_FAILED on
    # this machine's TLS-intercepting network; curl uses the system trust store.
    out = subprocess.run(
        ["curl", "-fsSL", "--max-time", "60", "-A", UA, url],
        capture_output=True,
        check=True,
    ).stdout
    return out if binary else out.decode("utf-8", "replace")


def pmid_to_pmcid(pmid):
    url = f"{IDCONV}?ids={pmid}&format=json&tool=lemma-meccog"
    data = json.loads(get(url))
    rec = (data.get("records") or [{}])[0]
    return rec.get("pmcid"), rec.get("doi")


def pubmed_record(pmid, email):
    url = f"{EUTILS}/efetch.fcgi?db=pubmed&id={pmid}&retmode=xml&tool=lemma-meccog"
    if email:
        url += f"&email={email}"
    return get(url, binary=True)


def pmc_fulltext(pmcid, email):
    pmcid_num = pmcid.replace("PMC", "")
    url = f"{EUTILS}/efetch.fcgi?db=pmc&id={pmcid_num}&retmode=xml&tool=lemma-meccog"
    if email:
        url += f"&email={email}"
    return get(url, binary=True)


def epmc_fulltext(pmcid):
    return get(f"{EPMC}/{pmcid}/fullTextXML", binary=True)


def text_of(el):
    return "".join(el.itertext())


def jats_to_text(xml_bytes):
    """JATS XML -> sectioned plain text. Figure/table captions are kept inline
    as 'FIG <label>: <caption>' because the submission's Data-location column
    cites them."""
    root = ET.fromstring(xml_bytes)
    lines = []

    title_el = root.find(".//article-title")
    if title_el is not None:
        lines.append(f"TITLE: {' '.join(text_of(title_el).split())}\n")

    for sec in root.iter():
        tag = sec.tag.split("}")[-1]
        if tag == "abstract":
            lines.append("## ABSTRACT")
            lines.append(" ".join(text_of(sec).split()))
            lines.append("")
            break

    body = None
    for el in root.iter():
        if el.tag.split("}")[-1] == "body":
            body = el
            break
    if body is None:
        return "\n".join(lines)

    def emit(el, depth=0):
        tag = el.tag.split("}")[-1]
        if tag == "sec":
            t = el.find("title")
            if t is not None:
                lines.append(
                    f"{'##' * min(depth + 1, 4)} {' '.join(text_of(t).split())}"
                )
            for child in el:
                ctag = child.tag.split("}")[-1]
                if ctag == "title":
                    continue
                emit(child, depth + 1)
        elif tag == "p":
            lines.append(" ".join(text_of(el).split()))
            lines.append("")
        elif tag in ("fig", "table-wrap"):
            label = el.find("label")
            cap = el.find("caption")
            lbl = text_of(label).strip() if label is not None else tag
            cpt = " ".join(text_of(cap).split()) if cap is not None else ""
            lines.append(f"[{tag.upper()} {lbl}] {cpt}")
            lines.append("")
        elif tag == "disp-quote":
            lines.append(" ".join(text_of(el).split()))
            lines.append("")

    for child in body:
        emit(child)

    # supplementary material descriptions (captions live in back matter)
    for el in root.iter():
        if el.tag.split("}")[-1] == "supplementary-material":
            lbl = el.find("label")
            cap = el.find("caption")
            s = f"[SUPPLEMENT {text_of(lbl).strip() if lbl is not None else ''}]"
            if cap is not None:
                s += " " + " ".join(text_of(cap).split())
            lines.append(s)

    return "\n".join(lines)


def meta_from_pubmed(xml_bytes):
    root = ET.fromstring(xml_bytes)
    art = root.find(".//Article")
    meta = {"title": None, "doi": None, "journal": None, "year": None}
    if art is not None:
        t = art.find(".//ArticleTitle")
        if t is not None:
            meta["title"] = " ".join(text_of(t).split())
        j = art.find(".//Journal/Title")
        if j is not None:
            meta["journal"] = text_of(j).strip()
        y = art.find(".//PubDate/Year")
        if y is not None:
            meta["year"] = y.text
    for aid in root.iter("ArticleId"):
        if aid.get("IdType") == "doi":
            meta["doi"] = (aid.text or "").strip()
    return meta


def seed_pmids():
    h = json.loads((ROOT / "meccog/resources/hypotheses.json").read_text())
    return [x["reference"]["pmid"] for x in h["hypotheses"]]


def fetch_one(pmid, email):
    d = OUT / pmid
    d.mkdir(parents=True, exist_ok=True)
    log = {"pmid": pmid, "routes": [], "full_text": None}

    pmcid, doi_idconv = None, None
    try:
        pmcid, doi_idconv = pmid_to_pmcid(pmid)
        log["routes"].append({"route": "idconv", "ok": True, "pmcid": pmcid})
    except Exception as e:
        log["routes"].append({"route": "idconv", "ok": False, "error": str(e)})
    time.sleep(0.4)

    try:
        xml = pubmed_record(pmid, email)
        (d / "abstract.xml").write_bytes(xml)
        meta = meta_from_pubmed(xml)
        meta["doi"] = meta["doi"] or doi_idconv
        log["routes"].append({"route": "pubmed_efetch", "ok": True})
    except Exception as e:
        meta = {"title": None, "doi": doi_idconv, "journal": None, "year": None}
        log["routes"].append({"route": "pubmed_efetch", "ok": False, "error": str(e)})
    time.sleep(0.4)

    meta["pmid"] = pmid
    meta["pmcid"] = pmcid
    (d / "meta.json").write_text(json.dumps(meta, indent=2))

    full = None
    if pmcid:
        for name, fn in (
            ("pmc_efetch", lambda: pmc_fulltext(pmcid, email)),
            ("europe_pmc", lambda: epmc_fulltext(pmcid)),
        ):
            try:
                full = fn()
                log["routes"].append({"route": name, "ok": True})
                break
            except Exception as e:
                log["routes"].append({"route": name, "ok": False, "error": str(e)})
            time.sleep(0.4)

    if full:
        (d / "fulltext.xml").write_bytes(full)
        try:
            txt = jats_to_text(full)
            (d / "fulltext.txt").write_text(txt)
            log["full_text"] = "pmc_jats"
        except ET.ParseError as e:
            log["routes"].append({"route": "jats_parse", "ok": False, "error": str(e)})
            log["full_text"] = "xml_saved_parse_failed"
    else:
        log["full_text"] = "unavailable_via_pmc" if not pmcid else "pmc_fetch_failed"

    (d / "fetch.json").write_text(json.dumps(log, indent=2))
    status = "FULL TEXT" if full else ("NO PMCID" if not pmcid else "FAILED")
    print(f"{pmid}: {status}  pmcid={pmcid}  title={(meta['title'] or '?')[:80]}")
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pmids", nargs="*")
    ap.add_argument(
        "--all", action="store_true", help="fetch all hypothesis seed PMIDs"
    )
    ap.add_argument("--email", default=None, help="NCBI courtesy email")
    a = ap.parse_args()
    pmids = list(a.pmids) + (seed_pmids() if a.all else [])
    if not pmids:
        ap.error("give PMIDs or --all")
    logs = [fetch_one(p, a.email) for p in dict.fromkeys(pmids)]
    ok = sum(1 for lg in logs if lg["full_text"] == "pmc_jats")
    print(f"\n{ok}/{len(logs)} full texts reached")


if __name__ == "__main__":
    sys.exit(main())
