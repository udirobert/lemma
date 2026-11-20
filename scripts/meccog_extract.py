#!/usr/bin/env python3
"""MecCog finding extractor — paper + hypothesis -> findings JSON.

One LLM call per (hypothesis, paper). The model writes candidate finding rows;
this script then verifies every quote verbatim against the fetched full text
(whitespace/dash/quote-normalized substring match) — the "distrust itself"
step. A quote that is not in the text is marked `quote_not_found` and the row
is still written, flagged, for a human or a re-extract to fix.

Usage:
    python scripts/meccog_extract.py --hyp H1 --pmid 29946028
    python scripts/meccog_extract.py --hyp H1 --pmid 29946028 --model claude-opus-5-5

Writes meccog/findings/<H>__<pmid>.json
"""

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
FINDINGS = ROOT / "meccog" / "findings"
FALLBACK_ENDPOINTS = ["DEEPSEEK", "KIMI", "WANDB", "HF", "OPENROUTER"]
load_dotenv(ROOT / ".env", override=True)

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = os.environ.get("LEMMA_MODEL", "claude-opus-5-5")

PROVENANCE_CODES = [
    "printed_in_text",
    "visual_only",
    "estimated_from_figure",
    "supplement_only",
    "not_found_in_source",
    "full_text_unavailable",
    "ambiguous_panel",
]

DIRECTNESS = """Directness classes (what the experiment DID, judged against the hypothesis
variables A = the cause in the hypothesis, B = the effect):
  1 Direct: A itself perturbed or naturally varying (APOE4 vs APOE3 genotype
    counts as natural perturbation), B itself measured.
  2 Indirect input: A perturbed through a surrogate A' (e.g. a drug/siRNA/kinase
    that is assumed to change A), B itself measured.
  3 Indirect output: A itself perturbed, B inferred from a surrogate B'
    (e.g. LysoSensor intensity as a proxy for endosomal pH, cholesterol efflux
    as a proxy for transporter abundance).
  4 Indirect input and output: surrogate at both ends (d unlikely below 3).
  5 Other: none fit; the description must say why.
Strength d (1 strongest .. 5 weakest) — how tight the experiment-to-hypothesis
link is, given possible alternative causal routes through the surrogate(s)."""

SYSTEM = """You are a meticulous evidence-extraction scientist working on the MecCog
open-science challenge. You extract experimental findings from a biomedical
paper that bear on a specific mechanism hypothesis, exactly as they are
reported — including findings that argue AGAINST the hypothesis or show no
effect. You never invent a quote, a statistic, or a data location; where the
paper does not report a statistic you say so via the provenance code rather
than leaving a blank. You are precise about what the experiment did versus
what the authors claimed it means."""

PROMPT = """HYPOTHESIS (the finding must bear on this, directly or through a surrogate):
{hypothesis}

TASK: Read the paper below. List every experimental finding that bears on the
hypothesis — for or against. A finding is one experiment's result (one panel or
one reported measurement), not the paper's conclusion. Include negative and
null results that are relevant. Order by relevance, most relevant first.

Return ONLY a JSON object, no prose:
{{"findings": [
  {{
    "finding_description": "what the experiment was and what it showed, one or two sentences. If directness class is 5, explain why no class fits.",
    "quote": "a short scholarly quotation (max 60 words) transcribed exactly as it appears in the paper text below — the sentence(s) reporting this finding.",
    "summary": "one sentence for a reader skimming the sheet",
    "relevance": 0.0,
    "experimental_system": "the system, e.g. 'humanized ApoE4/ApoE3 knock-in mouse primary astrocytes', 'iPSC-derived human astrocytes', '5xFAD mouse microglia in vivo'",
    "data_location": "panel-level location, e.g. 'Fig. 1B', 'Table 2', 'SI Appendix, Fig. S3'",
    "effect_size": "ONLY one of these shapes: '40%', '2.5x', 'd=3.7' (any name=number like 'r=0.51'), '~40%' only if you measured it off a figure, 'qualitative', or 'N/A'. A raw-unit change ('0.84 pH units', '>1 pH unit') is NOT allowed — write 'qualitative'; the number stays in description and quote.",
    "effect_size_prov": "one of the provenance codes",
    "p_value": "'0.03', '<0.001', '1.2e-10' (convert '× 10⁻ᴺ' to e-notation), the star-convention inequality from the legend, or 'N/A'",
    "p_value_prov": "provenance code — NEVER estimated_from_figure",
    "sample_size": "a bare whole number '12' or range '8-12' or 'N/A' — no words",
    "sample_size_prov": "provenance code — if the value is not reported in the text you were given, use not_found_in_source, even if you suspect it is in the supplement",
    "directness": "1",
    "strength": "2"
  }}
]}}

Rules:
- relevance is 0.0-1.0: how directly the finding bears on THIS hypothesis.
- provenance codes: {prov}
- {directness}
- quote must be an exact substring of the paper text (it is machine-checked
  against the source). Do not fix the paper's typos; do not paraphrase.
- Supplementary figures and tables were NOT fetched — you only have the main
  text. For a stat you cannot see, use not_found_in_source; never claim
  supplement_only for a value you did not read.
- If the paper has no finding bearing on the hypothesis, return {{"findings": []}}.

PAPER ({source_desc}):
---
{text}
---"""


def chunk_text(text, size, overlap=2000):
    chunks, i = [], 0
    while i < len(text):
        end = min(i + size, len(text))
        if end < len(text):
            nl = text.rfind("\n", i + size // 2, end)
            if nl > i:
                end = nl
        chunks.append(text[i:end])
        i = end - overlap if end < len(text) else end
    return chunks


def parse_findings_json(raw):
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        try:
            return json.loads(m.group(0), strict=False)
        except json.JSONDecodeError:
            return None


def normalize(s):
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s).strip().lower()


def load_hypothesis(hyp_id):
    h = json.loads((ROOT / "meccog/resources/hypotheses.json").read_text())
    for x in h["hypotheses"]:
        if x["id"] == hyp_id:
            return x
    raise SystemExit(f"unknown hypothesis {hyp_id}")


def extract_text(pm_response):
    for block in pm_response.get("content", []):
        if block.get("type") == "text":
            return block["text"]
    return ""


def call_openai_compat(base_url, model, system, user, api_key):
    r = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 16000,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=240,
    )
    if r.status_code != 200:
        raise SystemExit(f"openai-compat {r.status_code}: {r.text[:500]}")
    body = r.json()
    choice = (body.get("choices") or [{}])[0]
    fr = choice.get("finish_reason")
    if fr and fr != "stop":
        print(f"note: finish_reason={fr}", file=sys.stderr)
    return (choice.get("message") or {}).get("content") or "", fr


def call_anthropic(model, system, user, api_key):
    r = requests.post(
        ANTHROPIC_URL,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": model,
            "max_tokens": 16000,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        },
        timeout=600,
    )
    if r.status_code != 200:
        raise SystemExit(f"anthropic {r.status_code}: {r.text[:500]}")
    body = r.json()
    sr = body.get("stop_reason")
    if sr and sr != "end_turn":
        print(f"note: stop_reason={sr}", file=sys.stderr)
    return extract_text(body), sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp", required=True)
    ap.add_argument("--pmid", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-text-chars", type=int, default=140000)
    ap.add_argument(
        "--chunk-chars",
        type=int,
        default=0,
        help="if >0, split the paper into chunks of this size and extract each separately",
    )
    ap.add_argument(
        "--only-chunks",
        default="",
        help="comma-separated 1-based chunk indices to (re)run and merge into the existing findings file",
    )
    a = ap.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not in .env")

    hyp = load_hypothesis(a.hyp)
    paper_dir = ROOT / "meccog" / "papers" / a.pmid
    text_path = paper_dir / "fulltext.txt"
    if not text_path.exists():
        raise SystemExit(f"no full text at {text_path} — run meccog_fetch first")
    text = text_path.read_text()[: a.max_text_chars]
    meta = (
        json.loads((paper_dir / "meta.json").read_text())
        if (paper_dir / "meta.json").exists()
        else {}
    )

    source_desc = f"PMID {a.pmid} — {meta.get('title', '?')} ({meta.get('journal', '?')} {meta.get('year', '?')})"
    user = PROMPT.format(
        hypothesis=hyp["statement"],
        prov=", ".join(PROVENANCE_CODES),
        directness=DIRECTNESS,
        source_desc=source_desc,
        text=text,
    )

    print(f"extracting {a.hyp} x {a.pmid} with {a.model} ...", file=sys.stderr)
    model_used = a.model
    findings = []

    def run_one(user, label):
        nonlocal model_used
        data = None
        last_err = ""
        raw = ""
        for attempt in range(1, 5):
            raw, stop_reason = call_anthropic(a.model, SYSTEM, user, api_key)
            data = parse_findings_json(raw)
            if data is not None:
                break
            last_err = (
                f"no JSON (stop_reason={stop_reason})"
                if not re.search(r"\{.*\}", raw or "", re.S)
                else "unparseable JSON"
            )
            print(
                f"{label} attempt {attempt} failed: {last_err}; retrying",
                file=sys.stderr,
            )
            time.sleep(15 * attempt)
        if data is None:
            for ep in FALLBACK_ENDPOINTS:
                fb_key = os.environ.get(f"LEMMA_{ep}_API_KEY")
                fb_url = os.environ.get(f"LEMMA_{ep}_BASE_URL")
                fb_model = os.environ.get(f"LEMMA_{ep}_MODEL")
                if not (fb_key and fb_url and fb_model):
                    continue
                print(
                    f"{label} falling back to {ep.lower()} {fb_model}", file=sys.stderr
                )
                try:
                    raw, fr = call_openai_compat(fb_url, fb_model, SYSTEM, user, fb_key)
                except (SystemExit, requests.RequestException) as e:
                    print(f"{ep} failed: {e}", file=sys.stderr)
                    last_err = f"{ep} request error"
                    continue
                data = parse_findings_json(raw)
                if data is not None:
                    model_used = f"{ep.lower()}:{fb_model}"
                    break
                last_err = f"{ep} no usable JSON (finish_reason={fr})"
        if data is None:
            (FINDINGS / f"{a.hyp}__{a.pmid}.{label}.raw.txt").write_text(raw)
            print(f"{label} extraction failed: {last_err}; raw saved", file=sys.stderr)
            return []
        return data.get("findings", [])

    if a.chunk_chars > 0:
        chunks = chunk_text(text, a.chunk_chars)
        only = {int(x) for x in a.only_chunks.split(",") if x.strip()}
        if only:
            print(
                f"{len(chunks)} chunks of ~{a.chunk_chars} chars; rerunning {sorted(only)}",
                file=sys.stderr,
            )
        else:
            print(f"{len(chunks)} chunks of ~{a.chunk_chars} chars", file=sys.stderr)
        for ci, chunk in enumerate(chunks):
            if only and (ci + 1) not in only:
                continue
            user = PROMPT.format(
                hypothesis=hyp["statement"],
                prov=", ".join(PROVENANCE_CODES),
                directness=DIRECTNESS,
                source_desc=f"{source_desc} [excerpt {ci + 1}/{len(chunks)}]",
                text=chunk,
            )
            findings.extend(run_one(user, f"chunk{ci + 1}"))
        if only:
            dst = FINDINGS / f"{a.hyp}__{a.pmid}.json"
            if dst.exists():
                old = json.loads(dst.read_text()).get("findings", [])
                findings = old + findings
        # dedupe across overlapping chunks by normalized quote, else description
        seen, deduped = set(), []
        for f in findings:
            key = normalize(f.get("quote") or f.get("finding_description") or "")
            if key and key in seen:
                continue
            seen.add(key)
            deduped.append(f)
        if len(deduped) != len(findings):
            print(
                f"deduped {len(findings)} -> {len(deduped)} findings", file=sys.stderr
            )
        findings = deduped
    else:
        user = PROMPT.format(
            hypothesis=hyp["statement"],
            prov=", ".join(PROVENANCE_CODES),
            directness=DIRECTNESS,
            source_desc=source_desc,
            text=text,
        )
        findings = run_one(user, "main")
        if not findings:
            raise SystemExit("extraction produced no findings")

    norm_text = normalize(text)
    n_ok = 0
    for f in findings:
        q = f.get("quote", "") or ""
        found = normalize(q) in norm_text if q else False
        f["verification"] = (
            "verbatim" if found else ("missing" if not q else "quote_not_found")
        )
        n_ok += found

    out = {
        "hypothesis_id": a.hyp,
        "hypothesis": hyp["statement"],
        "pmid": a.pmid,
        "paper": meta,
        "model": model_used,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "quotes_verbatim": f"{n_ok}/{len(findings)}",
        "findings": findings,
    }
    FINDINGS.mkdir(parents=True, exist_ok=True)
    dst = FINDINGS / f"{a.hyp}__{a.pmid}.json"
    dst.write_text(json.dumps(out, indent=2))
    print(
        f"{len(findings)} findings, {n_ok} verbatim quotes -> {dst.relative_to(ROOT)}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    sys.exit(main())
