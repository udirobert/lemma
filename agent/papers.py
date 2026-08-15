"""Paper source resolution: arxiv id, openreview id, or local PDF -> text."""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

ARXIV_RE = re.compile(r"^(\d{4}\.\d{4,5})(v\d+)?$")
OPENREVIEW_RE = re.compile(r"^[A-Za-z0-9]{8,12}$")
MAX_TEXT_CHARS = 60_000  # keep prompts tractable


def resolve(source: str, workdir: Path) -> dict:
    """Return {paper_id, title_hint, source_kind, text, pdf_path}."""
    workdir.mkdir(parents=True, exist_ok=True)

    if Path(source).is_file():
        return _from_pdf(Path(source), workdir)

    m = ARXIV_RE.match(source.strip())
    if m:
        return _from_arxiv(m.group(1), workdir)

    if OPENREVIEW_RE.match(source.strip()):
        return _from_openreview(source.strip(), workdir)

    raise ValueError(
        f"Cannot resolve paper source {source!r}; expected arxiv id "
        "(e.g. 2601.19791), openreview id, or a local PDF path."
    )


def _from_pdf(pdf_path: Path, workdir: Path) -> dict:
    text = _extract_pdf_text(pdf_path)
    return {
        "paper_id": pdf_path.stem,
        "title_hint": pdf_path.stem,
        "source_kind": "local_pdf",
        "text": text,
        "pdf_path": str(pdf_path),
    }


def _from_arxiv(arxiv_id: str, workdir: Path) -> dict:
    pdf_path = workdir / f"arxiv-{arxiv_id}.pdf"
    if not pdf_path.is_file():
        url = f"https://arxiv.org/pdf/{arxiv_id}"
        urllib.request.urlretrieve(url, pdf_path)
    title = _arxiv_title(arxiv_id)
    text = _extract_pdf_text(pdf_path)
    return {
        "paper_id": f"arxiv-{arxiv_id}",
        "title_hint": title or arxiv_id,
        "source_kind": "arxiv",
        "text": text,
        "pdf_path": str(pdf_path),
    }


def _arxiv_title(arxiv_id: str) -> str | None:
    try:
        url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
        with urllib.request.urlopen(url, timeout=30) as resp:
            xml = resp.read().decode("utf-8", errors="replace")
        m = re.search(r"<title>([\s\S]*?)</title>", xml)
        entries = re.findall(r"<entry>[\s\S]*?</entry>", xml)
        if entries:
            t = re.search(r"<title>([\s\S]*?)</title>", entries[0])
            if t:
                return re.sub(r"\s+", " ", t.group(1)).strip()
        if m:
            return re.sub(r"\s+", " ", m.group(1)).strip()
    except OSError:
        pass
    return None


def _from_openreview(oid: str, workdir: Path) -> dict:
    pdf_path = workdir / f"openreview-{oid}.pdf"
    if not pdf_path.is_file():
        url = f"https://openreview.net/pdf?id={oid}"
        req = urllib.request.Request(url, headers={"User-Agent": "lemma-agent/0.1"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            pdf_path.write_bytes(resp.read())
    text = _extract_pdf_text(pdf_path)
    return {
        "paper_id": f"openreview-{oid}",
        "title_hint": oid,
        "source_kind": "openreview",
        "text": text,
        "pdf_path": str(pdf_path),
    }


def _extract_pdf_text(pdf_path: Path) -> str:
    # Prefer Firecrawl markdown (cleaner for LLM extraction) when keyed.
    try:
        from agent import firecrawl

        md = firecrawl.markdown_from_pdf(pdf_path)
        if md:
            return md[:MAX_TEXT_CHARS]
    except Exception:
        pass  # degrade to local extraction

    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    parts = [page.get_text() for page in doc]
    text = "\n".join(parts)
    return text[:MAX_TEXT_CHARS]
