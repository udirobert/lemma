#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Papers & Findings submission validator — MecCog / APOE4 challenge
# Author: Abigail Djossou
# Date:   2026-07-03, revised 2026-10-26 (version 2)
# -----------------------------------------------------------------------------
"""
Validator for MecCog "papers and findings" submission spreadsheets.

Based on the official spreadsheet description. Checks structure and per-column rules, and prints
a clear report telling the submitter exactly what to fix if the sheet is not valid.

Usage:
    python validate_submission.py submission.xlsx
    python validate_submission.py submission.xlsx --strict   # warnings -> errors

The spreadsheet layout (official):
  Row 1  : column labels (must all be present, correct columns)
  Row 2  : hypothesis being considered (must be present)
  Row 3+ : blocks, one per paper/source. Each block is:
             - a PAPER row  : DOI, source type, PubMed ID, ID = P1, P2, ...
             - one FINDING row per finding : ID = P1.F1, P1.F2, ...

Official columns (version 2):
  B DOI | C Paper/source type | D PubMed ID | E Paper/Finding ID |
  F Finding description | G Finding quote | H Finding summary |
  I Finding relevance (0..1) | J Experimental system | K Data location |
  L Effect size | M Effect size provenance |
  N P value | O P value provenance |
  P Sample size | Q Sample size provenance

What changed in version 2:
  - each statistic now carries a provenance code, so that a value the paper
    never reported is no longer written the same way as a value the submitter
    could not find. The codes are Nikolas's audit categories plus one for a
    value estimated off a figure.
  - three rules were relaxed because they rejected what papers actually
    report: a P value may be an inequality, an effect size may be a fold
    change or a standardized effect size or the word qualitative, and a
    sample size may be a range.

Messages are written for challenge participants: they say which cell is wrong
and what the expected format is, without referring to internal templates.
"""

import argparse
import re
import sys
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter, column_index_from_string

# ---- official spec ------------------------------------------------------

# The five source types, as they are written in the template.
SOURCE_TYPES = ["PubMed published", "PubMed preprint", "Web article",
                "Database", "Other"]
ALLOWED_SOURCE_TYPES = {t.lower() for t in SOURCE_TYPES}

# The template's header row, columns A to Q. Column A carries the hypothesis
# in row 2 and no data. Kept here so that the template builder and the
# converter read the format from one place, this file.
COLUMNS = [
    "Hypothesis", "DOI", "Paper/source type", "PubMed ID", "Paper/Finding ID",
    "Finding description", "Finding quote", "Finding summary",
    "Finding relevance", "Experimental system", "Data location",
    "Effect size", "Effect size provenance",
    "P value", "P value provenance",
    "Sample size", "Sample size provenance", "Directness class",
    "Directness strength",
]

# How directly a finding bears on the hypothesis: what the experiment did,
# not what the evidence is worth. The labels are those of the directness
# document, which participants are given in full.
DIRECTNESS = {
    "1": "Direct: A itself perturbed or naturally varying, B itself measured",
    "2": "Indirect input: A perturbed through a surrogate variable A', B itself measured",
    "3": "Indirect output: A itself perturbed, B inferred from a surrogate variable B'",
    "4": "Indirect input and output: a surrogate variable for A and for B",
    "5": "Other: none of the classes above fit, explained in Finding description",
}

# How strong the link is between the experiment and the hypothesis, 1 to 5,
# 1 the strongest. The same direction as the class, so a low number always
# means closer to the hypothesis. The word is carried in the cell because a
# bare number on a scale whose strong end is 1 gets inverted by readers.
STRENGTH = {
    "1": "very strong",
    "2": "strong",
    "3": "moderate",
    "4": "weak",
    "5": "very weak",
}
RE_STRENGTH = re.compile(r"([1-5])(?:\s*[.:-]?\s*(very strong|strong|moderate|"
                         r"very weak|weak))?$", re.I)


def read_strength(v):
    """Accept '2' or '2 strong', and return the number as a string, or None.

    Two forms because the template offers the labelled one but a participant
    building the file from the column list will write a bare number, and both
    say the same thing."""
    s = "" if v is None else str(v).strip()
    if s.endswith(".0"):
        s = s[:-2]
    m = RE_STRENGTH.match(s)
    if not m:
        return None
    n, word = m.group(1), m.group(2)
    if word and word.lower() != STRENGTH[n]:
        return False          # number and word disagree
    return n


# Where a statistic came from, or why there is none.
PROVENANCE = {
    "printed_in_text", "visual_only", "estimated_from_figure",
    "supplement_only", "not_found_in_source", "full_text_unavailable",
    "ambiguous_panel",
}
# The Method sheet: seven labelled rows, label in column A, value in column B.
METHOD_SHEET_NAMES = ("method", "methods")
# Sheets that never hold submitted data: the Method sheet and the guide.
NON_DATA_SHEETS = METHOD_SHEET_NAMES + ("how to fill this in",)
# Team leader and their email name the person the organizers contact about the
# submission. Registration and teams themselves are handled on the submission page.
METHOD_FIELDS = [
    ("Team", "team"),
    ("Team leader", "leader"),
    ("Team leader email", "email"),
    ("Method URL", "url"),
    ("Commit SHA", "sha"),
    ("Who did the work", "who"),
    ("Model and version", "model"),
]
# A deliberately loose check: something@something.something, no spaces. The
# address is confirmed by writing to it, not by a regular expression.
RE_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# What strip_contact.py leaves in the team leader cells of a published Entry.
# Kept here so the two scripts cannot drift apart.
PUBLISHED_PLACEHOLDER = "removed before publication"
# Who did the work. 'collab' covers several agents working together, whose
# models are then listed one per agent in Model and version.
WHO = ["person", "agent", "both", "collab"]
# Who did the work, for the submissions that used no model at all.
WHO_NO_MODEL = {"person"}
# What a submission writes in Model and version when no model was used. 'none'
# is the word the template asks for; N/A is accepted because every other cell
# in the format takes N/A, and a human-only team reaches for it.
NO_MODEL = {"none", "n/a"}

# The most papers one submission may carry. A safety rail rather than a control:
# in the Phase 0 collab this would have bound one submission out of 45. Nothing
# expects a team to reach it, and there is deliberately no cap on findings yet.
MAX_PAPERS = 50

# Where the method is published. A git host needs a commit SHA to pin what was
# there at submission time; an archive with a DOI is already immutable, so it
# takes N/A.
GIT_HOSTS = ("github.com", "gitlab.com", "bitbucket.org", "codeberg.org",
             "sourcehut.org", "git.")
RE_URL = re.compile(r"https://[^\s/]+\.[^\s/]+/\S*$", re.I)
RE_SHA = re.compile(r"[0-9a-f]{7,40}$", re.I)

# Codes that mean "there is no usable value".
PROVENANCE_ABSENT = {"not_found_in_source", "full_text_unavailable",
                     "ambiguous_panel"}

# Codes from earlier drafts of the format, and what they are now called.
PROVENANCE_RENAMED = {"none_in_source": "not_found_in_source"}

# Accepted shapes, with an optional leading ~ marking a value estimated from
# a figure.
EST = r"~?\s*"
RE_PERCENT = re.compile(EST + r"\d+(\.\d+)?%$")
RE_FOLD = re.compile(EST + r"\d+(\.\d+)?x$", re.I)
RE_STANDARDISED = re.compile(EST + r"[a-zA-Z]+\s*=\s*-?\d+(\.\d+)?$")
RE_QUALITATIVE = re.compile(r"qualitative$", re.I)
RE_P_EXACT = re.compile(EST + r"\d+(\.\d+)?([eE][-+]?\d+)?$")
RE_P_INEQ = re.compile(EST + r"[<>]=?\s*\d+(\.\d+)?([eE][-+]?\d+)?$")
RE_N_ONE = re.compile(EST + r"\d+$")
RE_N_RANGE = re.compile(EST + r"\d+\s*-\s*\d+$")

# canonical field -> list of header aliases (lowercased, stripped) we accept
FIELD_ALIASES = {
    "doi":            ["doi"],
    "source_type":    ["paper/source type", "paper type", "source type", "paper/source", "type"],
    "pmid":           ["pubmed id", "pmid", "pubmed"],
    "id":             ["paper or source id", "paper/finding id", "finding id", "id", "code"],
    "finding_desc":   ["finding description", "findings", "finding"],
    "quote":          ["finding quote", "quote"],
    "summary":        ["finding summary", "summary"],
    "relevance":      ["finding relevance", "relevance"],
    "exp_system":     ["experimental system", "exp system"],
    "data_location":  ["data location", "data source", "data location in the source", "location"],
    "effect_size":    ["effect size"],
    "p_value":        ["p value", "p-value", "pvalue"],
    "sample_size":    ["sample size", "n"],
    "effect_size_prov": ["effect size provenance", "effect provenance",
                         "effect size source"],
    "p_value_prov":     ["p value provenance", "p-value provenance",
                         "p value source"],
    "sample_size_prov": ["sample size provenance", "sample provenance",
                         "sample size source"],
    "directness":       ["directness class", "directness", "class",
                         "evidence directness", "directness of evidence"],
    "strength":         ["directness strength", "strength", "d",
                         "strength of directness"],
}

# official column letters (fallback if header mapping fails)
OFFICIAL_COLS = {
    "doi": "B", "source_type": "C", "pmid": "D", "id": "E",
    "finding_desc": "F", "quote": "G", "summary": "H", "relevance": "I",
    "exp_system": "J", "data_location": "K",
    "effect_size": "L", "effect_size_prov": "M",
    "p_value": "N", "p_value_prov": "O",
    "sample_size": "P", "sample_size_prov": "Q",
    "directness": "R", "strength": "S",
}

# fields required on a FINDING row
FINDING_REQUIRED = ["finding_desc", "relevance", "exp_system", "data_location"]


class Report:
    def __init__(self):
        self.errors = []
        self.warnings = []

    def err(self, row, col, msg):
        self.errors.append((row, col, msg))

    def warn(self, row, col, msg):
        self.warnings.append((row, col, msg))

    def render(self, strict=False):
        """Build the report as text. Returns (ok, text) and prints nothing.

        Anything calling this from a web app or a thread must use it rather
        than dump(): stdout belongs to the whole process, so two validations
        running at once would cross each other's output and a bad file could
        be shown someone else's 'SUBMISSION VALID'.
        """
        errs = list(self.errors)
        warns = list(self.warnings)
        if strict:
            errs += warns
            warns = []
        out = ["", "=" * 66]
        if not errs and not warns:
            out += ["SUBMISSION VALID — no problems found.", "=" * 66, "",
                    "Your spreadsheet passed all checks. You can submit it.", ""]
            return True, "\n".join(out)
        status = "NOT VALID" if errs else "VALID (with suggestions)"
        out.append(f"SUBMISSION {status} — "
                   f"{len(errs)} thing(s) to fix, {len(warns)} suggestion(s)")
        out.append("=" * 66)

        def lines(items):
            for row, col, msg in sorted(items, key=lambda x: (x[0] or 0, x[1] or "")):
                loc = (f"row {row}" if row else "whole file") + (f", column {col}" if col else "")
                yield f"  • [{loc}] {msg}"

        if errs:
            out.append("")
            out.append("MUST FIX (the submission will be rejected until these are corrected):")
            out += list(lines(errs))
        if warns:
            out.append("")
            out.append("PLEASE CHECK (allowed, but worth reviewing):")
            out += list(lines(warns))
        out.append("")
        return not errs, "\n".join(out)

    def dump(self, strict=False):
        """Print the report and return whether the file passed. For the
        command line. Code that needs the text should call render()."""
        ok, text = self.render(strict=strict)
        print(text)
        return ok


def norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower()) if s is not None else ""


def map_columns(ws, rpt):
    """Map canonical field -> column letter, by header name, falling back to
    official letters. Returns (colmap, header_ok)."""
    header_row = 1
    headers = {}
    headers_raw = {}
    for c in range(1, ws.max_column + 1):
        v = ws.cell(row=header_row, column=c).value
        if v is not None and str(v).strip():
            letter = get_column_letter(c)
            headers[letter] = norm(v)
            headers_raw[letter] = str(v).strip()

    colmap = {}
    matched_by_header = {}
    used_headers = False
    matched_letters = set()
    for field, aliases in FIELD_ALIASES.items():
        found = None
        for letter, htext in headers.items():
            if htext in aliases:
                found = letter
                break
        if found:
            colmap[field] = found
            matched_by_header[field] = True
            matched_letters.add(found)
            used_headers = True
        else:
            colmap[field] = OFFICIAL_COLS.get(field)
            matched_by_header[field] = False

    # ---- header conformance check (reported up front) ----
    header_ok = True
    OFFICIAL_LABELS = {
        "doi": "DOI", "source_type": "Paper/source type", "pmid": "PubMed ID",
        "id": "Paper/Finding ID", "finding_desc": "Finding description",
        "quote": "Finding quote", "summary": "Finding summary",
        "relevance": "Finding relevance", "exp_system": "Experimental system",
        "data_location": "Data location",
        "effect_size": "Effect size", "effect_size_prov": "Effect size provenance",
        "p_value": "P value", "p_value_prov": "P value provenance",
        "sample_size": "Sample size", "sample_size_prov": "Sample size provenance",
        "directness": "Directness class",
        "strength": "Directness strength",
    }
    if not used_headers:
        rpt.err(1, None, "The column headers in row 1 were not recognised. "
                         "Please use the official submission template so the "
                         "columns are: " + ", ".join(OFFICIAL_LABELS.values()) + ".")
        header_ok = False
    else:
        missing = [OFFICIAL_LABELS[f] for f in
                   ("doi", "source_type", "pmid", "id", "finding_desc", "quote",
                    "summary", "relevance", "exp_system", "data_location",
                    "effect_size", "effect_size_prov", "p_value",
                    "p_value_prov", "sample_size", "sample_size_prov",
                    "directness", "strength")
                   if not matched_by_header.get(f)]
        if missing:
            header_ok = False
            rpt.err(1, None, "These required columns are missing from row 1: "
                    + ", ".join(missing) + ". Please add them (use the official "
                    "submission template) so every column is present with its "
                    "exact heading.")

        # extra columns that aren't part of the official layout -> warn, ignored
        extra = []
        for letter, htext in headers.items():
            if letter == "A":
                continue  # column A carries the hypothesis (row 2), no data header
            if letter not in matched_letters and htext:
                extra.append((letter, htext))
        for letter, htext in sorted(extra):
            shown = headers_raw.get(letter, htext)
            rpt.warn(1, letter, f"Column '{shown}' is not an official column "
                                f"and will be ignored. If this was meant to be "
                                f"one of the required columns, please rename it "
                                f"to the exact official heading.")
    return colmap, header_ok, matched_by_header


def cell(ws, row, letter):
    if not letter:
        return None
    return ws.cell(row=row, column=column_index_from_string(letter)).value


def is_paper_id(v):
    return bool(re.fullmatch(r"P\d+", str(v).strip())) if v is not None else False


def is_finding_id(v):
    return bool(re.fullmatch(r"P\d+\.F\d+", str(v).strip())) if v is not None else False


def check_statistic(ws, r, cm, fid, field, label, shape, shape_text, rpt):
    """One statistic and its provenance, which have to tell the same story."""
    vcol, pcol = cm.get(field), cm.get(field + "_prov")
    v = cell(ws, r, vcol)
    pv = cell(ws, r, pcol)
    vs = "" if v is None else str(v).strip()
    ps = "" if pv is None else str(pv).strip()

    if not vs:
        rpt.err(r, vcol, f"{fid}: '{label}' must not be empty. Enter a value "
                         f"({shape_text}), or 'N/A' if there is none.")
    elif norm(vs) != "n/a" and not shape(vs):
        rpt.err(r, vcol, f"{fid}: {label.lower()} '{vs}' must be {shape_text}, "
                         f"or 'N/A'.")

    if not ps:
        rpt.err(r, pcol, f"{fid}: '{label} provenance' must not be empty. Say "
                         f"where the value came from, or why there is none, "
                         f"using one of: " + ", ".join(sorted(PROVENANCE)) + ".")
        return
    if ps in PROVENANCE_RENAMED:
        rpt.err(r, pcol, f"{fid}: '{ps}' was renamed. Use "
                         f"'{PROVENANCE_RENAMED[ps]}' instead.")
        return
    if ps not in PROVENANCE:
        rpt.err(r, pcol, f"{fid}: provenance '{ps}' is not allowed. Use one of: "
                + ", ".join(sorted(PROVENANCE)) + ".")
        return

    absent = ps in PROVENANCE_ABSENT
    if absent and norm(vs) != "n/a" and vs:
        rpt.err(r, pcol, f"{fid}: {label.lower()} is '{vs}' but its provenance "
                         f"'{ps}' means there is no usable value. Either give the "
                         f"provenance you took the value from, or set the value "
                         f"to 'N/A'.")
    if not absent and norm(vs) == "n/a":
        rpt.err(r, pcol, f"{fid}: {label.lower()} is 'N/A' but its provenance "
                         f"'{ps}' says where a value came from. Use "
                         f"'not_found_in_source' if the paper does not report "
                         f"it, or "
                         f"'full_text_unavailable' if you could not read it.")
    if ps == "estimated_from_figure":
        if field == "p_value":
            rpt.err(r, pcol, f"{fid}: a P value cannot be estimated from a "
                             f"figure. If the panel carries significance stars, "
                             f"record the inequality the paper's own legend "
                             f"gives them, with provenance 'visual_only'.")
        elif not vs.startswith("~"):
            rpt.err(r, vcol, f"{fid}: {label.lower()} '{vs}' is an estimate read "
                             f"off a figure, so write it with a leading ~ "
                             f"(for example ~40%).")
    elif vs.startswith("~"):
        rpt.err(r, vcol, f"{fid}: {label.lower()} '{vs}' starts with ~, which "
                         f"marks an estimate, but its provenance is '{ps}'. Use "
                         f"'estimated_from_figure', or drop the ~.")


def method_sheet_name(wb):
    """The Method sheet, or None. Every submission carries one and no other
    kind of spreadsheet does, so its absence is the surest sign that the file
    handed over is not a submission at all."""
    for sh in wb.sheetnames:
        if sh.strip().lower() in METHOD_SHEET_NAMES:
            return sh
    return None


def check_method_sheet(wb, rpt, published=False):
    """The Method sheet, which says who produced the submission and where the
    full method is published. It lives in the same workbook so that nothing has
    to be uploaded twice, and on its own sheet so that no value sits under a
    column heading that means something else."""
    name = method_sheet_name(wb)
    if name is None:
        rpt.err(None, None, "There is no 'Method' sheet in this workbook. Every "
                            "submission carries one, naming the team, where the "
                            "method is published, and what produced the "
                            "submission. Start from the official template.")
        return
    ms = wb[name]

    found = {}
    for r in range(1, ms.max_row + 1):
        label = norm(ms.cell(row=r, column=1).value)
        if not label:
            continue
        for official, key in METHOD_FIELDS:
            if label == official.lower():
                v = ms.cell(row=r, column=2).value
                found[key] = ("" if v is None else str(v).strip(), r)

    for official, key in METHOD_FIELDS:
        if key not in found:
            rpt.err(None, None, f"The Method sheet has no row labelled "
                                f"'{official}'. All {len(METHOD_FIELDS)} are "
                                f"required.")
    if len(found) < len(METHOD_FIELDS):
        return

    def val(k):
        return found[k][0]

    def row(k):
        return found[k][1]

    if not val("team"):
        rpt.err(row("team"), "B", "Method sheet: the team name is missing.")

    leader = val("leader")
    email = val("email")
    # A published Entry has had these two cells emptied by strip_contact.py, as
    # section 6 of the official rules requires, and must still pass this
    # validator or the archive fails the project's own check. The placeholder
    # counts only when the caller says it is checking a published file. On a
    # submission it is just a team that typed those words, and letting it pass
    # would accept an Entry with no way to reach anyone.
    stripped = (published
                and norm(leader) == PUBLISHED_PLACEHOLDER
                and norm(email) == PUBLISHED_PLACEHOLDER)

    if published and not stripped:
        rpt.err(row("leader"), "B", f"Checked as a published Entry, but the "
                f"team leader cells do not read '{PUBLISHED_PLACEHOLDER}'. Run "
                f"strip_contact.py on this file before publishing it.")
        return

    if not stripped:
        # Neither cell may carry the publication placeholder on a submission,
        # whatever the other one says. Half a placeholder is still a team the
        # organizers cannot identify.
        for key, label in (("leader", "team leader"),
                           ("email", "team leader's email")):
            if norm(val(key)) == PUBLISHED_PLACEHOLDER:
                rpt.err(row(key), "B", f"Method sheet: the {label} reads "
                        f"'{PUBLISHED_PLACEHOLDER}'. That wording is what the "
                        f"organizers put there when an Entry is published. "
                        f"Give the real one.")

        if not leader:
            rpt.err(row("leader"), "B", "Method sheet: the team leader is "
                    "missing. Name the person the organizers should contact "
                    "about this submission. A team of one names themselves.")

        if not email:
            rpt.err(row("email"), "B", "Method sheet: the team leader's email "
                    "is missing. It is how the organizers reach the team about "
                    "this submission.")
        elif not RE_EMAIL.match(email):
            rpt.err(row("email"), "B", f"Method sheet: '{email}' does not look "
                    f"like an email address.")

    url = val("url")
    if not url:
        rpt.err(row("url"), "B", "Method sheet: the method URL is missing. Every "
                                 "submission must point to a public repository or "
                                 "archive holding a README that describes how it "
                                 "was produced.")
    elif not RE_URL.match(url):
        rpt.err(row("url"), "B", f"Method sheet: '{url}' does not look like a "
                                 f"public https address, for example "
                                 f"https://github.com/yourname/your-repo.")

    sha = val("sha")
    if url and RE_URL.match(url):
        if any(h in url.lower() for h in GIT_HOSTS):
            if not sha:
                rpt.err(row("sha"), "B", "Method sheet: the commit SHA is "
                                         "missing. A repository can be rewritten "
                                         "after the deadline, so give the commit "
                                         "your submission was built from.")
            elif norm(sha) == "n/a":
                rpt.err(row("sha"), "B", "Method sheet: a git repository needs a "
                                         "commit SHA, not N/A. Run 'git rev-parse "
                                         "HEAD' in your repository.")
            elif not RE_SHA.match(sha):
                rpt.err(row("sha"), "B", f"Method sheet: '{sha}' should be 7 to 40 "
                                         f"hexadecimal characters.")
        elif not sha:
            rpt.err(row("sha"), "B", "Method sheet: put the commit SHA, or 'N/A' "
                                     "for an archive with a DOI, which is already "
                                     "immutable.")
        elif norm(sha) != "n/a" and not RE_SHA.match(sha):
            rpt.err(row("sha"), "B", f"Method sheet: '{sha}' is neither a commit "
                                     f"SHA nor 'N/A'.")

    who = val("who")
    if not who:
        rpt.err(row("who"), "B", "Method sheet: say who did the work: "
                + ", ".join(WHO) + ".")
    elif norm(who) not in WHO:
        rpt.err(row("who"), "B", f"Method sheet: '{who}' is not allowed. Use one "
                                 f"of: " + ", ".join(WHO) + ".")

    model = val("model")
    if not model:
        rpt.err(row("model"), "B", "Method sheet: give the model and its exact "
                                   "version, or 'none' if no model was used. "
                                   "List one per agent where several were "
                                   "involved. An alias such as 'the latest "
                                   "version' is not a version, because it "
                                   "moves.")
    elif norm(who) in WHO:
        no_model = norm(model) in NO_MODEL
        # The two cells have to tell the same story. An agent submission that
        # names no model, or a hand-made one that names a model, is either a
        # mistake in one cell or a method that is not what it says it is.
        if no_model and norm(who) not in WHO_NO_MODEL:
            rpt.err(row("model"), "B", f"Method sheet: who did the work is "
                                       f"'{who}', so a model was used. Give it "
                                       f"with its exact version. Write 'none' "
                                       f"only where the work used no model at "
                                       f"all, which is 'person'.")
        elif not no_model and norm(who) in WHO_NO_MODEL:
            rpt.warn(row("model"), "B", f"Method sheet: who did the work is "
                                        f"'person', which means a human working "
                                        f"without a model, but a model is named. "
                                        f"If a model did part of the work, this "
                                        f"is 'both'. If it did none, the model "
                                        f"is 'none'.")
        elif not no_model and norm(who) == "collab":
            # collab means several agents, so the cell should list several
            # models, one per agent, even where two agents ran the same one.
            if not any(sep in model for sep in (",", ";", "/", "\n", " and ")):
                rpt.warn(row("model"), "B", f"Method sheet: who did the work is "
                                            f"'collab', so list one model per "
                                            f"agent, separated by commas, even "
                                            f"where two agents ran the same "
                                            f"model. Only '{model}' is given.")


def validate_report(path, strict=False, published=False):
    """Validate and return (ok, report_text) without printing anything.

    This is the entry point for the submission app and for anything else that
    runs validations side by side. validate() wraps it for the command line.
    """
    rpt = Report()
    wb = load_workbook(path, data_only=True)
    # Not wb.active: a participant who saves with the Method sheet in front
    # would otherwise have the whole file read against the wrong sheet.
    data_sheets = [n for n in wb.sheetnames
                   if n.strip().lower() not in NON_DATA_SHEETS]
    for n in data_sheets:
        if n.strip().lower() == "submission":
            ws = wb[n]
            break
    else:
        ws = wb[data_sheets[0]] if data_sheets else wb.active
    check_method_sheet(wb, rpt, published=published)
    cm, header_ok, matched = map_columns(ws, rpt)

    # A version 1 file has no provenance columns. Falling back to column
    # letters would then read the P value as a provenance code and bury the
    # submitter in errors that explain nothing, so say it once and stop.
    PROV_FIELDS = ("effect_size_prov", "p_value_prov", "sample_size_prov")
    if not any(matched.get(f) for f in PROV_FIELDS):
        # No provenance columns and no Method sheet together mean this is not a
        # submission at all, most often the wrong file. Saying three columns are
        # missing would send someone looking for them in a spreadsheet that was
        # never meant to have them.
        if method_sheet_name(wb) is None:
            return False, "\n".join([
                "", "=" * 66,
                "NOT A SUBMISSION SPREADSHEET.",
                "=" * 66, "",
                "This workbook has no Method sheet and none of the submission",
                "columns, so it does not look like a submission at all. Check",
                "that you picked the right file.",
                "",
                "A submission has two sheets: Submission, holding the papers",
                "and findings, and Method, naming the team and where the",
                "method is published. Both are in submission_template.xlsx on",
                "the shared resources page.", "",
            ])
        text = "\n".join([
            "", "=" * 66,
            "SUBMISSION NOT VALID - three required columns are missing.",
            "=" * 66, "",
            "Every statistic is recorded with where it came from, or why",
            "there is none. Your spreadsheet has the statistics but not",
            "their provenance, so these three columns are missing:", "",
            "    M  Effect size provenance",
            "    O  P value provenance",
            "    Q  Sample size provenance", "",
            "Each takes exactly one of:",
            "    " + ", ".join(sorted(PROVENANCE)), "",
            "The surest fix is to start from submission_template.xlsx on the",
            "shared resources page, which has every column in place, and copy",
            "your rows across.", "",
        ])
        return False, text
    missing_prov = [f for f in PROV_FIELDS if not matched.get(f)]
    if missing_prov:
        rpt.err(1, None, "Some provenance columns are present and others are "
                         "not. All three are required: Effect size provenance, "
                         "P value provenance, Sample size provenance.")
        return rpt.render(strict=strict)

    # ---- row 2: hypothesis present ----
    # hypothesis is expected in row 2, column A (or first non-empty cell)
    hyp_val = ws.cell(row=2, column=1).value
    if not hyp_val or not str(hyp_val).strip():
        # try any cell in row 2
        row2 = [ws.cell(row=2, column=c).value for c in range(1, ws.max_column + 1)]
        if not any(v and str(v).strip() for v in row2):
            rpt.err(2, "A", "Hypothesis being considered is missing (row 2).")

    # ---- iterate blocks from row 3 ----
    expected_paper_n = 1
    current_paper = None            # e.g. "P1"
    finding_counters = {}           # paper -> next finding index expected
    seen_dois = {}

    r = 3
    max_r = ws.max_row
    while r <= max_r:
        idv = cell(ws, r, cm["id"])
        doi = cell(ws, r, cm["doi"])
        # skip fully empty rows
        rowvals = [ws.cell(row=r, column=c).value for c in range(1, ws.max_column + 1)]
        if not any(v is not None and str(v).strip() for v in rowvals):
            r += 1
            continue

        if is_paper_id(idv):
            # ---- PAPER row ----
            pid = str(idv).strip()
            num = int(pid[1:])
            if num != expected_paper_n:
                rpt.err(r, cm["id"], f"Paper ID '{pid}' out of sequence; "
                                     f"expected 'P{expected_paper_n}'.")
            expected_paper_n = num + 1
            current_paper = pid
            finding_counters[pid] = 1

            # DOI required
            if not doi or not str(doi).strip():
                rpt.err(r, cm["doi"], f"{pid}: DOI is missing.")
            else:
                d = str(doi).strip().lower()
                # N/A is not a DOI. The format allows sources that have none,
                # so it neither warns about its shape nor counts as a duplicate.
                if d == "n/a":
                    pass
                else:
                    if not re.match(r"10\.\d{4,9}/\S+", d):
                        rpt.warn(r, cm["doi"], f"{pid}: DOI '{doi}' does not look "
                                               f"like a standard DOI (10.xxxx/...).")
                    if d in seen_dois:
                        rpt.err(r, cm["doi"], f"{pid}: this DOI is already used by "
                                              f"{seen_dois[d]}. Each source must "
                                              f"have a unique DOI.")
                    else:
                        seen_dois[d] = pid

            # source type
            st = cell(ws, r, cm["source_type"])
            if not st or not str(st).strip():
                rpt.err(r, cm["source_type"], f"{pid}: Paper/source type missing.")
            elif norm(st) not in ALLOWED_SOURCE_TYPES:
                rpt.err(r, cm["source_type"],
                        f"{pid}: source type '{st}' is not allowed. Use one of: "
                        f"PubMed published, PubMed preprint, Web article, "
                        f"Database, Other.")

            # PMID (required for PubMed sources; recommended generally)
            pmid = cell(ws, r, cm["pmid"])
            st_norm = norm(st)
            if st_norm.startswith("pubmed"):
                if not pmid or not str(pmid).strip():
                    rpt.err(r, cm["pmid"], f"{pid}: PubMed ID required for "
                                           f"PubMed sources.")
                elif not re.fullmatch(r"\d+", str(pmid).strip()):
                    rpt.err(r, cm["pmid"], f"{pid}: PubMed ID '{pmid}' must be "
                                           f"digits only.")

            # Directness belongs to a finding, not to a paper: one paper can
            # hold findings of several classes.
            for key, label in (("directness", "directness class"),
                               ("strength", "directness strength")):
                dv = cell(ws, r, cm.get(key))
                if dv is not None and str(dv).strip():
                    rpt.err(r, cm.get(key), f"{pid}: a paper row carries no "
                            f"{label}. It describes one finding's relation to "
                            f"the hypothesis, and one paper can hold findings "
                            f"of several classes and strengths. Put it on the "
                            f"finding rows.")

        elif is_finding_id(idv):
            # ---- FINDING row ----
            fid = str(idv).strip()
            fpaper, findex = fid.split(".")
            if current_paper is None:
                rpt.err(r, cm["id"], f"Finding '{fid}' appears before any paper row.")
            elif fpaper != current_paper:
                rpt.err(r, cm["id"], f"Finding '{fid}' does not belong to current "
                                     f"paper '{current_paper}'.")
            else:
                exp = finding_counters.get(current_paper, 1)
                if int(findex[1:]) != exp:
                    rpt.err(r, cm["id"], f"Finding '{fid}' out of sequence; "
                                         f"expected '{current_paper}.F{exp}'.")
                finding_counters[current_paper] = int(findex[1:]) + 1

            # required finding fields (must be non-empty)
            READABLE = {
                "finding_desc": "Finding description",
                "relevance": "Finding relevance",
                "exp_system": "Experimental system",
                "data_location": "Data location",
            }
            for fld in FINDING_REQUIRED:
                v = cell(ws, r, cm[fld])
                if v is None or not str(v).strip():
                    rpt.err(r, cm[fld], f"{fid}: '{READABLE[fld]}' is required "
                            f"but empty. Please fill it in.")

            # quote and summary must contain a value OR the text 'N/A' (never blank)
            for fld, label in (("quote", "Finding quote"),
                               ("summary", "Finding summary")):
                v = cell(ws, r, cm[fld])
                if v is None or not str(v).strip():
                    rpt.err(r, cm[fld], f"{fid}: '{label}' must not be empty. "
                            f"Enter the {label.lower()}, or 'N/A' if there is none.")

            # relevance 0..1
            rel = cell(ws, r, cm["relevance"])
            if rel is not None and str(rel).strip():
                try:
                    rv = float(rel)
                    if not (0.0 <= rv <= 1.0):
                        rpt.err(r, cm["relevance"], f"{fid}: relevance must be a "
                                f"number between 0 and 1 (found {rv}).")
                except ValueError:
                    rpt.err(r, cm["relevance"], f"{fid}: relevance must be a "
                            f"number between 0 and 1 (found '{rel}').")

            # data location precision (warn if whole-figure only)
            loc = cell(ws, r, cm["data_location"])
            if loc and str(loc).strip():
                locs = str(loc).strip()
                if re.fullmatch(r"(fig(ure)?\s*\d+)", locs, flags=re.I):
                    rpt.warn(r, cm["data_location"],
                             f"{fid}: location '{locs}' may be imprecise "
                             f"(panel not specified, e.g. Fig4C).")

            check_statistic(
                ws, r, cm, fid, "effect_size", "Effect size",
                lambda x: (RE_PERCENT.match(x) or RE_FOLD.match(x)
                           or RE_STANDARDISED.match(x) or RE_QUALITATIVE.match(x)),
                "a percentage (40%), a fold change (2.5x), a standardized "
                "effect size (d=3.7), or the word qualitative", rpt)

            check_statistic(
                ws, r, cm, fid, "p_value", "P value",
                lambda x: RE_P_EXACT.match(x) or RE_P_INEQ.match(x),
                "a number (0.03) or an inequality (<0.001)", rpt)

            check_statistic(
                ws, r, cm, fid, "sample_size", "Sample size",
                lambda x: RE_N_ONE.match(x) or RE_N_RANGE.match(x),
                "a whole number (12) or the range the paper gives (8-12)", rpt)

            # directness: the shape of the experiment relative to this
            # hypothesis. A property of the finding, so it is required on
            # every finding row and meaningless on a paper row.
            dv = cell(ws, r, cm.get("directness"))
            ds = "" if dv is None else str(dv).strip()
            if ds.endswith(".0"):
                ds = ds[:-2]
            if not ds:
                rpt.err(r, cm.get("directness"), f"{fid}: 'Directness class' "
                        f"must not be empty. Give the class of this finding "
                        f"against the hypothesis, 1 to 5.")
            elif ds not in DIRECTNESS:
                rpt.err(r, cm.get("directness"), f"{fid}: directness class "
                        f"'{ds}' is not allowed. Use a number from 1 to 5: "
                        + "; ".join("%s %s" % (k, v.split(":")[0])
                                    for k, v in sorted(DIRECTNESS.items()))
                        + ".")

            # strength: how strong the link is, within the class
            sv = cell(ws, r, cm.get("strength"))
            ss = "" if sv is None else str(sv).strip()
            st = read_strength(sv)
            if not ss:
                rpt.err(r, cm.get("strength"), f"{fid}: 'Directness strength' "
                        f"must not be empty. How strong is the link between "
                        f"this experiment and the hypothesis, 1 to 5, where 1 "
                        f"is the strongest: "
                        + ", ".join("%s %s" % (k, STRENGTH[k])
                                    for k in sorted(STRENGTH)) + ".")
            elif st is False:
                rpt.err(r, cm.get("strength"), f"{fid}: '{ss}' pairs a number "
                        f"with the wrong word. The scale is "
                        + ", ".join("%s %s" % (k, STRENGTH[k])
                                    for k in sorted(STRENGTH)) + ".")
            elif st is None:
                rpt.err(r, cm.get("strength"), f"{fid}: directness strength "
                        f"'{ss}' is not allowed. Use a number from 1 to 5, or "
                        f"the number with its word, for example '2 strong'. "
                        f"1 is the strongest link and 5 the weakest.")
            elif ds in DIRECTNESS and ds == "1" and st in ("4", "5"):
                # Class 1 perturbs A itself and measures B itself, so there is
                # no surrogate variable for an alternative route to enter by. A weak d
                # here usually means the scale was read upside down.
                rpt.warn(r, cm.get("strength"), f"{fid}: class 1 with strength "
                         f"{st} ({STRENGTH[st]}). A class 1 finding has no "
                         f"surrogate variable at either end, so there is no alternative "
                         f"route for the perturbation to reach the readout by. "
                         f"Check the scale: 1 is the strongest link, not the "
                         f"weakest.")

        else:
            # a non-empty row whose ID cell isn't a valid P#/P#.F# code
            if idv is not None and str(idv).strip():
                rpt.err(r, cm["id"], f"ID '{idv}' is neither a paper ID (P1) "
                                     f"nor a finding ID (P1.F1).")
            else:
                rpt.warn(r, cm["id"], "Non-empty row with no ID in the ID column.")
        r += 1

    if expected_paper_n == 1:
        rpt.err(None, None, "No paper/source blocks found (expected P1, P2, ...).")

    n_papers = expected_paper_n - 1
    if n_papers > MAX_PAPERS:
        rpt.err(None, None, f"This submission carries {n_papers} papers. A "
                f"submission may carry at most {MAX_PAPERS}. Keep the papers "
                f"that bear most directly on the hypothesis and remove the "
                f"rest, with the findings that hang off them.")

    return rpt.render(strict=strict)


def validate(path, strict=False, published=False):
    """Validate, print the report, and return whether it passed."""
    ok, text = validate_report(path, strict=strict, published=published)
    print(text)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="submission .xlsx to validate")
    ap.add_argument("--strict", action="store_true",
                    help="treat warnings as errors")
    ap.add_argument("--published", action="store_true",
                    help="check an Entry that has already been through "
                         "strip_contact.py, where the team leader cells are "
                         "expected to carry the placeholder. Never use this "
                         "on an incoming submission.")
    args = ap.parse_args()
    ok = validate(args.file, strict=args.strict, published=args.published)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()