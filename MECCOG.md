# MecCog — lemma's entry

**Challenge:** [MecCog "Papers and Findings, Step 1"](https://huggingface.co/spaces/MecCog/meccog-challenge) —
open science pilot in preclinical biology. Opened **2026-11-17**, deadline **2027-01-15**.
Submit one spreadsheet per hypothesis (5 max), a public method README per team.
Organizers ran a 38-agent collab on a prior hypothesis set: agents found papers
well but **failed at extracting experimental detail** — that gap is lemma's
discipline, which is why this is a flagship entry.

## The five hypotheses (APOE4 → endosome function → LOAD risk)

All "in some types of non-Alzheimer's, non-aged human brain cells … compared with APOE3":

| ID | Short name | Seed reference (PMID) |
|----|-----------|----------------------|
| H1 | Lower endosome pH (e.g. astrocytes) | 29946028 (Prasad & Rao, PNAS 2018) |
| H2 | Impaired ARF6-related receptor recycling | 31641056 (Rawat et al., J Neurosci 2019) |
| H3 | Increased APOE4 aggregation in endo-lysosome | 39419029 (Kaji et al., Immunity 2024) |
| H4 | Tighter APOE4 adhesion to recycling receptors | 30375977 (Xian et al., eLife 2018) |
| H5 | Lipid-metabolism disruption → impaired late endosome/lysosome | 37777962 (Lee et al., Cell Rep 2023) |

Model-system findings count as relevant — the hypotheses are about human brain
cells, but most experiments are in model systems; extracting the experimental
specifics *is* the task.

## Submission contract (validator v2, `meccog/resources/`)

- **Submission sheet**: row 1 headers A–S; hypothesis text copied verbatim into A2;
  row 3+ one block per paper: PAPER row (`P1`: DOI, source type, PMID) then
  FINDING rows (`P1.F1`, `P1.F2`, …). ≤50 papers per sheet.
- **Per finding**: description, verbatim quote, summary, relevance 0–1,
  experimental system, data location (figure panel / table), effect size,
  p-value, sample size — each stat with a **provenance code**
  (`printed_in_text` / `visual_only` / `estimated_from_figure` /
  `supplement_only` / `not_found_in_source` / `full_text_unavailable` /
  `ambiguous_panel`), plus **directness class** (1–5: is A itself perturbed or a
  surrogate A′? is B itself measured or a surrogate B′?) and **directness
  strength** (1–5 words).
- **Method sheet**: team, leader, email, method URL, commit SHA, who did the
  work (`person`/`agent`/`both`/`collab`), exact model + version.
- **Method README** (their template in `meccog/resources/`): who did what,
  models+versions, how papers were found (include failed searches), how full
  text was obtained + unreachable fraction, what a human checked, judgement
  calls. Six honest sections.
- Registration: HF OAuth, auto-approved with an institution/company-verified
  email; leader submits; sheets replaceable until the deadline.

## Why lemma fits

The challenge's stated failure mode — finding papers is easy, extracting the
finding with its quote, figure panel and statistics is hard — is precisely the
extract → audit → evidence → judge discipline. `lemma search` (Paperclip/GXL
corpus) plus OpenAlex/Europe PMC covers the find half; the audit machinery
(sandboxed numeric checks, append-only trace, self-judge) becomes the
verification layer that decides whether an extracted row is trustworthy.

## Adaptation gaps (the work)

1. **Biomedical full text.** Current intake is arXiv/OpenReview/PDF via
   Firecrawl/PyMuPDF. MecCog papers live in PubMed/PMC (XML, figure captions,
   supplements) and publisher sites. Need a PMC-first fetcher with PDF fallback
   and an honest `full_text_unavailable` path — the README asks for the
   unreachable fraction.
2. **Findings, not claims.** Extraction target is the MecCog row: verbatim
   quote + figure/table location + stats with provenance, not a checkable
   claim + success criterion. Figure-panel localization is the hard part —
   captions and panel letters, plus supplement awareness.
3. **Directness grading.** Supported/falsified/inconclusive → directness class
   (A perturbed directly vs surrogate A′, B measured directly vs surrogate B′)
   and strength. This is a judgement rubric, closer to the judge stage than to
   numeric audit; kytos's biology knowledge is the reference for what counts as
   a surrogate (e.g. LysoSensor for pH, transferrin uptake for recycling).
4. **Emit the spreadsheet.** rows → xlsx in the official template →
   `validate_submission.py --strict` in the loop before every submission, and
   the Method sheet auto-filled (team, URL, SHA, models+versions) — the trace
   already knows them.

## Working layout

- `meccog/resources/` — vendored challenge files (validator, template, example,
  hypotheses.json, method README template, directness doc). Source of truth is
  the space; re-vendor if the validator version changes.
- `meccog/papers/<pmid>/` — fetched full text + `fetch.json` (route used, what
  was reachable — feeds Method README §4).
- `meccog/discovery/<H>.json` — `meccog_find.py` shortlists: deduped Europe
  PMC + OpenAlex candidates with LLM relevance scores and `screen_why`.
- `meccog/findings/<H>__<pmid>.json` — extracted rows per (hypothesis, paper),
  quotes machine-verified verbatim, `model` recorded per file.
- `meccog/submissions/` — built xlsx + validator reports.
- `scripts/meccog_*.py` — fetch / find / extract / build drivers; the build
  step runs the vendored validator and saves the report beside each xlsx.

## Milestones

1. ✅ Vendor resources; hypotheses.json final (approved 2026-11-09, John Moult).
2. ✅ Fetch the 5 seed-reference full texts end-to-end — PMC/JATS where
   available, repository PDFs for the two Cell Press papers; `fetch.json`
   per paper records the route.
3. ✅ Extract findings from all 5 seed papers → `meccog/findings/H*__*.json`,
   every quote machine-verified verbatim against the fetched text.
4. ✅ Build + validate all five seed workbooks —
   `meccog/submissions/H{1..5}.xlsx`, each `SUBMISSION VALID` (reports
   committed next to the xlsx). One workbook per hypothesis, one seed paper
   each so far.
5. 🟡 Paper-finding pass — `scripts/meccog_find.py` (Europe PMC fielded
   queries + OpenAlex cites/cited-by snowball per seed, LLM screens batches
   of abstracts 0–1 → `meccog/discovery/<H>.json`). First pass fetched the
   top ~3 real-PMID candidates per hypothesis; sheets now hold:
   H1 4 papers / 37 findings, H2 4 / 41, H3 2 / 21, H4 4 / 73, H5 4 / 78 —
   all `SUBMISSION VALID`. Shortlists have more candidates to fetch.
   Known gap: H3's strongest lookalikes were preprints of the seed itself;
   several top hits are EPMC preprint records (PPR ids) with no PubMed
   route — needs a bioRxiv/medRxiv fetch path.
6. Human spot-check pass (Method README §5); register/team; submit.

### Extraction notes (gotchas that cost us time)

- `claude-opus-5-5` intermittently returns `stop_reason=refusal` on some
  papers (H3/PMID 39419029 refused ~10× in a row whole-text, and still
  refused some 12k-char chunks). `--chunk-chars` + `--only-chunks` rerun the
  failing span; `FALLBACK_ENDPOINTS` in `meccog_extract.py` tries the
  OpenAI-compatible `LEMMA_*` endpoints (only WANDB/Kimi answered today —
  DEEPSEEK key was reset, KIMI endpoint gone, HF paused, OPENROUTER has no
  credits). The findings JSON records `model` per paper.
- Stats must arrive in validator shapes (`40%`, `2.5x`, `d=3.7`,
  `qualitative`, `N/A`); raw-unit strings get normalized to `qualitative`
  at sheet-build time and logged, but the extract prompt now asks for
  the right shapes natively.
- The builder verifies a stat's numeric core against the paper's
  `fulltext.txt` before upgrading `not_found_in_source` → `printed_in_text`;
  unverifiable numbers are dropped to `N/A` and flagged `REVIEW` rather than
  being given an invented provenance.

Status log lives at the bottom of this file; each milestone lands with its
validator report committed under `meccog/submissions/`.
