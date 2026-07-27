# scripts/

Helper scripts for the ICML 2026 Agent Reproductions hackathon.

| File | Purpose |
|------|---------|
| `validate_icml_logbook.py` | Canonical validator mirrored from the org Space. Run before `trackio logbook publish`; the judge runs the same checks. |
| `validate_logbook.sh` | Thin wrapper around the validator: `./scripts/validate_logbook.sh <owner>/repro-<slug>`. |

The org Space also provides a `paper_template.py` reference scaffold; we keep the per-paper working dir at `papers/<paper-orid>/` and copy it from `papers/_template/` so the agent can iterate locally.

## Posterly poster (special award)

The `Highest-Quality Human-in-the-Loop` award requires a `gradio-app/posterly` poster embed in the logbook's executive-summary page. The workflow lives inside each paper's `poster/` directory:

```bash
cd papers/<paper-id>/poster
# 1. Fill poster.html from a posterly template (e.g. portrait_2col_neutral)
# 2. Pass all gates (preflight, style, measure, polish --strict)
python3 /path/to/posterly/tools/run_gates.py poster.html --strict-polish --report GATE_REPORT.json
# 3. Render the preview PNG
python3 /path/to/posterly/tools/render_preview.py poster.html
# 4. Generate the self-contained embed (validates slugs against logbook.json)
python3 /path/to/posterly/tools/render_logbook_embed.py \
  poster.html poster_preview.png \
  --logbook-manifest ../../.trackio/logbook/logbook.json \
  --gate-report GATE_REPORT.json --out poster_embed.html
```

The resulting `poster_embed.html` replaces the figure cell in the executive-summary page before `trackio logbook publish`.

## Notes for the org script

`validate_icml_logbook.py` is mirrored verbatim so the local repo can run the same checks the judge will, even if the org Space restructures. If the org rotates the script, refresh it:

```bash
curl -sSL -o scripts/validate_icml_logbook.py \
  "https://huggingface.co/spaces/ICML-2026-agent-repro/challenge/raw/main/scripts/validate_icml_logbook.py"
chmod +x scripts/validate_icml_logbook.py
```
