# scripts/

Helper scripts for the ICML 2026 Agent Reproductions hackathon.

| File | Purpose |
|------|---------|
| `validate_icml_logbook.py` | Canonical validator mirrored from the org Space. Run before `trackio logbook publish`; the judge runs the same checks. |
| `validate_logbook.sh` | Thin wrapper around the validator: `./scripts/validate_logbook.sh <owner>/repro-<slug>`. |

The org Space also provides a `paper_template.py` reference scaffold; we keep the per-paper working dir at `papers/<paper-orid>/` and copy it from `papers/_template/` so the agent can iterate locally.

## Notes for the org script

`validate_icml_logbook.py` is mirrored verbatim so the local repo can run the same checks the judge will, even if the org Space restructures. If the org rotates the script, refresh it:

```bash
curl -sSL -o scripts/validate_icml_logbook.py \
  "https://huggingface.co/spaces/ICML-2026-agent-repro/challenge/raw/main/scripts/validate_icml_logbook.py"
chmod +x scripts/validate_icml_logbook.py
```
