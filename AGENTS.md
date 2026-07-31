# AGENTS.md — lemma

Guidance for AI agents (and humans) working on this project.

## What this is

A workspace for reproducing ICML 2026 papers using coding agents (Claude/Cursor/Codex) and publishing the trail as Trackio logbooks on Hugging Face Spaces.

The challenge: [ICML 2026 Agent Reproductions](https://huggingface.co/spaces/ICML-2026-agent-repro/challenge). Running **July 15 – August 2, 2026**.

## Core workflow

```
Pick paper → set up papers/<id>/ → drive reproduction with agent
  → log everything via Trackio → publish logbook to HF Space
  → Logbook Judge auto-verifies → leaderboard updates
```

Every claim in a paper gets a logbook page. Failures and "almost worked" notes are part of the trail.

## Working with paper directories

When starting a new paper:
```bash
cp -r papers/_template papers/<paper-id>
```

Each paper directory should have:
- `reproduce.py` — entry point that runs the reproduction
- `configs/` — hyperparameters, model configs, dataset configs
- `README.md` — the paper's claims and which ones you're testing
- `results/` — gitignored outputs (metrics, plots, model checkpoints)
- `notes.md` — running log of what the agent did, what worked, what didn't

## Compute strategy

Compute lives on **Hugging Face Jobs** under the applicant's own namespace (e.g. `Papajams`). The challenge judges reward HF-Job-scaled experiments with recorded Job URLs, GPU flavor, and command; the org does NOT grant `job.write`, so each applicant runs Jobs under their own HF account. Modal / local Docker is only for smoke tests. Keep experiments:
- **Tractable**: ≤2-3 hours on T4/L4 for a single claim reproduction (or smaller / CPU-only for theory papers)
- **Idempotent**: `reproduce.py` should be re-runnable from scratch
- **Logged**: every metric that matters goes to Trackio, not just stdout

Before designing GPU experiments, smoke-test the flavor with a `--timeout 3m` canary (`hf jobs run --flavor <gpu> --timeout 3m python:3.12 python -c "import torch; print(torch.cuda.is_available())"`). A 402 means add credits; a 403 means the token lacks `job.write`. **A `RUNNING` state is not proof of progress**, and a detached submit's exit 0 is not proof of completion — always poll `hf jobs logs <id>` until you see real output before recording a Job as done.

## Secrets and credentials

- All secrets in `.env` (gitignored) — copy from `.env.example`
- `HF_TOKEN` needs write access to publish logbook Spaces
- `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`) for coding-agent traces via Trackio
- Pre-commit hooks scan for leaked secrets on every commit

## Documentation discipline

The judge rewards **honest, transparent trails**. Document:
- What you tried (even if it failed)
- Why you tried it (your hypothesis)
- What actually happened (the result)
- What you'd do differently next time

A logbook with 5 failed attempts that lead to 1 successful reproduction is worth more than 1 attempt that mysteriously worked.

## Tone

This is research, not a production system. Be rigorous, be honest, document mistakes. The reproduction trail is the deliverable.

## Logbook shape (judge-validated)

Every published logbook must satisfy `scripts/validate_icml_logbook.py` (mirrored from the org Space). The required structure:
- One `Index` page with `# Reproduction: <paper title>` and a `## Pages` table linking each Page slug. No external paper link on the index.
- An `Executive summary` page with two **pinned** cells in this order: (1) `Executive summary` (outcome-first paragraph + `## Scope & cost` table comparing *This reproduction* vs *Full replication* with `≈$X (est. = wall × flavor-rate)` cost), (2) a self-contained `Reproduction poster` figure cell built with `Chenruishuo/posterly`'s `tools/render_logbook_embed.py` against a passing `--strict-polish` gate.
- One page per claim with the audit / experiment evidence; link every Hub asset (Models, Datasets, Spaces, Jobs, Buckets) and GitHub repo with full URLs.
- A `Conclusion` page summarizing supported / falsified / inconclusive claims.
- Run `./scripts/validate_logbook.sh <owner>/repro-<slug>` before `trackio logbook publish`; iterate until it passes.

Agent traces (Trackio ≥ 0.32.1) are required only for the two $500 special awards (`Highest-Quality Human-in-the-Loop`, `Best Falsification / Negative Result`), but empty Traces is the most common fail mode — attach at the start of work, not the end.

## Anti-patterns

- ❌ Cherry-pick results — report ALL runs, including failures
- ❌ Skip baselines — always run the paper's own baseline before "improvements"
- ❌ Hide compute costs — log wall-clock time and GPU hours
- ❌ Commit weights or large data — gitignored
- ❌ Copy paper code verbatim without understanding — explain what each piece does
- ❌ Trust the paper's hyperparameters blindly — verify they reproduce
