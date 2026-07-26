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

We have **Modal** as our GPU workhorse. T4/L4 for cheap runs, A100-40GB for heavier experiments. Keep experiments:
- **Tractable**: ≤2-3 hours on T4 for a single claim reproduction
- **Idempotent**: `reproduce.py` should be re-runnable from scratch
- **Logged**: every metric that matters goes to Trackio, not just stdout

## Secrets and credentials

- All secrets in `.env` (gitignored) — copy from `.env.example`
- `HF_TOKEN` needs write access to publish logbook Spaces
- `MODAL_TOKEN_ID` + `MODAL_TOKEN_SECRET` for GPU compute
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

## Anti-patterns

- ❌ Cherry-pick results — report ALL runs, including failures
- ❌ Skip baselines — always run the paper's own baseline before "improvements"
- ❌ Hide compute costs — log wall-clock time and GPU hours
- ❌ Commit weights or large data — gitignored
- ❌ Copy paper code verbatim without understanding — explain what each piece does
- ❌ Trust the paper's hyperparameters blindly — verify they reproduce
