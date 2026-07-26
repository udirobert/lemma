# lemma

> *A small step in a proof.* Each paper reproduction is a lemma toward verifiable ML.

Reproduction logbooks for ICML 2026 papers, driven by coding agents (Claude/Cursor/Codex). Submissions to the [ICML 2026 Agent Reproductions](https://huggingface.co/spaces/ICML-2026-agent-repro/challenge) challenge (July 15 – August 2, 2026).

## Quick start

```bash
# 1. Set up environment
cp .env.example .env
# edit .env — at minimum set HF_TOKEN and HF_USERNAME

# 2. Install pre-commit hooks (secrets + lint + format)
pip install pre-commit detect-secrets ruff
pre-commit install

# 3. Install Trackio (the submission format for this challenge)
pip install --upgrade "trackio>=0.32.1"
trackio skills add --claude   # or --cursor / --codex / --opencode / --pi

# 4. Authenticate with HF (needs write access for publishing logbooks)
hf auth login

# 5. Pick a paper from https://icml-2026-agent-repro-challenge.static.hf.space/papers.html
#    then create a working directory from the template
cp -r papers/_template papers/<paper-id>

# 6. Open and publish a logbook
trackio logbook open --title "Repro: <paper title>"
# ... do reproduction work, logging as you go ...
trackio logbook publish <your-hf-username>/<paper-id>
```

## Project structure

```
lemma/
├── papers/              # One directory per paper reproduction
│   ├── _template/       # Starting structure (copy when starting a new paper)
│   └── <paper-id>/      # reproduce.py, configs/, README.md, results/
├── logbooks/            # Local Trackio logbook outputs (gitignored)
├── scripts/             # Helper scripts (HF upload, logbook sync, etc.)
├── notebooks/           # Exploratory notebooks
├── data/                # Synthetic / small datasets only (large data is gitignored)
├── .env.example         # Template for .env (HF_TOKEN, MODAL_TOKEN_ID, etc.)
├── .pre-commit-config.yaml  # detect-secrets + ruff + file hygiene
├── pyproject.toml       # Ruff config
├── AGENTS.md            # Guidance for AI agents working on this project
└── README.md
```

## Submission flow

1. Pick a paper from the [challenge papers list](https://icml-2026-agent-repro-challenge.static.hf.space/papers.html)
2. Use a coding agent (Claude/Cursor/Codex) to reproduce its claims
3. Log each attempt, failure, and success with Trackio
4. One logbook page per claim when possible
5. Publish the logbook to a HF Space under your username
6. The Logbook Judge auto-verifies claims and updates the leaderboard
7. Independent attempts on the same paper are welcome

## Awards ($4,000 in HF GPU credits)

- 🥇 1st place — $2,000
- 🥈 2nd place — $1,000
- ⭐ Highest-Quality Human-in-the-Loop Reproduction Award — $500 (single logbook, public agent traces)
- 🔬 Best Falsification / Negative Result Award — $500 (single logbook, public agent traces)

GPU credit slots are fully allocated; prizes and challenge remain open.

## Strategy

- **Pick strategically**: computationally tractable papers (≤2-3 hours on T4/L4), clear claims, public code available
- **Document everything**: failed attempts are valuable; the Logbook Judge rewards honest, transparent trails
- **Agent traces**: required for the two special awards; use Trackio 0.32.1+ for trace capture
- **Diversify papers**: choose something not already heavily reproduced
- **One logbook page per claim**: makes verification easier

## References

- [Challenge Space](https://huggingface.co/spaces/ICML-2026-agent-repro/challenge)
- [Papers list](https://icml-2026-agent-repro-challenge.static.hf.space/papers.html)
- [Leaderboard](https://icml-2026-agent-repro-challenge.static.hf.space/leaderboard.html)
- [Trackio docs](https://huggingface.co/docs/trackio)
- [Logbook Judge](https://huggingface.co/spaces/ICML-2026-agent-repro/logbook-judge)
