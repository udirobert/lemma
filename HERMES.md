# HERMES.md — Lemma rides inside Hermes: the agent that audits what agents read

**Status:** planned — secondary entry behind elcaro (see `PROTEIN.md` for the
other new muscle). Sibling to `HACKATHON.md` / `SUBMISSION.md` / `PROTEIN.md`.
**Last updated:** 2026-11-05.

## The event

**Hermes Agent Plugin Hackathon** (Nous Research) · 7 days · **submissions due
Tuesday Nov 11, 5:00 PM EST** (internal target: Sun Nov 9).

- Prizes: $20,000 — 1st $10k · 2nd $5k · 3rd $2.5k · 4th $1.5k · 5th $1k ·
  runners-up ×5 (merch + $250 creds)
- Judged on **usefulness · reliability · polish · creativity**; explicit bias
  for "plugins that do more than a prompt or a skill could"
- Entry path: PR adding the plugin to the catalog → demo video on X tagging
  @NousResearch (≤90 s, must show the plugin working) →
  form: https://form.typeform.com/to/FwJLvOln (form can send as soon as the
  PR is open; catalog checks run when a maintainer starts, reviews every few
  days)

## Why the "audit this paper" niche is empty — and ours

Marketplace scouted 2026-11-05 (Tavily): **~976 plugins** since launch
(Official ~20, Certified 9). Search of the catalog returns only a bundled
*research-paper-writing* skill (authoring NeurIPS/ICML papers). **No plugin
extracts claims, runs audits, or preserves evidence.** Hermes users read
papers constantly; nothing in the agent can ground what's real in them.

Honest caveat: the pool of daily users who audit papers is thinner than the
pool who want security or phone control. That is why this is the secondary
entry, and why the demo must lead with a verdict, not a philosophy.

## The shape (native plugin, standalone-repo form)

```
lemma/hermes-plugin/
├── plugin.yaml            # name: lemma; provides_tools; python_dependencies
├── __init__.py            # register(ctx): tools + bundled skill
├── tools.py               # wrap agent.cli stages
├── skills/audit-this-paper/SKILL.md
└── fixture/               # demo paper with a verdict we know cold
```

| Tool | Wraps | Demo behavior |
|---|---|---|
| `lemma_quick_audit` | extract + audit 2 claims + judge, fast path | the ≤90 s video: paste arXiv ID → supported/falsified verdict + trail path in under a minute |
| `lemma_audit` | full pipeline (`python -m agent.cli audit`) | long runs; keyed off quick path |
| `lemma_judge` / `lemma_status` | existing judges + trail inspection | "open the evidence trail" — the money shot: a Hermes conversation ending in a preserved evidence folder, not a paragraph |

Integration requirements from the developer guide (verified): native
`plugin.yaml` + `register(ctx)`; state under
`plugin_data_dir("lemma")/papers` (never the install tree); Python deps in
`plugin.yaml`; `hermes plugins doctor --ci` as the gate — **no network at
registration**, so LLM calls must be lazy; ANTHROPIC_API_KEY via Hermes's
credential path, not lemma's `.env` habit.

## Path to winning, custom to lemma

1. **Reliability first:** the demo fixture is a paper whose verdict we have
   already reproduced end-to-end (grokking fixture qualifies). The video
   cannot flake — judges see it once.
2. **A "quick audit" mode** is the one genuine lemma change this entry needs:
   the pipeline was built for reviewer-runs, not one-shot plugin users.
   Cap first pass at 2 claims, surface progress per stage.
3. **Polish:** evidence trail surfaced so it's enumerable from the plugin
   (`lemma_status` lists per-claim verdicts + file paths). Desktop evidence
   pane only if elcaro is already on schedule — it is the first thing to cut.
4. **Creativity/honesty line:** "the agent distrusts itself and ships the
   receipts" — the `improve` re-audit loop (reviewer feedback → re-audit →
   before/after recorded) is real existing capability and maps to "does more
   than a prompt could" better than any wrapper we could write.

## Sequencing and dependencies

- **elcaro is the primary Hermes entry** (security niche, bigger user pool,
  clearer "more than a prompt" story — see `elcaro/docs/hermes-plugin.md`).
- lemma port is ~1.5 days. Decision gate: **Sun Nov 9** — if elcaro's video
  is recorded, lemma ships as a second entry the same day; if not, lemma is
  cut without ceremony.
- Do not let this collide with `PROTEIN.md` (Challenge 05, Nov 8). The
  protein smoke test has priority in any shared-time conflict; plugin work
  resumes after Sunday.

## Links

- Plugin developer guide: https://hermes-agent.nousresearch.com/docs/developer-guide/plugins
- Catalog submission: https://hermes-agent.nousresearch.com/docs/developer-guide/plugins/catalog-submission
- Marketplace: https://hermes-agent.nousresearch.com/marketplace
- Announcement: https://x.com/NousResearch/status/2118094256438936062
- Existing ChatGPT/MCP adapter work (reusable framing): `docs/CONNECT.md`, `agent/mcp_adapter.py`
