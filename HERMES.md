# HERMES.md — Lemma rides inside Hermes: the agent that audits what agents read

**Status:** scaffold built 2026-11-05 — `hermes-plugin/` is doctor-clean with
all 4 tools registering (`hermes plugins doctor hermes-plugin --ci` passes;
handlers verified against real workdirs). Secondary entry behind elcaro (see
`PROTEIN.md` for the other new muscle). Sibling to `HACKATHON.md` /
`SUBMISSION.md` / `PROTEIN.md`. **Last updated:** 2026-11-05.

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

## Competitive bar — scouted 2026-11-05 (marketplace + Tavily)

976 plugins (213 Official / 9 Certified / ~1,465 Community). Top tools by
stars: Oh My Hermes bridge 5.1k (model routing + **Desktop status pane** +
evidence probes), phone-harness 4.3k (real device control), sem-weave 3.4k,
LetsFG 2.2k, solana pay 1.8k. Official featured plugins ship Desktop pages
(Memory Wiki), approval UX loops (Telegram Business), and bundled skills
that install/drive real software (Blender). Every catalog entry carries a
"Disclosure —" blurb; bundled workflow skills are table stakes.

**Read:** a thin wrapper is mid-pack. Every top plugin adds a *surface* —
Desktop panes, gateway channels, ambient hooks — not just tools. Lemma's
moat is the evidence trail (persistent, inspectable, judged); the plugin
must *surface* it, not just wrap it.

## Path to winning, custom to lemma

Ranked by judge-visible value ("does more than a prompt or skill could"):

1. **Desktop page — Audit Room** (the differentiator). `agent/audit_room.py`
   already exports a typed evidence bundle (claims → attempts → figures →
   verdicts) for the site's `/room/`; a `desktop/plugin.js` page renders the
   same bundle over `plugin_data_dir("lemma")`. Demos the moat in 10 s:
   "the conversation ended in a preserved evidence folder you can inspect."
2. **Observer hook — arXiv/PDF link detection.** Ambient behavior: notice
   `arxiv.org/abs/…` in the conversation, offer `lemma_quick_audit`.
3. **`lemma_improve` tool** — wraps `agent/meta.py`: reviewer feedback →
   re-audit → before/after. "The agent distrusts itself and ships the
   receipts" — lemma's actual novelty, already built and hardened.
4. **`/audit <id>` slash command** — `register_command`, ~15 lines.
5. **Reliability:** the demo fixture is a paper whose verdict we have
   already reproduced end-to-end (grokking/1903.08560 fixture). The video
   cannot flake — `lemma_status` on a pre-seeded workdir is the fallback.
6. **Disclosure blurb** in the catalog entry — polish marker every real
   plugin has.
7. **Cut:** `lemma_search`, `hermes lemma` CLI subcommand, extra pane
   polish. Demo sells the pattern ("agent that audits what it reads"),
   not just the tool — paper-auditing is a thin-user niche and the
   judges know it.

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
