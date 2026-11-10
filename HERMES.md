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

## Beyond parity — 2026-11-05 evening

Parity reached (tools + hook + pane + skill). To *win*, sell the
epistemic process, not the verdict — only lemma can demo "the agent
discovering its own metric is broken":

1. **Live Audit Room** — pane shows a run *in progress* (stage,
   attempts, trace tail) via broadcast_plugin_event or a 3 s ctx.rest
   poll (rss-reader precedent). Video: link → hint → /audit → claims
   appear → control fails → `inconclusive`. Honesty as theater.
2. **Audit what the agent says** — `post_llm_call` scans
   assistant_response for paper citations → offer audit. "Audits what
   agents read — including its own answers." ~20 lines.
3. **Self-distrust as UI** — `verify_gain` flag (flip TO supported
   post-feedback = reward-hacking direction) and screen-integrity
   badge (control failed → can't convict → hold) surface in the pane.
4. **Stretch — audit certificates**: workdir manifest hash →
   machine-checkable verdict card via dispatch_tool for other plugins.
   Lemma as ecosystem trust oracle.

Build order: live pane → post_llm_call → badges → certificates.

## Sequencing and dependencies

## Status — built and verified (Nov 5)

- **Plugin complete**: 5 tools, `/audit` command, ambient `pre_llm_call`
  hook (user turns + agent's own citations), Desktop Audit Room pane with
  live `/runs` polling + `verify_gain`/`reference_disagreement` badges,
  disclosure blurb in `plugin.yaml`. `hermes plugins doctor --ci` clean;
  install-tested in a real `HERMES_HOME`.
- **ctx.rest verified live**: `dashboard/plugin_api.py` mounts at
  `/api/plugins/lemma/*` on the dashboard server (:9119) — proven via the
  OpenAPI schema after fixing the `from tools import` → hermes-agent
  `tools` package name collision (`bacb0ef`; sibling imports in plugin API
  files must load by path).
- **MERGED — live in the catalog**: `NousResearch/hermes-agent#149514`
  merged 2026-11-10 by teknium1; entry at `plugin-catalog/lemma.yaml`,
  pinned `98b061d`, `tier: community`, `version: 0.3.0`.
- **Maintainer review — two rounds, both resolved** (teknium1):
  - Round 1 (`bacb0ef`, changes requested): blocker = unsandboxed
    execution of LLM-written audit scripts reachable from model-callable
    tools. Full fix list: sandbox+minimal env for generated code (or
    per-run approval), minimal env for CLI children, ship pipeline
    inside the pin (agent/ outside hermes-plugin/), restrict local
    sources to .pdf-in-datadir, reject path-traversal paper values,
    Weave opt-in only, drop load_dotenv(override=True) for
    secret_scope.get_secret, remove sys.path.insert in plugin_api.py,
    README covering network endpoints + detached runs + generated-code
    exec.
  - Round 2 (`3f1e637c`, changes requested): `subdir:` installs only
    ship `hermes-plugin/` — `agent/` unreachable → vendored
    `lemma_pipeline/` + `trackio` python dep. Approved at `98b061d`.
  - Maintainer merge chore `0f036b77`: rewrote the disclosure blurb +
    added the `version` floor. They *removed* the Firecrawl mention —
    correctly: `_child_env` never forwards `FIRECRAWL_API_KEY` and
    `LEMMA_NO_DOTENV=1` blocks `.env`, so api.firecrawl.dev is
    unreachable under plugin install. **If that key is ever routed via
    `secret_scope`, the catalog disclosure must be updated to match —
    an undisclosed network call is delisting material.**
- **Video shipped**: 64 s HyperFrames film (product-first: plugin card →
  problem → terminal beat → rupture → stats → flag → showcase → card) in
  `~/Downloads/lemma-culture-dish.mp4`; source in `/tmp/lemma-film` (kept
  out of the repo by decision).
- **Site**: `index.astro` hermes-band live (`1c0c5bf`); builds clean.
- **Submitted**: typeform entry (blurb: "audits what agents read" —
  news, benchmarks, GitHub issues, papers) + demo video posted on X
  tagging @NousResearch — both sent while the PR was open, ahead of the
  merge. Remaining stretch: audit certificates.

- elcaro remains the primary entry; lemma ships alongside. `PROTEIN.md`
  round-2 finished **hold** (screen invalid per positive control) —
  Proteinbase bundle regenerated, method note declares the limit.

## Links

- Plugin developer guide: https://hermes-agent.nousresearch.com/docs/developer-guide/plugins
- Catalog submission: https://hermes-agent.nousresearch.com/docs/developer-guide/plugins/catalog-submission
- Marketplace: https://hermes-agent.nousresearch.com/marketplace
- Announcement: https://x.com/NousResearch/status/2118094256438936062
- Existing ChatGPT/MCP adapter work (reusable framing): `docs/CONNECT.md`, `agent/mcp_adapter.py`
