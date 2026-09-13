
## 2026-08-17 — GRAM (arXiv 2607.08077) C1 GPU reproduction on Modal
- Attempted GPU rental marketplace first — user reports it didn't work as expected; switched to Modal (workspace `ungethe`, credentials in .env verified via `modal app list`).
- No public GRAM code found; reimplemented from paper text in `scripts/gram/modal_train.py`:
  - 26M decoder-only transformer: 8 layers, 8 heads, d_model=512, seq 256, vocab 4096.
  - Baseline MLP hidden 2048; GRAM = core MLP 1856 + 4 aux MLPs of width 192 (aux ≈1.58M params ≈ 6%).
  - Routing: p_af=1.0, p_as=0.3, p_cr=0.5, heterogeneous accumulation (requires_grad toggling per group).
  - 1 epoch, batch 128, AdamW β1=0.9 β2=0.95 wd=0.1, LR 5e-3 WSD (10/80/10), grad clip 1.0, bf16.
  - Eval: per-category loss; compute ratio via power-law fit on baseline learning curve (Appendix M).
  - Elicit: 75 steps fine-tune on 128 seqs from each forget category (seed fixed), re-measure core-only loss.
- Smoke test (2k samples, 20 steps, CPU): 3 iterations to pass.
  - Run 1 FAIL: `datasets` Column object passed to tokenizer → ValueError. Fixed with `[str(s) for s in ...]` in `build_eval_tensors` and `load_eval_tensors`.
  - Run 2 FAIL: same Column issue in `make_batches`. Fixed there too.
  - Run 3 FAIL: double-shift in `shifted_loss` (logits 255 vs labels 256). Fixed by removing redundant `[:, :-1, :]` slice since model input is already `input_ids[:, :-1]`.
  - Run 4 PASS.
- Scaled run (150k samples, ~1171 steps, A10G): PASSED, ~6 min/model. Aggregate CR: core 0.868, retain 0.886, forget 0.894, elicit 0.852. Retain/forget gap is visible (0.886 vs 0.894 is weak at this scale — undertrained; full run needed). Paper reference: core 0.938, retain 0.952, forget 0.766, elicit 0.855.
- Saved: papers/arxiv-2607.08077/results/c1/scaled_results.json
- Full run launched: `modal run scripts/gram/modal_train.py --mode full`, log `runs/gram_full_run.log`, app https://modal.com/apps/ungethe/main/ap-EqyB7x0g4gDA7fjsL273t8
- Full run pace (A10G): ~0.23 s/step → baseline ~63 min for 16,528 steps, both models + eval + elicit ETA ~2.5 h. Loss converging normally (8.38 → 2.2 by step 1k).
- Scaled GPU wall: 350 s baseline / 357 s GRAM for 1,171 steps (confirms routing overhead ≈ baseline FLOPs, consistent with C6).
- Docs staged for commit: per-paper `.gitignore` (trace.jsonl + PDF ignored), `papers/arxiv-2607.08077/README.md` (claims table + repro log), `scripts/gram/README.md`, GRAM row in root README submission table.

## 2026-08-17 (later) — C1 full-run results landed: SUPPORTED
- Full 2.1M-sample run completed on A10G: baseline 3839.4 s, GRAM 3852.5 s (wall-time ratio 1.003 — empirical corroboration of C6 cost independence).
- Aggregate compute ratios (single seed): core 0.975, retain 0.976, forget 0.789, elicit 0.836.
- Success criterion check: core ≥ 0.9 ✓ (0.975); retain within ~0.02 of data filtering ✓ (0.976 vs paper filtering 0.962, diff 0.014); forget well below retain ✓ (0.789 vs 0.976, gap 0.187 vs paper gap 0.186). Verdict: supported.
- Closest per-category match to paper: "a deadline or time limit" forget core-only 0.606 (paper cites ~0.61); aliens retained 1.002 (paper 0.99).
- Caveats recorded in `audit_summary.json`: single seed vs paper's 3; no independently trained data-filtering baseline (comparison uses paper-reported filtering values); text-only reimplementation.
- Artifacts: `results/c1/full_results.json`, `audit_summary.json`, `run_attempt1.json`, `audit_attempt1.py` (copy of the runner), `c1_comparison.png`. Merged into `results/audit_report.json`.

## 2026-08-17 (late) — GRAM judge run: PASS 5/5
- `./lemma judge papers/arxiv-2607.08077` → PASS 5/5 (structure, evidence, integrity, cost, trace).
- Trace now 29 events incl. the manual C1 reproduction run-id (smoke iterations preserved honestly: 3 failed smokes + 1 pass, scaled, full).
- Committed: 554e114 (C1 full evidence) + 57018e4 (judge_report.json).

## 2026-08-17 (night) — multi-paper architecture refactor
- Goal: site + pipeline scale to N papers without hardcoded edits; user-facing structure for going live.
- Added `papers/<id>/meta.json` (curated slug/title/links/blurb/xylo labels) for all 4 papers.
- New: `scripts/build_paper_index.py` → `papers/_index.json` (claims + report + judge + trace stats + figures).
- New: `scripts/build_site_data.py` → `site/src/data/papers.json` + copies figures to `site/public/papers/<slug>/figures/` (2 MB cap, dedupe by basename).
- Refactored `scripts/build_trace_data.py`: index-driven, slug-keyed traces, legacy ca/icl aliases kept; note renderer now reads the `message` key too.
- Site: landing counters/artifacts/xylo (24 bars, auto-width) /trace tabs all from papers.json; new `/papers/` registry + `/papers/<slug>/` detail pages (claims table with anchors, figure gallery, single-tab trace player).
- Moved `scripts/gram/` → `papers/arxiv-2607.08077/reproduce/` (paper-specific code convention); `.env` fallback path fixed to 3 levels up.
- AGENTS.md + scripts/README.md document the registry → site pipeline; anti-pattern added: never hand-edit site data.
- Build verified: 6 pages, 24 xylo bars, all detail pages render claims/figures/trace correctly.
- Ran evidence stage for GRAM and published logbook: https://huggingface.co/spaces/Papajams/repro-gram-modular-pretraining (public trace/artifacts). Updated meta.json logbook link, regenerated _index.json/papers.json/traces, rebuilt site.
- Fixed post-refactor site regression: Xylophone bars were missing data-freq (used b.f instead of b.freq), so clicking bars made no sound; also added data-state for accurate readouts and scaled helix spacing for 24 bars.

## 2026-09-08 — UX/product pass: industry context + interactivity + utility

Feedback received: the site should contextualize lemma within the wider AI-science
industry, and improve interactivity, engagement and utility. Three-part response:

**1. Industry context (landing):** new `Landscape` section — "everyone is
 generating, nobody is checking." Led by the reproducibility-crisis numbers
 (Nature 2016 survey: 70% failed to reproduce another lab, >50% their own),
 then an interactive comparison of the AI-science landscape: generators
 (Sakana / Google co-scientist), literature copilots (FutureHouse / Elicit),
 benchmarks & harnesses (PaperBench / ICML Agent Reproductions), and lemma as
 the audit layer ("you are here"). Each category shows "what it produces" and
 "who checks the checker". Sources linked directly.

**2. Interactivity (landing + paper pages):**
 - New `PipelineExplorer`: extract → audit → evidence → judge as clickable
   stages, each rendering a REAL artifact from the registry (grokking-ca
   claims.json lines, audit metrics incl. nu_fitted + control_pass, logbook
   cells + failures preserved, judge rubric rows). Data-driven at build time.
 - Paper detail: claims are now expandable dossiers (disclosure buttons) —
   auditor notes, metrics grid, attempt count, positive-control badge; fallback
   text when a claim has no summary in the registry snapshot. New "why the
   judge trusts this trail" card renders the 5-dimension rubric with
   pass/fail dots + details.
 - Registry: "every claim, one table" — a cross-paper claim explorer with
   verdict filter chips + title search; each row links to the claim dossier.

**3. Utility / honesty disclosure:**
 - Cost-of-honesty receipts on landing (wall minutes, model calls, tool runs,
   audit attempts) + per-paper stats row (model calls / tool runs / wall min /
   failures kept).

**Data pipeline changes (build_paper_index.py / build_site_data.py):**
 - Claims now carry `notes`, `metrics`, `control_pass` (from results/audit_report.json
   summaries) into _index.json → papers.json; judge carries full `rubric`.
 - Totals gain `llm_calls`, `tool_runs`, `wall_min`, `attempts`.
 - **Fresh-clone preservation:** several artifacts are intentionally local-only
   (5nNNVY8NW4-grokking gitignores results/; some trace.jsonl not committed),
   so a bare rebuild used to silently downgrade the registry (grokking-ridge
   went 6S→0S). Both builders now fall back to the prior committed entry when
   a local artifact is absent (claims, trace stats, figures, failure counts,
   judge). Verified: rebuilt totals match the committed baseline exactly
   (24 claims, 16 supported, 4 inconclusive, 4 not_audited, 54 failures).

Verified: `npm run build` 6 pages green; `npm test` 10/10; builders idempotent.

## 2026-09-13 — Audit Room live reruns on Modal (production backend)

**Goal:** `lemma room` only serves the rerun API on localhost and the macOS
sandbox fails closed (RLIMIT_AS rejected), so lemmabio.netlify.app was
recorded-only. Added a Modal-hosted runner behind a Netlify proxy.

- `agent/room_jobs.py` (new, pure): manifest verification delegates to
  `room_server.load_allowed` (identical digest checks), `build_attempt`
  emits the roomModel.Attempt shape (uuid id, number+1, source/execution_mode
  "live_rerun", replay_of, null criterion, tail-bounded stdout/stderr
  artifacts, data-URI figures), `figure_data_uris`, `synthesize_events`,
  `rate_ok` sliding-window helper.
- `agent/room_modal.py` (new, deploy entry): app `lemma-room`; worker
  `run_attempt` (single_use_containers, block_network, restrict_modal_access,
  240s timeout) replays the allowlisted script under the same rlimit
  launcher as room_sandbox minus sandbox-exec — RLIMIT_AS works on Linux.
  Always returns {events, attempt, error}. ASGI `api` function serves
  GET /api/session, POST /api/reruns (spawn -> FunctionCall object_id),
  GET /api/reruns/{job_id} (get(timeout=0); modal TimeoutError -> running,
  NotFound -> 404, OutputExpired -> failed). Token + per-IP/day counters +
  inflight ledger in modal.Dict "lemma-room-state".
- Image: debian_slim py3.12 + numpy/matplotlib/python-dotenv/fastapi —
  dotenv is the only real dep of the agent.auditor import chain
  (anthropic/openai are lazy imports in llm.py, never executed).
- `netlify.toml`: `/api/*` -> `https://thepapajams--lemma-room-api.modal.run/api/:splat`
  (status 200, force) so the API stays same-origin, no CORS.
- `site/src/room.ts`: removed the isLocal short-circuit so the client
  actually calls /api/session in prod (the proxy only helps if the client
  asks); copy tweaks for non-local live mode.
- Rate limits: 10 jobs/day per IP (X-Forwarded-For first hop), 4 concurrent
  globally (inflight ledger pruned at the 240s worker timeout).
- Verified live: session available:true; real C6 legacy-4 rerun completed in
  3.7s -> honest inconclusive + control:failed + fig.png data URI
  (/tmp/modal-rerun-result.json); 403/400/413/415/429/404 guards all hold;
  5-way burst hit the 4-concurrent cap. scripts/test_room_jobs.py 13 tests,
  all existing suites + npm build green.
- Note: prod /api only goes live when the new netlify.toml reaches main —
  no commit made.

### Follow-up — shipped and verified in prod (same day)

- Committed `d3f9fdb`, pushed to main; Netlify rebuild picked up the
  `/api/*` proxy automatically.
- Prod verified end-to-end through the proxy (no dev server, curl only):
  - `GET lemmabio.netlify.app/api/session` → `available:true`, both
    manifest entries, token issued.
  - `POST /api/reruns` (icl-bayesian/C6/legacy-4) → `fc-…` queued → poll
    → `completed`: `source=live_rerun`, `replay_of=legacy-4`, verdict
    `inconclusive`, control `failed`, 1 figure data URI, exit 0, ~3s.
    Same honest outcome as the recorded attempt — the replay agrees.
  - Deployed room JS carries the `RECORDED + LIVE RERUN` pill.
- **Modal workspace: `thepapajams`** (confirmed by owner as the intended
  prod workspace). App name `lemma-room`; endpoint
  `https://thepapajams--lemma-room-api.modal.run`; shared state in
  `modal.Dict` `lemma-room-state` (session token, ip_counters, inflight
  ledger). Redeploy: `modal deploy agent/room_modal.py` from repo root.
- Rate-limit counters hold test jobs we spawned during verification
  (sliding 24h window, self-heals).
