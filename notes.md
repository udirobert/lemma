
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

## 2026-10-03 — Omnigent headline run: arxiv 1912.07242 (sample-wise double descent)

First full agent-orchestrated discovery loop on a fresh paper, for the
Hack-Nation x Databricks Omnigent challenge. Bundle: `omnigent/lab/`
(lemma-lab director + auditor + proposer sub-agents, pi harness).
Sessions: `c62c9b43...` (died on provider 402 mid-run) then
`a475c559...` (continuation; transcript exported to
`runs/omni-session-1912.jsonl`).

**What happened**
- Extract: 6 cpu-fast claims (C1 peak-at-n=d; C2-C6 theorem-level).
- Round 1 (Kimi-K2.7-Code/WANDB): C1 inconclusive (3x 20-min timeouts —
  every generated MC used d=1000, ~10k lstsq solves); C2 FALSIFIED,
  control_pass=true (bias rel_err 24.2% at d=1000/T=500). Auditor
  sub-agent died mid-run (upstream 500); underlying lemma process
  survived and completed — results read from disk, not the sub-agent.
- Human-in-the-loop: wrote reviewer feedback.md for C1+C2 (d<=300
  designs; for C2 argue error-decay with d, not one huge d).
- Provider died mid-run: WANDB 402 insufficient_quota; ORCA key invalid;
  KIMI 503; HF paused. Rebound bundle to RUNINFRA (qwen3-8-27b),
  continuation session picked up from the workdir state.
- Round 2 (feedback applied): C1 SUPPORTED — peak exactly at n=200=d at
  d=200, T=20, control (d=50 peak at n=50) passed, ~1 attempt/~2 min.
  C2 SUPPORTED at d=150 (B~0, V<=1.8%, R<=1.3%, control 0.5%).
  Round 2 total: ~4.5 min vs ~65 min unguided — the feedback effect.
- Proposer (sub-agent flaky; director wrote proposals.json from
  evidence, provenance disclosed): P1 promoted to C7
  (kind: generated-hypothesis) — finite-d risk valley at gamma*=0.90.
  P2 parked pending human approval (independent MC check of E[beta_hat]
  — repairs C2 round-2's circular bias check, which hard-coded
  E[beta_hat]=gamma*beta).
- C7 audit: SUPPORTED — valley at gamma=0.90, strict min, 4.1% rel err;
  control (sigma=0.5 -> analytic gamma*=0.75) recovered at 1.7%.
- evidence+judge: PASS 5/5 (155 trace events, 7 failed attempts
  preserved). Regression fixture also PASS 5/5.

**Wall time**: ~2h20m total incl. provider-failure recovery;
continuation session ~24 min (22:57-23:21).

**Caveats for the demo**: C2's falsified->supported flip is a round-1
MC-noise artifact at d=1000 (T=500 too small for mean-of-beta_hat
bias), disclosed in the director's report; C2's bias metric is circular
(P2 filed for independent verification, gated on human approval); C1's
peak scale is tail-dominated (location verified, magnitude diverges —
matches the paper's own Fig 1).

**Next time**: pin a budget-cap d in the auditor prompt for MC claims
(d=1000 kept regenerating across all 3 attempts); the 25-min move-on
budget needs a round-level cap the director can enforce; Omnigent
0.16.0 client has a 120s subscribe-after-post race — one-shot -p mode
crashes while the session survives; `omni session export` recovers the
transcript.

## 2026-10-04 — corpus sweep: from auditing a paper to auditing a literature

Tried: expanded the discovery loop from the flagship paper to a corpus —
Hastie et al. (1903.08560), Bartlett et al. (1906.11300), d'Ascoli et al.
triple descent (2005.13924). Extract via `lemma audit <id> --stages extract`
in parallel; flagship deepened in parallel (C3–C6 + P2 promoted to C8).

Why: breakthrough-potential was the weak rubric axis; a "verification map"
of the literature's claims is a result only this infrastructure produces.

What happened:
- Hastie: 6 testable claims, all cpu-fast. Bartlett: 6 claims (bounds +
  effective-rank identities). Triple-descent: PDF fetch truncated twice
  (ContentTooShortError) → 0 claims; dropped from corpus.
- Bartlett C1 falsified (control passed) at n=300 — the benign-overfitting
  iff-claim may be a small-n artifact (log-slow decay needs bigger n) —
  flagged for honest caveat, same class as flagship C2 round-1.
- Bartlett C6 falsified with control: the generated script independently
  re-derived the paper's true identity r_k² = r_k(Σ²)·R_k, then found 4/36
  boundary violations of the extracted chain — a nuanced finding.
- Institutional-memory test: corpus claims got feedback.md pre-seeded with
  the d≤300 cost lesson BEFORE their first attempt; flagship c3/c4/c6 got
  it mid-run for re-audit rounds.

Next: Omnigent corpus-triage session → proposer round targeting a real
new result (finite-d correction rate), then verification map + demo.

### C9 outcome (later, same day)

C9 (agent-generated, promoted by the corpus-triage director): 3 attempts
crashed/timed-out on the T=4000 x d<=800 design → feedback.md seeded →
re-audit at T<=16000-adaptive/d<=300 completed in ~8.5 min. Verdict:
inconclusive (control failed at the noise floor) — but the measurement
found the C2 approximation is within ~0.1% at d=80. Errors are at the MC
floor; no decay rate is measurable. Honest label for a real finding.

## 2026-10-22 — Build-with-Beam competition entry (HF × Reflection, $20K, deadline Nov 2)

What we tried: enter the Reflection × HF "Build with Beam" competition
(discussions/5 on reflection/Beam-1.0-501B-A23B) with Lemma as the vehicle —
"Beam as Lemma's auditor brain", clean-conditions re-audits of both recorded
papers plus a ground-truth benchmark. Hypothesis: Beam's science-reasoning
profile should crack claims that stymied 27B-class models.

What happened:
- Wired THREE Beam routes into agent/llm.py's endpoint chain: REFLECTION
  (api.reflection.ai, key pending), BEAMHF (router.huggingface.co via
  HF_TOKEN), OPENROUTER (user-provided key). All smoke-tested.
- Key integration facts (Reflection docs): reasoning is MANDATORY
  (low..max, no "none"), reasoning tokens count against the output budget,
  org default budget 32768. Set LEMMA_*_MAX_TOKENS=32768 floors.
- Hardened llm.py's empty-content error to report finish_reason (the budget
  trap) and bakeoff_codegen.py to honour the endpoint floor — Beam's first
  bake-off attempt burned all 6000 tokens on reasoning (content empty,
  finish_reason=length); with the floor it produced ν=0.9974 in one
  generation (27B models: ν≈6000-6700; Kimi K3: 0.9911).
- CA clean round (no feedback/references): C3+C6 supported first attempt;
  C1 ν=1.476 and C2 ν=0.075 honestly falsified (controls passed —
  implementation misses, not physics); C4/C5 inconclusive. Judge PASS 5/5.
- ICL clean round: C1 supported in ONE attempt (rel_diff 0.0008; the
  recorded 27B rounds took 8 attempts + a reviewer reference), and C2+C5 —
  inconclusive through 7 recorded rounds — flipped to supported. C3/C4
  honestly falsified; C6/C7 inconclusive with controls failed. Judge PASS 5/5.
- Beam Audit Bench packaged (scripts/build_beam_audit_bench.py) and published:
  https://huggingface.co/datasets/Papajams/beam-audit-bench
  13 tasks, 8 keyed, 5 with hand-verified references; attribution tags
  reviewer-reference attempts from trace events so model-only scores exclude
  them. Logbooks published (--public traces):
  https://huggingface.co/spaces/Papajams/beam-audit-ca-grokking
  https://huggingface.co/spaces/Papajams/beam-audit-icl-bayesian
- Bug found + fixed: Beam's honest inconclusive C6 summary contained NaN
  metrics; python's json emits bare NaN tokens which break the site's JS
  build. build_paper_index.py now sanitises non-finite floats to null.

What I'd do differently: none — clean workdirs kept the recorded evidence
untouched and made the comparison honest.

Next: discussions/5 entry comment (drafted, user posts), then optional
REFLECTION-API-key route test + one more Beam round on CA C1/C2 with the
feedback loop (would show Beam+human-reviewer closing the gap like Kimi did).

### Same day, later — differentiators beyond the first entry

Hypothesis: the competition thread was filling with single-axis benchmarks;
three things only we have — the self-improvement loop, ground-truth grading
without an LLM judge, and the unclaimed "RL environment" category.

What happened:
1. **RL environment** (scripts/beam_audit_rl_env.py): gym-style reset/step
   over bench tasks; reward = the deterministic grader (0/0.1/0.25/0.5/
   0.7/1.0 ladder), observation = exit code + failed key checks. Two
   reward-hacking guards added and verified: hardcoded answer-key
   literals capped at 0.25; decisive metrics with no computation (pure
   fabrication) capped at 0.25. Clean script earns 1.0.
2. **Rollouts**: with a control-missing signal added to the observation,
   all four episodes (Beam + RUNINFRA 27B × ca_c1 + icl_c1) reach 1.0 —
   Beam attempt 1 on both, 27B attempt 3 on both (its first passes
   recovered the numbers but omitted the control; the env observation
   taught it). Trajectories in the dataset under rl_env/.
3. **Self-improvement round** on the Beam CA workdir (LEMMA_IMPROVE_
   PROVIDER=BEAMHF LEMMA_AUDIT_PROVIDER=BEAMHF ./lemma improve --claims
   C1,C2,C4,C5): Beam-as-reviewer diagnosed C1's exact bug (unshifted
   training distribution → threshold never crosses at finite time; fit
   window too wide) and Beam-as-auditor fixed it: nu=1.0044, control
   slope 1.0000, falsified→supported. Zero human feedback in that
   workdir — feedback.auto.md is Beam-authored. verify_gain correctly
   flags the flip (that's the reward-hacking direction). C2/C4/C5
   unchanged. Judge still PASS 5/5 (228 events), eval scorers clean.
4. Dataset republished as v2 with rl_env/ + improved README; CA Space
   republished with the improve round; site rebuilt (CA now 3S/1F/2I).

Open item: WANDB endpoint is 402 (quota) — no second frontier competitor
for the RL episodes yet; RUNINFRA 27B is the stand-in comparison.

### Long-context ablation + guard refinement (same day, latest)

Hypothesis: Beam's headline feature nobody in the thread has tested is the
1M context; our pipeline feeds only a 20k-char excerpt. CA C2's recorded
failure mode ("model keeps re-deriving the 1D ball, not the D-ball") lives
PAST the excerpt window. Test: RL env with full_context=True (whole paper,
119,673 chars) vs excerpt (20,000) on ca-grokking_c2.

First run result: full-context gt_pass=True (exact nu_D2/D5/D10 = 1.49/3.00/
5.51 series) where excerpt never passed — BUT reward capped at 0.25 by the
v1 literal-presence guard: full-context scripts legitimately hold the paper's
stated predictions (1.5, 3.0, 5.5) as comparison constants.

Guard refined (v2): source literals now informational (key_literals_in_source);
the cap triggers only when a REPORTED metric exactly equals a source literal
(_hardcoded_reported, alias-aware via _get_metric). A genuine fit lands at
full precision (1.4939295367...) and cannot equal the rounded target.
Regression-verified: fabricated exact-match=0.25, clean fit=1.0,
no-computation=0.25, legit comparison-literal fit=1.0. Trajectories now
record full script source per step (inspectability gap — the first capped
episode could not be audited because the script wasn't saved).

Open: re-run on fixed env in flight; then repackage dataset v3 with ablation
rollouts + updated reward_spec, republish, update entry draft + notes.

Result of the re-run on the fixed env: excerpt arm best 0.0 (att3 reports
nu=2/5/10 — the 1D-ball exponents, the exact recorded failure mode); full-
context arm attempt 3 = 1.5001/3.0174/5.4117 vs targets 1.5/3.0/5.5, reward
1.0, control passed, literals present in source as comparison constants but
NOT exact-equal to reported metrics so the v2 guard correctly does not cap.
Trajectories now save every script source. Dataset republished as v3 (6
rollouts incl. both ablation arms, reward_spec v2); entry draft updated with
the ablation headline. ICL bench rate is 1/2 and stable (C3 honestly
falsified by Beam's own implementations; only C1 is a recovery).

Paused: re-running the four C1 rollouts on the env that records script
sources was killed when the laptop had to travel (episode 1 of 4 done:
ca-grokking_c1 BEAMHF reward 1.0, scripts recorded). Three remain:
ca-grokking_c1 RUNINFRA (interrupted mid-episode), both icl-bayes_c1 runs.
Resume with ./tmp/resume_c1_rollouts.sh, then repackage + republish the
dataset as v4. Nothing else is blocked: commit staging (255 paths) and the
entry draft are ready; v3 is live with the ablation.

### Back from travel — episodes r2 finished, dataset v4

Resume: RUNINFRA ca_c1 r2 had completed as an orphan (crash, missing nu,
nu=0.5 → best 0.0). Recovered v3 episodes from the Hub as r1, indexed new
runs as r2, re-ran both ICL r2 episodes (BEAMHF 1.0@att2, RUNINFRA 1.0@att2).
Final episode table (all published): Beam 4/4 at 1.0; RUNINFRA 3/4 at 1.0.
rl_episode_lines() in the packager now generates the table + aggregates
from the rollout files, so the README cannot drift. Dataset republished v4.

### Kimi K3 RL episodes — blocked (2026-10-22 evening)

Goal: third model in the RL env episodes (bake-off had shown Kimi K3
nu=0.9911 vs 27B nu~6000). Both routes dead:
1. LEMMA_KIMI (user-provided Modal proxy ep-kimi-k3-server): 503 "no
   upstreams available"; the app no longer appears in `modal app list`
   and its server code was never in this repo.
2. LEMMA_KIMIOR (added, moonshotai/kimi-k3 via OpenRouter): 402 — the
   OpenRouter account has never purchased credits (the Beam entry
   apparently rode free/credit balance that is now exhausted or
   org-scoped).
Dataset v5 published WITHOUT Kimi: Beam 4/4, 27B-class 3/4 remains the
episode comparison; v5 adds the "training against this environment"
section and a provider-agnostic episode table. To add Kimi later:
redeploy the Modal server or add OpenRouter credits, then
`scripts/beam_audit_rl_env.py ca-grokking_c1 --provider KIMIOR` (+
icl-bayes_c1), repackage, republish.

### Kimi K3 unblocked via the HF router (zero-cost) — dataset v6

The Modal Kimi proxy was gone and OpenRouter wanted $11.94/M out tokens,
but the HF inference router hosts moonshotai/Kimi-K3 and accepts the same
HF_TOKEN — added LEMMA_KIMIHF (router.huggingface.co/v1, reasoning_effort
low, 16384 cap), smoke-tested, ran both keyed episodes:
- ca-grokking_c1: att1 falsified (honest), att2 nu=1.0278 → reward 1.0
- icl-bayes_c1: att1 falsified, att2 rel_diff=6.5e-16 → reward 1.0
Three-way episode table now: Beam 4/4, Kimi 2/2, 27B-class 3/4. Dataset
republished v6 (14 rl_env files); entry draft updated; rl_episode_lines()
aggregated providers automatically — no README hand-editing needed.

### Claude Opus 5.5 closed-frontier episodes — the headline result

8 episodes via ANTHROPIC_API_KEY (user-provided, key supports opus-5-5,
sonnet-5-5, sonnet-4-5; LEMMA_MODEL=claude-opus-5-5). Fixed agent/llm.py:
anthropic SDK 1.11 removed `temperature` from messages.create() (the first
launch attempt failed client-side, zero tokens billed).

Results — keyed C1 episode success: ANTHROPIC 0/4, BEAMHF 4/4, KIMI 2/2,
RUNINFRA 3/4. Opus overall 1/8 (single 1.0: ca_c2 excerpt r1, genuine
full-precision D-ball fit from the 20k excerpt). Failure analysis from the
preserved scripts: CONTRACT failure not science — Opus computed physics
(median_nu, nu_se_bootstrap, rel_diff=not_computed) but never emitted the
key metric under a graded alias in any C1 attempt, or declined to compute
(claim_quantity_computed=false). The env observation named the failed
check; 3 attempts didn't fix the naming.

Honesty notes folded into README/draft: (1) Opus did ca_c2 from the
excerpt — so the long-context unlock is Beam-specific, not a universal
property of the task; stated as such. (2) ca_c2 full-context Opus runs
scored 0.5 (recovered metrics, no passing control) and 0.0.
Estimated spend ~$2-5 at Opus pricing (usage not logged per call — note
for next time: record resp.usage in the trajectory).

Dataset v7 published (22 rl_env files). Entry draft rewritten around the
four-way ladder + contract-failure analysis.

### Presentation layer (same evening)

Built the "full package": 3 data-driven dark-mode figures
(scripts/make_beam_figures.py, site palette, dark-mode first for HF) —
episode ladder, context ablation, improve arc. Figures ship inside the
dataset (figures/) so links never rot. Caught + fixed a real bug before it
shipped: ladder figure keyed on provider "KIMI" but trajectories record
"KIMIHF" — Kimi's bar was silently missing; regenerated + re-uploaded
(dataset v9).

/beam landing page: scripts/build_beam_page_data.py -> site/src/data/beam.json
(20 episodes, aggregates, bakeoff, improve numbers, links) ->
site/src/pages/beam.astro (ladder, failure gallery, bake-off table, figures,
episode grid with attempt dots). ClipMenu links it. Site builds clean.

Entry comment rewritten hook-first: Opus 4-to-0 opener, figure embeds,
reading paths (60s / 5min / deep dive), failure gallery table, honest
ablation footnote. X thread draft at tmp/beam-x-post-draft.md (post with
the ladder + ablation figures, weekday US morning, right after the
discussion comment).
