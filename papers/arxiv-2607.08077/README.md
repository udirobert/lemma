# GRAM: Gradient-Routed Accumulation of Modules (arXiv 2607.08077)

**Paper:** *GRAM — capability isolation via gradient-masked auxiliary MLPs*
**arXiv:** https://arxiv.org/abs/2607.08077
**Code link:** none public — full reimplementation from paper text
**Audit context:** re:AGENT Track A build; reproduction prepared as a showcase
for the AIAF pitch.

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | GRAM approximates multiple data-filtered models in a single run (Simple Stories, 26M) | **supported** — full 2.1M-sample GPU reproduction (see `results/c1/`) |
| C2 | GRAM isolates realistic dual-use capabilities and matches filtering at 800M scale | not_audited — compute scope (800M, 3 seeds × 6 runs) |
| C3 | Isolation improves with scale; GRAM tracks filtering 50M→5B | not_audited — compute scope (5B) |
| C4 | GRAM composes arbitrary capability subsets without degradation, unlike FT-LoRA | not_audited — compute scope (800M + LoRA baselines) |
| C5 | GRAM outperforms filtering/FT-LoRA on capability removal under partial labeling | not_audited — compute scope (800M) |
| C6 | Training cost independent of profile count; ~5× savings over filtering | **supported** — analytic FLOPs/param accounting, 1 attempt (see `results/c6/`) |

C2–C5 are recorded as `not_audited` with an explicit compute justification —
never silently skipped (AGENTS.md policy).

## Reproduction log

### 2026-08-17: C6 analytic audit (CPU)
- **Approach:** analytic `6·N·T` FLOPs model with paper hyperparameters
  (80/20 core/aux token mix, p_cr=0.5, 4 aux modules at 0.1× core MLP).
- **Result:** GRAM train-FLOPs ratio 1.04× vs filtering 5.0× → 4.81× savings.
  Active MLP per served profile 1.10× ≤ claimed 1.1×. Positive controls exact.
- **Verdict:** supported, 1 attempt. Files: `results/c6/`.

### 2026-08-17: C1 reimplementation + GPU reproduction (Modal A10G)
- **Hypothesis:** a from-scratch 26M reproduction of Table 1 should show
  retain ≈ 1, forget substantially below retain, core ≥ 0.9, per the
  paper's compute-ratio evaluation (Appendix M power-law inversion).
- **Implementation:** `scripts/gram/modal_train.py` — decoder-only
  Transformer (8 layers, 8 heads, d=512, seq 256, vocab 4096, ~27.4M params).
  Baseline MLP hidden 2048; GRAM = core 1856 + 4× aux 192 (aux ≈1.58M,
  ~6% of total, matching the paper's accounting). Routing: p_af=1.0,
  p_as=0.3, p_cr=0.5, heterogeneous accumulation via per-group
  `requires_grad` toggling. 1 epoch, batch 128, AdamW (β1=0.9, β2=0.95,
  wd=0.1), LR 5e-3 WSD 10/80/10, grad clip 1.0, bf16.
- **Evaluation:** per-category loss on held-out test split; compute ratios
  via power-law fit of the baseline learning curve and step-equivalent
  inversion (Appendix M); elicitation = 75-step fine-tune on 128 sequences
  from each forget category, re-measure core-only loss.
- **Validation ladder:**
  1. Smoke (2k samples, 20 steps, Modal CPU): end-to-end pass. Bugs found
     and fixed: `datasets` returns `Column` not `list` (tokenizer
     ValueError → explicit str coercion); double-shift in `shifted_loss`
     (logits 255 vs labels 256 → removed second shift).
  2. Scaled (150k samples, ~1171 steps, A10G): ~6 min/model, aggregate
     compute ratios core 0.868 / retain 0.886 / forget 0.894 / elicit
     0.852. Directionally sensible but undertrained at this scale
     (retain/forget gap not yet separated). Saved: `results/c1/scaled_results.json`.
  3. **Full (2,115,696 samples, 16,528 steps, A10G): COMPLETE, 2026-08-17.**
     Baseline wall 3839 s, GRAM wall 3853 s (ratio 1.003 — training-cost
     independence corroborates C6 empirically). Modal app
     https://modal.com/apps/ungethe/main/ap-EqyB7x0g4gDA7fjsL273t8.
     **Aggregate compute ratios (1 seed):** core 0.975 / retain 0.976 /
     forget 0.789 / elicit 0.836. All three success-criterion thresholds
     met: core ≥ 0.9 ✓; retain within 0.014 of paper filtering 0.962
     (threshold ~0.02) ✓; forget well below retain (gap 0.187, matching
     paper gap 0.186) ✓. Elicit 0.836 ≈ paper 0.855 with partial-recovery
     pattern reproduced (elicit > forget). Saved: `results/c1/full_results.json`,
     `results/c1/c1_comparison.png`, `results/c1/audit_summary.json`.
- **Paper reference (Table 1):** GRAM core 0.938 / retain 0.952 /
  forget 0.766 / elicit 0.855. Filtering: 0.961 / 0.962 / 0.780 / 0.870.
- **Known deviations (disclosed):** reimplementation from text only — no
  official code; exact tokenizer/data-order details may differ; aux
  categories = first 4 alphabetical topics (paper's convention).

## Files

- `claims.json` — extracted claims (stage 1)
- `results/audit_report.json` — per-claim outcomes (stage 2)
- `results/c6/` — C6 audit script, summary, figure
- `results/c1/` — C1 training results: scaled + full run JSONs, comparison figure, audit summary
- `../../scripts/gram/modal_train.py` — the Modal training/eval runner
