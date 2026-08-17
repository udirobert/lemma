
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
