# reproduce/ — GRAM (arXiv 2607.08077) C1 reproduction runner

From-scratch reimplementation of the Simple Stories experiment (no public
GRAM code exists). Audit state and results live one level up
(`papers/arxiv-2607.08077/`).

## modal_train.py (current)

Self-contained Modal runner: defines the 26M decoder-only Transformer
(8 layers, 8 heads, d=512, seq 256, vocab 4096), baseline MLP 2048 vs
GRAM core 1856 + 4× aux 192, trains 1 epoch on `SimpleStories/SimpleStories`
with the paper's recipe (batch 128, AdamW β=0.9/0.95 wd 0.1, LR 5e-3 WSD
10/80/10, bf16, grad clip 1.0), evaluates per-category loss on the held-out
test split, and computes Appendix-M compute ratios (power-law fit on the
baseline learning curve + step-equivalent inversion) including the 75-step
elicitation fine-tune.

```bash
# from the repo root, venv activated, Modal token configured
modal run papers/arxiv-2607.08077/reproduce/modal_train.py --mode smoke    # CPU, 2k samples, pipeline check
modal run papers/arxiv-2607.08077/reproduce/modal_train.py --mode scaled   # A10G, 150k samples (~6 min/model)
modal run papers/arxiv-2607.08077/reproduce/modal_train.py --mode full     # A10G, full 2.1M dataset (~1h/model)
```

Results JSON is written to the `lemma-gram-results` Modal volume at
`/root/results/gram_audit_results.json` and echoed as `SUMMARY_JSON=`.
HF token is read from `HF_TOKEN` env or `.env` at launch and passed as a
Modal Secret.

## train.py (historical)

Earlier local CPU prototype of the same pipeline; kept for reference.
Use `modal_train.py` for runs.

## Pitfalls already hit (don't regress)

- `datasets` column access returns a `Column`, not a `list` — the HF
  tokenizer rejects it. Always coerce: `[str(s) for s in ds.select(ids)["story"]]`.
- The model is fed `input_ids[:, :-1]`, so `shifted_loss` must NOT slice
  logits again (double-shift → 255 vs 256 batch mismatch).
