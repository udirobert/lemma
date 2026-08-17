"""
GRAM Audit — GPU Training Reproduction (Modal)
================================================
Reproduces the Simple Stories experiment from arXiv 2607.08077:
  - Baseline: 26M dense Transformer (8 layers, 8 heads, d=512, MLP=2048)
  - GRAM: core MLP d=1856 + 4 aux modules d=192, p_as=0.3, p_cr=0.5
  - 1 epoch, batch 128, seq 256, AdamW(β1=0.9, β2=0.95, wd=0.1)
  - LR 5e-3 WSD schedule (10% warmup, 80% stable, 10% decay)
  - bf16, grad clip 1.0

Outputs learning curves + final eval losses + compute ratios.
"""

import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# CONFIG
# ============================================================
CONFIG = {
    "vocab_size": 4096,
    "d_model": 512,
    "n_heads": 8,
    "n_layers": 8,
    "seq_len": 256,
    "batch_size": 128,
    "lr": 5e-3,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "grad_clip": 1.0,
    "warmup_frac": 0.10,
    "stable_frac": 0.80,
    "decay_frac": 0.10,
    # GRAM specifics
    "d_core": 1856,
    "d_aux": 192,
    "n_aux": 4,
    "p_as": 0.3,
    "p_cr": 0.5,
    # Aux categories (first 4 alphabetically from 48 topics)
    "aux_categories": [
        "A Deadline or Time Limit",
        "Alien Encounters",
        "Bygone Eras",
        "Cultural Traditions",
    ],
    # Eval
    "eval_every_frac": 0.01,  # ~100 eval points per epoch
    # Smoke test overrides
    "max_steps": None,
    "max_samples": None,
}

AUX_CATS = CONFIG["aux_categories"]


# ============================================================
# MODEL
# ============================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, seq_len):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=True)
        self.proj = nn.Linear(d_model, d_model, bias=True)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(seq_len, seq_len)).view(1, 1, seq_len, seq_len),
        )

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, nh, T, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class MLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, seq_len, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GRAMBlock(nn.Module):
    """Transformer block with GRAM MLP routing: core + N aux modules."""

    def __init__(self, d_model, n_heads, seq_len, d_core, d_aux, n_aux):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.core_mlp = MLP(d_model, d_core)
        self.aux_mlps = nn.ModuleList([MLP(d_model, d_aux) for _ in range(n_aux)])
        self.n_aux = n_aux

    def forward(self, x, active_aux=-1, freeze_core_mlp=False, freeze_aux_idx=-1):
        """
        active_aux: index of aux module to activate (-1 = none)
        freeze_core_mlp: if True, core MLP forward uses detached params (no grad)
        freeze_aux_idx: if >= 0, that aux module forward uses detached params
        """
        x = x + self.attn(self.ln1(x))
        h = self.ln2(x)

        if freeze_core_mlp:
            core_out = _frozen_forward(self.core_mlp, h)
        else:
            core_out = self.core_mlp(h)

        out = core_out
        if active_aux >= 0:
            if active_aux == freeze_aux_idx:
                out = out + _frozen_forward(self.aux_mlps[active_aux], h)
            else:
                out = out + self.aux_mlps[active_aux](h)

        return x + out


def _frozen_forward(module, x):
    """Forward pass with detached params — output is numerically identical
    but no gradient accumulates on module parameters (Appendix H freeze())."""
    params = {n: p.detach() for n, p in module.named_parameters()}
    buffers = dict(module.named_buffers())
    return torch.func.functional_call(module, {**params, **buffers}, (x,))


class GPTBaseline(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["d_model"])
        self.pos_emb = nn.Embedding(cfg["seq_len"], cfg["d_model"])
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(cfg["d_model"], cfg["n_heads"], cfg["seq_len"], 2048)
                for _ in range(cfg["n_layers"])
            ]
        )
        self.ln_f = nn.LayerNorm(cfg["d_model"])
        self.head = nn.Linear(cfg["d_model"], cfg["vocab_size"], bias=False)
        # Weight tying
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        _B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.head(x)

    def get_loss(self, idx):
        logits = self.forward(idx)
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), idx.view(-1), reduction="mean"
        )


class GPTGram(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["d_model"])
        self.pos_emb = nn.Embedding(cfg["seq_len"], cfg["d_model"])
        self.blocks = nn.ModuleList(
            [
                GRAMBlock(
                    cfg["d_model"],
                    cfg["n_heads"],
                    cfg["seq_len"],
                    cfg["d_core"],
                    cfg["d_aux"],
                    cfg["n_aux"],
                )
                for _ in range(cfg["n_layers"])
            ]
        )
        self.ln_f = nn.LayerNorm(cfg["d_model"])
        self.head = nn.Linear(cfg["d_model"], cfg["vocab_size"], bias=False)
        self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, active_aux=-1, freeze_core=False, freeze_aux_idx=-1):
        _B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x, active_aux, freeze_core, freeze_aux_idx)
        x = self.ln_f(x)
        return self.head(x)

    def get_loss(self, idx, active_aux=-1, freeze_core=False, freeze_aux_idx=-1):
        logits = self.forward(idx, active_aux, freeze_core, freeze_aux_idx)
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)), idx.view(-1), reduction="mean"
        )


# ============================================================
# DATA
# ============================================================
def load_tokenizer(hf_token):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        "SimpleStories/SimpleStories-1.25M", token=hf_token
    )
    assert tok.vocab_size <= CONFIG["vocab_size"], (
        f"vocab {tok.vocab_size} > {CONFIG['vocab_size']}"
    )
    return tok


def load_dataset_splits(hf_token, max_samples=None):
    """Load Simple Stories, tokenize, split into core + 4 aux categories."""
    from datasets import load_dataset

    ds = load_dataset(
        "SimpleStories/SimpleStories",
        split="test" if max_samples else "train",
        token=hf_token,
        streaming=False,
    )
    if max_samples:
        ds = ds.shuffle(seed=42).select(range(min(max_samples, len(ds))))

    tok = load_tokenizer(hf_token)
    seq_len = CONFIG["seq_len"]

    buckets = {"core": [], **{c: [] for c in AUX_CATS}}
    for row in ds:
        topic = (row.get("topic") or "").strip()
        story = row.get("story") or row.get("text") or ""
        if not story:
            continue
        key = topic if topic in buckets else ("core" if topic else None)
        if key is None:
            continue
        ids = tok.encode(
            story, add_special_tokens=False, truncation=True, max_length=seq_len
        )
        if len(ids) < 8:
            continue
        # pad to seq_len
        ids = ids + [tok.eos_token_id or 0] * (seq_len - len(ids))
        buckets[key].append(ids[:seq_len])

    counts = {k: len(v) for k, v in buckets.items()}
    print(f"[data] sequences per category: {counts}", flush=True)
    return buckets, tok


class CategorySampler:
    """Yields (batch_tensor, category_index) batches with natural-frequency mixing.
    category index: 0 = core, 1..4 = aux categories."""

    def __init__(self, buckets, batch_size, seed, max_steps=None):
        rng = random.Random(seed)
        self.batch_size = batch_size
        self.max_steps = max_steps
        # build per-category shuffled index pools
        self.pools = {}
        self.cats = ["core", *AUX_CATS]
        for c in self.cats:
            seqs = buckets.get(c, [])
            if not seqs:
                continue
            order = list(range(len(seqs)))
            rng.shuffle(order)
            self.pools[c] = {"seqs": seqs, "order": order, "pos": 0}
        # sample categories by natural frequency (p_af = 1.0)
        total = sum(len(p["seqs"]) for p in self.pools.values())
        self.weights = {c: len(p["seqs"]) / total for c, p in self.pools.items()}

    def _next_from(self, cat):
        p = self.pools[cat]
        if p["pos"] + self.batch_size > len(p["order"]):
            p["pos"] = 0  # wrap (small categories like aux need this)
        idxs = p["order"][p["pos"] : p["pos"] + self.batch_size]
        p["pos"] += self.batch_size
        return torch.tensor([p["seqs"][i] for i in idxs], dtype=torch.long)

    def __iter__(self):
        rng = random.Random(self.max_steps if self.max_steps else 0)
        step = 0
        while True:
            if self.max_steps and step >= self.max_steps:
                return
            cat = rng.choices(
                list(self.weights.keys()), weights=list(self.weights.values())
            )[0]
            batch = self._next_from(cat)
            yield batch, self.cats.index(cat)
            step += 1


# ============================================================
# SCHEDULE
# ============================================================
def wsd_lr(step, total_steps, peak):
    warmup = int(total_steps * CONFIG["warmup_frac"])
    decay_start = int(total_steps * (CONFIG["warmup_frac"] + CONFIG["stable_frac"]))
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    if step < decay_start:
        return peak
    frac = (step - decay_start) / max(1, total_steps - decay_start)
    return peak * max(0.0, 1.0 - frac)


# ============================================================
# TRAIN
# ============================================================
def train(model_kind, seed, device, buckets, eval_sets, out_dir, max_steps=None):
    """Train baseline or gram. Returns dict with learning curve + final losses."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    cfg = CONFIG
    model = GPTBaseline(cfg) if model_kind == "baseline" else GPTGram(cfg)
    model = model.to(
        device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32
    )

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        params,
        lr=cfg["lr"],
        betas=(cfg["beta1"], cfg["beta2"]),
        weight_decay=cfg["weight_decay"],
    )

    # total steps estimate = 1 epoch over all sequences
    total_seqs = sum(len(v) for v in buckets.values())
    total_steps = max_steps or max(1, total_seqs // cfg["batch_size"])
    eval_every = max(1, int(total_steps * cfg["eval_every_frac"]))

    sampler = CategorySampler(buckets, cfg["batch_size"], seed, max_steps=max_steps)
    curve = []  # (step, {cat: loss})
    t0 = time.time()
    model.train()

    for step, (batch, cat_idx) in enumerate(sampler):
        batch = batch.to(device)
        lr = wsd_lr(step, total_steps, cfg["lr"])
        for g in opt.param_groups:
            g["lr"] = lr

        if model_kind == "baseline":
            loss = model.get_loss(batch)
        else:
            # GRAM routing
            if cat_idx == 0:
                # core batch: core always active+updated; random aux w/ prob p_cr
                active = -1
                if random.random() < cfg["p_cr"]:
                    active = random.randrange(cfg["n_aux"])
                loss = model.get_loss(batch, active_aux=active)
            else:
                # aux batch: core + aux[cat_idx-1] forward; aux always updated,
                # core updated with prob p_as (else frozen)
                ai = cat_idx - 1
                freeze_core = random.random() >= cfg["p_as"]
                loss = model.get_loss(batch, active_aux=ai, freeze_core=freeze_core)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
        opt.step()

        if step % eval_every == 0 or step == total_steps - 1:
            losses = evaluate(model, model_kind, eval_sets, device)
            curve.append({"step": step, "losses": losses})
            print(
                f"[{model_kind}] step {step}/{total_steps} loss={loss.item():.4f} eval={ {k: round(v, 3) for k, v in losses.items()} } elapsed={time.time() - t0:.0f}s",
                flush=True,
            )

        if step + 1 >= total_steps:
            break

    final = evaluate(model, model_kind, eval_sets, device)
    ckpt = os.path.join(out_dir, f"{model_kind}_seed{seed}.pt")
    torch.save(model.state_dict(), ckpt)
    return {
        "model": model_kind,
        "seed": seed,
        "steps": total_steps,
        "wall_seconds": round(time.time() - t0, 1),
        "curve": curve,
        "final_losses": final,
        "checkpoint": ckpt,
    }


@torch.no_grad()
def evaluate(model, model_kind, eval_sets, device):
    """Per-category mean CE loss. GRAM evaluated core-only (all aux ablated)
    for 'core', and with matching aux module active for each aux category."""
    model.eval()
    out = {}
    for cat, tensor in eval_sets.items():
        if tensor.numel() == 0:
            continue
        # chunked eval
        losses = []
        idx = 0
        chunk = 64
        while idx < tensor.size(0):
            b = tensor[idx : idx + chunk].to(device)
            if model_kind == "baseline":
                loss = model.get_loss(b)
            else:
                if cat == "core":
                    loss = model.get_loss(b, active_aux=-1)  # core-only profile
                else:
                    ai = AUX_CATS.index(cat)
                    loss = model.get_loss(b, active_aux=ai)  # retain profile
            losses.append(loss.item())
            idx += chunk
        out[cat] = float(np.mean(losses))
    model.train()
    return out


# ============================================================
# COMPUTE RATIO (Appendix M)
# ============================================================
def fit_power_law(curve_points, cat):
    """Fit L(s) = A (s + s0)^-alpha on baseline curve for category `cat`.
    Least squares in log space, nonlinear in s0 -> grid/scan over s0."""
    pts = [
        (p["step"] + 1, p["losses"][cat])
        for p in curve_points
        if cat in p["losses"] and p["losses"][cat] > 0
    ]
    if len(pts) < 5:
        return None
    ss = np.array([p[0] for p in pts], dtype=np.float64)
    ls = np.array([p[1] for p in pts], dtype=np.float64)
    best = None
    for s0 in np.concatenate([np.linspace(1, 500, 60), np.logspace(2.8, 4.5, 40)]):
        X = np.log(ss + s0)
        A_mat = np.vstack([np.ones_like(X), -X]).T
        coef, *_ = np.linalg.lstsq(A_mat, np.log(ls), rcond=None)
        resid = np.log(ls) - A_mat @ coef
        sse = float((resid**2).sum())
        if best is None or sse < best["sse"]:
            best = {
                "s0": float(s0),
                "logA": float(coef[0]),
                "alpha": float(coef[1]),
                "sse": sse,
            }
    best["A"] = math.exp(best["logA"])
    return best


def step_equivalent(fit, loss):
    if fit is None or loss <= 0:
        return None
    return (fit["A"] / loss) ** (1.0 / fit["alpha"]) - fit["s0"]


def compute_ratios(baseline_curve, baseline_final, gram_final, total_steps):
    """CR = step_equiv(variant) / mean(step_equiv(baseline finals))."""
    ratios = {}
    detail = {}
    for cat in ["core", *AUX_CATS]:
        fit = fit_power_law(baseline_curve, cat)
        bl_ref = step_equivalent(fit, baseline_final.get(cat, float("nan")))
        gr = step_equivalent(fit, gram_final.get(cat, float("nan")))
        if fit and bl_ref and gr and bl_ref > 0:
            ratios[cat] = float(gr / bl_ref)
        detail[cat] = {
            "fit_A": fit["A"] if fit else None,
            "fit_alpha": fit["alpha"] if fit else None,
            "fit_s0": fit["s0"] if fit else None,
            "baseline_final_loss": baseline_final.get(cat),
            "gram_final_loss": gram_final.get(cat),
            "baseline_step_equiv": bl_ref,
            "gram_step_equiv": gr,
        }
    return ratios, detail
