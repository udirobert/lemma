"""
GRAM audit - GPU reproduction of the Simple Stories experiment (arXiv 2607.08077).
Runs on Modal. Modes: smoke (CPU, tiny), scaled (A10G, capped samples), full (A10G, all data).

Paper setup (Simple Stories):
  - decoder-only Transformer, 8 layers, 8 heads, d_model 512, seq 256, vocab 4096
  - baseline MLP hidden 2048 (~26M params)
  - GRAM: core MLP 1856 + 4 aux MLPs of 192; p_as=0.3, p_cr=0.5, p_af=1.0
  - 1 epoch, batch 128, AdamW(b1=.9, b2=.95, wd=.1), LR 5e-3 WSD(10/80/10), clip 1.0, bf16
  - aux categories = first 4 alphabetically of 48 topics; remaining 44 = core
"""

import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    "d_core": 1856,
    "d_aux": 192,
    "n_aux": 4,
    "p_as": 0.3,
    "p_cr": 0.5,
    "aux_categories": [
        "A Deadline or Time Limit",
        "Alien Encounters",
        "Bygone Eras",
        "Cultural Traditions",
    ],
    "eval_per_cat": 256,
    "eval_chunk": 128,
    "elicit_steps": 75,
    "pad_id": 0,
}
AUX_CATS = CONFIG["aux_categories"]
PAPER_TABLE1 = {  # Table 1: compute ratios on Simple Stories (mean, 90% CI), N=3 seeds
    "GRAM": {"core": 0.938, "retain": 0.952, "forget": 0.766, "elicit": 0.855},
    "Filtering": {"core": 0.961, "retain": 0.962, "forget": 0.780, "elicit": 0.870},
}


# ------------------------- model -------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, d, h, seq):
        super().__init__()
        self.h = h
        self.hd = d // h
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.register_buffer(
            "mask", torch.tril(torch.ones(seq, seq)).view(1, 1, seq, seq)
        )

    def forward(self, x, attention_mask=None):
        B, T, C = x.size()
        qkv = self.qkv(x).view(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) * (self.hd**-0.5)
        causal = self.mask[:, :, :T, :T] == 0
        if attention_mask is not None:
            pad = (attention_mask == 0).view(B, 1, 1, T)
            att = att.masked_fill(causal | pad, float("-inf"))
        else:
            att = att.masked_fill(causal, float("-inf"))
        att = F.softmax(att, dim=-1)
        return self.proj((att @ v).transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(self, d, ff):
        super().__init__()
        self.fc1 = nn.Linear(d, ff)
        self.fc2 = nn.Linear(ff, d)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class GRAMMLP(nn.Module):
    """Core MLP plus N small auxiliary modules; output = core + sum(active aux)."""

    def __init__(self, d, d_core, d_aux, n_aux):
        super().__init__()
        self.core = MLP(d, d_core)
        self.aux = nn.ModuleList([MLP(d, d_aux) for _ in range(n_aux)])

    def forward(self, x, aux_active):
        out = self.core(x)
        for i, m in enumerate(self.aux):
            if aux_active[i]:
                out = out + m(x)
        return out


class Block(nn.Module):
    def __init__(self, cfg, gram=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg["d_model"])
        self.attn = CausalSelfAttention(cfg["d_model"], cfg["n_heads"], cfg["seq_len"])
        self.ln2 = nn.LayerNorm(cfg["d_model"])
        self.gram = gram
        if gram:
            self.mlp = GRAMMLP(
                cfg["d_model"], cfg["d_core"], cfg["d_aux"], cfg["n_aux"]
            )
        else:
            self.mlp = MLP(cfg["d_model"], 2048)

    def forward(self, x, aux_active=None, attention_mask=None):
        x = x + self.attn(self.ln1(x), attention_mask)
        h = self.ln2(x)
        x = x + self.mlp(h, aux_active) if self.gram else x + self.mlp(h)
        return x


class TransformerLM(nn.Module):
    def __init__(self, cfg, gram=False):
        super().__init__()
        self.cfg = cfg
        self.gram = gram
        self.tok = nn.Embedding(
            cfg["vocab_size"], cfg["d_model"], padding_idx=cfg["pad_id"]
        )
        self.pos = nn.Embedding(cfg["seq_len"], cfg["d_model"])
        self.blocks = nn.ModuleList([Block(cfg, gram) for _ in range(cfg["n_layers"])])
        self.ln_f = nn.LayerNorm(cfg["d_model"])
        self.head = nn.Linear(cfg["d_model"], cfg["vocab_size"], bias=False)
        self.head.weight = self.tok.weight  # weight tying
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.padding_idx is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].fill_(0)

    def forward(self, idx, aux_active=None, attention_mask=None):
        _B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok(idx) + self.pos(pos)
        for b in self.blocks:
            x = b(x, aux_active, attention_mask)
        return self.head(self.ln_f(x))


def build_model(cfg, gram):
    m = TransformerLM(cfg, gram=gram)
    n = sum(p.numel() for p in m.parameters())
    return m, n


# ------------------------- data -------------------------
def load_data(hf_token, max_samples=None):
    """Load SimpleStories, build per-category index lists, return dataset + tokenizer."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        "SimpleStories/SimpleStories-1.25M", token=hf_token
    )
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    CONFIG["pad_id"] = tok.pad_token_id

    split = "test" if max_samples and max_samples < 50000 else "train"
    ds = load_dataset("SimpleStories/SimpleStories", split=split, token=hf_token)
    if max_samples and max_samples < len(ds):
        ds = ds.shuffle(seed=42).select(range(max_samples))

    # Discover topics and map aux categories
    topics = sorted(set(ds["topic"]))
    aux_found = [t for t in AUX_CATS if t in topics]
    if len(aux_found) < 4:
        print(
            f"[warn] Expected aux topics not all found. Using first 4 alphabetical: {topics[:4]}",
            flush=True,
        )
        aux_found = topics[:4]
    aux_map = {t: i for i, t in enumerate(aux_found)}

    # Build index lists: -1 = core, 0..3 = aux (skip rows with empty/None story)
    idx_by_cat = {i: [] for i in range(-1, 4)}
    topic_col = ds["topic"]
    story_col = ds["story"]
    skipped = 0
    for i, (t, s) in enumerate(zip(topic_col, story_col, strict=False)):
        if not s or not str(s).strip():
            skipped += 1
            continue
        if t in aux_map:
            idx_by_cat[aux_map[t]].append(i)
        else:
            idx_by_cat[-1].append(i)

    counts = {"core": len(idx_by_cat[-1])}
    for i, name in enumerate(aux_found):
        counts[name] = len(idx_by_cat[i])
    print(
        f"[data] split={split} rows={len(ds)} topics={len(topics)} skipped_empty={skipped}",
        flush=True,
    )
    print(f"[data] category counts: {counts}", flush=True)
    return ds, tok, idx_by_cat, aux_found


def make_batches(ds, tok, idx_by_cat, cfg, seed, max_steps=None, aux_only=False):
    """Yield (input_ids, attention_mask, cat_label) batches.
    cat_label: 0=core, 1..4=aux index+1. Natural frequency sampling."""
    rng = random.Random(seed)
    # Shuffle each category's indices
    for k in idx_by_cat:
        rng.shuffle(idx_by_cat[k])
    pos = dict.fromkeys(idx_by_cat, 0)
    B = cfg["batch_size"]
    L = cfg["seq_len"] + 1  # +1 for shifted target
    keys = [k for k in idx_by_cat if (not aux_only or k >= 0)]
    total = sum(len(idx_by_cat[k]) for k in keys)
    steps = max_steps if max_steps else max(1, total // B)

    for _ in range(steps):
        # Pick category weighted by remaining samples
        alive = [
            (k, len(idx_by_cat[k]) - pos[k])
            for k in keys
            if pos[k] < len(idx_by_cat[k])
        ]
        if not alive:
            # reset pools if exhausted
            for k in keys:
                pos[k] = 0
            alive = [(k, len(idx_by_cat[k])) for k in keys if len(idx_by_cat[k]) > 0]
        if not alive:
            break
        cats, weights = zip(*alive, strict=False)
        k = rng.choices(cats, weights=weights)[0]

        ids = idx_by_cat[k]
        start = pos[k]
        sel = ids[start : start + B]
        pos[k] += len(sel)

        # Wrap if we don't have enough (small aux categories)
        while len(sel) < B:
            rng.shuffle(ids)
            need = B - len(sel)
            sel.extend(ids[:need])
            pos[k] = need

        sel = list(sel)
        texts = [str(s) for s in ds.select([int(i) for i in sel])["story"]]
        enc = tok(
            texts,
            truncation=True,
            max_length=L,
            padding="max_length",
            return_tensors="pt",
        )
        yield (
            enc["input_ids"],
            enc["attention_mask"],
            (k + 1),
        )  # k=-1 -> 0 core, aux 0..3 -> 1..4


# ------------------------- training -------------------------
def wsd_lr(step, total, peak, warmup_frac, stable_frac):
    warmup_end = int(total * warmup_frac)
    decay_start = int(total * (warmup_frac + stable_frac))
    if step < warmup_end:
        return peak * (step + 1) / max(1, warmup_end)
    if step < decay_start:
        return peak
    frac = (step - decay_start) / max(1, total - decay_start)
    return peak * max(0.0, 1.0 - frac)


def gram_param_groups(model):
    """Split params into core vs per-aux groups for gradient routing."""
    core, aux_groups = [], [[] for _ in range(CONFIG["n_aux"])]
    for name, p in model.named_parameters():
        if ".mlp.aux." in name:
            idx = int(name.split(".mlp.aux.")[1].split(".")[0])
            aux_groups[idx].append(p)
        else:
            core.append(p)
    return core, aux_groups


def set_grad(params, flag):
    for p in params:
        p.requires_grad_(flag)


@torch.no_grad()
def eval_loss(model, gram, tensor, device, aux_active=-1):
    """Mean CE loss over a tensor of token sequences for a given aux config."""
    if tensor is None or tensor.numel() == 0:
        return None
    model.eval()
    losses = []
    chunk = CONFIG["eval_chunk"]
    pad = CONFIG["pad_id"]
    for i in range(0, tensor.size(0), chunk):
        b = tensor[i : i + chunk].to(device)
        attn = b != pad
        mask = None
        if gram:
            mask = (
                [j == aux_active for j in range(CONFIG["n_aux"])]
                if aux_active >= 0
                else [False] * CONFIG["n_aux"]
            )
        logits = model(b[:, :-1], mask, attn[:, :-1])
        losses.append(shifted_loss(logits, b, attn, pad).item())
    model.train()
    return float(np.mean(losses))


def shifted_loss(logits, input_ids, attention_mask, ignore_index):
    # logits already come from model(input_ids[:, :-1]); targets are input_ids[:, 1:]
    labels = input_ids[:, 1:].contiguous()
    mask = attention_mask[:, 1:].contiguous().float()
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).reshape(labels.shape)
    return (loss * mask).sum() / mask.sum().clamp(min=1)


# ------------------------- eval tensors -------------------------
def load_eval_tensors(hf_token, aux_names, cfg):
    """Load the held-out test split and build per-category eval tensors."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        "SimpleStories/SimpleStories-1.25M", token=hf_token
    )
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    ds = load_dataset("SimpleStories/SimpleStories", split="test", token=hf_token)
    aux_map = {t: i for i, t in enumerate(aux_names)}
    idx_by_cat = {i: [] for i in range(-1, len(aux_names))}
    topic_col = ds["topic"]
    story_col = ds["story"]
    for i, (t, s) in enumerate(zip(topic_col, story_col, strict=False)):
        if not s or not str(s).strip():
            continue
        if t in aux_map:
            idx_by_cat[aux_map[t]].append(i)
        else:
            idx_by_cat[-1].append(i)
    tensors = {}
    L = cfg["seq_len"] + 1
    per_cat = cfg["eval_per_cat"]
    for i in range(-1, len(aux_names)):
        key = "core" if i == -1 else aux_names[i]
        ids = idx_by_cat[i][:per_cat]
        if not ids:
            tensors[key] = None
            continue
        texts = [
            str(s) for s in ds.select(ids)["story"] if s is not None and str(s).strip()
        ]
        if not texts:
            tensors[key] = None
            continue
        enc = tok(
            texts,
            truncation=True,
            max_length=L,
            padding="max_length",
            return_tensors="pt",
        )
        tensors[key] = enc["input_ids"]
    counts = {k: (0 if v is None else v.size(0)) for k, v in tensors.items()}
    print(f"[eval] test-split eval tensors built: {counts}", flush=True)
    return tensors


def build_eval_tensors(ds, tok, idx_by_cat, aux, cfg):
    """Fallback: build eval tensors from the same dataset (smoke mode only)."""
    tensors = {}
    L = cfg["seq_len"] + 1  # +1 for shifted target
    per_cat = cfg["eval_per_cat"]
    for i in range(-1, len(aux)):
        key = "core" if i == -1 else aux[i]
        ids = idx_by_cat[i][:per_cat]
        if not ids:
            tensors[key] = None
            continue
        texts = [
            str(s) for s in ds.select(ids)["story"] if s is not None and str(s).strip()
        ]
        if not texts:
            tensors[key] = None
            continue
        enc = tok(
            texts,
            truncation=True,
            max_length=L,
            padding="max_length",
            return_tensors="pt",
        )
        tensors[key] = enc["input_ids"]
    return tensors


# ------------------------- training loop -------------------------
def train_one(
    kind,
    seed,
    device,
    ds,
    tok,
    idx_by_cat,
    aux,
    cfg,
    total_steps,
    eval_tensors,
    eval_every,
):
    gram = kind == "gram"
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model, nparams = build_model(cfg, gram)
    model = model.to(device)
    if device.type == "cuda":
        model = model.to(torch.bfloat16)
    print(f"[{kind}] params={nparams:,} device={device}", flush=True)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        betas=(cfg["beta1"], cfg["beta2"]),
        weight_decay=cfg["weight_decay"],
    )
    if gram:
        core_p, aux_p = gram_param_groups(model)
        all_aux = [p for g in aux_p for p in g]

    feeder = make_batches(ds, tok, idx_by_cat, cfg, seed, max_steps=total_steps)
    curve = []
    t0 = time.time()
    model.train()
    pad = cfg["pad_id"]

    for step, (input_ids, attn, cat) in enumerate(feeder):
        input_ids = input_ids.to(device)
        attn = attn.to(device)
        lr = wsd_lr(
            step, total_steps, cfg["lr"], cfg["warmup_frac"], cfg["stable_frac"]
        )
        for g in opt.param_groups:
            g["lr"] = lr

        if gram:
            if cat == 0:  # core batch
                active = -1
                if random.random() < cfg["p_cr"]:
                    active = random.randrange(cfg["n_aux"])
                mask = [j == active for j in range(cfg["n_aux"])]
                set_grad(core_p, True)
                set_grad(all_aux, False)
                if active >= 0:
                    set_grad(aux_p[active], True)
            else:  # aux batch
                ai = cat - 1
                mask = [j == ai for j in range(cfg["n_aux"])]
                set_grad(core_p, random.random() < cfg["p_as"])
                set_grad(all_aux, False)
                set_grad(aux_p[ai], True)
            logits = model(input_ids[:, :-1], mask, attn[:, :-1])
        else:
            logits = model(input_ids[:, :-1], None, attn[:, :-1])

        loss = shifted_loss(logits, input_ids, attn, pad)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()

        if gram:
            set_grad(core_p, True)
            set_grad(all_aux, True)

        if step % eval_every == 0 or step == total_steps - 1:
            row = {"step": step, "train_loss": round(float(loss.item()), 4)}
            for cname, t in eval_tensors.items():
                if gram:
                    row[cname] = eval_loss(model, True, t, device, aux_active=-1)
                else:
                    row[cname] = eval_loss(model, False, t, device)
            curve.append(row)
            elapsed = time.time() - t0
            core_l = row.get("core")
            print(
                f"[{kind}] step {step}/{total_steps} loss={loss.item():.4f} "
                f"core={'N/A' if core_l is None else f'{core_l:.4f}'} "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )

    # Final eval across all profiles for GRAM
    final = {}
    if gram:
        profiles = {"core_only": -1}
        for i, name in enumerate(aux):
            profiles[f"retain_{name}"] = i
        for pname, aux_idx in profiles.items():
            final[pname] = {}
            for cname, t in eval_tensors.items():
                final[pname][cname] = eval_loss(
                    model, True, t, device, aux_active=aux_idx
                )
    else:
        final["all"] = {}
        for cname, t in eval_tensors.items():
            final["all"][cname] = eval_loss(model, False, t, device)

    wall = time.time() - t0
    print(f"[{kind}] done in {wall:.0f}s", flush=True)
    return {
        "kind": kind,
        "seed": seed,
        "n_params": nparams,
        "steps_completed": step + 1,
        "wall_seconds": round(wall, 1),
        "curve": curve,
        "final_losses": final,
        "model": model,
    }


def elicit_finetune(model, device, ds, tok, idx_by_cat, cfg, steps):
    """Adversarial elicitation (Appendix A.1): finetune the GRAM model for
    `steps` steps on 128 sequences from each forget (aux) category, all
    params trainable, then return the fresh losses. LR: paper does not state
    the elicit finetune LR; we use the final post-decay-region magnitude
    (peak/10) as a conservative choice and disclose it."""
    rng_seed = 1234
    feeder = make_batches(
        ds, tok, idx_by_cat, cfg, rng_seed, max_steps=steps, aux_only=True
    )
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"] / 10.0,
        betas=(cfg["beta1"], cfg["beta2"]),
        weight_decay=cfg["weight_decay"],
    )
    pad = cfg["pad_id"]
    model.train()
    for step, (input_ids, attn, _cat) in enumerate(feeder):  # noqa: B007 - step used after loop
        input_ids = input_ids.to(device)
        attn = attn.to(device)
        mask = [False] * cfg["n_aux"]  # core-only forward (modules ablated)
        logits = model(input_ids[:, :-1], mask, attn[:, :-1])
        loss = shifted_loss(logits, input_ids, attn, pad)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
    print(f"[elicit] finetuned {step + 1} steps", flush=True)


# ------------------------- compute ratio (Appendix M) -------------------------
def fit_power_law(curve, cat):
    """Fit L(s) = A*(s+s0)^-alpha via log-space least squares scanning s0."""
    pts = [
        (p["step"] + 1, p[cat]) for p in curve if p.get(cat) is not None and p[cat] > 0
    ]
    if len(pts) < 5:
        return None
    ss = np.array([p[0] for p in pts], dtype=np.float64)
    ls = np.array([p[1] for p in pts], dtype=np.float64)
    best = None
    s0_candidates = np.concatenate([np.linspace(1, 500, 60), np.logspace(2.8, 4.5, 40)])
    for s0 in s0_candidates:
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
    if fit is None or loss is None or loss <= 0:
        return None
    val = (fit["A"] / loss) ** (1.0 / fit["alpha"]) - fit["s0"]
    return max(0.0, val)


def compute_ratios(
    baseline_curve, baseline_final, gram_final, aux_names, elicit_losses=None
):
    fits = {}
    for cat in ["core", *aux_names]:
        fits[cat] = fit_power_law(baseline_curve, cat)

    def cr(cat, loss):
        fit = fits.get(cat)
        bl_loss = baseline_final.get("all", {}).get(cat)
        bl_se = step_equivalent(fit, bl_loss)
        g_se = step_equivalent(fit, loss)
        if bl_se is None or g_se is None or bl_se <= 0:
            return None
        return g_se / bl_se

    results = {"per_profile": {}, "aggregate": {}}
    # baseline self-check
    results["per_profile"]["baseline"] = {
        cat: cr(cat, baseline_final["all"].get(cat)) for cat in ["core", *aux_names]
    }

    # GRAM profiles
    core_crs, retain_crs, forget_crs = [], [], []
    for pname, cat_losses in gram_final.items():
        prof = {}
        for cat, loss in cat_losses.items():
            prof[cat] = cr(cat, loss)
        results["per_profile"][pname] = prof
        if pname == "core_only":
            if prof.get("core") is not None:
                core_crs.append(prof["core"])
            for a in aux_names:
                if prof.get(a) is not None:
                    forget_crs.append(prof[a])
        elif pname.startswith("retain_"):
            retained_cat = pname[len("retain_") :]
            if prof.get("core") is not None:
                core_crs.append(prof["core"])
            if prof.get(retained_cat) is not None:
                retain_crs.append(prof[retained_cat])
            for a in aux_names:
                if a != retained_cat and prof.get(a) is not None:
                    forget_crs.append(prof[a])

    def safe_mean(xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else None

    results["aggregate"] = {
        "core": safe_mean(core_crs),
        "retain": safe_mean(retain_crs),
        "forget": safe_mean(forget_crs),
    }
    if elicit_losses:
        el_crs = [cr(a, elicit_losses.get(a)) for a in aux_names]
        results["aggregate"]["elicit"] = safe_mean(el_crs)
        results["per_profile"]["elicited_core_only"] = dict(
            zip(aux_names, el_crs, strict=False)
        )
    return results


# ------------------------- main audit run -------------------------
def run_audit(mode="smoke"):
    import torch as th

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    hf_token = os.environ.get("HF_TOKEN", "")

    if mode == "smoke":
        max_samples = 2000
        max_steps = 20
        eval_per_cat = 32
    elif mode == "scaled":
        max_samples = 150000
        max_steps = None
        eval_per_cat = 256
    else:  # full
        max_samples = None
        max_steps = None
        eval_per_cat = 256

    CONFIG["eval_per_cat"] = eval_per_cat
    print(
        f"[setup] mode={mode}, device={device}, max_samples={max_samples}", flush=True
    )

    ds, tok, idx_by_cat, aux_names = load_data(hf_token, max_samples=max_samples)
    if mode == "smoke":
        # smoke uses test split for training too; eval from same split is fine for pipeline validation
        eval_tensors = build_eval_tensors(ds, tok, idx_by_cat, aux_names, CONFIG)
        train_split = "test (smoke)"
    else:
        eval_tensors = load_eval_tensors(hf_token, aux_names, CONFIG)
        train_split = "train"
    eval_split = "test"

    total_seqs = sum(len(v) for v in idx_by_cat.values())
    total_steps = max_steps or max(1, total_seqs // CONFIG["batch_size"])
    eval_every = max(1, total_steps // 100)
    print(f"[setup] total_steps={total_steps}, eval_every={eval_every}", flush=True)

    # Train baseline
    print("=== Training Baseline ===", flush=True)
    baseline = train_one(
        "baseline",
        seed=42,
        device=device,
        ds=ds,
        tok=tok,
        idx_by_cat=idx_by_cat,
        aux=aux_names,
        cfg=CONFIG,
        total_steps=total_steps,
        eval_tensors=eval_tensors,
        eval_every=eval_every,
    )

    # Train GRAM
    print("=== Training GRAM ===", flush=True)
    gram = train_one(
        "gram",
        seed=42,
        device=device,
        ds=ds,
        tok=tok,
        idx_by_cat=idx_by_cat,
        aux=aux_names,
        cfg=CONFIG,
        total_steps=total_steps,
        eval_tensors=eval_tensors,
        eval_every=eval_every,
    )

    # Compute ratios
    elicit_losses = None
    if mode != "smoke" and CONFIG["elicit_steps"] > 0:
        elicit_finetune(
            gram["model"], device, ds, tok, idx_by_cat, CONFIG, CONFIG["elicit_steps"]
        )
        elicit_losses = {}
        for cname, t in eval_tensors.items():
            if cname == "core":
                continue
            elicit_losses[cname] = eval_loss(
                gram["model"], True, t, device, aux_active=-1
            )
        print(
            f"[elicit] post-finetune aux losses (core-only profile): {elicit_losses}",
            flush=True,
        )

    ratios = compute_ratios(
        baseline["curve"],
        baseline["final_losses"],
        gram["final_losses"],
        aux_names,
        elicit_losses,
    )

    # drop torch models from the returned payload
    baseline.pop("model", None)
    gram.pop("model", None)

    result = {
        "mode": mode,
        "train_split": train_split,
        "eval_split": eval_split,
        "baseline_params": baseline["n_params"],
        "gram_params": gram["n_params"],
        "total_steps": total_steps,
        "baseline_wall_s": baseline["wall_seconds"],
        "gram_wall_s": gram["wall_seconds"],
        "compute_ratios": ratios,
        "paper_reference": PAPER_TABLE1,
    }

    out_path = (
        "/root/results/gram_audit_results.json"
        if os.path.isdir("/root/results")
        else "gram_audit_results.json"
    )
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSUMMARY_JSON={json.dumps(result, default=str)}", flush=True)
    return result


# ------------------------- Modal wiring -------------------------
import modal  # noqa: E402 - placed after model/training definitions by design

app = modal.App("lemma-gram-audit")


def _load_hf_token():
    tok = os.environ.get("HF_TOKEN", "")
    if tok:
        return tok
    env_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".env"
    )
    try:
        with open(env_path) as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    return line.strip().split("=", 1)[1]
    except FileNotFoundError:
        pass
    return ""


image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "transformers", "datasets", "numpy", "scipy"
)

hf_secret = modal.Secret.from_dict({"HF_TOKEN": _load_hf_token()})
results_vol = modal.Volume.from_name("lemma-gram-results", create_if_missing=True)


@app.function(
    image=image,
    secrets=[hf_secret],
    volumes={"/root/results": results_vol},
    timeout=6 * 3600,
    cpu=8.0,
    memory=32768,
)
def run_cpu(mode: str = "smoke"):
    return run_audit(mode)


@app.function(
    image=image,
    gpu="A10G",
    secrets=[hf_secret],
    volumes={"/root/results": results_vol},
    timeout=12 * 3600,
    cpu=16.0,
    memory=65536,
)
def run_gpu(mode: str = "scaled"):
    return run_audit(mode)


@app.local_entrypoint()
def main(mode: str = "smoke"):
    result = run_cpu.remote(mode) if mode == "smoke" else run_gpu.remote(mode)
    print(json.dumps(result, indent=2, default=str))
