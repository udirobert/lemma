"""Numerical replication of Figures 3 & 4 from 'To Grok Grokking' (ICML 2026).

C5 — Two-layer ReLU experiments that qualitatively reproduce the predicted
grokking-time dependence on hyperparameters beyond the linear setting.

Section 5.2 / Figure 3: Random-features ReLU network (output layer trained only).
  N(x; a) = Σ_j a_j relu(<w_j, x>), w_j fixed after init.
  Teacher: x → relu(<w*, x>) with ‖w*‖₂ = 1 (non-realizable but approximately so).
  Init: a_j ~ N(0,1), w_j ~ N(0, ν²/d · I_d).
  Defaults: d=100, η=1, n=100, λ=1e-5, m=10000, ν²=1.

Section 5.3 / Figure 4: Full two-layer ReLU network (both layers trained).
  N(x; W, a) = Σ_j a_j relu(<w_j, x>).
  Teacher: zero function.
  Init: a_j ~ N(0, 1/m), w_j ~ N(0, ν²/d · I_d).
  Defaults: η=1e-4, d=50, n=50, m=1000, ν²=1, λ=0.05.

Outputs:
  results/c5_figure3.png   — 4-panel sweep for random-features ReLU (Figure 3)
  results/c5_figure4.png   — 4-panel sweep for full ReLU network (Figure 4)
  results/c5_demo.png      — single-run grokking demonstration (Figure 1 right style)
  results/c5_summary.json  — raw t₁/t₂ data for all sweeps
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def find_t1_t2(
    train: np.ndarray,
    test: np.ndarray,
    epsilon: float = 0.01,
    c: float = 0.01,
    test_every: int = 1,
) -> tuple[int, int]:
    """Return (t₁, t₂) in actual step units."""
    t1_idx = next((i for i, v in enumerate(train) if v < epsilon), len(train) - 1)
    t2_idx = next((i for i, v in enumerate(test) if v < c), len(test) - 1)
    return t1_idx * test_every, t2_idx * test_every


# ---------------------------------------------------------------------------
# Part A: Random-features ReLU network (Section 5.2 / Figure 3)
# ---------------------------------------------------------------------------


def build_relu_features(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Φ_{ij} = ReLU(⟨w_j, x_i⟩) / √m.  X: (n, d), W: (m, d) → Φ: (n, m).

    The 1/√m normalization matches the feature scale in the linear case
    (Section 5.1: x ~ N(0, I_m/m)), keeping λ_max(ΦᵀΦ/n) = O(1) so that
    η = 1 is stable, consistent with the paper's stated hyperparameters.
    """
    m = W.shape[0]
    return relu(X @ W.T) / math.sqrt(m)


def gd_random_features_trace(
    Phi_train: np.ndarray,
    y_train: np.ndarray,
    Phi_test: np.ndarray,
    y_test: np.ndarray,
    a_init: np.ndarray,
    lam: float,
    eta: float,
    steps: int,
    test_every: int = 100,
) -> tuple[np.ndarray, np.ndarray, float]:
    """GD on output weights a for random-features model.

    Returns (train_loss, test_loss, eta_effective).
    eta is clamped to 1.9 / (L + 2λ) for stability, where L = λ_max(ΦᵀΦ/n).
    """
    n, _m = Phi_train.shape
    # Compute Lipschitz constant for stable GD
    # Use power iteration for speed on large m
    L = _spectral_norm_phit_phi_over_n(Phi_train)
    eta_eff = min(eta, 1.9 / (L + 2 * lam))
    a = a_init.copy()
    train_losses, test_losses = [], []
    for t in range(steps):
        residual = Phi_train @ a - y_train
        grad = Phi_train.T @ residual / n + lam * a
        a -= eta_eff * grad
        if t % test_every == 0:
            train_losses.append(0.5 * float(np.mean(residual**2)))
            test_res = Phi_test @ a - y_test
            test_losses.append(0.5 * float(np.mean(test_res**2)))
    return np.array(train_losses), np.array(test_losses), eta_eff


def _spectral_norm_phit_phi_over_n(Phi: np.ndarray, n_iter: int = 50) -> float:
    """Estimate λ_max(ΦᵀΦ/n) via power iteration (avoids forming m-by-m matrix)."""
    n, m = Phi.shape
    rng = np.random.default_rng(0)
    v = rng.standard_normal(m)
    v /= np.linalg.norm(v)
    for _ in range(n_iter):
        # ΦᵀΦ v / n = Φᵀ(Φv)/n
        u = Phi @ v  # (n,)
        v_new = Phi.T @ u / n  # (m,)
        norm = np.linalg.norm(v_new)
        if norm < 1e-15:
            return 0.0
        v = v_new / norm
    return float(norm)


def run_figure3_sweep(
    axis: str,
    values: list,
    d: int = 100,
    eta: float = 1.0,
    n: int = 100,
    lam: float = 1e-5,
    m: int = 10000,
    nu_sq: float = 1.0,
    steps: int = 80000,
    epsilon: float = 0.01,
    c: float = 0.01,
    seed: int = 42,
    test_every: int = 100,
    n_test: int = 5000,
) -> dict:
    """Sweep one hyperparameter axis for the random-features ReLU experiment."""
    rows = []
    for val in values:
        t0 = time.time()
        # Resolve per-run parameters
        n_use = int(val) if axis == "n" else n
        m_use = int(val) if axis == "m" else m
        lam_use = float(val) if axis == "lambda" else lam
        nu_use = float(val) if axis == "nu_sq" else nu_sq

        rng = np.random.default_rng(seed)
        # Data
        X_train = rng.standard_normal(size=(n_use, d))
        X_test = rng.standard_normal(size=(n_test, d))
        # Teacher
        w_star = rng.standard_normal(size=d)
        w_star /= np.linalg.norm(w_star)
        y_train = relu(X_train @ w_star)
        y_test = relu(X_test @ w_star)
        # Hidden weights (fixed random features)
        W = rng.standard_normal(size=(m_use, d)) * math.sqrt(nu_use / d)
        # Feature matrices
        Phi_train = build_relu_features(X_train, W)
        Phi_test = build_relu_features(X_test, W)
        # Output layer init
        a_init = rng.standard_normal(size=m_use)

        train, test, eta_eff = gd_random_features_trace(
            Phi_train,
            y_train,
            Phi_test,
            y_test,
            a_init,
            lam_use,
            eta,
            steps,
            test_every,
        )
        t1, t2 = find_t1_t2(train, test, epsilon, c, test_every)
        wall = time.time() - t0
        rows.append(
            {
                axis: val,
                "t1_emp": int(t1),
                "t2_emp": int(t2),
                "grokking_time": int(t2 - t1),
                "final_train": float(train[-1]),
                "final_test": float(test[-1]),
                "eta_eff": float(eta_eff),
                "wall_sec": round(wall, 1),
            }
        )
        print(
            f"  {axis}={val:>10.4g}  t1={t1:>7d}  t2={t2:>7d}  "
            f"grok={t2 - t1:>7d}  train={train[-1]:.6f}  test={test[-1]:.6f}  "
            f"η_eff={eta_eff:.6f}  ({wall:.1f}s)"
        )
    return {"axis": axis, "rows": rows}


# ---------------------------------------------------------------------------
# Part B: Full two-layer ReLU network (Section 5.3 / Figure 4)
# ---------------------------------------------------------------------------


def gd_relu_network_trace(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    W_init: np.ndarray,
    a_init: np.ndarray,
    lam: float,
    eta: float,
    steps: int,
    test_every: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Full-batch GD training both layers of a two-layer ReLU net.

    N(x; W, a) = Σ_j a_j · ReLU(⟨w_j, x⟩)
    W: (m, d), a: (m,)
    """
    n, _d = X_train.shape
    W = W_init.copy()
    a = a_init.copy()
    train_losses, test_losses = [], []

    for t in range(steps):
        # Forward
        H = relu(X_train @ W.T)  # (n, m)
        pred = H @ a  # (n,)
        residual = pred - y_train  # (n,)

        # Record losses
        if t % test_every == 0:
            train_losses.append(0.5 * float(np.mean(residual**2)))
            H_test = relu(X_test @ W.T)
            pred_test = H_test @ a
            test_losses.append(0.5 * float(np.mean((pred_test - y_test) ** 2)))

        # Backward
        # ∂L/∂a = Hᵀ residual / n + λ a
        grad_a = H.T @ residual / n + lam * a  # (m,)
        # ∂L/∂W = diag(a) · (H>0)ᵀ · diag(residual) · X / n + λ W
        #       = (a[:, None] * ((residual[:, None] * (H > 0)).T @ X)) / n + λ W
        mask = (H > 0).astype(float)  # (n, m)
        grad_W = (a[:, None] * ((residual[:, None] * mask).T @ X_train)) / n + lam * W

        W -= eta * grad_W
        a -= eta * grad_a

    return np.array(train_losses), np.array(test_losses)


def run_figure4_sweep(
    axis: str,
    values: list,
    d: int = 50,
    eta: float = 1e-4,
    n: int = 50,
    lam: float = 0.05,
    m: int = 1000,
    nu_sq: float = 1.0,
    steps: int = 150000,
    epsilon: float = 0.01,
    c: float = 0.01,
    seed: int = 42,
    test_every: int = 200,
    n_test: int = 5000,
) -> dict:
    """Sweep one hyperparameter axis for the full two-layer ReLU experiment."""
    rows = []
    for val in values:
        t0 = time.time()
        n_use = int(val) if axis == "n" else n
        m_use = int(val) if axis == "m" else m
        lam_use = float(val) if axis == "lambda" else lam
        nu_use = float(val) if axis == "nu_sq" else nu_sq

        rng = np.random.default_rng(seed)
        X_train = rng.standard_normal(size=(n_use, d))
        X_test = rng.standard_normal(size=(n_test, d))
        y_train = np.zeros(n_use)  # zero teacher
        y_test = np.zeros(n_test)

        # Init: a_j ~ N(0, 1/m), w_j ~ N(0, ν²/d · I_d)
        W_init = rng.standard_normal(size=(m_use, d)) * math.sqrt(nu_use / d)
        a_init = rng.standard_normal(size=m_use) / math.sqrt(m_use)

        train, test = gd_relu_network_trace(
            X_train,
            y_train,
            X_test,
            y_test,
            W_init,
            a_init,
            lam_use,
            eta,
            steps,
            test_every,
        )
        t1, t2 = find_t1_t2(train, test, epsilon, c, test_every)
        wall = time.time() - t0
        rows.append(
            {
                axis: val,
                "t1_emp": int(t1),
                "t2_emp": int(t2),
                "grokking_time": int(t2 - t1),
                "final_train": float(train[-1]),
                "final_test": float(test[-1]),
                "wall_sec": round(wall, 1),
            }
        )
        print(
            f"  {axis}={val:>10.4g}  t1={t1:>7d}  t2={t2:>7d}  "
            f"grok={t2 - t1:>7d}  train={train[-1]:.6f}  test={test[-1]:.6f}  "
            f"({wall:.1f}s)"
        )
    return {"axis": axis, "rows": rows}


# ---------------------------------------------------------------------------
# Demo: single-run grokking demonstration (Figure 1 right style)
# ---------------------------------------------------------------------------


def run_demo(
    d: int = 50,
    eta: float = 1e-4,
    n: int = 50,
    lam: float = 0.1,
    m: int = 1000,
    nu_sq: float = 1.0,
    steps: int = 200000,
    seed: int = 0,
    test_every: int = 200,
    n_test: int = 5000,
) -> dict:
    """Single run showing grokking in a two-layer ReLU net (zero teacher)."""
    rng = np.random.default_rng(seed)
    X_train = rng.standard_normal(size=(n, d))
    X_test = rng.standard_normal(size=(n_test, d))
    y_train = np.zeros(n)
    y_test = np.zeros(n_test)
    W_init = rng.standard_normal(size=(m, d)) * math.sqrt(nu_sq / d)
    a_init = rng.standard_normal(size=m) / math.sqrt(m)

    t0 = time.time()
    train, test = gd_relu_network_trace(
        X_train,
        y_train,
        X_test,
        y_test,
        W_init,
        a_init,
        lam,
        eta,
        steps,
        test_every,
    )
    wall = time.time() - t0
    t1, t2 = find_t1_t2(train, test, 0.01, 0.01, test_every)
    print(f"  Demo: t1={t1}, t2={t2}, grokking_time={t2 - t1}, wall={wall:.1f}s")
    return {
        "steps_recorded": len(train),
        "test_every": test_every,
        "train_loss": train.tolist(),
        "test_loss": test.tolist(),
        "t1": int(t1),
        "t2": int(t2),
        "grokking_time": int(t2 - t1),
        "wall_sec": round(wall, 1),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def render_sweep_figure(sweeps: dict, title: str, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    panel_cfg = [
        ("lambda", axes[0, 0], "λ (weight decay)", True),
        ("n", axes[0, 1], "n (sample size)", False),
        ("m", axes[1, 0], "m (width)", False),
        ("nu_sq", axes[1, 1], "ν² (init scale)", False),
    ]
    subtitles = {
        "lambda": "↓λ → longer grokking (t₂ ∝ 1/λ)",
        "n": "↓n → earlier overfit (↓t₁)",
        "m": "m has minor effect on t₁, t₂",
        "nu_sq": "↑ν² → ↑t₁, ↑t₂ (log rate)",
    }
    row_key = {"lambda": "lambda", "n": "n", "m": "m", "nu_sq": "nu_sq"}
    for key, ax, xlabel, log_x in panel_cfg:
        if key not in sweeps:
            ax.set_visible(False)
            continue
        rows = sweeps[key]["rows"]
        rk = row_key[key]
        xs = [r[rk] for r in rows]
        ax.plot(xs, [r["t1_emp"] for r in rows], "b-o", label="t₁ (overfit)", ms=5)
        ax.plot(xs, [r["t2_emp"] for r in rows], "r-s", label="t₂ (generalize)", ms=5)
        if log_x:
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("training steps")
        ax.set_title(subtitles[key], fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Wrote {out}")


def render_demo_figure(demo: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    te = demo["test_every"]
    steps = np.arange(demo["steps_recorded"]) * te
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogx(steps, demo["train_loss"], "b-", label="Train loss", alpha=0.8)
    ax.semilogx(steps, demo["test_loss"], "r-", label="Test loss", alpha=0.8)
    if demo["t1"] > 0:
        ax.axvline(demo["t1"], color="b", ls="--", alpha=0.5, label=f"t₁={demo['t1']}")
    if demo["t2"] > 0:
        ax.axvline(demo["t2"], color="r", ls="--", alpha=0.5, label=f"t₂={demo['t2']}")
    ax.set_xlabel("Training step (log scale)")
    ax.set_ylabel("Squared loss")
    ax.set_title(
        "C5 demo — Grokking in a two-layer ReLU network (zero teacher)\n"
        "η=1e-4, d=50, n=50, m=1000, λ=0.1, ν²=1",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Wrote {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"figure3": {}, "figure4": {}, "demo": {}}

    # ---- Demo (Figure 1 right style) ----
    print("=== Demo: two-layer ReLU grokking (zero teacher) ===")
    demo = run_demo(steps=200000, test_every=200)
    summary["demo"] = {
        k: v for k, v in demo.items() if k not in ("train_loss", "test_loss")
    }
    render_demo_figure(demo, out_dir / "c5_demo.png")

    # ---- Figure 3: Random-features ReLU (Section 5.2) ----
    # NOTE: Features are normalized by 1/√m to keep η=1 stable (matching the
    # linear case's feature scale). This shifts the effective λ range relative
    # to the paper's stated values; we sweep the range where grokking is
    # observable within our step budget.
    print("\n=== Figure 3: Random-features ReLU sweep ===")
    f3_steps = 60000
    f3_te = 100
    fig3 = {
        "lambda": run_figure3_sweep(
            "lambda",
            [5e-5, 1e-4, 2e-4, 5e-4, 1e-3],
            steps=f3_steps,
            test_every=f3_te,
        ),
        "n": run_figure3_sweep(
            "n",
            [25, 50, 100, 200, 400],
            steps=f3_steps,
            test_every=f3_te,
        ),
        "m": run_figure3_sweep(
            "m",
            [500, 1000, 2000, 5000, 10000],
            steps=f3_steps,
            test_every=f3_te,
        ),
        "nu_sq": run_figure3_sweep(
            "nu_sq",
            [0.25, 0.5, 1.0, 2.0, 4.0],
            steps=f3_steps,
            test_every=f3_te,
        ),
    }
    summary["figure3"] = fig3
    render_sweep_figure(
        fig3,
        "C5 / Figure 3 — Random-features ReLU: hyperparameter effects on grokking",
        out_dir / "c5_figure3.png",
    )

    # ---- Figure 4: Full two-layer ReLU (Section 5.3) ----
    print("\n=== Figure 4: Full two-layer ReLU sweep ===")
    f4_steps = 100000
    f4_te = 200
    fig4 = {
        "lambda": run_figure4_sweep(
            "lambda",
            [0.01, 0.05, 0.1, 0.2, 0.5],
            steps=f4_steps,
            test_every=f4_te,
        ),
        "n": run_figure4_sweep(
            "n",
            [10, 25, 50, 100, 200],
            steps=f4_steps,
            test_every=f4_te,
        ),
        "m": run_figure4_sweep(
            "m",
            [100, 250, 500, 1000, 2000],
            steps=f4_steps,
            test_every=f4_te,
        ),
        "nu_sq": run_figure4_sweep(
            "nu_sq",
            [0.25, 0.5, 1.0, 2.0, 4.0],
            steps=f4_steps,
            test_every=f4_te,
        ),
    }
    summary["figure4"] = fig4
    render_sweep_figure(
        fig4,
        "C5 / Figure 4 — Full two-layer ReLU: hyperparameter effects on grokking",
        out_dir / "c5_figure4.png",
    )

    # ---- Save summary ----
    (out_dir / "c5_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nWrote results/c5_summary.json")

    # ---- Verdict ----
    print("\n" + "=" * 60)
    print("C5 VERDICT")
    print("=" * 60)
    for fig_name, fig_data in [
        ("Figure 3 (random features)", fig3),
        ("Figure 4 (full network)", fig4),
    ]:
        print(f"\n{fig_name}:")
        for axis_name, sweep in fig_data.items():
            rows = sweep["rows"]
            t2s = [r["t2_emp"] for r in rows]
            grok = [r["grokking_time"] for r in rows]
            print(
                f"  {axis_name:>8s}: t2 range [{min(t2s)}, {max(t2s)}], "
                f"grokking range [{min(grok)}, {max(grok)}]"
            )


if __name__ == "__main__":
    main()
