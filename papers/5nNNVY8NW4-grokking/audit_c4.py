"""Numerical replication of Figure 2 from 'To Grok Grokking: Provable Grokking in Ridge Regression'.

Section 5.1 / Figure 2 of the paper plots how each of (λ, n, m, ν²) affects the empirical
grokking time t_2 - t_1 (more specifically t_1 and t_2 separately) in realizable ridge
regression with i.i.d. Gaussian features. We sweep each axis independently, holding
the others at the paper's defaults (n=100, m=1000, ν²=1, λ=1e-4, η=1), and overlay the
empirical t_1, t_2 trajectories with the theoretical upper/lower bounds from Eq. 8
(Remark A.14) of the paper:

  t_1 ≤  n · ln(14 m ν² / ε) / (2 η λ_min⁺(ΦᵀΦ))
  t_2 ≥  ln((m-n) ν² / (8 m ε)) / (2.02 η λ)

Outputs:
  results/c4_figure2.png       — 2x2 panel matching the paper's Figure 2 layout
  results/c4_sweep.json         — raw data + bounds per (param, value)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def gd_ridge_trace(
    Phi: np.ndarray,
    theta_init: np.ndarray,
    theta_star: np.ndarray,
    lam: float,
    eta: float,
    steps: int,
    test_every: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (train_loss, test_loss) sampled every `test_every` steps."""
    n, m = Phi.shape
    train = []
    test = []
    theta = theta_init.copy()
    for t in range(steps):
        err = theta - theta_star
        grad = (Phi.T @ (Phi @ err)) / n + lam * theta
        theta = theta - eta * grad
        if t % test_every == 0:
            train.append(0.5 * float(np.mean((Phi @ err) ** 2)))
            test.append(float(np.dot(err, err)) / m)
    return np.array(train), np.array(test)


def find_grokking_steps(
    train: np.ndarray, test: np.ndarray, epsilon: float, c: float, test_every: int
) -> tuple[int, int]:
    """Return (t_1, t_2) in actual step units (not subsample index)."""
    t1_idx = next((i for i, v in enumerate(train) if v < epsilon), len(train) - 1)
    t2_idx = next((i for i, v in enumerate(test) if v < c), len(test) - 1)
    return t1_idx * test_every, t2_idx * test_every


def sweep_lambda(
    lambdas: list[float],
    n: int = 100,
    m: int = 1000,
    nu_sq: float = 1.0,
    eta: float = 1.0,
    steps: int = 8000,
    epsilon: float = 0.01,
    c: float = 0.01,
    seed: int = 0,
    test_every: int = 10,
) -> dict:
    rng = np.random.default_rng(seed)
    Phi = rng.standard_normal(size=(n, m)) / math.sqrt(m)
    theta_init = rng.standard_normal(size=m) * math.sqrt(nu_sq)
    theta_star = rng.standard_normal(size=m)
    theta_star /= np.linalg.norm(theta_star)

    eigvals = np.linalg.eigvalsh(Phi.T @ Phi)
    nz = eigvals[eigvals > 1e-12]
    lam_min_plus = float(nz.min())

    rows = []
    for lam in lambdas:
        train, test = gd_ridge_trace(
            Phi, theta_init, theta_star, lam, eta, steps, test_every
        )
        t1_emp, t2_emp = find_grokking_steps(train, test, epsilon, c, test_every)
        # eq.8 / Remark A.14 bounds
        t1_theory = (n * math.log(14 * m * nu_sq / epsilon)) / (2 * eta * lam_min_plus)
        t2_theory = math.log(max((m - n) * nu_sq / (8 * m * epsilon), 1.0001)) / (
            2.02 * eta * lam
        )
        rows.append(
            {
                "lam": lam,
                "t1_emp": int(t1_emp),
                "t2_emp": int(t2_emp),
                "t1_theory": float(t1_theory),
                "t2_theory": float(t2_theory),
            }
        )
    return {"axis": "lambda", "rows": rows}


def sweep_param(
    name: str,
    values: list[int],
    n: int = 100,
    m: int = 1000,
    nu_sq: float = 1.0,
    lam: float = 1e-4,
    eta: float = 1.0,
    steps: int = 8000,
    epsilon: float = 0.01,
    c: float = 0.01,
    seed: int = 0,
    test_every: int = 10,
) -> dict:
    """Sweep n or m (over-parameterization gap)."""
    rows = []
    for val in values:
        if name == "n":
            Phi = np.random.default_rng(seed).standard_normal(
                size=(val, m)
            ) / math.sqrt(m)
            n_use = val
        else:  # m
            Phi = np.random.default_rng(seed).standard_normal(
                size=(n, val)
            ) / math.sqrt(val)
            n_use = n
        rng = np.random.default_rng(seed + val)
        theta_init = rng.standard_normal(size=Phi.shape[1]) * math.sqrt(nu_sq)
        theta_star = rng.standard_normal(size=Phi.shape[1])
        theta_star /= np.linalg.norm(theta_star)
        eigvals = np.linalg.eigvalsh(Phi.T @ Phi)
        nz = eigvals[eigvals > 1e-12]
        lam_min_plus = float(nz.min())

        train, test = gd_ridge_trace(
            Phi, theta_init, theta_star, lam, eta, steps, test_every
        )
        t1_emp, t2_emp = find_grokking_steps(train, test, epsilon, c, test_every)
        t1_theory = (n_use * math.log(14 * Phi.shape[1] * nu_sq / epsilon)) / (
            2 * eta * lam_min_plus
        )
        t2_theory = math.log(
            max((Phi.shape[1] - n_use) * nu_sq / (8 * Phi.shape[1] * epsilon), 1.0001)
        ) / (2.02 * eta * lam)
        rows.append(
            {
                name: val,
                "t1_emp": int(t1_emp),
                "t2_emp": int(t2_emp),
                "t1_theory": float(t1_theory),
                "t2_theory": float(t2_theory),
            }
        )
    return {"axis": name, "rows": rows}


def sweep_nu_sq(
    nu_sqs: list[float],
    n: int = 100,
    m: int = 1000,
    lam: float = 1e-4,
    eta: float = 1.0,
    steps: int = 8000,
    epsilon: float = 0.01,
    c: float = 0.01,
    seed: int = 0,
    test_every: int = 10,
) -> dict:
    rng = np.random.default_rng(seed)
    Phi = rng.standard_normal(size=(n, m)) / math.sqrt(m)
    theta_star = rng.standard_normal(size=m)
    theta_star /= np.linalg.norm(theta_star)

    eigvals = np.linalg.eigvalsh(Phi.T @ Phi)
    nz = eigvals[eigvals > 1e-12]
    lam_min_plus = float(nz.min())

    rows = []
    for nu_sq in nu_sqs:
        rng2 = np.random.default_rng(seed + int(nu_sq * 10))
        theta_init = rng2.standard_normal(size=m) * math.sqrt(nu_sq)
        train, test = gd_ridge_trace(
            Phi, theta_init, theta_star, lam, eta, steps, test_every
        )
        t1_emp, t2_emp = find_grokking_steps(train, test, epsilon, c, test_every)
        t1_theory = (n * math.log(14 * m * nu_sq / epsilon)) / (2 * eta * lam_min_plus)
        t2_theory = math.log(max((m - n) * nu_sq / (8 * m * epsilon), 1.0001)) / (
            2.02 * eta * lam
        )
        rows.append(
            {
                "nu_sq": nu_sq,
                "t1_emp": int(t1_emp),
                "t2_emp": int(t2_emp),
                "t1_theory": float(t1_theory),
                "t2_theory": float(t2_theory),
            }
        )
    return {"axis": "nu_sq", "rows": rows}


def render_figure(sweeps: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    titles = [
        "λ sweep (decreasing weight decay → longer grokking)",
        "n sweep (decreasing sample size → earlier training converge)",
        "m sweep (feature dimension; minor effect on t_1, t_2)",
        "ν² sweep (initialization scale; t_1, t_2 both ∝ ln(ν²))",
    ]
    panels = [
        ("lambda", axes[0, 0], "λ", True),
        ("n", axes[0, 1], "n", False),
        ("m", axes[1, 0], "m", False),
        ("nu_sq", axes[1, 1], "ν²", False),
    ]
    row_key_by_axis = {"lambda": "lam", "n": "n", "m": "m", "nu_sq": "nu_sq"}
    for (key, ax, xlabel, log_x), title in zip(panels, titles, strict=True):
        sweep = sweeps[key]
        rows = sweep["rows"]
        row_key = row_key_by_axis[key]
        x_vals = [r[row_key] for r in rows]
        ax.plot(x_vals, [r["t1_emp"] for r in rows], "b-o", label="t_1 empirical", ms=4)
        ax.plot(x_vals, [r["t2_emp"] for r in rows], "r-s", label="t_2 empirical", ms=4)
        ax.plot(
            x_vals,
            [r["t1_theory"] for r in rows],
            "b--^",
            label="t_1 theory (eq.8 upper)",
            ms=4,
            alpha=0.6,
        )
        ax.plot(
            x_vals,
            [r["t2_theory"] for r in rows],
            "r--v",
            label="t_2 theory (eq.8 lower)",
            ms=4,
            alpha=0.6,
        )
        if log_x:
            ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("training steps")
        ax.set_title(title, fontsize=10)
        ax.legend(loc="best", fontsize=7)
        ax.grid(True, which="both", alpha=0.3)

    fig.suptitle("Claim 4 — Figure 2 replication: how hyperparameters amplify grokking")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    sweeps = {
        "lambda": sweep_lambda(
            lambdas=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
            steps=4000,
            test_every=10,
        ),
        "n": sweep_param(
            "n",
            values=[50, 100, 150, 200, 250, 300, 350],
            steps=4000,
            test_every=10,
        ),
        "m": sweep_param(
            "m",
            values=[200, 400, 600, 800, 1000, 1500, 2000],
            steps=4000,
            test_every=10,
        ),
        "nu_sq": sweep_nu_sq(
            nu_sqs=[0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
            steps=4000,
            test_every=10,
        ),
    }
    (out_dir / "c4_sweep.json").write_text(json.dumps(sweeps, indent=2))
    render_figure(sweeps, out_dir / "c4_figure2.png")
    print("Wrote results/c4_figure2.png and results/c4_sweep.json")
    for k, v in sweeps.items():
        print(f"\n=== {k} ===")
        for row in v["rows"]:
            key_name = next(iter(row.keys()))
            print(
                f"  {key_name}={row[key_name]:>8.2g}  "
                f"t1_emp={row['t1_emp']:>6d}  t1_th={row['t1_theory']:>8.0f}   "
                f"t2_emp={row['t2_emp']:>6d}  t2_th={row['t2_theory']:>8.0f}"
            )


if __name__ == "__main__":
    main()
