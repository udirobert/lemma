"""Falsification experiment: do the Eq. 8 simplified bounds survive non-Gaussian features?

The paper's Theorem 4.2 is distribution-free (uses actual lambda_min(Sigma) and
lambda_min^+(Phi^T Phi)). But the simplified bounds in Eq. 8 (Remark A.14) assume
phi(x) ~ N(0, 1/m * I_m), giving lambda_min(Sigma) = 1/m.

We test three feature distributions:
  1. Gaussian isotropic  (baseline): phi ~ N(0, 1/m * I_m),  lambda_min(Sigma) = 1/m
  2. Uniform isotropic   (same covariance, different higher moments)
  3. Gaussian non-isotropic: phi ~ N(0, Sigma) where Sigma = diag(1/(k*H_m)),
     so lambda_min(Sigma) = 1/(m * H_m) << 1/m

Hypothesis: The general Theorem 4.1 bounds hold for all three (they use the actual
lambda_min(Sigma)). The Eq. 8 simplified bounds hold for (1) and (2) but FAIL for (3),
because they assume lambda_min(Sigma) = 1/m while the true value is much smaller.

This is a targeted falsification of the simplified bounds under violated distributional
assumptions — not a falsification of the main theorem, which is distribution-free.

Outputs:
  results/falsification.png          -- 3-panel comparison (one per distribution)
  results/falsification_summary.json -- raw data + violation counts
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def harmonic_number(m: int) -> float:
    """H_m = 1 + 1/2 + ... + 1/m."""
    return sum(1.0 / k for k in range(1, m + 1))


def generate_features(
    dist: str, n: int, m: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (Phi, Sigma_true) for the requested distribution.

    Phi: (n, m) empirical feature matrix.
    Sigma_true: (m, m) population covariance (used for exact test loss).
    """
    rng = np.random.default_rng(seed)
    if dist == "gaussian_isotropic":
        Phi = rng.standard_normal(size=(n, m)) / math.sqrt(m)
        Sigma = np.eye(m) / m
    elif dist == "uniform_isotropic":
        # Uniform(-sqrt(3/m), sqrt(3/m)): variance = 1/m per component
        Phi = (rng.uniform(size=(n, m)) - 0.5) * 2 * math.sqrt(3.0 / m)
        Sigma = np.eye(m) / m
    elif dist == "gaussian_nonisotropic":
        # Sigma = diag(1/(k * H_m)) so trace = 1, lambda_min = 1/(m * H_m)
        H_m = harmonic_number(m)
        variances = np.array([1.0 / (k * H_m) for k in range(1, m + 1)])
        stds = np.sqrt(variances)
        Phi = rng.standard_normal(size=(n, m)) * stds
        Sigma = np.diag(variances)
    elif dist == "spike_slab":
        # Spike-and-slab: first n directions have variance ~1/n (carry all the
        # trace), remaining m-n directions have variance epsilon (near-zero).
        # The row space of Phi (n samples) spans the high-variance spike,
        # leaving the null space in the low-variance slab. This forces the
        # orthogonal component's test loss to be governed by lambda_min(Sigma),
        # not the average 1/m -- so the Eq. 8 bound (which assumes 1/m) should
        # be violated.
        eps = 1e-6
        spike_var = (1.0 - (m - n) * eps) / n  # normalize trace to 1
        variances = np.full(m, eps)
        variances[:n] = spike_var
        stds = np.sqrt(variances)
        Phi = rng.standard_normal(size=(n, m)) * stds
        Sigma = np.diag(variances)
    else:
        raise ValueError(f"unknown distribution {dist!r}")
    return Phi, Sigma


def gd_ridge_with_sigma(
    Phi: np.ndarray,
    Sigma: np.ndarray,
    theta_init: np.ndarray,
    theta_star: np.ndarray,
    lam: float,
    eta: float,
    steps: int,
) -> tuple[list[float], list[float]]:
    """GD on L_n(theta; lambda) with weight decay.

    Returns (train_loss, test_loss) where test loss uses the TRUE population
    covariance Sigma: L(theta) = (theta - theta*)^T Sigma (theta - theta*).
    """
    n = Phi.shape[0]
    train, test = [], []
    theta = theta_init.copy()
    for _ in range(steps):
        err = theta - theta_star
        grad = (Phi.T @ (Phi @ err)) / n + lam * theta
        theta = theta - eta * grad
        train.append(0.5 * float(np.mean((Phi @ err) ** 2)))
        test.append(float(err @ (Sigma @ err)))
    return train, test


def audit_distribution(
    dist: str,
    n: int = 100,
    m: int = 1000,
    nu_sq: float = 1.0,
    lam: float = 1e-4,
    eta: float = 1.0,
    steps: int = 8000,
    epsilon: float = 0.01,
    c: float = 0.01,
    seed: int = 0,
) -> dict:
    """Run the C1-style audit for a given feature distribution."""
    Phi, Sigma = generate_features(dist, n, m, seed)
    rng = np.random.default_rng(seed + 100)
    theta_init = rng.standard_normal(size=m) * math.sqrt(nu_sq)
    theta_star = np.zeros(m)  # zero teacher

    train, test = gd_ridge_with_sigma(
        Phi, Sigma, theta_init, theta_star, lam, eta, steps
    )

    # Theory ingredients
    eigvals = np.linalg.eigvalsh(Phi.T @ Phi)
    nz = eigvals[eigvals > 1e-12]
    lam_min_plus = float(nz.min()) if nz.size else 0.0

    # Actual lambda_min(Sigma) — from the true population covariance
    sigma_eigvals = np.linalg.eigvalsh(Sigma)
    lam_min_sigma_actual = float(sigma_eigvals.min())
    lam_min_sigma_eq8 = 1.0 / m  # what Eq. 8 assumes

    theta_sq = float(np.dot(theta_init, theta_init))

    # Theorem 4.1(iii): ||theta||^2 <= (1 - eta*lam)^(2t) * ||theta(0)||^2
    env_norm_upper = [(1 - eta * lam) ** (2 * t) * theta_sq for t in range(steps)]

    # General Theorem 4.1(ii): L(theta) >= lam_min(Sigma) * (1-eta*lam)^(2t) * (m-n)*nu^2/2
    # Uses the ACTUAL lambda_min(Sigma)
    env_test_lower_general = [
        lam_min_sigma_actual * (1 - eta * lam) ** (2 * t) * (m - n) * nu_sq / 2
        for t in range(steps)
    ]

    # Eq. 8 simplified bound: uses lambda_min(Sigma) = 1/m (Gaussian assumption)
    env_test_lower_eq8 = [
        lam_min_sigma_eq8 * (1 - eta * lam) ** (2 * t) * (m - n) * nu_sq / 2
        for t in range(steps)
    ]

    # Theorem 4.1(i): L_n(theta) <= (L/2) * (1 - eta*lam_min^+/n - eta*lam)^(2t) * ||theta(0)||^2
    L = 1.0  # approximation
    env_train_upper = [
        (L / 2) * (1 - eta * lam_min_plus / n - eta * lam) ** (2 * t) * theta_sq
        for t in range(steps)
    ]

    # Empirical t1, t2
    t1 = next((t for t, v in enumerate(train) if v < epsilon), steps)
    t2 = next((t for t, v in enumerate(test) if v < c), steps)

    # Check violations (substantive: emp < 0.5 * envelope AND envelope > 1e-4)
    general_violations = sum(
        1
        for emp, env in zip(test, env_test_lower_general, strict=True)
        if emp < 0.5 * env and env > 1e-4
    )
    eq8_violations = sum(
        1
        for emp, env in zip(test, env_test_lower_eq8, strict=True)
        if emp < 0.5 * env and env > 1e-4
    )
    train_violations = sum(
        1
        for emp, env in zip(train, env_train_upper, strict=True)
        if emp > 2.0 * env and env > 1e-6
    )

    # Eq. 8 t2 lower bound
    t2_eq8_lower = math.log(max((m - n) * nu_sq / (8 * m * epsilon), 1.0001)) / (
        2.02 * eta * lam
    )

    # General t2 lower bound (uses actual lambda_min(Sigma))
    t2_general_lower = math.log(
        max(
            (m - n) * nu_sq / 2 * (math.sqrt(c / lam_min_sigma_actual)) ** -2,
            1.0001,
        )
    ) / (2.02 * eta * lam)

    return {
        "distribution": dist,
        "config": {
            "n": n,
            "m": m,
            "nu_sq": nu_sq,
            "lam": lam,
            "eta": eta,
            "steps": steps,
            "epsilon": epsilon,
            "c": c,
            "seed": seed,
        },
        "theory_constants": {
            "lam_min_plus_PhiTPhi": lam_min_plus,
            "lam_min_Sigma_actual": lam_min_sigma_actual,
            "lam_min_Sigma_eq8_assumed": lam_min_sigma_eq8,
            "ratio_eq8_over_actual": lam_min_sigma_eq8 / lam_min_sigma_actual,
            "theta_sq_norm": theta_sq,
        },
        "t1_empirical": t1,
        "t2_empirical": t2,
        "grokking_time": max(t2 - t1, 0),
        "t2_eq8_lower_bound": t2_eq8_lower,
        "t2_general_lower_bound": t2_general_lower,
        "bound_checks": {
            "train_upper_violations": train_violations,
            "test_lower_general_violations": general_violations,
            "test_lower_eq8_violations": eq8_violations,
        },
        "general_bound_holds": general_violations == 0,
        "eq8_bound_holds": eq8_violations == 0,
        "trace": {
            "train": train[:: max(1, steps // 200)],
            "test": test[:: max(1, steps // 200)],
            "env_test_lower_general": env_test_lower_general[:: max(1, steps // 200)],
            "env_test_lower_eq8": env_test_lower_eq8[:: max(1, steps // 200)],
            "env_train_upper": env_train_upper[:: max(1, steps // 200)],
            "env_norm_upper": env_norm_upper[:: max(1, steps // 200)],
        },
    }


def render_figure(results: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    titles = {
        "gaussian_isotropic": "Gaussian isotropic (baseline)\nEq. 8 assumption holds",
        "uniform_isotropic": "Uniform isotropic (same covariance)\nEq. 8 should still hold",
        "gaussian_nonisotropic": "Gaussian non-isotropic (smooth)\nEq. 8 too loose to violate",
        "spike_slab": "Spike-and-slab (extreme anisotropy)\nEq. 8 VIOLATED",
    }

    for ax, (key, result) in zip(axes, results.items(), strict=True):
        trace = result["trace"]
        steps_total = result["config"]["steps"]
        n_pts = len(trace["train"])
        step_axis = np.linspace(0, steps_total, n_pts)

        ax.semilogy(step_axis, trace["test"], "r-", label="empirical test loss", lw=1.5)
        ax.semilogy(
            step_axis,
            trace["env_test_lower_general"],
            "g--",
            label="general bound (actual min Sigma)",
            lw=1.5,
        )
        ax.semilogy(
            step_axis,
            trace["env_test_lower_eq8"],
            "b--",
            label="Eq. 8 bound (assumes 1/m)",
            lw=1.5,
        )
        ax.axhline(0.01, color="gray", ls=":", lw=1, label="c = 0.01")

        eq8_v = result["bound_checks"]["test_lower_eq8_violations"]
        gen_v = result["bound_checks"]["test_lower_general_violations"]
        ax.set_title(titles[key], fontsize=10)
        ax.set_xlabel("training step (log)")
        ax.set_ylabel("test loss L(theta)")
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, which="both", alpha=0.3)
        # Annotate violations
        ax.text(
            0.95,
            0.95,
            f"Eq. 8 violations: {eq8_v}\nGeneral violations: {gen_v}",
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            ha="right",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

    fig.suptitle(
        "Falsification: Eq. 8 simplified bounds vs. general Theorem 4.1 bounds\n"
        "under non-Gaussian feature distributions (zero teacher, n=100, m=1000)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    distributions = [
        "gaussian_isotropic",
        "uniform_isotropic",
        "gaussian_nonisotropic",
        "spike_slab",
    ]

    results = {}
    for dist in distributions:
        print(f"\n=== {dist} ===")
        r = audit_distribution(dist, seed=0)
        results[dist] = r
        tc = r["theory_constants"]
        bc = r["bound_checks"]
        print(f"  lambda_min(Sigma) actual = {tc['lam_min_Sigma_actual']:.6e}")
        print(f"  lambda_min(Sigma) Eq.8   = {tc['lam_min_Sigma_eq8_assumed']:.6e}")
        print(f"  ratio (Eq.8 / actual)    = {tc['ratio_eq8_over_actual']:.2f}x")
        print(
            f"  t1={r['t1_empirical']}, t2={r['t2_empirical']}, grokking={r['grokking_time']}"
        )
        print(f"  t2_eq8_lower   = {r['t2_eq8_lower_bound']:.0f}")
        print(f"  t2_general_lower= {r['t2_general_lower_bound']:.0f}")
        print(f"  general bound violations: {bc['test_lower_general_violations']}")
        print(f"  Eq.8 bound violations:    {bc['test_lower_eq8_violations']}")
        print(f"  general holds: {r['general_bound_holds']}")
        print(f"  Eq.8 holds:    {r['eq8_bound_holds']}")

    summary = {
        "experiment": "Falsification: Eq. 8 simplified bounds under non-Gaussian features",
        "hypothesis": (
            "The general Theorem 4.1 bounds (using actual lambda_min(Sigma)) hold for "
            "all distributions. The Eq. 8 simplified bounds (assuming lambda_min(Sigma)=1/m) "
            "hold for isotropic distributions but FAIL for non-isotropic features where "
            "lambda_min(Sigma) << 1/m."
        ),
        "results": results,
    }
    (out_dir / "falsification_summary.json").write_text(json.dumps(summary, indent=2))
    render_figure(results, out_dir / "falsification.png")

    print("\n" + "=" * 60)
    print("FALSIFICATION VERDICT")
    print("=" * 60)
    for dist, r in results.items():
        bc = r["bound_checks"]
        print(f"\n{dist}:")
        print(
            f"  General Theorem 4.1 bound: {'HOLDS' if r['general_bound_holds'] else 'VIOLATED'} "
            f"({bc['test_lower_general_violations']} violations)"
        )
        print(
            f"  Eq. 8 simplified bound:    {'HOLDS' if r['eq8_bound_holds'] else 'VIOLATED'} "
            f"({bc['test_lower_eq8_violations']} violations)"
        )


if __name__ == "__main__":
    main()
