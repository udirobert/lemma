"""Reviewer-provided reference implementation for ICL paper claim C3.

Verifies the Bayesian mechanism behind Theorem 3.3 (arXiv 2510.10981):
in a mixture of task types, the posterior over the task index concentrates
on the true task as context length k grows, and the Bayes-optimal mixture
predictor converges to the true-family oracle predictor.

Task families follow Definition 2.1 / the paper's Hermite example:
  F1 (linear):  f(x) = w . x,                    w ~ N(0, I_2)
  F2 (degree 2): f(x) = c . phi(x),               c ~ N(0, I_4),
       phi(x) = [x1, x2, x1^2 - 1, x2^2 - 1]  (Hermite H1, H2 basis)
  x ~ N(0, I_2), y = f(x) + eps, eps ~ N(0, sigma_e^2).

Metrics (all closed-form Gaussian posteriors; no iterative training):
  p_true(k)   = mean posterior probability of the true family    in [0,1]
  excess_mix(k)    = E[(mbar(Xq) - f_true(Xq))^2]   (mixture = M_Bayes)
  excess_oracle(k) = E[(oracle(Xq) - f_true(Xq))^2] (knows true family)
  ratio(k)    = excess_mix / excess_oracle. By Rao-Blackwell the oracle
                (which conditions on MORE information) has smaller excess
                risk, so ratio >= 1; concentration drives ratio -> 1.

Success criterion (re-scoped by reviewer — the original "Transformer MSE"
test needs GPU training, out of CPU-audit scope):
  p_true(5) >= 0.7 AND p_true(15) >= 0.9 (concentration toward 1) AND
  ratio(15) <= 1.05 (convergence to the oracle) AND ratio decreasing on
  the tail window k=5..15 (slope <= 0, r^2 > 0.7) AND the T=1 control
  gives ratio == 1, p_true == 1 exactly.
"""

import json

import numpy as np

D = 2
SIGMA_E = 0.5
P = 15
M_PROMPTS = 2000
SEED = 7


def _phi(X):
    """Hermite features for family 2: (k, D) -> (k, 2D)."""
    return np.hstack([X, X**2 - 1.0])


def _posterior_family1(X, y):
    """F1: y = w.x, w ~ N(0, I). Returns mu (D,), logml."""
    k = X.shape[0]
    se2 = SIGMA_E**2
    P1 = np.eye(D) + X.T @ X / se2
    r = X.T @ y / se2
    mu = np.linalg.solve(P1, r)
    syy = float(y @ y)
    logdet = k * np.log(se2) + np.log(np.linalg.det(P1))
    quad = syy / se2 - float(r @ mu)
    logml = -0.5 * (k * np.log(2 * np.pi) + logdet + quad)
    return mu, logml


def _posterior_family2(X, y):
    """F2: y = c.phi(x), c ~ N(0, I). Returns mu (2D,), logml."""
    A = _phi(X)
    k = A.shape[0]
    se2 = SIGMA_E**2
    P2 = np.eye(A.shape[1]) + A.T @ A / se2
    r = A.T @ y / se2
    mu = np.linalg.solve(P2, r)
    syy = float(y @ y)
    logdet = k * np.log(se2) + np.log(np.linalg.det(P2))
    quad = syy / se2 - float(r @ mu)
    logml = -0.5 * (k * np.log(2 * np.pi) + logdet + quad)
    return mu, logml


def run(alpha: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    two_families = bool(alpha[1] > 0)
    I_true = rng.choice(len(alpha), M_PROMPTS, p=alpha)
    w_lin = rng.normal(0.0, 1.0, (M_PROMPTS, D))  # F1 coefficients
    w_quad = rng.normal(0.0, 1.0, (M_PROMPTS, 2 * D))  # F2 coefficients
    X = rng.normal(0.0, 1.0, (M_PROMPTS, P, D))
    f_lin = np.einsum("mkd,md->mk", X, w_lin)
    phi_X = _phi(X.reshape(-1, D)).reshape(M_PROMPTS, P, 2 * D)
    f_quad = np.einsum("mkd,md->mk", phi_X, w_quad)
    y = np.where((I_true == 1)[:, None], f_quad, f_lin) + rng.normal(
        0.0, SIGMA_E, (M_PROMPTS, P)
    )
    Xq = rng.normal(0.0, 1.0, (M_PROMPTS, D))  # fresh query inputs
    fq_lin = np.einsum("md,md->m", Xq, w_lin)
    fq_quad = np.einsum("md,md->m", _phi(Xq), w_quad)
    f_true_q = np.where(I_true == 1, fq_quad, fq_lin)
    is2 = (I_true == 1).astype(float)

    p_true_curve, mix_curve, oracle_curve = [], [], []
    for k in range(1, P + 1):
        mu1 = np.zeros((M_PROMPTS, D))
        mu2 = np.zeros((M_PROMPTS, 2 * D))
        logml1 = np.zeros(M_PROMPTS)
        logml2 = np.zeros(M_PROMPTS)
        for t in range(M_PROMPTS):
            m1, l1 = _posterior_family1(X[t, :k, :], y[t, :k])
            mu1[t], logml1[t] = m1, l1
            if two_families:
                m2, l2 = _posterior_family2(X[t, :k, :], y[t, :k])
                mu2[t], logml2[t] = m2, l2
        if two_families:
            lw = np.stack(
                [np.log(alpha[0]) + logml1, np.log(alpha[1]) + logml2], axis=-1
            )
            lw -= lw.max(axis=-1, keepdims=True)
            wts = np.exp(lw)
            pi = wts / wts.sum(axis=-1, keepdims=True)
            pi1, pi2 = pi[:, 0], pi[:, 1]
        else:
            pi1, pi2 = np.ones(M_PROMPTS), np.zeros(M_PROMPTS)

        pred_mix = (
            pi1 * np.einsum("md,md->m", Xq, mu1)
            + pi2 * np.einsum("md,md->m", _phi(Xq), mu2)
        )
        pred_oracle = (1 - is2) * np.einsum("md,md->m", Xq, mu1) + is2 * np.einsum(
            "md,md->m", _phi(Xq), mu2
        )
        p_true_curve.append(float(np.mean(pi1 * (1 - is2) + pi2 * is2)))
        mix_curve.append(float(np.mean((pred_mix - f_true_q) ** 2)))
        oracle_curve.append(float(np.mean((pred_oracle - f_true_q) ** 2)))

    ratio = [m / max(o, 1e-12) for m, o in zip(mix_curve, oracle_curve)]
    return {"p_true": p_true_curve, "ratio": ratio}


def main() -> dict:
    main_res = run(np.array([0.5, 0.5]))
    ks = np.arange(1, P + 1)
    # tail-window fit (k=5..15): the mixture is confused early, then
    # concentration drives ratio -> 1; measure the convergent regime
    tail = np.arange(4, P)  # 0-indexed k=5..15
    slope, _ = np.polyfit(ks[tail], np.array(main_res["ratio"])[tail], 1)
    pred = slope * ks[tail] + np.polyfit(ks[tail], np.array(main_res["ratio"])[tail], 1)[1]
    ss_res = float(np.sum((np.array(main_res["ratio"])[tail] - pred) ** 2))
    ss_tot = float(
        np.sum(
            (
                np.array(main_res["ratio"])[tail]
                - np.mean(np.array(main_res["ratio"])[tail])
            )
            ** 2
        )
    )
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    checks = {
        "p_true_5_ge_0.7": main_res["p_true"][4] >= 0.7,
        "p_true_15_ge_0.9": main_res["p_true"][14] >= 0.9,
        "ratio_15_le_1.05": main_res["ratio"][14] <= 1.05,
        "p_true_in_01": bool(np.all(np.array(main_res["p_true"]) <= 1.0 + 1e-9)),
        "ratio_ge_1": bool(np.all(np.array(main_res["ratio"]) >= 1.0 - 1e-9)),
        "ratio_tail_decreasing": slope <= 0,
        "ratio_tail_r2_gt_0.7": r2 > 0.7,
    }

    ctrl = run(np.array([1.0, 0.0]))
    ctrl_ratio_ok = bool(np.allclose(ctrl["ratio"], 1.0, atol=1e-6))
    ctrl_p_ok = bool(np.allclose(ctrl["p_true"], 1.0, atol=1e-9))
    control_pass = ctrl_ratio_ok and ctrl_p_ok
    checks["control_pass"] = control_pass

    supported = all(checks.values())
    metrics = {
        "mean_p_true_5": float(main_res["p_true"][4]),
        "mean_p_true_15": float(main_res["p_true"][14]),
        "ratio_1": float(main_res["ratio"][0]),
        "ratio_5": float(main_res["ratio"][4]),
        "ratio_15": float(main_res["ratio"][14]),
        "slope_ratio": float(slope),
        "r_squared": float(r2),
        "control_ratio_is_1": ctrl_ratio_ok,
        "control_p_true_is_1": ctrl_p_ok,
        "control_pass": bool(control_pass),
        "checks": {k: bool(v) for k, v in checks.items()},
    }
    notes = (
        "Closed-form verification of the Theorem-3.3 mechanism on the "
        "paper's task-mixture class (linear vs degree-2 Hermite families, "
        "Definition 2.1): the mixture posterior concentrates on the true "
        "task and the Bayes-optimal mixture predictor converges to the "
        "true-family oracle, ratio(k) = excess_mix/excess_oracle >= 1 "
        "(Rao-Blackwell) decreasing to 1. The Transformer training "
        "experiment of Sec 4 needs GPU training and is out of CPU-audit "
        "scope; this verifies the Bayesian mechanism it demonstrates. "
        "Reviewer-provided reference implementation."
    )
    return {
        "claim_id": "C3",
        "status": "supported" if supported else "inconclusive",
        "metrics": metrics,
        "notes": notes,
    }


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os

    result = main()
    m = result["metrics"]
    ks = np.arange(1, P + 1)
    curve = run(np.array([0.5, 0.5]))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].plot(ks, curve["p_true"], marker="o")
    ax[0].set_xlabel("context length k")
    ax[0].set_ylabel("mean posterior prob. of true task")
    ax[0].set_title("Posterior concentration (Thm 3.3 mechanism)")
    ax[1].plot(ks, curve["ratio"], marker="o")
    ax[1].axhline(1.0, ls="--", c="gray", lw=1)
    ax[1].set_xlabel("context length k")
    ax[1].set_ylabel("excess_mix / excess_oracle")
    ax[1].set_title(f"ratio(k) -> 1 (ratio_5={m['ratio_5']:.3f})")
    fig.tight_layout()
    out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig.png")
    fig.savefig(out_png, dpi=110)
    print("SUMMARY_JSON=" + json.dumps(result))
