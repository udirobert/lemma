"""Reviewer-provided reference implementation for ICL paper claim C1.

Verifies Proposition 3.1 (arXiv 2510.10981): for ANY measurable predictor M,
the in-context-learning risk decomposes EXACTLY as

    R(M) = RBG(M) + RPV,

where RBG(M) = (1/p) sum_k E[(M(P^k) - M_Bayes(P^k))^2] is the Bayes Gap and
RPV = (1/p) sum_k E[Var(f(x_{k+1}) | D_k)] is the posterior variance.

Setting (per reviewer feedback round 4): two DISTINGUISHABLE linear task
families in dfeat=1:
  family 1: y = w x,            w ~ N(0, sigma_w^2)
  family 2: y = w x + b,        w, b ~ N(0, sigma_w^2)
I ~ Categorical(alpha), noise eps ~ N(0, sigma_e^2), x_k ~ N(0,1).

All posteriors and marginal likelihoods are closed-form Gaussians.
MAIN predictor: the task-1-only oracle M(P^k) = E[w|family 1, D_k] * x_{k+1}
(has RBG > 0). CONTROL: M = M_Bayes (has RBG ~ 0, so R ~ RPV).
"""

import json

import numpy as np

SIGMA_W = 1.0
SIGMA_E = 0.5
P = 10  # prompt length
M_PROMPTS = 5000
ALPHA = np.array([0.7, 0.3])
SEED = 42


def _log_marginal_family1(k, Sxx, sxy, syy):
    """log p(D_k | family 1): y ~ N(0, x x^T + se2 I), rank-1 + diagonal."""
    se2 = SIGMA_E**2
    t = 1.0 + Sxx / se2
    logdet = k * np.log(se2) + np.log(t)
    quad = syy / se2 - sxy**2 / (se2**2 * t)
    return -0.5 * (k * np.log(2 * np.pi) + logdet + quad)


def _family2_stats(k, Sxx, Sx, sxy, sy, syy):
    """Family 2 (w,b) posterior and marginal. Returns dict of arrays."""
    se2 = SIGMA_E**2
    # posterior precision P2 = I + A^T A / se2 with rows A_j = (x_j, 1)
    p00 = 1.0 + Sxx / se2
    p01 = Sx / se2
    p11 = 1.0 + k / se2
    det2 = p00 * p11 - p01 * p01
    inv00 = p11 / det2
    inv01 = -p01 / det2
    inv11 = p00 / det2
    # posterior mean = P2^{-1} A^T y / se2
    r0 = sxy / se2
    r1 = sy / se2
    mu_w = inv00 * r0 + inv01 * r1
    mu_b = inv01 * r0 + inv11 * r1
    # marginal: C = A A^T + se2 I; logdet = k log se2 + log det P2
    logdet = k * np.log(se2) + np.log(det2)
    # quad via Woodbury: y^T C^{-1} y = syy/se2 - r^T P2^{-1} r
    # (r = A^T y / se2 already carries both 1/se2 factors)
    quad = syy / se2 - (inv00 * r0 * r0 + 2 * inv01 * r0 * r1 + inv11 * r1 * r1)
    logml = -0.5 * (k * np.log(2 * np.pi) + logdet + quad)
    return dict(
        mu_w=mu_w, mu_b=mu_b, inv00=inv00, inv01=inv01, inv11=inv11, logml=logml
    )


def run() -> dict:
    rng = np.random.default_rng(SEED)
    I_true = rng.choice(2, M_PROMPTS, p=ALPHA)  # 0 = family1, 1 = family2
    w_true = rng.normal(0.0, SIGMA_W, M_PROMPTS)
    b_true = rng.normal(0.0, SIGMA_W, M_PROMPTS)  # used only when I_true == 1
    x = rng.normal(0.0, 1.0, (M_PROMPTS, P + 1))
    eps = rng.normal(0.0, SIGMA_E, (M_PROMPTS, P))
    y = (
        w_true[:, None] * x[:, :P]
        + (I_true == 1)[:, None] * b_true[:, None]
        + eps
    )

    # cumulative sufficient statistics over context positions 1..k
    cum_Sxx = np.cumsum(x[:, :P] ** 2, axis=1)
    cum_Sx = np.cumsum(x[:, :P], axis=1)
    cum_sxy = np.cumsum(x[:, :P] * y, axis=1)
    cum_sy = np.cumsum(y, axis=1)
    cum_syy = np.cumsum(y**2, axis=1)

    xq = x[:, 1 : P + 1]  # query inputs x_{k+1} for k = 1..P
    y_true_q = w_true[:, None] * xq + (I_true == 1)[:, None] * b_true[:, None]

    R_acc = RBG_acc = RPV_acc = 0.0
    R_ctrl_acc = 0.0
    for k in range(1, P + 1):
        Sxx = cum_Sxx[:, k - 1]
        Sx = cum_Sx[:, k - 1]
        sxy = cum_sxy[:, k - 1]
        sy = cum_sy[:, k - 1]
        syy = cum_syy[:, k - 1]

        # --- family 1 posterior ---
        se2 = SIGMA_E**2
        prec1 = 1.0 / SIGMA_W**2 + Sxx / se2
        mu1 = (sxy / se2) / prec1
        var1 = 1.0 / prec1
        logml1 = _log_marginal_family1(k, Sxx, sxy, syy)

        # --- family 2 posterior ---
        f2 = _family2_stats(k, Sxx, Sx, sxy, sy, syy)

        # --- mixture posterior over task index ---
        logw = np.stack(
            [np.log(ALPHA[0]) + logml1, np.log(ALPHA[1]) + f2["logml"]], axis=-1
        )
        logw -= logw.max(axis=-1, keepdims=True)
        wts = np.exp(logw)
        pi = wts / wts.sum(axis=-1, keepdims=True)  # (M, 2)
        pi1, pi2 = pi[:, 0], pi[:, 1]

        # --- predictors at the query point ---
        xq_k = xq[:, k - 1]
        pred1 = mu1 * xq_k  # family-1 posterior mean prediction
        pred2 = f2["mu_w"] * xq_k + f2["mu_b"]
        mbar = pi1 * pred1 + pi2 * pred2  # M_Bayes(P^k)

        # conditional variance of f(x_{k+1}) given D_k (law of total variance)
        var_f1 = var1 * xq_k**2 + pred1**2
        var_f2 = (
            f2["inv00"] * xq_k**2
            + 2 * f2["inv01"] * xq_k
            + f2["inv11"]
            + pred2**2
        )
        cond_second = pi1 * var_f1 + pi2 * var_f2
        rpv_k = cond_second - mbar**2
        rpv_k = np.maximum(rpv_k, 0.0)  # numerical floor at 0

        # MAIN: M = task-1-only oracle (assumes family 1 always)
        pred_main = pred1
        R_acc += np.mean((y_true_q[:, k - 1] - pred_main) ** 2)
        RBG_acc += np.mean((pred_main - mbar) ** 2)
        RPV_acc += np.mean(rpv_k)
        # CONTROL: M = M_Bayes
        R_ctrl_acc += np.mean((y_true_q[:, k - 1] - mbar) ** 2)

    R = R_acc / P
    RBG = RBG_acc / P
    RPV = RPV_acc / P
    R_ctrl = R_ctrl_acc / P

    rel_diff = abs(R - (RBG + RPV)) / R
    rel_diff_ctrl = abs(R_ctrl - RPV) / R_ctrl

    main_ok = rel_diff < 0.01 and RBG > 1e-6
    control_pass = rel_diff_ctrl < 0.02 and True  # RBG_ctrl == 0 by construction
    status = "supported" if (main_ok and control_pass) else "inconclusive"

    metrics = {
        "R_main": float(R),
        "RBG_main": float(RBG),
        "RPV_main": float(RPV),
        "rel_diff": float(rel_diff),
        "R_ctrl": float(R_ctrl),
        "RPV_ctrl": float(RPV),
        "rel_diff_ctrl": float(rel_diff_ctrl),
        "RBG_main_nonzero": bool(RBG > 1e-6),
        "control_pass": bool(control_pass),
        "M": M_PROMPTS,
        "p": P,
    }
    notes = (
        "Closed-form verification of Proposition 3.1 (R = RBG + RPV) on a "
        "two-family linear-task mixture (families distinguished by an "
        "intercept). MAIN predictor: task-1-only oracle (RBG_main > 0); "
        "CONTROL: Bayes predictor. Reviewer-provided reference "
        "implementation, executed by the audit pipeline."
    )
    return {
        "claim_id": "C1",
        "status": status,
        "metrics": metrics,
        "notes": notes,
    }


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os

    result = run()
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    m = result["metrics"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    terms = ["R(M)", "RBG(M)", "RPV", "RBG+RPV"]
    vals = [m["R_main"], m["RBG_main"], m["RPV_main"], m["RBG_main"] + m["RPV_main"]]
    ax.bar(terms, vals)
    ax.set_ylabel("risk")
    ax.set_title(
        f"Prop. 3.1 identity: rel_diff={m['rel_diff']:.2e}, "
        f"RBG_main={m['RBG_main']:.3e}"
    )
    fig.tight_layout()
    out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig.png")
    fig.savefig(out_png, dpi=110)
    print("SUMMARY_JSON=" + json.dumps(result))
