import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
rng = np.random.default_rng(SEED)

# Hyperparameters
N_PROMPTS = 15000
P = 5
TAU2 = 1.0
SIGMA2 = 0.5
ALPHA = np.array([0.5, 0.5])


def phi_linear(x):
    return np.array([1.0, float(x)])


def phi_quadratic(x):
    x = float(x)
    return np.array([1.0, x, x * x])


def type_posterior(xs, ys, x_star, phi, tau2, sigma2):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    k = len(xs)
    X = np.vstack([phi(x) for x in xs])
    # Marginal likelihood of y under this task type
    C = sigma2 * np.eye(k) + tau2 * (X @ X.T)
    Cinv_y = np.linalg.solve(C, ys)
    quad = float(ys @ Cinv_y)
    sign, logdet = np.linalg.slogdet(C)
    loglik = -0.5 * (quad + float(logdet) + k * np.log(2.0 * np.pi))
    # Posterior mean of coefficients
    A = X.T @ X + (sigma2 / tau2) * np.eye(X.shape[1])
    beta = np.linalg.solve(A, X.T @ ys)
    pred = float(phi(x_star) @ beta)
    return loglik, pred


def bayes_predict(xs, ys, x_star, tau2, sigma2, alpha):
    phis = [phi_linear, phi_quadratic]
    logliks = []
    preds = []
    for phi in phis:
        ll, pred = type_posterior(xs, ys, x_star, phi, tau2, sigma2)
        logliks.append(ll)
        preds.append(pred)
    logliks = np.asarray(logliks, dtype=float)
    preds = np.asarray(preds, dtype=float)
    logws = np.log(alpha) + logliks
    m = np.max(logws)
    ws = np.exp(logws - m)
    ws = ws / np.sum(ws)
    return float(ws @ preds)


# Accumulators
sum_R = 0.0
sum_RBG = 0.0
sum_RPV = 0.0
sum_R_c = 0.0
sum_RBG_c = 0.0
sum_RPV_c = 0.0
count = 0

for _ in range(N_PROMPTS):
    # Sample task type and function
    type_id = 0 if rng.random() < ALPHA[0] else 1
    if type_id == 0:
        beta = rng.normal(0.0, np.sqrt(TAU2), size=2)
        def f(x, b=beta):
            return b[0] + b[1] * x
    else:
        beta = rng.normal(0.0, np.sqrt(TAU2), size=3)
        def f(x, b=beta):
            return b[0] + b[1] * x + b[2] * x * x

    xs = rng.uniform(-1.0, 1.0, size=P)
    eps = rng.normal(0.0, np.sqrt(SIGMA2), size=P)
    ys = f(xs) + eps
    x_star = float(rng.uniform(-1.0, 1.0))
    f_star = float(f(x_star))

    for k in range(1, P + 1):
        B = bayes_predict(xs[:k], ys[:k], x_star, TAU2, SIGMA2, ALPHA)
        M = float(np.mean(ys[:k]))

        sum_R += (f_star - M) ** 2
        sum_RBG += (B - M) ** 2
        sum_RPV += (f_star - B) ** 2

        # Positive control: model equals Bayes predictor, so RBG = 0 exactly
        sum_R_c += (f_star - B) ** 2
        sum_RBG_c += 0.0
        sum_RPV_c += (f_star - B) ** 2

        count += 1

R = sum_R / count
RBG = sum_RBG / count
RPV = sum_RPV / count
abs_diff = abs(R - (RBG + RPV))
threshold = 0.01 * R

R_c = sum_R_c / count
RBG_c = sum_RBG_c / count
RPV_c = sum_RPV_c / count
abs_diff_c = abs(R_c - (RBG_c + RPV_c))
control_pass = bool(abs_diff_c < 1e-9)

if control_pass and abs_diff < threshold:
    status = "supported"
elif control_pass:
    status = "falsified"
else:
    status = "inconclusive"

os.makedirs("results/c1", exist_ok=True)
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["R", "RBG+RPV", "RBG", "RPV"], [R, RBG + RPV, RBG, RPV], color=["#4C78A8", "#F58518", "#54A24B", "#E45756"])
ax.set_ylabel("Risk")
ax.set_title("C1: Risk decomposition identity")
fig.tight_layout()
fig.savefig("results/c1/fig.png", dpi=150)
plt.close(fig)

summary = {
    "claim_id": "C1",
    "status": status,
    "metrics": {
        "R": R,
        "RBG": RBG,
        "RPV": RPV,
        "RBG_plus_RPV": RBG + RPV,
        "abs_diff": abs_diff,
        "rel_diff": abs_diff / R,
        "threshold": threshold,
        "control_abs_diff": abs_diff_c,
        "control_pass": control_pass,
        "n_prompts": N_PROMPTS,
        "context_lengths": P,
        "n_evaluations": count,
    },
    "notes": "Empirical squared-loss decomposition over a linear-vs-quadratic task mixture; Bayes predictor is the analytic posterior mean. Positive control uses M equal to the Bayes predictor.",
}
print("SUMMARY_JSON=" + json.dumps(summary, default=str))
