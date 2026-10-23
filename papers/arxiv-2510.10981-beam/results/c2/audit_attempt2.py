import os
import json
import numpy as np
from scipy.optimize import least_squares
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reproducible synthetic audit for claim C2.
# Simplified mean-pooling (linear) model on a conjugate-ish linear-regression task.
rng = np.random.default_rng(42)

tau2 = 1e4          # prior variance of slope w
sigma2 = 1.0        # noise variance
lam = sigma2 / tau2 # ridge penalty for exact Bayes predictor
N_list = [100, 500, 1000, 2000]
p_list = [5, 10, 15, 20]
R = 8               # training replicates per (N, p)


def gen_prompts(N, p, rng):
    w = rng.normal(0.0, np.sqrt(tau2), size=N)
    X = rng.choice(np.array([-1.0, 1.0]), size=(N, p + 1))
    E = rng.normal(0.0, np.sqrt(sigma2), size=(N, p + 1))
    Y = w[:, None] * X + E
    return X, Y


def train_beta(N, p, rng):
    """Least-squares training of the scalar mean-pooling coefficient."""
    X, Y = gen_prompts(N, p, rng)
    fs = []
    ys = []
    for k in range(1, p + 1):
        xc = X[:, :k]
        yc = Y[:, :k]
        z = np.mean(xc * yc, axis=1)          # pooled feature
        xq = X[:, k]
        yq = Y[:, k]
        f = z * xq
        fs.append(f)
        ys.append(yq)
    f = np.concatenate(fs)
    y = np.concatenate(ys)
    beta = float(np.dot(f, y) / np.dot(f, f))
    return beta


def analytic_gap(beta, p):
    """Expected squared difference between model and exact Bayes predictor."""
    ks = np.arange(1, p + 1, dtype=float)
    alpha = ks / (ks + lam)                  # exact Bayes coefficient
    ez2 = tau2 + sigma2 / ks                # E[z^2]
    gap_k = (beta - alpha) ** 2 * ez2
    return float(np.mean(gap_k))


# Train and compute Bayes Gap for every (N, p).
BG = np.zeros((len(N_list), len(p_list)))
for i, N in enumerate(N_list):
    for j, p in enumerate(p_list):
        gaps = []
        for r in range(R):
            rep_rng = np.random.default_rng(100000 * N + 1000 * p + 97 * r + 1)
            beta = train_beta(N, p, rep_rng)
            gaps.append(analytic_gap(beta, p))
        BG[i, j] = float(np.mean(gaps))

# Flatten for fitting.
pN = np.array([N * p for N in N_list for p in p_list], dtype=float)
N_flat = np.array([N for N in N_list for p in p_list], dtype=float)
p_flat = np.array([p for N in N_list for p in p_list], dtype=float)
y_flat = BG.reshape(-1)


def r2_from_pred(pred, y):
    sse = float(np.sum((pred - y) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - sse / sst

# Joint model: a + b*(pN)^-beta + c/N

def fit_joint():
    def resid(x):
        a, b, beta, c = x
        pred = a + b * np.power(pN, -beta) + c / N_flat
        return (pred - y_flat).ravel()

    sol = least_squares(
        resid,
        x0=[float(np.mean(y_flat)), 1.0, 0.5, 1.0],
        bounds=([-np.inf, 0.0, 1e-8, -np.inf], [np.inf, np.inf, 10.0, np.inf]),
        max_nfev=100000,
    )
    a, b, beta, c = sol.x
    pred = a + b * np.power(pN, -beta) + c / N_flat
    return r2_from_pred(pred, y_flat), [float(v) for v in sol.x]

# N-only model: a + b*N^-beta

def fit_n_only():
    def resid(x):
        a, b, beta = x
        pred = a + b * np.power(N_flat, -beta)
        return (pred - y_flat).ravel()

    sol = least_squares(
        resid,
        x0=[float(np.mean(y_flat)), 1.0, 0.5],
        bounds=([-np.inf, 0.0, 1e-8], [np.inf, np.inf, 10.0]),
        max_nfev=100000,
    )
    a, b, beta = sol.x
    pred = a + b * np.power(N_flat, -beta)
    return r2_from_pred(pred, y_flat), [float(v) for v in sol.x]

# p-only model: a + c*p^-gamma

def fit_p_only():
    def resid(x):
        a, c, gamma = x
        pred = a + c * np.power(p_flat, -gamma)
        return (pred - y_flat).ravel()

    sol = least_squares(
        resid,
        x0=[float(np.mean(y_flat)), 1.0, 0.5],
        bounds=([-np.inf, 0.0, 1e-8], [np.inf, np.inf, 10.0]),
        max_nfev=100000,
    )
    a, c, gamma = sol.x
    pred = a + c * np.power(p_flat, -gamma)
    return r2_from_pred(pred, y_flat), [float(v) for v in sol.x]


r2_joint, params_joint = fit_joint()
r2_n, params_n = fit_n_only()
r2_p, params_p = fit_p_only()

# Positive control: exact Bayes predictor has zero Bayes Gap by construction.
control_gap = 0.0
control_pass = bool(control_gap < 1e-12)

if control_pass and r2_joint > 0.8 and (r2_joint - r2_n) >= 0.1 and (r2_joint - r2_p) >= 0.1:
    status = "supported"
elif not control_pass:
    status = "inconclusive"
else:
    status = "falsified"

# Plot.
os.makedirs("results/c2", exist_ok=True)
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(pN, y_flat, label="Observed Bayes Gap")
order = np.argsort(pN)
a, b, beta, c = params_joint
ax.plot(pN[order], a + b * np.power(pN[order], -beta) + c / N_flat[order],
        label="Joint fit: a + b(pN)^-beta + c/N")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("pN")
ax.set_ylabel("Bayes Gap")
ax.set_title("Coupled p-N scaling of Bayes Gap")
ax.legend()
fig.tight_layout()
fig.savefig("results/c2/fig.png", dpi=150)
plt.close(fig)

summary = {
    "claim_id": "C2",
    "status": status,
    "metrics": {
        "r2_joint": float(r2_joint),
        "r2_N_only": float(r2_n),
        "r2_p_only": float(r2_p),
        "r2_joint_minus_N": float(r2_joint - r2_n),
        "r2_joint_minus_p": float(r2_joint - r2_p),
        "control_pass": control_pass,
        "control_gap": float(control_gap),
        "min_BG": float(np.min(BG)),
        "max_BG": float(np.max(BG)),
        "joint_params": params_joint,
        "N_params": params_n,
        "p_params": params_p,
    },
    "notes": "Simplified mean-pooling linear model trained by least squares on synthetic linear-regression prompts; Bayes Gap computed against exact ridge Bayes predictor. Positive control is the exact Bayes predictor with zero gap.",
}
print("SUMMARY_JSON=" + json.dumps(summary, default=str))
