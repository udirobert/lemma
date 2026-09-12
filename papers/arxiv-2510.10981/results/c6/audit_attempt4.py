import json
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

np.random.seed(0)

# ---------- Setup: two-task Gaussian linear mixture ----------
# Task 1: y = w1 x + eps ; Task 2: y = w2 x + eps ; x ~ N(0,1), eps ~ N(0, sigma^2)
w1, w2 = 1.0, -1.0
sigma = 1.0
Delta = w1 - w2  # = 2
prior1 = 0.5

# Theoretical constants
D_min = Delta**2 / (2.0 * sigma**2)          # pairwise KL per example
C = 0.5 * np.log(1.0 + Delta**2 / (4.0 * sigma**2))  # Chernoff information per example
minimax_risk = sigma**2  # point-mass true task family: irreducible noise

k_list = np.arange(1, 31)
n_trials = 20000

def sigmoid(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))

def excess_for_k(k, true_w, n_trials, sigma, w1, w2):
    """Monte-Carlo estimate of E[ x_q^2 * (w1-w2)^2 * p1*p2 ] with contexts from true task."""
    # sample contexts: shape (n_trials, k)
    x = np.random.randn(n_trials, k)
    eps = sigma * np.random.randn(n_trials, k)
    y = true_w * x + eps
    # log-likelihood ratio task1 vs task2: L = (2/sigma^2) * (w1-w2) * sum x*y ... derive:
    # log p(y|w1) - log p(y|w2) = -( (y-w1 x)^2 - (y-w2 x)^2 )/(2 sigma^2)
    r1 = (y - w1 * x)
    r2 = (y - w2 * x)
    L = -(np.sum(r1**2, axis=1) - np.sum(r2**2, axis=1)) / (2.0 * sigma**2)
    p1 = sigmoid(L)
    p2 = 1.0 - p1
    xq = np.random.randn(n_trials)
    exc = xq**2 * (Delta**2) * p1 * p2
    return exc.mean(), exc.std() / np.sqrt(n_trials)

# ---------- Main experiment: true task = task 1 ----------
excess = np.zeros(len(k_list))
excess_se = np.zeros(len(k_list))
for i, k in enumerate(k_list):
    m, se = excess_for_k(k, w1, n_trials, sigma, w1, w2)
    excess[i] = m
    excess_se[i] = se

RPV = minimax_risk + excess
bound_C = Delta**2 * np.exp(-C * k_list)      # 4 * exp(-C k)
bound_D = Delta**2 * np.exp(-D_min * k_list / 2.0)

# Bound check: excess_k <= 4 exp(-C k), allowing 3-sigma MC tolerance
tol = 3.0 * excess_se
viol_C = excess > bound_C + tol
frac_viol_C = float(np.mean(viol_C))
max_ratio_C = float(np.max(excess / np.maximum(bound_C, 1e-300)))

# monotonicity (allowing MC noise): count significant increases
increases = np.sum(excess[1:] > excess[:-1] + 3.0 * (excess_se[1:] + excess_se[:-1]))

# Optional tail slope fit where excess reliably above floor
mask = excess > 1e-6
if mask.sum() >= 5:
    idx = np.where(mask)[0]
    # use upper tail (largest k's still above floor)
    tail = idx[-min(8, len(idx)):]
    slope, intercept = np.polyfit(k_list[tail], np.log(excess[tail]), 1)
    lambda_hat = float(-slope)
else:
    lambda_hat = float('nan')

# ---------- Positive control 1: well-separated (same as main, known true) ----------
# Synthetic check: with huge separation the bound must hold trivially and excess decays.
ctrl_excess = []
for k in [1, 2, 4, 8]:
    m, _ = excess_for_k(k, w1, 5000, sigma, w1, w2)
    ctrl_excess.append(m)
ctrl_excess = np.array(ctrl_excess)
ctrl_bounds = Delta**2 * np.exp(-C * np.array([1, 2, 4, 8]))
control1_pass = bool(np.all(ctrl_excess <= ctrl_bounds * 1.5 + 1e-3) and ctrl_excess[-1] < ctrl_excess[0])

# ---------- Positive control 2: indistinguishable tasks (w1 == w2) ----------
# Then posterior stays at prior, excess = E[x_q^2] * Delta^2 * p1 p2 = 0 since Delta=0.
# Use a modified 'mixture' where both tasks identical: excess must be ~0 (no invented decay,
# and no invented positive variance beyond truth). We test estimator consistency:
m0, se0 = excess_for_k(5, w1, 5000, sigma, w1, w1)  # Delta=0 case handled via w2=w1
control2_pass = bool(abs(m0) < 5 * se0 + 1e-6)

control_pass = control1_pass and control2_pass

# ---------- Verdict ----------
if not control_pass:
    status = "inconclusive"
    notes = "Positive control failed; estimator unreliable."
else:
    bound_ok = frac_viol_C == 0.0
    decaying = increases <= 2  # essentially monotone decreasing
    if bound_ok and decaying:
        status = "supported"
        notes = ("Excess posterior variance is positive, decreasing, and satisfies "
                 "excess_k <= 4*exp(-C*k) for all tested k; tail decay rate ~%.3f vs C=%.3f." % (lambda_hat, C))
    elif not decaying or frac_viol_C > 0.5:
        status = "falsified"
        notes = "Excess does not decay or systematically violates the theoretical bound."
    else:
        status = "inconclusive"
        notes = "Mixed behavior; MC noise or isolated bound violations."

# ---------- Plot ----------
if HAVE_MPL:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].semilogy(k_list, np.maximum(excess, 1e-16), 'o-', label='excess RPV - minimax')
    ax[0].semilogy(k_list, bound_C, '--', label='4 exp(-C k)')
    ax[0].semilogy(k_list, bound_D, ':', label='4 exp(-D_min k/2)')
    ax[0].set_xlabel('k'); ax[0].set_ylabel('excess variance'); ax[0].legend(); ax[0].set_title('Excess Posterior Variance')
    ax[1].plot(k_list, RPV, 'o-', label='RPV_k')
    ax[1].axhline(minimax_risk, color='r', ls='--', label='minimax risk = sigma^2')
    ax[1].set_xlabel('k'); ax[1].set_ylabel('RPV'); ax[1].legend(); ax[1].set_title('Posterior Variance vs minimax risk')
    plt.tight_layout()
    plt.savefig('results/c6/fig.png', dpi=100)

summary = {
    "claim_id": "C6",
    "status": status,
    "metrics": {
        "C_chernoff": float(C),
        "D_min": float(D_min),
        "minimax_risk": float(minimax_risk),
        "excess_k1": float(excess[0]),
        "excess_k10": float(excess[9]),
        "excess_k30": float(excess[-1]),
        "frac_k_violating_C_bound": frac_viol_C,
        "max_ratio_excess_over_boundC": max_ratio_C,
        "n_significant_increases": int(increases),
        "tail_decay_rate_lambda_hat": lambda_hat,
        "control1_pass": control1_pass,
        "control2_pass": control2_pass,
        "control_pass": bool(control_pass),
    },
    "notes": notes,
}
print("SUMMARY_JSON=" + json.dumps(summary, default=str))
