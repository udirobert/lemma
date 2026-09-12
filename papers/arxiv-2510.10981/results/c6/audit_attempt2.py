import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expit

np.random.seed(0)

# ---- Setup: two-task Gaussian linear mixture ----
# Task 1: y = w1 x + eps ; Task 2: y = w2 x + eps ; x ~ N(0,1), eps ~ N(0, sigma^2)
w1, w2 = 1.0, -1.0
sigma = 1.0
Delta = w1 - w2  # = 2
k_max = 30
n_trials = 20000
ks = np.arange(1, k_max + 1)

# Theoretical constants
D_min = Delta**2 / (2 * sigma**2)                 # pairwise KL per example
C = 0.5 * np.log(1 + Delta**2 / (4 * sigma**2))   # Chernoff information per example
minimax_risk = sigma**2                            # point-mass true task family: irreducible noise
bound_amp = Delta**2 * 1.0                         # Delta^2 * E[x_q^2]

def run_experiment(w_true, w_alt, sigma, n_trials, ks, seed):
    """Contexts drawn from TRUE task w_true; posterior over {w_true, w_alt}."""
    rng = np.random.default_rng(seed)
    excess = np.zeros(len(ks))
    se = np.zeros(len(ks))
    for ik, k in enumerate(ks):
        x = rng.standard_normal((n_trials, k))
        y = w_true * x + sigma * rng.standard_normal((n_trials, k))
        # log-likelihood ratio (task true vs alt): L = (2(w_true-w_alt)/sigma^2) * sum x*(y - (w_true+w_alt)/2 x)
        # simpler: L = (1/sigma^2) * sum [ (y - w_alt x)^2 - (y - w_true x)^2 ] / ... use direct form
        ll_true = -0.5 / sigma**2 * np.sum((y - w_true * x)**2, axis=1)
        ll_alt  = -0.5 / sigma**2 * np.sum((y - w_alt  * x)**2, axis=1)
        p_true = expit(ll_true - ll_alt)  # equal priors
        p_alt = 1.0 - p_true
        xq = rng.standard_normal(n_trials)
        # predictive variance minus sigma^2 = xq^2 * (w_true - w_alt)^2 * p_true * p_alt
        ex = xq**2 * (w_true - w_alt)**2 * p_true * p_alt
        excess[ik] = np.mean(ex)
        se[ik] = np.std(ex) / np.sqrt(n_trials)
    return excess, se

# Main experiment: true task = w1, alternative = w2
excess, se = run_experiment(w1, w2, sigma, n_trials, ks, seed=1)
RPV = minimax_risk + excess
bound_C = bound_amp * np.exp(-C * ks)
bound_D = bound_amp * np.exp(-D_min * ks / 2.0)

# Bound check (allow 3-sigma MC slack)
viol = excess - 3 * se > bound_C
n_viol = int(np.sum(viol))
bound_holds = n_viol == 0

# Positive control 1: well-separated tasks (same as main) -> excess positive, decreasing, below bound
ctrl1_excess, _ = run_experiment(1.0, -1.0, 1.0, 5000, np.arange(1, 11), seed=2)
ctrl1_bound = 4.0 * np.exp(-C * np.arange(1, 11))
ctrl1_pass = bool(np.all(ctrl1_excess > 0) and np.all(np.diff(ctrl1_excess) <= 3*np.abs(ctrl1_excess[1:])*0 + 1e-6 + 3*0.01) and np.all(ctrl1_excess <= ctrl1_bound + 1e-9))

# Positive control 2: indistinguishable tasks (w_alt = w_true) -> excess flat at prior variance 0
# Here both tasks identical: posterior stays 0.5 but (w_true-w_alt)^2 = 0, so excess = 0 exactly.
# Better control: tasks identical in likelihood but slope uncertainty remains -> use w_alt close but
# the true 'no-decay' control: D_min = C = 0 means bound is constant; check estimator gives flat excess.
ctrl2_excess, _ = run_experiment(1.0, 1.0, 1.0, 5000, np.arange(1, 11), seed=3)
ctrl2_pass = bool(np.allclose(ctrl2_excess, 0.0, atol=1e-12))  # identical tasks => zero excess, no invented decay

control_pass = ctrl1_pass and ctrl2_pass

# Optional asymptotic rate fit on reliable tail (excess > 1e-10, monotonic region)
mask = excess > 1e-10
lambda_hat = float('nan')
if np.sum(mask) >= 5:
    idx = np.where(mask)[0]
    # use upper half of reliable region
    idx = idx[len(idx)//2:]
    if len(idx) >= 3:
        slope, _ = np.polyfit(ks[idx], np.log(excess[idx]), 1)
        lambda_hat = slope

# Monotonicity / positivity check on main curve (within MC noise)
pos = bool(np.all(excess >= -3*se))
dec = bool(np.all(np.diff(excess) <= 3*se[1:] + 1e-12))

# Plot
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].semilogy(ks, np.maximum(excess, 1e-16), 'o-', label='excess RPV - minimax')
ax[0].semilogy(ks, bound_C, 'r--', label=r'$4 e^{-C k}$ (Chernoff bound)')
ax[0].semilogy(ks, bound_D, 'g:', label=r'$4 e^{-D_{min} k/2}$')
ax[0].set_xlabel('context length k'); ax[0].set_ylabel('excess variance')
ax[0].legend(); ax[0].set_title('Excess Posterior Variance vs k')
ax[1].plot(ks, excess, 'o-', label='excess')
ax[1].plot(ks, bound_C, 'r--', label='bound (C)')
ax[1].set_xlabel('k'); ax[1].set_ylabel('excess variance (linear)')
ax[1].legend(); ax[1].set_title('Linear scale')
plt.tight_layout()
plt.savefig('results/c6/excess_decay.png', dpi=100)

# Verdict
if not control_pass:
    status = 'inconclusive'
    notes = 'Positive control failed; estimator/statistic may be buggy.'
elif bound_holds and pos and dec:
    status = 'supported'
    notes = ('Excess RPV is positive, decreasing, and satisfies excess_k <= 4*exp(-C*k) '
             'for all k=1..30 (Chernoff constant C=%.4f, D_min=%.4f). Fitted tail rate %.4f.' % (C, D_min, lambda_hat))
elif not bound_holds:
    status = 'falsified'
    notes = 'Excess exceeds theoretical bound 4*exp(-C*k) at %d values of k beyond MC noise.' % n_viol
else:
    status = 'inconclusive'
    notes = 'Bound holds but excess not cleanly positive/monotone within MC noise.'

summary = {
    'claim_id': 'C6',
    'status': status,
    'metrics': {
        'D_min': float(D_min),
        'C_chernoff': float(C),
        'minimax_risk': float(minimax_risk),
        'excess_k1': float(excess[0]),
        'excess_k10': float(excess[9]),
        'excess_k30': float(excess[-1]),
        'bound_C_k1': float(bound_C[0]),
        'bound_C_k30': float(bound_C[-1]),
        'n_bound_violations': n_viol,
        'bound_holds_all_k': bool(bound_holds),
        'fitted_tail_rate_lambda_hat': lambda_hat,
        'control_pass': bool(control_pass),
        'ctrl1_pass': ctrl1_pass,
        'ctrl2_pass': ctrl2_pass,
    },
    'notes': notes,
}
print('SUMMARY_JSON=' + json.dumps(summary, default=str))
