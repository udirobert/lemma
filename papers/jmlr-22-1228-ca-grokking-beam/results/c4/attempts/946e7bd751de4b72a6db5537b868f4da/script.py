import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Fixed audit parameters from the claim/test plan
N = 10
M = 2 * N
EPSILON = 1.01
A = EPSILON
LAMBDA2 = 0.1
DS = [5, 10, 20, 50]
TRIALS = 1000
RNG = np.random.default_rng(20240724)


def sample_ball(rng, n, d):
    """Uniform samples in the d-dimensional unit ball."""
    vec = rng.normal(size=(n, d))
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    unit = vec / np.where(norms == 0, 1.0, norms)
    radius = rng.random(n) ** (1.0 / d)
    return unit * radius[:, None]


def trial_groks(d):
    pos = sample_ball(RNG, N, d) + np.array([A] + [0.0] * (d - 1))
    neg = sample_ball(RNG, N, d) - np.array([A] + [0.0] * (d - 1))
    X = np.vstack([pos, neg])
    y = np.array([1.0] * N + [-1.0] * N)

    bar_x = X.mean(axis=0)
    Xc = X - bar_x
    c = (Xc.T @ y) / M
    A_mat = (Xc.T @ Xc) / M + LAMBDA2 * np.eye(d)
    w = np.linalg.solve(A_mat, c)
    b = -float(w @ bar_x)

    w1 = float(w[0])
    norm_w = float(np.linalg.norm(w))
    margin = A * w1 - norm_w
    return bool(w1 > 0.0 and margin >= abs(b) - 1e-12)


# Main simulation
P = {}
for d in DS:
    count = 0
    for _ in range(TRIALS):
        if trial_groks(d):
            count += 1
    P[d] = count / TRIALS

p_values = [P[d] for d in DS]
monotonic_decrease = bool(all(p_values[i] > p_values[i + 1] for i in range(len(p_values) - 1)))
monotonic_increase = bool(all(p_values[i] < p_values[i + 1] for i in range(len(p_values) - 1)))

# Log-linear fit on nonzero estimates
nonzero = [(d, P[d]) for d in DS if P[d] > 0]
if len(nonzero) >= 2:
    xs = np.array([z[0] for z in nonzero], dtype=float)
    ys = np.log(np.array([z[1] for z in nonzero], dtype=float))
    slope, intercept = np.polyfit(xs, ys, 1)
    pred = slope * xs + intercept
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2_log = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0
else:
    slope = float('nan')
    intercept = float('nan')
    r2_log = float('nan')

pass_main = bool(
    monotonic_decrease
    and np.isfinite(slope)
    and slope < 0
    and np.isfinite(r2_log)
    and r2_log >= 0.9
)

# Positive control: exact exponential decay, which must satisfy the same trend test
control_ds = [5, 10, 20, 50]
control_p = [float(np.exp(-0.1 * d)) for d in control_ds]
control_monotonic = bool(all(control_p[i] > control_p[i + 1] for i in range(len(control_p) - 1)))
cx = np.array(control_ds, dtype=float)
cy = np.log(np.array(control_p, dtype=float))
c_slope, c_intercept = np.polyfit(cx, cy, 1)
c_pred = c_slope * cx + c_intercept
c_ss_res = float(np.sum((cy - c_pred) ** 2))
c_ss_tot = float(np.sum((cy - cy.mean()) ** 2))
c_r2 = float(1.0 - c_ss_res / c_ss_tot)
control_pass = bool(control_monotonic and c_slope < 0 and c_r2 >= 0.99)

# Status decision
if not control_pass:
    status = 'inconclusive'
elif pass_main:
    status = 'supported'
elif monotonic_increase:
    status = 'falsified'
else:
    status = 'inconclusive'

# Plot
os.makedirs('results/c4', exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(DS, p_values, 'o-', label='estimated P(D)')
axes[0].set_xlabel('D')
axes[0].set_ylabel('grokking probability')
axes[0].set_title('Grokking probability vs D')
axes[0].grid(True, alpha=0.3)

nz_xs = [z[0] for z in nonzero]
nz_ys = [z[1] for z in nonzero]
if len(nz_xs) >= 2:
    axes[1].plot(nz_xs, np.log(nz_ys), 'o-', label='ln P(D)')
    fit_xs = np.array(nz_xs, dtype=float)
    axes[1].plot(fit_xs, slope * fit_xs + intercept, '--', label='linear fit')
axes[1].set_xlabel('D')
axes[1].set_ylabel('ln P(D)')
axes[1].set_title('Log probability vs D')
axes[1].grid(True, alpha=0.3)
axes[1].legend()
fig.tight_layout()
fig.savefig('results/c4/fig.png', dpi=150)
plt.close(fig)

summary = {
    'claim_id': 'C4',
    'status': status,
    'metrics': {
        'P_D5': float(P[5]),
        'P_D10': float(P[10]),
        'P_D20': float(P[20]),
        'P_D50': float(P[50]),
        'monotonic_decrease': monotonic_decrease,
        'log_slope': float(slope),
        'log_r2': float(r2_log),
        'trials_per_D': TRIALS,
        'control_pass': bool(control_pass),
        'control_r2': float(c_r2),
    },
    'notes': 'Full D-dimensional ridge-regression minimizer test-zero condition; control is exact exponential decay.',
}
print('SUMMARY_JSON=' + json.dumps(summary, default=str))
