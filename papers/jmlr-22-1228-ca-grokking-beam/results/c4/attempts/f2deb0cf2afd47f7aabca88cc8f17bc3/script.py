import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLAIM_ID = 'C4'
OUT_DIR = 'results/c4'
os.makedirs(OUT_DIR, exist_ok=True)

# Fixed parameters from the claim/test plan.
N = 10
EPSILON = 1.01
LAMBDA2 = 0.1
LAMBDA1 = 0.0
D_VALUES = [5, 10, 20, 50]
TRIALS_PER_D = 100000
BATCH_SIZE = 500
RNG_SEED_BASE = 20240724


def sample_batch(rng, batch, n, d, eps):
    m = 2 * n
    total = batch * m
    dirs = rng.normal(size=(total, d))
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / norms
    u = rng.random(total)
    r = u ** (1.0 / d)
    pts = (dirs * r[:, None]).reshape(batch, m, d)
    pts[:, :n, 0] += eps
    pts[:, n:, 0] -= eps
    labels = np.ones((batch, m), dtype=float)
    labels[:, n:] = -1.0
    return pts, labels


def wilson_ci(count, n, z=1.959963984540054):
    if n <= 0:
        return (0.0, 0.0)
    p = count / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (float(center - half), float(center + half))


def fit_log_linear(xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    A = np.vstack([xs, np.ones_like(xs)]).T
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(coef[0]), float(coef[1]), float(r2)


counts = []
probabilities = []
cis = []

for d in D_VALUES:
    rng = np.random.default_rng(RNG_SEED_BASE + int(d))
    count = 0
    n_batches = (TRIALS_PER_D + BATCH_SIZE - 1) // BATCH_SIZE
    for _ in range(n_batches):
        this_batch = min(BATCH_SIZE, TRIALS_PER_D - _ * BATCH_SIZE)
        if this_batch <= 0:
            break
        pts, labels = sample_batch(rng, this_batch, N, d, EPSILON)
        for i in range(this_batch):
            X = pts[i]
            y = labels[i]
            m = 2 * N
            xbar = X.mean(axis=0)
            g = X.T @ y
            Z = X - xbar[None, :]
            A_mat = np.eye(m) + Z @ Z.T
            q = np.linalg.solve(A_mat, Z @ g)
            w = (g - Z.T @ q) / (N * LAMBDA2)
            b = float(-(w @ xbar))
            margin = EPSILON * float(w[0]) - float(np.linalg.norm(w))
            if w[0] > 0.0 and margin >= abs(b) - 1e-12:
                count += 1
    p = count / TRIALS_PER_D
    counts.append(int(count))
    probabilities.append(float(p))
    cis.append(wilson_ci(count, TRIALS_PER_D))

probabilities = np.asarray(probabilities, dtype=float)
monotonic_strict = bool(all(probabilities[i] > probabilities[i + 1] for i in range(len(probabilities) - 1)))
nonzero_dimensions = int(np.sum(np.asarray(counts) > 0))
min_count = int(min(counts)) if counts else 0

if nonzero_dimensions == len(D_VALUES):
    log_slope, log_intercept, log_r2 = fit_log_linear(D_VALUES, np.log(probabilities))
else:
    log_slope, log_intercept, log_r2 = (None, None, None)

# Positive control: exact exponential trend.
control_ds = np.asarray(D_VALUES, dtype=float)
control_p = np.exp(-0.1 * control_ds)
control_monotonic = bool(all(control_p[i] > control_p[i + 1] for i in range(len(control_p) - 1)))
control_slope, control_intercept, control_r2 = fit_log_linear(control_ds, np.log(control_p))
control_pass = bool(control_monotonic and control_slope < 0 and control_r2 >= 0.9)

# Status logic.
if not control_pass:
    status = 'inconclusive'
elif nonzero_dimensions < len(D_VALUES):
    status = 'inconclusive'
elif min_count < 20:
    status = 'inconclusive'
elif not monotonic_strict:
    status = 'falsified'
elif log_slope is None or log_slope >= 0 or log_r2 < 0.9:
    status = 'falsified'
else:
    status = 'supported'

# Plot.
fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(
    D_VALUES,
    probabilities,
    yerr=[
        [probabilities[i] - cis[i][0] for i in range(len(D_VALUES))],
        [cis[i][1] - probabilities[i] for i in range(len(D_VALUES))],
    ],
    fmt='o-', capsize=4, label='Estimated P(D)',
)
ax.set_yscale('log')
ax.set_xlabel('Dimension D')
ax.set_ylabel('Grokking probability P(D)')
ax.set_title('C4: Grokking probability vs. dimension')
ax.grid(True, which='both', alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig.png'), dpi=150)
plt.close(fig)

metrics = {
    'claim_id': CLAIM_ID,
    'trials_per_D': int(TRIALS_PER_D),
    'D_values': [int(x) for x in D_VALUES],
    'count_D5': int(counts[0]),
    'count_D10': int(counts[1]),
    'count_D20': int(counts[2]),
    'count_D50': int(counts[3]),
    'P_D5': float(probabilities[0]),
    'P_D10': float(probabilities[1]),
    'P_D20': float(probabilities[2]),
    'P_D50': float(probabilities[3]),
    'expected_successes_D5': float(TRIALS_PER_D * probabilities[0]),
    'expected_successes_D10': float(TRIALS_PER_D * probabilities[1]),
    'expected_successes_D20': float(TRIALS_PER_D * probabilities[2]),
    'expected_successes_D50': float(TRIALS_PER_D * probabilities[3]),
    'CI95_D5': [float(cis[0][0]), float(cis[0][1])],
    'CI95_D10': [float(cis[1][0]), float(cis[1][1])],
    'CI95_D20': [float(cis[2][0]), float(cis[2][1])],
    'CI95_D50': [float(cis[3][0]), float(cis[3][1])],
    'monotonic_decrease': bool(monotonic_strict),
    'nonzero_dimensions': int(nonzero_dimensions),
    'min_count': int(min_count),
    'log_slope': None if log_slope is None else float(log_slope),
    'log_intercept': None if log_intercept is None else float(log_intercept),
    'log_r2': None if log_r2 is None else float(log_r2),
    'control_pass': bool(control_pass),
    'control_r2': float(control_r2),
    'control_slope': float(control_slope),
}

summary = {
    'claim_id': CLAIM_ID,
    'status': status,
    'metrics': metrics,
    'notes': 'Ridge-regression fixed point for lambda1=0; uniform balls centered at +/-epsilon with radius 1; grokking iff w1>0 and epsilon*w1 - ||w|| >= |b|.',
}
print('SUMMARY_JSON=' + json.dumps(summary, default=str))
