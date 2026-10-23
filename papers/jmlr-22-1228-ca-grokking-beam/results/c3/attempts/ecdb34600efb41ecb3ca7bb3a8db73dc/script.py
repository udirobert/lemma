import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLAIM_ID = 'C3'
OUT_DIR = 'results/c3'
os.makedirs(OUT_DIR, exist_ok=True)


def sample_ball(rng, n, D, center, radius=1.0):
    # Uniform sampling in a D-dimensional ball.
    Z = rng.normal(size=(n, D))
    norms = np.linalg.norm(Z, axis=1)
    norms[norms == 0] = 1.0
    U = rng.random(n) ** (1.0 / D)
    pts = (Z / norms[:, None]) * (U * radius)[:, None]
    return pts + np.asarray(center, dtype=float)[None, :]


def make_data(rng, N, D, eps, radius=1.0):
    pos = sample_ball(rng, N, D, np.array([eps] + [0.0] * (D - 1)), radius)
    neg = sample_ball(rng, N, D, np.array([-eps] + [0.0] * (D - 1)), radius)
    X = np.vstack([pos, neg])
    y = np.concatenate([np.ones(N), -np.ones(N)])
    return X, y


def train_l2(X, y, lam2, N):
    M, D = X.shape
    A = np.column_stack([X, np.ones(M)])
    # (A^T A / N + lam2 * diag(1..1,0)) theta = A^T y / N
    Mmat = (A.T @ A) / N
    Mmat[:D, :D] += lam2 * np.eye(D)
    rhs = (A.T @ y) / N
    theta = np.linalg.solve(Mmat, rhs)
    return theta[:D], float(theta[D])


def soft_threshold(z, kappa):
    return np.sign(z) * np.maximum(np.abs(z) - kappa, 0.0)


def train_l1(X, y, lam1, lam2, N, max_iter=3000, tol=1e-10):
    M, D = X.shape
    # Lipschitz constant of smooth part gradient.
    A = np.column_stack([X, np.ones(M)])
    H = (A.T @ A) / N
    H[:D, :D] += lam2 * np.eye(D)
    L = float(np.linalg.eigvalsh(H).max())
    eta = 0.99 / max(L, 1e-12)

    w = np.zeros(D)
    b = 0.0
    for _ in range(max_iter):
        res = X @ w + b - y
        grad_w = (X.T @ res) / N + lam2 * w
        grad_b = np.sum(res) / N
        w_new = soft_threshold(w - eta * grad_w, eta * lam1)
        b_new = b - eta * grad_b
        if np.max(np.abs(w_new - w)) < tol and abs(b_new - b) < tol:
            w, b = w_new, b_new
            break
        w, b = w_new, b_new
    return w, float(b)


def zero_test_error(w, b, eps, radius=1.0):
    # Exact condition for a linear classifier to separate two D-balls.
    nw = float(np.linalg.norm(w))
    if nw < 1e-12:
        return False
    # Positive ball: center +eps e1, radius r.
    min_pos = eps * w[0] - radius * nw + b
    # Negative ball: center -eps e1, radius r.
    max_neg = -eps * w[0] + radius * nw + b
    return bool(min_pos >= -1e-7 and max_neg <= 1e-7)


def run_experiment(D, N, eps, lam1, lam2, n_trials, seed):
    rng = np.random.default_rng(seed)
    grok = 0
    for _ in range(n_trials):
        X, y = make_data(rng, N, D, eps)
        if lam1 == 0.0:
            w, b = train_l2(X, y, lam2, N)
        else:
            w, b = train_l1(X, y, lam1, lam2, N)
        if zero_test_error(w, b, eps):
            grok += 1
    return grok / n_trials


# Main experiment from the claim/test plan.
D, N, eps = 10, 20, 1.01
lam2 = 0.1
n_trials = 300
p_l2 = run_experiment(D, N, eps, 0.0, lam2, n_trials, 12345)
p_l1 = run_experiment(D, N, eps, 0.1, lam2, n_trials, 67890)

# Positive control: a regime where the paper's trend is even stronger
# (larger D), so L1 should clearly exceed L2.
Dc, Nc, epsc = 20, 10, 1.01
n_control = 100
p_l2_c = run_experiment(Dc, Nc, epsc, 0.0, lam2, n_control, 11111)
p_l1_c = run_experiment(Dc, Nc, epsc, 0.1, lam2, n_control, 22222)
control_pass = bool(p_l1_c > p_l2_c)

# Plot.
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(['L2 (lambda1=0)', 'L1 (lambda1=0.1)'], [p_l2, p_l1], color=['tab:blue', 'tab:orange'])
ax.set_ylabel('Grokking probability')
ax.set_title(f'D={D}, N={N}, eps={eps}, n={n_trials}')
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig.png'), dpi=150)
plt.close(fig)

if not control_pass:
    status = 'inconclusive'
    notes = 'Positive control failed; statistic/implementation may be buggy.'
elif p_l1 > p_l2:
    status = 'supported'
    notes = 'Empirical grokking probability with L1 exceeds that with L2 for the tested parameters.'
else:
    status = 'falsified'
    notes = 'Empirical grokking probability with L1 did not exceed that with L2.'

summary = {
    'claim_id': CLAIM_ID,
    'status': status,
    'metrics': {
        'p_grok_l1': float(p_l1),
        'p_grok_l2': float(p_l2),
        'n_trials': int(n_trials),
        'D': int(D),
        'N': int(N),
        'epsilon': float(eps),
        'lambda2': float(lam2),
        'ratio_l1_over_l2': float(p_l1 / (p_l2 + 1e-12)),
        'control_p_grok_l1': float(p_l1_c),
        'control_p_grok_l2': float(p_l2_c),
        'control_pass': bool(control_pass),
    },
    'notes': notes,
}
print('SUMMARY_JSON=' + json.dumps(summary, default=str))
