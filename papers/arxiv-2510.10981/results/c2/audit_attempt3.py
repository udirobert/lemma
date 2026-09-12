import json, os
import numpy as np
from scipy.optimize import curve_fit

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

np.random.seed(0)
OUT = 'results/c2'

# ---------------- Data-generating process (Definition 2.1, single linear family) ----------------
D = 4            # input dimension
SIG = 0.3        # noise std
TAU2 = 1.0       # prior variance of w

def sample_prompt(rng, p):
    w = rng.normal(0, np.sqrt(TAU2), size=D)
    b = rng.normal(0, 1.0)
    X = rng.normal(0, 1, size=(p, D))
    y = X @ w + b + rng.normal(0, SIG, size=p)
    xq = rng.normal(0, 1, size=D)
    yq = xq @ w + b + rng.normal(0, SIG)
    return X, y, xq, yq

def bayes_pred(X, y, xq):
    # exact posterior mean predictor for linear model with Gaussian prior
    A = X.T @ X + (SIG**2 / TAU2) * np.eye(D)
    w_post = np.linalg.solve(A, X.T @ y)
    b_post = np.mean(y - X @ w_post)
    return xq @ w_post + b_post

# ---------------- Uniform-attention model (Definition 2.2 simplified) ----------------
# phi(x,y) fixed feature map; model: yhat = theta^T psi(phi_bar, xq); theta meta-trained on N prompts.
def phi_map(x, y):
    return np.concatenate([x * y, [y, y * y, 1.0]])

def psi_map(phi_bar, xq):
    # features allowing representation of ridge-like estimator: phi_bar (dim D+3), xq, and
    # cross terms xq_j * phi_bar_j (so model can learn w_hat from context moments)
    pb = phi_bar
    cross = xq * pb[:D]          # xq_j * (x_j y)_bar  -> key term for OLS-like readout
    return np.concatenate([pb, xq, cross, [np.sum(xq * pb[:D])]])

PSI_DIM = (D + 3) + D + D + 1

def meta_train(N, p, rng, queries_per_prompt=4):
    # build training set of (context, query) pairs from N prompts
    Psis, Ys = [], []
    for _ in range(N):
        X, y, _, _ = sample_prompt(rng, p)
        for _ in range(queries_per_prompt):
            xq = rng.normal(0, 1, size=D)
            # target: noiseless query response is not observable; train on noisy yq
            w_true = None
            yq = None
            # regenerate yq consistent with prompt's task: resample task implicitly via bayes target
            # simpler: use the Bayes predictor target is circular; instead sample fresh task data properly
            Psis.append(None); Ys.append(None)
            break
        break
    return None  # placeholder (replaced below)

# The placeholder above is unused; proper implementation:
def build_train(N, p, rng, qpp=4):
    Psis, Ys = [], []
    for _ in range(N):
        w = rng.normal(0, np.sqrt(TAU2), size=D)
        b = rng.normal(0, 1.0)
        X = rng.normal(0, 1, size=(p, D))
        y = X @ w + b + rng.normal(0, SIG, size=p)
        phi_bar = np.mean([phi_map(X[i], y[i]) for i in range(p)], axis=0)
        for _ in range(qpp):
            xq = rng.normal(0, 1, size=D)
            yq = xq @ w + b + rng.normal(0, SIG)
            Psis.append(psi_map(phi_bar, xq))
            Ys.append(yq)
    return np.array(Psis), np.array(Ys)

def fit_theta(Psis, Ys, lam=1e-3):
    A = Psis.T @ Psis + lam * np.eye(PSI_DIM)
    return np.linalg.solve(A, Psis.T @ Ys)

def model_pred(theta, X, y, xq):
    p = X.shape[0]
    phi_bar = np.mean([phi_map(X[i], y[i]) for i in range(p)], axis=0)
    return psi_map(phi_bar, xq) @ theta

def compute_bayes_gap(N, p, seed, n_eval=300):
    rng = np.random.default_rng(seed)
    Psis, Ys = build_train(N, p, rng)
    theta = fit_theta(Psis, Ys)
    rng_e = np.random.default_rng(seed + 999)
    gaps = []
    for _ in range(n_eval):
        X, y, xq, yq = sample_prompt(rng_e, p)
        yh_m = model_pred(theta, X, y, xq)
        yh_b = bayes_pred(X, y, xq)
        # gap in excess risk: E[(yq-yh_m)^2 - (yq-yh_b)^2]; per-sample diff, averaged
        gaps.append((yq - yh_m)**2 - (yq - yh_b)**2)
    gaps = np.array(gaps)
    return float(np.mean(gaps)), float(np.std(gaps) / np.sqrt(len(gaps)))

# ---------------- Sweep grid ----------------
Ns = [100, 500, 1000, 2000]
Ps = [5, 10, 15, 20]
bg, bg_se, gridN, gridP = [], [], [], []
for N in Ns:
    for p in Ps:
        m, s = compute_bayes_gap(N, p, seed=1000 + N * 7 + p)
        bg.append(m); bg_se.append(s); gridN.append(N); gridP.append(p)
        print(f'N={N} p={p} BG={m:.5f} +- {s:.5f}')
bg = np.array(bg); gridN = np.array(gridN, float); gridP = np.array(gridP, float)
pN = gridN * gridP

def r2(y, yhat):
    ss_res = np.sum((y - yhat)**2); ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - ss_res / ss_tot

# ---------------- Robust joint fit: grid over beta, exact linear LS for (a,b,c) ----------------
def fit_joint(y, pN, N):
    best = None
    for beta in np.linspace(0.05, 2.0, 80):
        Xd = np.column_stack([np.ones_like(pN), pN**(-beta), 1.0 / N])
        coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        rss = np.sum((y - Xd @ coef)**2)
        if best is None or rss < best[0]:
            best = (rss, beta, coef, Xd @ coef)
    return best

def fit_single(y, x):
    best = None
    for g in np.linspace(0.05, 2.0, 80):
        Xd = np.column_stack([np.ones_like(x), x**(-g)])
        coef, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        rss = np.sum((y - Xd @ coef)**2)
        if best is None or rss < best[0]:
            best = (rss, g, coef, Xd @ coef)
    return best

rss_j, beta_j, coef_j, yh_j = fit_joint(bg, pN, gridN)
r2_joint = r2(bg, yh_j)
rss_n, g_n, coef_n, yh_n = fit_single(bg, gridN)
r2_n = r2(bg, yh_n)
rss_p, g_p, coef_p, yh_p = fit_single(bg, gridP)
r2_p = r2(bg, yh_p)

# ---------------- Positive control: synthetic data from the claimed law ----------------
rngc = np.random.default_rng(42)
a_t, b_t, beta_t, c_t = 0.05, 2.0, 0.5, 1.0
bg_true = a_t + b_t * pN**(-beta_t) + c_t / gridN
bg_syn = bg_true + rngc.normal(0, 0.01, size=len(bg_true))
rss_c, beta_c, coef_c, yh_c = fit_joint(bg_syn, pN, gridN)
r2_control = r2(bg_syn, yh_c)
control_pass = bool(r2_control > 0.9 and abs(beta_c - beta_t) <= 0.15)

# ---------------- Verdict ----------------
if not control_pass:
    status = 'inconclusive'
    notes = 'Positive control failed; fitting pipeline untrustworthy.'
else:
    crit = (r2_joint > 0.8) and (r2_joint - r2_n > 0.1) and (r2_joint - r2_p > 0.1)
    status = 'supported' if crit else 'falsified'
    notes = (f'Joint R2={r2_joint:.3f}, N-only R2={r2_n:.3f}, p-only R2={r2_p:.3f}. '
             f'Criterion: joint>0.8 and beats both single-var models by >0.1 -> {crit}.')

if HAVE_MPL:
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(pN, bg, c='k', label='measured BG')
    xs = np.sort(pN)
    ax[0].plot(xs, coef_j[0] + coef_j[1]*xs**(-beta_j) + coef_j[2]/np.sort(gridN)[0]*0 + 0, 'r--', alpha=0.3)
    ax[0].set_xscale('log'); ax[0].set_xlabel('pN'); ax[0].set_ylabel('Bayes Gap'); ax[0].legend()
    ax[1].bar(['joint', 'N-only', 'p-only'], [r2_joint, r2_n, r2_p])
    ax[1].axhline(0.8, color='r', ls='--'); ax[1].set_ylabel('R^2')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig.png'), dpi=100)

summary = {
    'claim_id': 'C2', 'status': status,
    'metrics': {
        'r2_joint': float(r2_joint), 'r2_n_only': float(r2_n), 'r2_p_only': float(r2_p),
        'joint_a': float(coef_j[0]), 'joint_b': float(coef_j[1]), 'joint_beta': float(beta_j), 'joint_c': float(coef_j[2]),
        'bg_values': [float(v) for v in bg], 'bg_mc_stderr_mean': float(np.mean(bg_se)),
        'control_r2': float(r2_control), 'control_a': float(coef_c[0]), 'control_b': float(coef_c[1]),
        'control_beta': float(beta_c), 'control_c': float(coef_c[2]), 'control_pass': control_pass,
    },
    'notes': notes,
}
print('SUMMARY_JSON=' + json.dumps(summary, default=str))
