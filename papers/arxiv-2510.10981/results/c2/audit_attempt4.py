import json, os
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

np.random.seed(0)
OUT = 'results/c2'

# ---------------- Synthetic meta-learning setup ----------------
# Task family: linear regression y = w.x + b + eps, w ~ N(0, I_d), b ~ N(0,1)
# Uniform-attention model: phi_bar = mean of features phi(x,y) = [y*x, y, x];
# predictor y_hat = theta . psi(phi_bar, x_q) with a feature map rich enough to
# represent ridge/OLS-like estimators. theta is meta-trained by least squares
# over N prompts (this is where N-dependence enters).

d = 4
SIG = 0.5          # noise std
N_EVAL = 400       # eval prompts per grid point
Q_EVAL = 8         # fresh queries per eval prompt

def gen_prompt(rng, p):
    w = rng.normal(size=d)
    b = rng.normal()
    X = rng.normal(size=(p, d))
    y = X @ w + b + SIG * rng.normal(size=p)
    return w, b, X, y

def phi_feats(X, y):
    # per-example features: y*x (d), y (1), x (d)
    return np.concatenate([y[:, None] * X, y[:, None], X], axis=1)

def psi_feats(phi_bar, xq):
    # readout features: phi_bar .* [xq,1,xq] blocks -> lets model form (sum yx).xq etc.
    pb1 = phi_bar[:d]          # mean y*x
    pb2 = phi_bar[d:d+1]       # mean y
    pb3 = phi_bar[d+1:]        # mean x
    return np.concatenate([pb1, pb2, pb3, pb1 * xq, pb3 * xq, xq, [1.0]])

PSI_DIM = d + 1 + d + d + d + d + 1

def meta_train(rng, N, p, ridge=1e-3):
    A, Y = [], []
    for _ in range(N):
        w, b, X, y = gen_prompt(rng, p)
        phibar = phi_feats(X, y).mean(axis=0)
        for _ in range(4):
            xq = rng.normal(size=d)
            yq = xq @ w + b + SIG * rng.normal()
            A.append(psi_feats(phibar, xq))
            Y.append(yq)
    A = np.array(A); Y = np.array(Y)
    theta = np.linalg.solve(A.T @ A + ridge * np.eye(PSI_DIM), A.T @ Y)
    return theta

def bayes_pred(X, y, xq):
    # exact posterior mean predictor for linear model with N(0,I) prior
    A = X.T @ X + (SIG ** 2) * np.eye(d)
    w_post = np.linalg.solve(A, X.T @ y)
    b_post = y.mean() - X.mean(axis=0) @ w_post  # crude intercept via centering
    # proper joint posterior with intercept prior var 1:
    Z = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    pr = np.diag([1.0 / (SIG ** 2)] * d + [1.0 / (SIG ** 2)]) * (SIG ** 2)
    M = Z.T @ Z + SIG ** 2 * np.eye(d + 1)
    coef = np.linalg.solve(M, Z.T @ y)
    return np.concatenate([xq, [1.0]]) @ coef

def bayes_gap(rng, theta, p):
    sq_m, sq_b = [], []
    for _ in range(N_EVAL):
        w, b, X, y = gen_prompt(rng, p)
        phibar = phi_feats(X, y).mean(axis=0)
        for _ in range(Q_EVAL):
            xq = rng.normal(size=d)
            yq = xq @ w + b + SIG * rng.normal()
            ym = psi_feats(phibar, xq) @ theta
            yb = bayes_pred(X, y, xq)
            sq_m.append((yq - ym) ** 2)
            sq_b.append((yq - yb) ** 2)
    sq_m = np.array(sq_m); sq_b = np.array(sq_b)
    diff = sq_m - sq_b
    return diff.mean(), diff.std() / np.sqrt(len(diff))

# ---------------- Sweep grid ----------------
Ns = [100, 500, 1000, 2000]
ps = [5, 10, 15, 20]
BG = np.zeros((len(Ns), len(ps)))
SE = np.zeros_like(BG)
rng = np.random.default_rng(12345)
for i, N in enumerate(Ns):
    for j, p in enumerate(ps):
        theta = meta_train(rng, N, p)
        bg, se = bayes_gap(rng, theta, p)
        BG[i, j] = bg
        SE[i, j] = se

Nv = np.repeat(np.array(Ns, float), len(ps))
pv = np.tile(np.array(ps, float), len(Ns))
yv = BG.ravel()

# ---------------- Model fits (deterministic: grid over exponent + linear LS) ----------------
def fit_powerlaw(x1, x2, y, betas):
    # y = a + b*x1^-beta (+ c*x2 if x2 given)
    best = None
    for beta in betas:
        cols = [np.ones_like(y), x1 ** (-beta)]
        if x2 is not None:
            cols.append(x2)
        A = np.column_stack(cols)
        coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
        rss = float(((y - A @ coef) ** 2).sum())
        if best is None or rss < best[0]:
            best = (rss, beta, coef)
    rss, beta, coef = best
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - rss / tss
    return r2, beta, coef

betas = np.linspace(0.05, 2.0, 80)
r2_joint, beta_j, coef_j = fit_powerlaw(Nv * pv, 1.0 / Nv, yv, betas)
r2_n, beta_n, coef_n = fit_powerlaw(Nv, None, yv, betas)
r2_p, beta_p, coef_p = fit_powerlaw(pv, None, yv, betas)

# ---------------- Positive control ----------------
# Synthetic data known to follow the joint law exactly + small noise
rngc = np.random.default_rng(7)
a_t, b_t, beta_t, c_t = 0.05, 2.0, 0.5, 1.0
yc = a_t + b_t * (Nv * pv) ** (-beta_t) + c_t / Nv + 0.005 * rngc.normal(size=len(Nv))
r2_c, beta_c, coef_c = fit_powerlaw(Nv * pv, 1.0 / Nv, yc, betas)
r2_c_n, _, _ = fit_powerlaw(Nv, None, yc, betas)
r2_c_p, _, _ = fit_powerlaw(pv, None, yc, betas)
control_pass = bool(r2_c > 0.9 and abs(beta_c - beta_t) <= 0.15 and
                    r2_c - max(r2_c_n, r2_c_p) > 0.05)

# ---------------- Verdict ----------------
if not control_pass:
    status = 'inconclusive'
    notes = 'Positive control failed; fit pipeline untrustworthy.'
else:
    crit = (r2_joint > 0.8) and (r2_joint - r2_n > 0.1) and (r2_joint - r2_p > 0.1)
    status = 'supported' if crit else 'falsified'
    notes = (f'Joint R2={r2_joint:.3f}, N-only R2={r2_n:.3f}, p-only R2={r2_p:.3f}. '
             f'Criterion requires joint>0.8 and >0.1 above both single-variable models.')

metrics = {
    'r2_joint': float(r2_joint), 'r2_n_only': float(r2_n), 'r2_p_only': float(r2_p),
    'joint_a': float(coef_j[0]), 'joint_b': float(coef_j[1]),
    'joint_beta': float(beta_j), 'joint_c': float(coef_j[2]),
    'bg_values': [float(v) for v in yv],
    'bg_mc_stderr_max': float(SE.max()),
    'control_r2': float(r2_c), 'control_beta_recovered': float(beta_c),
    'control_beta_true': beta_t, 'control_pass': control_pass,
}
summary = {'claim_id': 'C2', 'status': status, 'metrics': metrics, 'notes': notes}

if HAVE_MPL:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, N in enumerate(Ns):
        axes[0].plot(ps, BG[i], 'o-', label=f'N={N}')
    axes[0].set_xlabel('p'); axes[0].set_ylabel('Bayes Gap'); axes[0].legend(); axes[0].set_title('BG vs p')
    axes[1].bar(['joint', 'N-only', 'p-only'], [r2_joint, r2_n, r2_p])
    axes[1].axhline(0.8, color='r', ls='--'); axes[1].set_ylabel('R^2'); axes[1].set_title('Model comparison')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig.png'), dpi=100)

print('SUMMARY_JSON=' + json.dumps(summary, default=str))
