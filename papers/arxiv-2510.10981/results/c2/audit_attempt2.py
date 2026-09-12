import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(0)

# ---------------- Setup ----------------
d = 4              # input dimension
sigma = 0.5        # label noise std
N_list = [100, 500, 1000, 2000]
p_list = [5, 10, 15, 20]
Q_train = 5        # queries per training prompt
N_EVAL = 200       # eval prompts per (N,p)
Q_EVAL = 5
N_SEEDS = 3

def gen_prompt(rng, p, nq):
    w = rng.normal(size=d)
    b = rng.normal()
    Xc = rng.normal(size=(p, d))
    yc = Xc @ w + b + sigma * rng.normal(size=p)
    Xq = rng.normal(size=(nq, d))
    yq = Xq @ w + b + sigma * rng.normal(size=nq)
    return Xc, yc, Xq, yq

def features(Xc, yc, Xq):
    # model: yhat = x_q^T W m + c^T x_q + e*my ; linear in params
    m = (Xc * yc[:, None]).mean(axis=0)   # d
    my = yc.mean()
    nq = Xq.shape[0]
    F = np.zeros((nq, d * d + d + 1))
    F[:, :d * d] = np.einsum('qd,e->qde', Xq, m).reshape(nq, d * d)
    F[:, d * d:d * d + d] = Xq
    F[:, -1] = my
    return F

def bayes_pred(Xc, yc, Xq):
    p = Xc.shape[0]
    A = Xc.T @ Xc + sigma**2 * np.eye(d)
    # include intercept via augmented feature
    Xa = np.hstack([Xc, np.ones((p, 1))])
    Aa = Xa.T @ Xa + sigma**2 * np.eye(d + 1)
    theta_post = np.linalg.solve(Aa, Xa.T @ yc)
    Xqa = np.hstack([Xq, np.ones((Xq.shape[0], 1))])
    return Xqa @ theta_post

def compute_bg(N, p, seed):
    rng = np.random.default_rng(seed)
    # meta-train: least squares over N prompts
    Fs, ys = [], []
    for _ in range(N):
        Xc, yc, Xq, yq = gen_prompt(rng, p, Q_train)
        Fs.append(features(Xc, yc, Xq))
        ys.append(yq)
    F = np.vstack(Fs); y = np.concatenate(ys)
    theta, *_ = np.linalg.lstsq(F, y, rcond=None)
    # out-of-sample eval
    se_m, se_b = [], []
    for _ in range(N_EVAL):
        Xc, yc, Xq, yq = gen_prompt(rng, p, Q_EVAL)
        yh_m = features(Xc, yc, Xq) @ theta
        yh_b = bayes_pred(Xc, yc, Xq)
        se_m.append(np.mean((yq - yh_m) ** 2))
        se_b.append(np.mean((yq - yh_b) ** 2))
    return np.mean(se_m) - np.mean(se_b)

# ---------------- Fit helpers (grid over exponent, exact linear LSQ) ----------------
def fit_power(x, y, extra=None):
    # y = a + b*x^-beta (+ c*extra). Returns best R2 and params.
    best = None
    for beta in np.linspace(0.05, 2.0, 80):
        cols = [np.ones_like(x), x ** (-beta)]
        if extra is not None:
            cols.append(extra)
        A = np.column_stack(cols)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        rss = np.sum(resid ** 2)
        if best is None or rss < best[0]:
            best = (rss, beta, coef)
    rss, beta, coef = best
    tss = np.sum((y - y.mean()) ** 2)
    r2 = 1 - rss / tss
    return r2, beta, coef

# ---------------- Positive control ----------------
rngc = np.random.default_rng(123)
Ns_c = np.array(N_list * len(p_list), dtype=float)
ps_c = np.array(sorted(p_list * len(N_list)), dtype=float)
a_t, b_t, beta_t, c_t = 0.05, 0.5, 0.5, 2.0
y_ctrl = a_t + b_t * (ps_c * Ns_c) ** (-beta_t) + c_t / Ns_c + rngc.normal(scale=0.005, size=len(Ns_c))
r2c, betac, coefc = fit_power(ps_c * Ns_c, y_ctrl, extra=1.0 / Ns_c)
control_pass = bool(r2c > 0.9 and abs(betac - beta_t) < 0.15)

# ---------------- Real experiment ----------------
Ns, ps, bgs, bg_se = [], [], [], []
for N in N_list:
    for p in p_list:
        vals = [compute_bg(N, p, seed=1000 + s) for s in range(N_SEEDS)]
        Ns.append(N); ps.append(p)
        bgs.append(np.mean(vals))
        bg_se.append(np.std(vals) / np.sqrt(N_SEEDS))
Ns = np.array(Ns, dtype=float); ps = np.array(ps, dtype=float)
bgs = np.array(bgs); bg_se = np.array(bg_se)

r2_joint, beta_j, coef_j = fit_power(ps * Ns, bgs, extra=1.0 / Ns)
r2_n, beta_n, _ = fit_power(Ns, bgs)
r2_p, beta_p, _ = fit_power(ps, bgs)

# ---------------- Verdict ----------------
if not control_pass:
    status = 'inconclusive'
    notes = 'Positive control failed (fitter unreliable); cannot judge claim.'
elif r2_joint > 0.8 and (r2_joint - r2_n) > 0.1 and (r2_joint - r2_p) > 0.1:
    status = 'supported'
    notes = 'Joint pN model fits well and beats both single-variable models by >0.1 R2.'
else:
    status = 'falsified'
    notes = 'Joint model does not meet criterion (R2>0.8 and >0.1 above both single-variable fits).'

# ---------------- Plot ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for N in N_list:
    msk = Ns == N
    axes[0].errorbar(ps[msk], bgs[msk], yerr=bg_se[msk], marker='o', label=f'N={N}')
axes[0].set_xlabel('p'); axes[0].set_ylabel('Bayes Gap'); axes[0].legend(); axes[0].set_title('BG vs p')
axes[1].scatter(ps * Ns, bgs, c=Ns, cmap='viridis')
pn_grid = np.linspace((ps * Ns).min(), (ps * Ns).max(), 200)
axes[1].set_xscale('log'); axes[1].set_xlabel('pN'); axes[1].set_ylabel('Bayes Gap')
axes[1].set_title(f'Joint fit R2={r2_joint:.3f} (N-only {r2_n:.3f}, p-only {r2_p:.3f})')
plt.colorbar(axes[1].collections[0], ax=axes[1], label='N')
plt.tight_layout()
plt.savefig('results/c2/bayes_gap_scaling.png', dpi=100)

summary = {
    'claim_id': 'C2',
    'status': status,
    'metrics': {
        'r2_joint': float(r2_joint),
        'r2_n_only': float(r2_n),
        'r2_p_only': float(r2_p),
        'joint_beta': float(beta_j),
        'joint_a': float(coef_j[0]), 'joint_b': float(coef_j[1]), 'joint_c': float(coef_j[2]),
        'bg_values': [float(v) for v in bgs],
        'bg_mc_stderr_mean': float(bg_se.mean()),
        'control_r2': float(r2c),
        'control_beta_recovered': float(betac),
        'control_pass': control_pass,
    },
    'notes': notes,
}
print('SUMMARY_JSON=' + json.dumps(summary, default=str))
