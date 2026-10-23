import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

rng = np.random.default_rng(42)

# Constant-mean task: f ~ N(0, tau^2), y = f + N(0, sigma^2).
tau2 = 1.0
tau = np.sqrt(tau2)
sigma2 = 0.25
sigma = np.sqrt(sigma2)

# Matched (N, p) grid from the claim.
cells = [(100,20),(200,10),(400,5),(500,8),(1000,4),(2000,2),(1000,8)]
R = 50
T_eval = 2000

def bayes_risk(p):
    # Posterior variance of f given p observations.
    return tau2 * sigma2 / (sigma2 + tau2 * p)

cell_pn = []
cell_mean_bg = []
cell_rep_bg = { (N,p): [] for N,p in cells }

for N, p in cells:
    pn = N * p
    br = bayes_risk(p)
    reps = []
    for _ in range(R):
        # Pretraining data: N prompts, each with p+1 observations.
        f = rng.normal(0.0, tau, size=N)
        noise = rng.normal(0.0, sigma, size=(N, p+1))
        Y = f[:, None] + noise

        # All p prefixes per prompt -> pN training examples.
        X_all = []
        Y_all = []
        for k in range(1, p+1):
            Xk = Y[:, :k].mean(axis=1)
            Yk = Y[:, k]
            X_all.append(Xk)
            Y_all.append(Yk)
        X_all = np.concatenate(X_all)
        Y_all = np.concatenate(Y_all)
        denom = float(np.dot(X_all, X_all))
        alpha_hat = float(np.dot(X_all, Y_all) / denom) if denom > 0 else 0.0

        # Fresh query points: evaluate against latent f (noise-free target).
        f_test = rng.normal(0.0, tau, size=T_eval)
        noise_test = rng.normal(0.0, sigma, size=(T_eval, p))
        Y_test = f_test[:, None] + noise_test
        Xbar_test = Y_test.mean(axis=1)
        pred = alpha_hat * Xbar_test
        mse = float(np.mean((pred - f_test)**2))
        bg = mse - br
        reps.append(bg)
        cell_rep_bg[(N,p)].append(bg)

    cell_pn.append(pn)
    cell_mean_bg.append(float(np.mean(reps)))

cell_pn = np.array(cell_pn, dtype=float)
cell_mean_bg = np.array(cell_mean_bg, dtype=float)

# Spearman across the grid (cell-level means).
sp_cells = float(spearmanr(cell_pn, cell_mean_bg).correlation)

# Replicate-level Spearman.
rep_pn = []
rep_bg = []
for N, p in cells:
    for bg in cell_rep_bg[(N,p)]:
        rep_pn.append(N*p)
        rep_bg.append(bg)
sp_reps = float(spearmanr(rep_pn, rep_bg).correlation)

# Within-pN coefficient of variation.
groups = {}
for (N,p), pn in zip(cells, cell_pn):
    groups.setdefault(float(pn), []).append(cell_mean_bg[cells.index((N,p))])
cvs = []
for pn, vals in groups.items():
    vals = np.array(vals, dtype=float)
    if len(vals) > 1 and np.mean(vals) != 0:
        cvs.append(float(np.std(vals, ddof=1) / np.mean(vals)))
mean_cv = float(np.mean(cvs)) if cvs else float('nan')

# Synthetic positive control: BG generated from a known pN-only law.
ctrl_cell_means = []
ctrl_rep_bg = []
for N, p in cells:
    pn = N * p
    reps = []
    for _ in range(R):
        bg = 100.0 / pn * float(np.exp(rng.normal(0.0, 0.05)))
        reps.append(bg)
        ctrl_rep_bg.append((pn, bg))
    ctrl_cell_means.append(float(np.mean(reps)))

cell_pn_arr = np.array([N*p for N,p in cells], dtype=float)
ctrl_cell_means = np.array(ctrl_cell_means, dtype=float)
ctrl_sp_cells = float(spearmanr(cell_pn_arr, ctrl_cell_means).correlation)

groups_ctrl = {}
for (N,p), m in zip(cells, ctrl_cell_means):
    groups_ctrl.setdefault(float(N*p), []).append(m)
cvs_ctrl = []
for pn, vals in groups_ctrl.items():
    vals = np.array(vals, dtype=float)
    if len(vals) > 1 and np.mean(vals) != 0:
        cvs_ctrl.append(float(np.std(vals, ddof=1) / np.mean(vals)))
ctrl_cv = float(np.mean(cvs_ctrl)) if cvs_ctrl else float('nan')

control_pass = bool((ctrl_sp_cells > 0.80) and (ctrl_cv < 0.25))
real_pass = bool((sp_cells > 0.80) and (mean_cv < 0.25))

if control_pass and real_pass:
    status = 'supported'
elif control_pass and not real_pass:
    status = 'falsified'
else:
    status = 'inconclusive'

# Plot.
os.makedirs('results/c7', exist_ok=True)
plt.figure(figsize=(7,5))
for pn in sorted(set(cell_pn.tolist())):
    idx = np.where(cell_pn == pn)[0]
    plt.errorbar(pn, cell_mean_bg[idx].mean(), yerr=cell_mean_bg[idx].std(ddof=1),
                 fmt='o', capsize=4, label=f'pN={int(pn)}')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('pN')
plt.ylabel('Bayes Gap')
plt.title('Bayes Gap vs pN')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/c7/fig.png')
plt.close()

summary = {
    'claim_id': 'C7',
    'status': status,
    'metrics': {
        'spearman_cells': sp_cells,
        'spearman_replicates': sp_reps,
        'mean_within_pn_cv': mean_cv,
        'control_spearman': ctrl_sp_cells,
        'control_cv': ctrl_cv,
        'control_pass': control_pass,
        'real_pass': real_pass,
        'cell_bg_means': {f'N{N}_p{p}': float(m) for (N,p), m in zip(cells, cell_mean_bg)},
    },
    'notes': 'Constant-mean meta-learning task; meta-learner is least squares on all pN prefix examples using mean-pooled context. Bayes Gap = out-of-sample MSE minus exact posterior variance.'
}
print('SUMMARY_JSON=' + json.dumps(summary, default=str))
