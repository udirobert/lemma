import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

rng = np.random.default_rng(42)
outdir = 'results/c5'
os.makedirs(outdir, exist_ok=True)

# --- Synthetic Bayesian linear regression setup ---
tau = 1.0          # prior std of slope w
sigma = 1.0        # noise std
lam = sigma**2 / tau**2
p = 5              # context length
n_mc = 30000       # Monte Carlo trials


def sample_bg_pv(xdist_sampler, n_mc, p, rng):
    """Estimate Bayes Gap (model=0) and Posterior Variance for an input distribution."""
    # sample tasks
    w = rng.normal(0.0, tau, size=n_mc)
    # context
    Xc = xdist_sampler(n_mc, p, rng)          # (n_mc, p)
    eps = rng.normal(0.0, sigma, size=(n_mc, p))
    Yc = w[:, None] * Xc + eps
    # query
    xq = xdist_sampler(n_mc, 1, rng).ravel()  # (n_mc,)
    # Bayes posterior mean and variance of f(xq)
    S = np.sum(Xc**2, axis=1)                 # (n_mc,)
    SY = np.sum(Xc * Yc, axis=1)
    mb = SY / (S + lam)                       # posterior mean slope
    pred = mb * xq                            # Bayes predictor value
    pv = (sigma**2 * xq**2) / (S + lam)       # posterior variance of f(xq)
    # model is fixed at 0 (pretrained source model baseline)
    bg = (0.0 - pred)**2
    return float(np.mean(bg)), float(np.mean(pv))


def src_sampler(n, k, rng):
    return rng.normal(0.0, 1.0, size=(n, k))


deltas = np.linspace(0.0, 3.0, 13)
# source estimates
bg_src, pv_src = sample_bg_pv(src_sampler, n_mc, p, rng)

Ws = []
DBG = []
DPV = []
for d in deltas:
    def tgt_sampler(n, k, rng, d=d):
        return rng.normal(d, 1.0, size=(n, k))
    bg_t, pv_t = sample_bg_pv(tgt_sampler, n_mc, p, rng)
    # empirical W1 between input distributions
    xs = src_sampler(5000, 1, rng).ravel()
    xt = tgt_sampler(5000, 1, rng).ravel()
    w1 = float(wasserstein_distance(xs, xt))
    Ws.append(w1)
    DBG.append(bg_t - bg_src)
    DPV.append(pv_t - pv_src)

Ws = np.array(Ws)
DBG = np.array(DBG)
DPV = np.array(DPV)

corr_bg = float(np.corrcoef(Ws, DBG)[0, 1])
corr_pv = float(np.corrcoef(Ws, DPV)[0, 1])
mean_abs_dbg = float(np.mean(np.abs(DBG)))
mean_abs_dpv = float(np.mean(np.abs(DPV)))
ratio_pv_bg = float(mean_abs_dpv / (mean_abs_dbg + 1e-12))

# --- Positive control: synthetic known-linear BG, uncorrelated PV ---
rngc = np.random.default_rng(7)
Wc = np.linspace(0.0, 3.0, 13)
DBGc = 2.0 * Wc + rngc.normal(0.0, 0.05, size=Wc.shape)
DPVc = rngc.normal(0.0, 0.05, size=Wc.shape)
corr_bg_c = float(np.corrcoef(Wc, DBGc)[0, 1])
corr_pv_c = float(np.corrcoef(Wc, DPVc)[0, 1])
control_pass = bool((corr_bg_c > 0.9) and (abs(corr_pv_c) < 0.5))

# status
pv_ok = bool((abs(corr_pv) < 0.5) or (ratio_pv_bg < 0.5))
status = 'supported' if (corr_bg > 0.9 and pv_ok and control_pass) else ('inconclusive' if not control_pass else 'falsified')

# plot
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(Ws, DBG, label='Bayes Gap change')
axes[0].plot(Ws, np.poly1d(np.polyfit(Ws, DBG, 1))(Ws), 'r--', label='linear fit')
axes[0].set_xlabel('W_1(prompt distributions)')
axes[0].set_ylabel('Change in Bayes Gap')
axes[0].set_title(f'corr={corr_bg:.3f}')
axes[0].legend()
axes[1].scatter(Ws, DPV, label='Posterior Variance change')
axes[1].set_xlabel('W_1(prompt distributions)')
axes[1].set_ylabel('Change in Posterior Variance')
axes[1].set_title(f'corr={corr_pv:.3f}')
axes[1].legend()
fig.tight_layout()
fig.savefig(os.path.join(outdir, 'fig.png'), dpi=120)
plt.close(fig)

summary = {
    'claim_id': 'C5',
    'status': status,
    'metrics': {
        'corr_wasserstein_bayes_gap': corr_bg,
        'corr_wasserstein_posterior_variance': corr_pv,
        'mean_abs_change_bayes_gap': mean_abs_dbg,
        'mean_abs_change_posterior_variance': mean_abs_dpv,
        'ratio_pv_bg': ratio_pv_bg,
        'control_pass': control_pass,
        'control_corr_bg': corr_bg_c,
        'control_corr_pv': corr_pv_c,
        'bg_source': bg_src,
        'pv_source': pv_src,
    },
    'notes': 'Monte Carlo Bayesian linear regression; model fixed at 0; W1 estimated empirically between input distributions.'
}
print('SUMMARY_JSON=' + json.dumps(summary, default=str))
