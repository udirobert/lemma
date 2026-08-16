import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(0)

# ---------------- Model parameters (Fig. 9 of the paper) ----------------
D = 5
eps = 2.0
lam2 = 0.01
N_PER_CLASS = 12          # N positive and N negative training samples
N_TRIALS = 10000
DT = 0.005
T_MAX = 12.0
STEPS = int(T_MAX / DT)

# Analytic slow-relaxation grokking time: t_slow = (1/(2*lam2)) * ln(eps^4/(eps^4-1))
T_SLOW = (1.0 / (2.0 * lam2)) * np.log(eps**4 / (eps**4 - 1.0))

# ---------------- Data: two disjoint uniform balls ----------------
# Positive class: uniform in unit ball centered at +c; negative at -c, c=(eps,0,...).
def sample_ball(n, center):
    g = np.random.randn(n, D)
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    r = np.random.rand(n, 1) ** (1.0 / D)
    return center + r * g

c = np.zeros(D); c[0] = eps
Xp = sample_ball(N_PER_CLASS, c)
Xn = sample_ball(N_PER_CLASS, -c)
X = np.vstack([Xp, Xn])                      # (2N, D)
y = np.concatenate([np.ones(N_PER_CLASS), -np.ones(N_PER_CLASS)])  # (2N,)

# ---------------- Vectorized gradient-flow simulation over trials ----------------
# Loss: R = mean(exp(-y w.x)) + (lam2/2)|w|^2  (exponential loss + L2)
w = np.random.randn(N_TRIALS, D)             # w(0) ~ N(0, I)

t_train0 = np.full(N_TRIALS, np.nan)         # first time train error = 0
t_test0 = np.full(N_TRIALS, np.nan)          # first time test error = 0

for k in range(STEPS):
    t = k * DT
    S = w @ X.T                              # (trials, 2N)
    m = y[None, :] * S                       # margins
    # train error zero?
    done_train = np.isnan(t_train0)
    sep = np.all(m > 0.0, axis=1)
    t_train0[done_train & sep] = t
    # test error zero (analytic): min over +ball of w.x = w.c - |w| > 0
    wc = w @ c
    wn = np.linalg.norm(w, axis=1)
    done_test = np.isnan(t_test0)
    gen = (wc - wn) > 0.0
    t_test0[done_test & gen] = t
    # gradient step (cap exp for numerical safety)
    coeff = y[None, :] * np.exp(-np.clip(m, -30.0, 30.0))   # (trials, 2N)
    grad_desc = -(coeff @ X) / (2.0 * N_PER_CLASS) + lam2 * w
    w = w - DT * grad_desc

tG = t_test0 - t_train0
valid = ~np.isnan(tG)
tG = tG[valid]
n_grok = int(valid.sum())

# ---------------- 1D 2-means clustering (bimodality statistic) ----------------
def two_means(x, iters=100):
    c1, c2 = np.quantile(x, 0.25), np.quantile(x, 0.75)
    for _ in range(iters):
        lab = (np.abs(x - c2) < np.abs(x - c1)).astype(int)
        n1, n2 = (lab == 0).sum(), (lab == 1).sum()
        if n1 == 0 or n2 == 0:
            break
        nc1, nc2 = x[lab == 0].mean(), x[lab == 1].mean()
        if abs(nc1 - c1) < 1e-12 and abs(nc2 - c2) < 1e-12:
            c1, c2 = nc1, nc2
            break
        c1, c2 = nc1, nc2
    lab = (np.abs(x - c2) < np.abs(x - c1)).astype(int)
    return lab, c1, c2

def bimodality_stats(x):
    lab, c1, c2 = two_means(x)
    if c1 > c2:
        lab = 1 - lab
        c1, c2 = c2, c1
    fast, slow = x[lab == 0], x[lab == 1]
    if len(fast) < 10 or len(slow) < 10:
        return None
    mu_f, sd_f = fast.mean(), fast.std()
    mu_s, sd_s = slow.mean(), slow.std()
    sep_stds = (mu_s - mu_f) / max(sd_f, sd_s, 1e-12)
    bimodal = bool(sep_stds > 2.0)
    return dict(mu_f=float(mu_f), sd_f=float(sd_f), mu_s=float(mu_s),
                sd_s=float(sd_s), n_f=int(len(fast)), n_s=int(len(slow)),
                sep_stds=float(sep_stds), bimodal=bimodal)

stats = bimodality_stats(tG)

# ---------------- POSITIVE CONTROL: synthetic known-bimodal mixture ----------------
# 90% broad continuous (fast-like) + 10% sharp peak at analytic t_slow
ctrl_fast = np.random.gamma(shape=2.0, scale=0.3, size=9000)
ctrl_slow = np.random.normal(T_SLOW, 0.05, size=1000)
ctrl = np.concatenate([ctrl_fast, ctrl_slow])
ctrl_stats = bimodality_stats(ctrl)
control_pass = bool(ctrl_stats is not None and ctrl_stats['bimodal']
                    and abs(ctrl_stats['mu_s'] - T_SLOW) / T_SLOW < 0.2)

# ---------------- Verdict ----------------
if stats is None:
    status = 'inconclusive'
    notes = 'Clustering failed (degenerate clusters).'
    metrics = {'n_grok': n_grok, 'control_pass': control_pass}
elif not control_pass:
    status = 'inconclusive'
    notes = 'Positive control failed: bimodality statistic is buggy; cannot judge claim.'
    metrics = {'n_grok': n_grok, 'control_pass': control_pass}
else:
    rel_err = abs(stats['mu_s'] - T_SLOW) / T_SLOW
    slow_sharp = stats['sd_s'] / abs(stats['mu_s'])
    fast_rel_spread = stats['sd_f'] / abs(stats['mu_f']) if stats['mu_f'] != 0 else np.inf
    bimodal = stats['bimodal']
    passed = bimodal and rel_err < 0.2
    status = 'supported' if passed else 'falsified'
    notes = (f"Two clusters: fast mean={stats['mu_f']:.3f} (std={stats['sd_f']:.3f}), "
             f"slow mean={stats['mu_s']:.3f} (std={stats['sd_s']:.3f}), "
             f"separation={stats['sep_stds']:.2f} stds; analytic t_slow={T_SLOW:.3f}, "
             f"rel err={rel_err:.3f}. Slow cluster sharp (std/mean={slow_sharp:.3f}) "
             f"vs fast continuous (std/mean={fast_rel_spread:.3f}).")
    metrics = {
        'n_trials': N_TRIALS, 'n_grok': n_grok,
        'fast_mean': stats['mu_f'], 'fast_std': stats['sd_f'], 'n_fast': stats['n_f'],
        'slow_mean': stats['mu_s'], 'slow_std': stats['sd_s'], 'n_slow': stats['n_s'],
        'separation_stds': stats['sep_stds'], 'bimodal': bimodal,
        't_slow_analytic': float(T_SLOW), 'slow_rel_error': float(rel_err),
        'slow_sharpness_std_over_mean': float(slow_sharp),
        'fast_spread_std_over_mean': float(fast_rel_spread),
        'control_pass': control_pass,
        'control_sep_stds': ctrl_stats['sep_stds'],
        'control_slow_rel_error': float(abs(ctrl_stats['mu_s'] - T_SLOW) / T_SLOW),
    }

# ---------------- Plot ----------------
plt.figure(figsize=(8, 5))
plt.hist(tG, bins=120, density=True, alpha=0.7, color='steelblue', label='empirical $t_G$')
plt.axvline(T_SLOW, color='red', lw=2, label=f'analytic $t_{{slow}}$={T_SLOW:.3f}')
if stats is not None:
    plt.axvline(stats['mu_s'], color='darkred', lw=1.5, ls='--',
                label=f"slow cluster mean={stats['mu_s']:.3f}")
    plt.axvline(stats['mu_f'], color='navy', lw=1.5, ls=':',
                label=f"fast cluster mean={stats['mu_f']:.3f}")
plt.xlabel('grokking time $t_G$')
plt.ylabel('PDF')
plt.title(f'D={D}, eps={eps}, lambda2={lam2}: grokking-time distribution')
plt.legend()
plt.tight_layout()
os.makedirs('results/c6', exist_ok=True)
plt.savefig('results/c6/fig.png', dpi=120)

summary = {'claim_id': 'C6', 'status': status, 'metrics': metrics, 'notes': notes}
print('SUMMARY_JSON=' + json.dumps(summary, default=str))
