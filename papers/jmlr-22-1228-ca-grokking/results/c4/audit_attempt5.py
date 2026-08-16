import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)
OUT = 'results/c4'
os.makedirs(OUT, exist_ok=True)

N = 10          # N positive + N negative samples
LAM2 = 0.1
N_MC = 2000     # datasets per (D, eps)
Ds = [2, 5, 10, 20]
EPSS = [1.005, 1.01, 1.02, 1.05, 1.1, 1.2]

def sample_ball(n, D):
    """Uniform samples in unit D-ball: E[x_j^2] = 1/(D+2)."""
    g = np.random.randn(n, D)
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    r = np.random.rand(n, 1) ** (1.0 / D)
    return g * r

def dataset_stats(D):
    """One synthetic dataset: N pos (shifted +eps*e1), N neg (-eps*e1).
    Returns xbar1, x1sq, xbarj (D-1 vector), x1xj (D-1 vector)."""
    eps = dataset_stats.eps
    xp = sample_ball(N, D); xp[:, 0] += eps
    xn = sample_ball(N, D); xn[:, 0] -= eps
    X = np.vstack([xp, xn])
    x1 = X[:, 0]
    xbar1 = x1.mean()
    x1sq = (x1 ** 2).mean()
    xbarj = X[:, 1:].mean(axis=0)
    x1xj = (x1[:, None] * X[:, 1:]).mean(axis=0)
    return xbar1, x1sq, xbarj, x1xj

def grok_condition(D, eps, lam2=LAM2):
    """Monte-Carlo estimate of P_grok via Appendix B.1 (lambda_1=0) formulas."""
    lam_D = 1.0 / (D + 2) + lam2
    L = lam_D + eps ** 2
    beta = eps / L + eps / (L ** 2 * (D + 2))
    a1 = 1.0 / L - 2 * eps ** 2 / L ** 2
    a2 = -eps / L ** 2
    a3 = 1.0 / lam_D - eps / (lam_D * L)
    a4 = -eps / (lam_D * L)
    dataset_stats.eps = eps
    cnt = 0
    for _ in range(N_MC):
        xbar1, x1sq, xbarj, x1xj = dataset_stats(D)
        w1 = beta + a1 * xbar1 + a2 * x1sq
        wj = a3 * xbarj + a4 * x1xj
        if w1 > 0 and (eps ** 2 - 1) * w1 ** 2 >= np.sum(wj ** 2):
            cnt += 1
    return cnt / N_MC

# ---- Sweep P_grok(D, eps) ----
table = {}
for D in Ds:
    for eps in EPSS:
        table[(D, eps)] = grok_condition(D, eps)

# Per-D discriminating window: eps with 0 < P < 1 (prefer P closest to 0.5)
chosen = {}
for D in Ds:
    cands = [(e, table[(D, e)]) for e in EPSS if 0.0 < table[(D, e)] < 1.0]
    if cands:
        chosen[D] = min(cands, key=lambda t: abs(t[1] - 0.5))

# Verdict logic (shared with positive control)
def verdict(seq):
    """seq: list of P values across increasing D. True iff strictly decreasing
    across >=3 points with overall drop >= 0.2."""
    if len(seq) < 3:
        return False
    strictly_dec = all(seq[i] > seq[i + 1] for i in range(len(seq) - 1))
    return strictly_dec and (seq[0] - seq[-1]) >= 0.2

# Positive control: synthetic exponentially decreasing sequence must be detected
ctrl_seq = [0.9, 0.6, 0.35, 0.15]
control_pass = bool(verdict(ctrl_seq))

# Main verdict on per-D window values
Ds_used = sorted(chosen.keys())
P_seq = [chosen[D][1] for D in Ds_used]
eps_used = {str(D): chosen[D][0] for D in Ds_used}
main_pass = verdict(P_seq) if len(Ds_used) >= 3 else False

# Also check common-eps rows for a decreasing trend (secondary evidence)
common_eps_trend = {}
for e in EPSS:
    row = [table[(D, e)] for D in Ds]
    common_eps_trend[str(e)] = row

# Log-linearity fit on the per-D window values (where 0<P<1)
log_slope = None
if len(Ds_used) >= 3:
    xs = np.array(Ds_used, dtype=float)
    ys = np.log(np.array(P_seq))
    A = np.vstack([xs, np.ones_like(xs)]).T
    log_slope, _ = np.linalg.lstsq(A, ys, rcond=None)[0]

# ---- Plot ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for e in EPSS:
    ax[0].plot(Ds, [table[(D, e)] for D in Ds], 'o-', label=f'eps={e}')
ax[0].set_xlabel('D'); ax[0].set_ylabel('P_grok'); ax[0].legend(fontsize=7)
ax[0].set_title('P_grok(D) per eps (Appendix B.1, lambda1=0)')
if Ds_used:
    ax[1].plot(Ds_used, P_seq, 's-', color='crimson', label='per-D window')
    ax[1].set_yscale('log')
ax[1].set_xlabel('D'); ax[1].set_ylabel('P_grok (log scale)')
ax[1].set_title('Discriminating-window P_grok vs D')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'pgrok_vs_D.png'), dpi=100)

# ---- Status ----
if not control_pass:
    status = 'inconclusive'
    notes = 'Positive control failed: verdict logic buggy.'
elif len(Ds_used) < 3:
    status = 'inconclusive'
    notes = ('Fewer than 3 D values had a discriminating eps window (0<P<1); '
             'N>>1 limit too sharp for the eps sweep. Full table reported.')
elif main_pass:
    status = 'supported'
    notes = (f'P_grok strictly decreasing across Ds {Ds_used} at per-D window eps '
             f'{eps_used}; drop={P_seq[0]-P_seq[-1]:.3f}; log-slope={log_slope:.4f}.')
else:
    status = 'falsified'
    notes = (f'Discriminating-window P_grok not strictly decreasing with >=0.2 drop: '
             f'{P_seq} at Ds {Ds_used}.')

metrics = {
    'control_pass': control_pass,
    'Ds_used': Ds_used,
    'eps_used_per_D': eps_used,
    'P_grok_window_seq': [float(p) for p in P_seq],
    'log_slope': (float(log_slope) if log_slope is not None else 'NA'),
    'P_grok_table': {f'D{D}_eps{e}': float(table[(D, e)]) for D in Ds for e in EPSS},
    'common_eps_rows': {k: [float(v) for v in row] for k, row in common_eps_trend.items()},
}
summary = {'claim_id': 'C4', 'status': status, 'metrics': metrics, 'notes': notes}
print('SUMMARY_JSON=' + json.dumps(summary, default=str))
