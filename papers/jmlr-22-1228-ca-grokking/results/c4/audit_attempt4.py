import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(12345)

LAMBDA2 = 0.1
N_HALF = 10          # N positive and N negative samples (N=10 per class)
N_DATASETS = 4000    # Monte-Carlo datasets per (D, eps)
DS = [2, 5, 10, 20]
EPSS = [1.005, 1.01, 1.02, 1.05, 1.1, 1.2]

def sample_half_ball(D, n, positive, rng):
    # uniform in D-dim unit ball, conditioned on sign(x1)
    g = rng.standard_normal((n, D))
    r = rng.random(n) ** (1.0 / D)
    x = g / np.linalg.norm(g, axis=1, keepdims=True) * r[:, None]
    if positive:
        x[:, 0] = np.abs(x[:, 0])
    else:
        x[:, 0] = -np.abs(x[:, 0])
    return x

def dataset_stats(D, rng):
    xp = sample_half_ball(D, N_HALF, True, rng)
    xn = sample_half_ball(D, N_HALF, False, rng)
    X = np.vstack([xp, xn])
    y = np.concatenate([np.ones(N_HALF), -np.ones(N_HALF)])
    m = 2 * N_HALF
    xbar1 = np.mean(y * X[:, 0])
    xbarj = np.array([np.mean(y * X[:, j]) for j in range(1, D)])
    x1sq = np.mean(X[:, 0] ** 2)
    x1xj = np.array([np.mean(X[:, 0] * X[:, j]) for j in range(1, D)])
    return xbar1, xbarj, x1sq, x1xj

def grok_condition(D, eps, rng):
    lam_D = 1.0 / (D + 2) + LAMBDA2
    L = lam_D + eps ** 2
    beta = eps / L + eps / (L ** 2 * (D + 2))
    a1 = 1.0 / L - 2 * eps ** 2 / L ** 2
    a2 = -eps / L ** 2
    a3 = 1.0 / lam_D - eps / (lam_D * L)
    a4 = -eps / (lam_D * L)
    xbar1, xbarj, x1sq, x1xj = dataset_stats(D, rng)
    w1 = beta + a1 * xbar1 + a2 * x1sq
    wj = a3 * xbarj + a4 * x1xj
    return (w1 > 0) and ((eps ** 2 - 1) * w1 ** 2 >= np.sum(wj ** 2))

def p_grok(D, eps, n=N_DATASETS):
    rng = np.random.default_rng(1000 + D * 100 + int(eps * 1000))
    cnt = 0
    for _ in range(n):
        if grok_condition(D, eps, rng):
            cnt += 1
    return cnt / n

# ---- main sweep ----
table = {}
for D in DS:
    for eps in EPSS:
        table[(D, eps)] = p_grok(D, eps)

print('P_grok(D, eps) table:')
for D in DS:
    row = ' '.join(f'{table[(D, e)]:.4f}' for e in EPSS)
    print(f'D={D:3d}: {row}')

# ---- verdict logic ----
def verdict(pvals):
    # pvals: list of probabilities at increasing D
    dec_pairs = sum(1 for i in range(len(pvals) - 1) if pvals[i + 1] < pvals[i])
    strictly_dec = all(pvals[i + 1] < pvals[i] for i in range(len(pvals) - 1))
    drop = pvals[0] - pvals[-1]
    return strictly_dec and drop >= 0.2, dec_pairs, drop

# positive control: synthetic clearly decreasing sequence must be detected
ctrl_seq = [0.9, 0.6, 0.3, 0.1]
control_pass, _, _ = verdict(ctrl_seq)
ctrl_seq2 = [0.5, 0.5, 0.5, 0.5]
control_pass2, _, _ = verdict(ctrl_seq2)
control_pass = bool(control_pass and (not control_pass2))

# 1) common-eps analysis: for each eps, count D values in discriminating window
best = None
for eps in EPSS:
    pvals = [table[(D, eps)] for D in DS]
    in_win = [i for i, p in enumerate(pvals) if 0.0 < p < 1.0]
    if len(in_win) >= 3:
        sub = [pvals[i] for i in in_win]
        ok, dec_pairs, drop = verdict(sub)
        if best is None or (ok and not best[0]) or (ok == best[0] and drop > best[2]):
            best = (ok, eps, drop, dec_pairs, pvals, in_win)

# 2) per-D window fallback: pick per D the eps closest to P=0.5 within window
perD_eps = {}
perD_p = {}
for D in DS:
    cands = [(abs(table[(D, e)] - 0.5), e) for e in EPSS if 0.0 < table[(D, e)] < 1.0]
    if cands:
        cands.sort()
        perD_eps[D] = cands[0][1]
        perD_p[D] = table[(D, cands[0][1])]

perD_ok = False
perD_drop = 0.0
if len(perD_p) >= 3:
    ds_sorted = sorted(perD_p)
    pv = [perD_p[d] for d in ds_sorted]
    perD_ok, _, perD_drop = verdict(pv)

if best is not None:
    common_ok, common_eps, common_drop, common_decpairs, common_pvals, common_win = best
else:
    common_ok, common_eps, common_drop, common_decpairs, common_pvals, common_win = False, None, 0.0, 0, [], []

any_window = any(0.0 < table[(D, e)] < 1.0 for D in DS for e in EPSS)

if not control_pass:
    status = 'inconclusive'
    notes = 'Positive control failed: verdict logic buggy.'
elif not any_window:
    status = 'inconclusive'
    notes = 'No eps gave a discriminating window (0<P<1) for any D; N>>1 limit too sharp for this sweep.'
elif common_ok or perD_ok:
    status = 'supported'
    which = 'common eps' if common_ok else 'per-D window'
    notes = f'P_grok strictly decreasing across >=3 D values with drop>=0.2 ({which}).'
else:
    status = 'falsified'
    notes = 'Discriminating window exists but P_grok is not strictly decreasing with drop>=0.2 across >=3 D values.'

# ---- plots ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for D in DS:
    axes[0].plot(EPSS, [table[(D, e)] for e in EPSS], 'o-', label=f'D={D}')
axes[0].set_xlabel('epsilon'); axes[0].set_ylabel('P_grok')
axes[0].set_title('Grokking probability vs epsilon')
axes[0].legend(); axes[0].grid(alpha=0.3)
for eps in EPSS:
    axes[1].semilogy(DS, [max(table[(D, eps)], 1e-4) for D in DS], 's-', label=f'eps={eps}')
axes[1].set_xlabel('D'); axes[1].set_ylabel('P_grok (log)')
axes[1].set_title('Grokking probability vs D')
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('results/c4/p_grok_sweep.png', dpi=120)

metrics = {
    'control_pass': control_pass,
    'any_discriminating_window': bool(any_window),
    'common_eps_used': common_eps if common_eps is not None else 'none',
    'common_eps_pvals': [round(float(p), 4) for p in common_pvals],
    'common_eps_window_idx': common_win,
    'common_eps_drop': round(float(common_drop), 4),
    'perD_eps': {str(d): perD_eps[d] for d in perD_eps},
    'perD_pvals': {str(d): round(float(perD_p[d]), 4) for d in perD_p},
    'perD_drop': round(float(perD_drop), 4),
}
for D in DS:
    for e in EPSS:
        metrics[f'P_D{D}_eps{e}'] = round(float(table[(D, e)]), 4)

summary = {'claim_id': 'C4', 'status': status, 'metrics': metrics, 'notes': notes}
print('SUMMARY_JSON=' + json.dumps(summary, default=str))
