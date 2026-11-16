import os, json, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'results/c2'
os.makedirs(OUT, exist_ok=True)
t_start = time.time()

r = 5
alphas = 10.0 ** np.array([0, -0.5, -1, -1.5, -2, -2.5, -3])
ETA_MAIN, T_MAIN = 0.025, 1000   # same eta*T = 25 as paper (0.0025 x 1e4), for CPU budget
ETA_PAPER, T_PAPER = 0.0025, 10000
N_SEEDS = 3


def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def setup(d, seed):
    rng = np.random.default_rng(1000 * d + seed)
    m = 5 * d * r
    iu = np.triu_indices(d)
    w = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0)).astype(np.float32)
    # Symmetric A_i = (Q+Q^T)/2: in coordinates [X_ii, sqrt2 X_ij] the measurement rows are iid N(0,1)
    G = rng.standard_normal((m, len(iu[0])), dtype=np.float32)
    Q, _ = np.linalg.qr(rng.standard_normal((d, r)))
    Xs = (Q @ Q.T)  # rank r PSD, spectral norm 1
    xs = (Xs[iu] * w).astype(np.float32)
    y = G @ xs
    return G, iu, w, Xs, y, m


def vecs(Xb, iu, w):
    return (Xb[:, iu[0], iu[1]] * w).T.astype(np.float32)  # (p,k)


def errors(Xb, G, iu, w, Xs, y):
    pred = G @ vecs(Xb, iu, w)
    train = np.linalg.norm(pred - y[:, None], axis=0) / np.linalg.norm(y)
    test = np.array([np.linalg.norm(X - Xs) for X in Xb]) / np.linalg.norm(Xs)
    return train, test


def run_gd(G, iu, w, Xs, y, m, d, alist, eta, T):
    k = len(alist)
    U = np.stack([a * np.eye(d) for a in alist]).astype(np.float32)
    for t in range(T):
        X = U @ np.transpose(U, (0, 2, 1))
        res = G @ vecs(X, iu, w) - y[:, None]  # (m,k)
        g = (G.T @ res) / m  # (p,k)
        M = np.zeros((k, d, d), dtype=np.float32)
        M[:, iu[0], iu[1]] = (g / w[:, None]).T
        M = M + np.transpose(M, (0, 2, 1)) - M * np.eye(d, dtype=np.float32)[None]
        U = U - eta * (M @ U)  # paper's update U_{t+1} = (I - eta M_t) U_t
        if not np.all(np.isfinite(U)):
            break
    X = (U @ np.transpose(U, (0, 2, 1))).astype(np.float64)
    return errors(X, G, iu, w, Xs, y)


metrics = {}
results = {}
ctrl_ok = []
exact_check = None
for d in [100, 200]:
    gaps, trains, tests = [], [], []
    for s in range(N_SEEDS):
        G, iu, w, Xs, y, m = setup(d, s)
        tr, te = run_gd(G, iu, w, Xs, y, m, d, alphas, ETA_MAIN, T_MAIN)
        trains.append(tr); tests.append(te); gaps.append(te - tr)
        if s == 0:
            # positive control: synthetic estimates X* + a*E (E full rank PSD noise) -> gap must grow with a
            rngc = np.random.default_rng(7)
            E = rngc.standard_normal((d, d)); E = E @ E.T / d; E /= np.linalg.norm(E)
            Xc = np.stack([Xs + a * E for a in alphas])
            trc, tec = errors(Xc, G, iu, w, Xs, y)
            gc = tec - trc
            rho_c = spearman(np.log10(alphas), gc)
            ctrl_ok.append(bool(gc[-1] < gc[0] and rho_c > 0.9 and abs(errors(Xs[None], G, iu, w, Xs, y)[1][0]) < 1e-8))
            metrics[f'd{d}_control_spearman'] = rho_c
            if d == 100:
                trp, tep = run_gd(G, iu, w, Xs, y, m, d, alphas, ETA_PAPER, T_PAPER)
                exact_check = (tep - trp)
        del G
    gaps = np.array(gaps); trains = np.array(trains); tests = np.array(tests)
    mg = np.nanmean(gaps, axis=0)
    rho = spearman(np.log10(alphas), mg) if np.all(np.isfinite(mg)) else float('nan')
    results[d] = dict(gap=gaps, train=trains, test=tests, mean_gap=mg, rho=rho)
    metrics[f'd{d}_mean_gap_by_alpha'] = [float(x) for x in mg]
    metrics[f'd{d}_mean_test_by_alpha'] = [float(x) for x in np.nanmean(tests, 0)]
    metrics[f'd{d}_mean_train_by_alpha'] = [float(x) for x in np.nanmean(trains, 0)]
    metrics[f'd{d}_gap_alpha1'] = float(mg[0]); metrics[f'd{d}_gap_alpha1e-3'] = float(mg[-1])
    metrics[f'd{d}_spearman_logalpha_vs_gap'] = rho

if exact_check is not None:
    metrics['d100_seed0_exact_paper_eta0.0025_T1e4_gap_by_alpha'] = [float(x) for x in exact_check]

control_pass = bool(all(ctrl_ok))
metrics['control_pass'] = control_pass
pass_d = {}
for d in [100, 200]:
    R = results[d]
    pass_d[d] = bool(np.isfinite(R['rho']) and R['mean_gap'][-1] < R['mean_gap'][0] and R['rho'] >= 0.7)
    metrics[f'd{d}_pass'] = pass_d[d]
metrics['runtime_s'] = time.time() - t_start

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for d in [100, 200]:
    R = results[d]
    ax[0].errorbar(alphas, R['mean_gap'], yerr=np.nanstd(R['gap'], 0), marker='o', label=f'd={d}')
    ax[1].plot(alphas, np.nanmean(R['test'], 0), 'o-', label=f'test d={d}')
    ax[1].plot(alphas, np.nanmean(R['train'], 0), 's--', label=f'train d={d}')
if exact_check is not None:
    ax[0].plot(alphas, exact_check, 'k^:', label='d=100 exact eta=0.0025,T=1e4')
for a in ax:
    a.set_xscale('log'); a.set_xlabel('alpha'); a.legend(fontsize=7)
ax[0].set_ylabel('test - train relative error'); ax[1].set_yscale('log')
plt.tight_layout(); plt.savefig(os.path.join(OUT, 'gap_vs_alpha.png'), dpi=120)

if not control_pass:
    status = 'inconclusive'; note = 'Positive control failed; statistic unreliable.'
elif all(pass_d.values()):
    status = 'supported'; note = 'Gap at alpha=1e-3 smaller than at alpha=1 and decreasing trend (Spearman>=0.7) for d=100,200.'
else:
    status = 'falsified'; note = 'Gap trend criterion failed for d=' + str([d for d in pass_d if not pass_d[d]])
note += ' Main runs use eta=0.025,T=1000 (same eta*T as paper) for CPU budget; d=100 seed0 also run at exact paper eta=0.0025,T=1e4. X*=QQ^T (kappa=1), symmetrized Gaussian A_i, m=5dr.'
summary = dict(claim_id='C2', status=status, metrics=metrics, notes=note)
print('SUMMARY_JSON=' + json.dumps(summary, default=str))
