import numpy as np, json, time, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'results/c5'
os.makedirs(OUT, exist_ok=True)
ETA = 0.0025
TMAX = 10000
TRAIN_TOL = 1e-3
SQ2 = np.float32(np.sqrt(2.0))
T0 = time.time()
DEADLINE = 820.0

def make_ops(d):
    iu = np.triu_indices(d, 1)
    di = np.diag_indices(d)
    def vec(X):
        return np.concatenate([np.diag(X), SQ2 * X[iu]]).astype(np.float32)
    def unvec(v):
        X = np.zeros((d, d))
        X[iu] = v[d:] / SQ2
        X = X + X.T
        X[di] = v[:d]
        return X
    return vec, unvec

def run(d, m, seed):
    r = np.random.default_rng(seed)
    n = d * (d + 1) // 2
    vec, unvec = make_ops(d)
    # symmetric Gaussian A=(Q+Q^T)/2 <=> iid N(0,1) entries in sqrt2-weighted upper-tri embedding
    A = r.standard_normal((m, n), dtype=np.float32)
    u = r.standard_normal(d); u /= np.linalg.norm(u)
    Xs = np.outer(u, u)
    y = (A @ vec(Xs)).astype(np.float64)
    alpha = 1.0 / d
    U = alpha * np.eye(d)
    Xp = alpha ** 2 * np.eye(d)
    act = [True, True]
    its = [TMAX, TMAX]
    trl = [np.nan, np.nan]
    for t in range(TMAX):
        cols = []
        if act[0]: cols.append(vec(U @ U.T))
        if act[1]: cols.append(vec(Xp))
        V = np.stack(cols, axis=1)
        R = (A @ V).astype(np.float64) - y[:, None]
        loss = 0.5 * np.mean(R ** 2, axis=0)
        k = 0; idx = []
        for j in range(2):
            if act[j]:
                trl[j] = float(loss[k])
                if (not np.isfinite(trl[j])) or trl[j] < TRAIN_TOL:
                    act[j] = False; its[j] = t
                else:
                    idx.append((j, k))
                k += 1
        if not idx:
            break
        Rs = np.stack([R[:, k] for (_, k) in idx], axis=1).astype(np.float32)
        G = (A.T @ Rs).astype(np.float64) / m
        for c, (j, _) in enumerate(idx):
            M = unvec(G[:, c])
            if j == 0:
                U = U - ETA * (M @ U)
            else:
                Y = Xp - ETA * M
                Y = 0.5 * (Y + Y.T)
                w, Q = np.linalg.eigh(Y)
                Xp = (Q * np.clip(w, 0, None)) @ Q.T
    te_f = float(np.sum((U @ U.T - Xs) ** 2))
    te_p = float(np.sum((Xp - Xs) ** 2))
    del A
    return dict(d=d, m=m, te_gd=te_f, te_pgd=te_p, it_gd=its[0], it_pgd=its[1], tr_gd=trl[0], tr_pgd=trl[1])

# ---------- positive control: overdetermined (m > d(d+1)/2) -> both methods must recover X* ----------
ctrl = run(20, 700, 123)
control_pass = bool(ctrl['te_gd'] < 0.05 and ctrl['te_pgd'] < 0.05)

# ---------- main experiment ----------
mults = [5, 10, 20, 35]
ds = [100, 150]
res = []
skipped = []
for k in mults:
    for d in ds:
        if time.time() - T0 > DEADLINE:
            skipped.append((d, k * d)); continue
        res.append(run(d, k * d, 1000 * d + k))

def get(d, k):
    for r in res:
        if r['d'] == d and r['m'] == k * d: return r
    return None

LOW = 0.1
metrics = {'control_pass': control_pass, 'control_te_gd': ctrl['te_gd'], 'control_te_pgd': ctrl['te_pgd'],
           'low_threshold': LOW, 'n_configs_done': len(res), 'skipped': str(skipped)}
for r in res:
    tag = 'd%d_m%dd' % (r['d'], r['m'] // r['d'])
    metrics[tag + '_te_gd'] = r['te_gd']; metrics[tag + '_te_pgd'] = r['te_pgd']
    metrics[tag + '_it_gd'] = r['it_gd']; metrics[tag + '_it_pgd'] = r['it_pgd']

r100, r150 = get(100, 5), get(150, 5)
if r100 is None or r150 is None or not control_pass:
    status = 'inconclusive'
    notes = 'Control failed or smallest-m configs missing.' if not control_pass else 'Missing configs.'
else:
    gd_low = all(r['te_gd'] < LOW for r in res)
    pgd_worse_small = (r100['te_pgd'] > r100['te_gd']) and (r150['te_pgd'] > r150['te_gd'])
    pairs = [(get(100, k), get(150, k)) for k in mults if get(100, k) and get(150, k)]
    mean100 = float(np.mean([a['te_pgd'] for a, b in pairs])); mean150 = float(np.mean([b['te_pgd'] for a, b in pairs]))
    pgd_d_worse = (r150['te_pgd'] > r100['te_pgd']) and (mean150 > mean100)
    metrics.update(gd_low_all_m=gd_low, pgd_worse_than_gd_small_m=pgd_worse_small,
                   pgd_d150_worse_than_d100=pgd_d_worse, pgd_mean_te_d100=mean100, pgd_mean_te_d150=mean150)
    ok = gd_low and pgd_worse_small and pgd_d_worse
    status = 'supported' if ok else 'falsified'
    notes = ('Rank-1, eta=0.0025, stop at train<1e-3 or 1e4 iters, alpha=1/d, one seed per config, m/d in %s. '
             'GD low across m: %s; PGD>GD at m=5d: %s; PGD d=150 worse than d=100: %s.' % (mults, gd_low, pgd_worse_small, pgd_d_worse))
    if skipped: notes += ' Some configs skipped for time: %s.' % skipped

try:
    plt.figure(figsize=(6, 4))
    for d, c in zip(ds, ['C0', 'C1']):
        rr = sorted([r for r in res if r['d'] == d], key=lambda r: r['m'])
        if not rr: continue
        x = [r['m'] / d for r in rr]
        plt.plot(x, [r['te_gd'] for r in rr], 'o-', color=c, label='factorized GD d=%d' % d)
        plt.plot(x, [r['te_pgd'] for r in rr], 's--', color=c, label='PGD d=%d' % d)
    plt.xlabel('m / d'); plt.ylabel('test error ||X-X*||_F^2'); plt.yscale('log'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'fig4_gd_vs_pgd.png'), dpi=120)
except Exception as e:
    metrics['plot_error'] = str(e)

metrics['runtime_s'] = time.time() - T0
print('SUMMARY_JSON=' + json.dumps({'claim_id': 'C5', 'status': status, 'metrics': metrics, 'notes': notes}, default=str))
