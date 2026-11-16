import numpy as np, json, os, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'results/c5'
ETA = 0.0025
TMAX = 10000
TOL = 1e-3
SQ2 = np.sqrt(2.0)
T0 = time.time()

def run(d, m, seed, alpha=None):
    r = np.random.default_rng(seed)
    iu = np.triu_indices(d, 1)
    n = d * (d + 1) // 2
    def vec(X):
        return np.concatenate([np.diag(X), SQ2 * X[iu]])
    def unvec(v):
        X = np.zeros((d, d))
        X[iu] = v[d:] / SQ2
        X = X + X.T
        X[np.diag_indices(d)] = v[:d]
        return X
    # A=(Q+Q^T)/2 symmetric Gaussian <=> iid N(0,1) rows in sqrt2-weighted upper-tri embedding
    A = r.standard_normal((m, n), dtype=np.float32)
    u = r.standard_normal(d); u /= np.linalg.norm(u)
    Xs = np.outer(u, u)
    y = (A @ vec(Xs).astype(np.float32)).astype(np.float64)
    if alpha is None:
        alpha = 1.0 / d
    U = alpha * np.eye(d)
    Xp = alpha ** 2 * np.eye(d)
    sf = sp = False
    itf = itp = TMAX
    trf = trp = np.nan
    for t in range(TMAX):
        cols = []
        if not sf: cols.append(vec(U @ U.T))
        if not sp: cols.append(vec(Xp))
        V = np.stack(cols, axis=1).astype(np.float32)
        R = (A @ V).astype(np.float64) - y[:, None]
        loss = 0.5 * np.mean(R ** 2, axis=0)
        k = 0
        if not sf:
            trf = loss[k]; k += 1
            if (not np.isfinite(trf)) or trf < TOL:
                sf = True; itf = t
        if not sp:
            trp = loss[k]
            if (not np.isfinite(trp)) or trp < TOL:
                sp = True; itp = t
        if sf and sp:
            break
        G = (A.T @ R.astype(np.float32)).astype(np.float64) / m
        k = 0
        if len(cols) == 2 or not sf:
            pass
        idx = 0
        if not (itf < t + 1 and sf and itf <= t):
            pass
        # map gradient columns to methods active at start of this iteration
        active_f = (len(cols) == 2) or (len(cols) == 1 and itf > t and not (sp and itp < t))
        gi = 0
        if len(cols) == 2:
            gf, gp = G[:, 0], G[:, 1]
        else:
            gf = G[:, 0] if active_f else None
            gp = None if active_f else G[:, 0]
        if gf is not None and not sf:
            M = unvec(gf)
            U = U - ETA * (M @ U)  # paper eq (2.2)
        if gp is not None and not sp:
            Y = Xp - ETA * unvec(gp)
            Y = 0.5 * (Y + Y.T)
            w, Q = np.linalg.eigh(Y)
            Xp = (Q * np.clip(w, 0, None)) @ Q.T
    tef = float(np.sum((U @ U.T - Xs) ** 2))
    tep = float(np.sum((Xp - Xs) ** 2))
    return tef, tep, int(itf), int(itp), float(trf), float(trp)

# Positive control: heavily overdetermined (m = 2 * d(d+1)/2), both methods must recover X*
cd = 15
cm = cd * (cd + 1)
ctf, ctp, _, _, _, _ = run(cd, cm, 123)
control_pass = bool(ctf < 0.01 and ctp < 0.01)

mults = [5, 10, 20, 35]
res = {}
for d in [100, 150]:
    for k in mults:
        res[(d, k)] = run(d, k * d, 1000 * d + k)

GD_LOW = 0.05
gd_low_all = all(res[(d, k)][0] < GD_LOW for d in [100, 150] for k in mults)
pgd_worse_small = all(res[(d, 5)][1] > res[(d, 5)][0] for d in [100, 150])
pgd_mean100 = float(np.mean([res[(100, k)][1] for k in mults]))
pgd_mean150 = float(np.mean([res[(150, k)][1] for k in mults]))
pgd_150_worse = pgd_mean150 > pgd_mean100

metrics = {'control_pass': control_pass, 'control_gd_test': ctf, 'control_pgd_test': ctp,
           'gd_low_all_m': gd_low_all, 'gd_low_threshold': GD_LOW,
           'pgd_worse_than_gd_at_5d': pgd_worse_small,
           'pgd_mean_test_d100': pgd_mean100, 'pgd_mean_test_d150': pgd_mean150,
           'pgd_d150_worse_than_d100': pgd_150_worse, 'runtime_s': time.time() - T0}
for (d, k), v in res.items():
    metrics['d%d_m%dd' % (d, k)] = {'gd_test': v[0], 'pgd_test': v[1], 'gd_iters': v[2], 'pgd_iters': v[3], 'gd_train': v[4], 'pgd_train': v[5]}

plt.figure(figsize=(6, 4))
for d, c in [(100, 'b'), (150, 'r')]:
    plt.semilogy(mults, [max(res[(d, k)][0], 1e-12) for k in mults], c + 'o-', label='factorized GD d=%d' % d)
    plt.semilogy(mults, [max(res[(d, k)][1], 1e-12) for k in mults], c + 's--', label='PGD d=%d' % d)
plt.xlabel('m / d'); plt.ylabel('test error ||X-X*||_F^2'); plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig4_gd_vs_pgd.png'), dpi=120)

if not control_pass:
    status = 'inconclusive'
    notes = 'Positive control (overdetermined recovery) failed; statistic/implementation suspect.'
elif gd_low_all and pgd_worse_small and pgd_150_worse:
    status = 'supported'
    notes = 'Factorized GD recovers X* at all m; PGD worse at m=5d and worse on average at d=150 than d=100.'
else:
    status = 'falsified'
    notes = 'Criterion failed: gd_low_all=%s, pgd_worse_at_5d=%s, pgd_d150_worse=%s.' % (gd_low_all, pgd_worse_small, pgd_150_worse)
notes += ' Grid m in {5,10,20,35}d, 1 seed each, alpha=1/d, eta=0.0025, stop train<1e-3 or 1e4 iters.'

summary = {'claim_id': 'C5', 'status': status, 'metrics': metrics, 'notes': notes}
print('SUMMARY_JSON=' + json.dumps(summary, default=str))
