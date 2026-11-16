import numpy as np, json, time, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.fft import dctn, idctn

OUT = 'results/c4'
os.makedirs(OUT, exist_ok=True)
np.random.seed(0)

# NOTE: dense Gaussian A_i (m x d^2) is computationally infeasible for 1e5 GD iterations on CPU
# (attempt 1 timed out). We use a structured random measurement operator with the same
# isometry normalization (E[(1/m)sum <A_i,X>^2] = ||X||_F^2): random sign flip + orthonormal 2D DCT
# + random row subsampling (a standard RIP ensemble).

def make_op(d, m, rng):
    D = rng.choice([-1.0, 1.0], size=(d, d))
    idx = rng.choice(d * d, size=m, replace=False)
    def A(X):
        return d * dctn(D * X, norm='ortho').ravel()[idx]
    def Aadj(r):
        Z = np.zeros(d * d); Z[idx] = r
        return d * D * idctn(Z.reshape(d, d), norm='ortho')
    return A, Aadj

def run(d, r, m, eta, alpha, T, log_every, rng):
    Q, _ = np.linalg.qr(rng.standard_normal((d, r)))
    Xs = Q @ Q.T  # rank r, spectral norm 1, kappa = 1
    A, Aadj = make_op(d, m, rng)
    U = alpha * np.eye(d)
    its, test, rel, train = [], [], [], []
    nXs = np.sum(Xs ** 2)
    for t in range(T + 1):
        E = U @ U.T - Xs
        res = A(E)
        if t % log_every == 0:
            te = float(np.sum(E ** 2))
            its.append(t); test.append(te); rel.append(te / nXs); train.append(float(0.5 * np.mean(res ** 2)))
            if not np.isfinite(te) or te > 1e6:
                break
        if t == T:
            break
        M = Aadj(res) / m
        M = 0.5 * (M + M.T)
        U = U - eta * (M @ U)  # paper update (2.2): U_{t+1} = (I - eta M_t) U_t
    return np.array(its), np.array(test), np.array(rel), np.array(train)

def judge(its, test, T, chk=20000, thr=0.03):
    finite = bool(np.all(np.isfinite(test))) and its[-1] == T
    if not finite:
        return False, {}, False
    e_chk = test[its == chk][0]; e_fin = test[-1]
    after = test[its >= chk]
    nodiv = bool(np.max(after) <= 2.0 * e_chk + 1e-12)
    mono = bool(e_fin <= e_chk * (1 + 1e-9))
    small = bool(e_fin <= thr)
    return (mono and nodiv and small), dict(err_at_2e4=e_chk, err_final=e_fin, max_after_2e4=float(np.max(after))), True

t0 = time.time()
metrics = {}
# ---- positive control: full measurements (exact isometry, population GD), small d ----
rngc = np.random.default_rng(1)
dc, rc = 30, 2
its_c, te_c, rel_c, tr_c = run(dc, rc, dc * dc, 0.0025, 0.01, 30000, 1000, rngc)
ok_c, mc, fin_c = judge(its_c, te_c, 30000, chk=20000, thr=0.03)
control_pass = bool(ok_c)
metrics['control_final_test_err'] = float(te_c[-1])
metrics['control_pass'] = control_pass

T = 100000
res_all = {}
all_ok = True
for d in [100, 200]:
    rng = np.random.default_rng(100 + d)
    r = 5; m = 5 * d * r
    ts = time.time()
    its, te, rel, tr = run(d, r, m, 0.0025, 0.01, T, 1000, rng)
    ok, md, fin = judge(its, te, T)
    all_ok = all_ok and ok
    res_all[d] = (its, te, rel, tr)
    metrics[f'd{d}_m'] = m
    metrics[f'd{d}_completed'] = bool(fin)
    for k, v in md.items():
        metrics[f'd{d}_{k}'] = float(v)
    metrics[f'd{d}_final_rel_err'] = float(rel[-1])
    metrics[f'd{d}_min_test_err'] = float(np.min(te))
    metrics[f'd{d}_argmin_iter'] = int(its[np.argmin(te)])
    metrics[f'd{d}_final_train_loss'] = float(tr[-1])
    metrics[f'd{d}_pass'] = bool(ok)
    metrics[f'd{d}_seconds'] = round(time.time() - ts, 1)

plt.figure(figsize=(7, 4.5))
for d, (its, te, rel, tr) in res_all.items():
    plt.semilogy(its, te, label=f'test ||UU^T-X*||_F^2, d={d}')
    plt.semilogy(its, tr, '--', label=f'train loss, d={d}')
plt.semilogy(its_c, te_c, ':', label='control (full meas., d=30)')
plt.xlabel('iteration'); plt.ylabel('error'); plt.legend(fontsize=8); plt.title('C4: GD, U0=0.01I, m=5dr, r=5, eta=0.0025')
plt.tight_layout(); plt.savefig(f'{OUT}/test_error_vs_iter.png', dpi=110)

if not control_pass:
    status = 'inconclusive'; notes = 'Positive control failed; statistic/pipeline unreliable.'
elif all_ok:
    status = 'supported'; notes = 'Test error (||UU^T-X*||_F^2) at 1e5 <= at 2e4, no divergence, final <= 0.03 for d=100,200.'
else:
    status = 'falsified'; notes = 'At least one d fails: error at 1e5 > error at 2e4, divergence, or final error > 0.03.'
notes += ' Measurement operator is randomized subsampled DCT (RIP ensemble) instead of dense Gaussian for tractability; X*=QQ^T, Q orthonormal d x 5.'
metrics['total_seconds'] = round(time.time() - t0, 1)
summary = dict(claim_id='C4', status=status, metrics=metrics, notes=notes)
print('SUMMARY_JSON=' + json.dumps(summary, default=str))
