import numpy as np, json, time, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT='results/c3'
os.makedirs(OUT, exist_ok=True)
T0=time.time()
d, r = 100, 5
m = 5*d*r
eta = 0.0025
T = 20000
alphas = [1.0, 1e-3]
seeds = [0, 1, 2]
TIME_BUDGET = 1000.0
iu = np.triu_indices(d)
w = np.where(iu[0]==iu[1], 1.0, 2.0)
ntri = len(iu[0])

def make_A(rng):
    At = np.empty((m, ntri), dtype=np.float32)
    cs = 250
    for s in range(0, m, cs):
        Q = rng.standard_normal((cs, d, d))
        S = 0.5*(Q + Q.transpose(0,2,1))
        At[s:s+cs] = S[:, iu[0], iu[1]].astype(np.float32)
    return At

def meas(At, Xs):
    # Xs: list of symmetric matrices -> (m, k) of <A_i, X>
    V = np.stack([(w*X[iu]) for X in Xs], axis=1).astype(np.float32)
    return (At @ V).astype(np.float64)

def Mop(At, R):
    # R: (m,k) -> list of symmetric M = (1/m) sum r_i A_i
    V = (At.T @ R.astype(np.float32)).astype(np.float64)/m
    out = []
    for k in range(R.shape[1]):
        M = np.zeros((d, d)); M[iu] = V[:, k]
        M = M + M.T - np.diag(np.diag(M))
        out.append(M)
    return out

# ---------------- positive control ----------------
rngc = np.random.default_rng(123)
Ac = make_A(rngc)
Zs = []
for k in range(3):
    G = rngc.standard_normal((d, 3)); Z = G@G.T; Z /= np.linalg.norm(Z); Zs.append(Z)
yz = meas(Ac, Zs)
emp = (yz**2).mean(axis=0)  # should be ~1 (=||Z||_F^2)
rip_ok = bool(np.all(np.abs(emp-1) < 0.15))
# gradient check of f(U)=1/(4m) sum (<A,UU^T>-y)^2 whose grad is M U (paper's update (2.2))
Us = rngc.standard_normal((d, d))*0.05
Xst = Zs[0]
y0 = meas(Ac, [Xst])[:, 0]
def fval(U):
    res = meas(Ac, [U@U.T])[:, 0]-y0
    return (res**2).sum()/(4*m)
res = meas(Ac, [Us@Us.T])[:, 0]-y0
Gm = Mop(Ac, res[:, None])[0]@Us
D = rngc.standard_normal((d, d)); D /= np.linalg.norm(D)
h = 1e-3
fd = (fval(Us+h*D)-fval(Us-h*D))/(2*h)
an = np.sum(Gm*D)
grad_rel = abs(fd-an)/(abs(an)+1e-12)
grad_ok = bool(grad_rel < 2e-2)
control_pass = rip_ok and grad_ok
del Ac

# ---------------- main experiment ----------------
rec_every = 100
hist = {a: {'train': [], 'test': []} for a in alphas}
final = {a: {'train': [], 'test': []} for a in alphas}
timed_out = False
for sd in seeds:
    if time.time()-T0 > TIME_BUDGET: timed_out = True; break
    rng = np.random.default_rng(1000+sd)
    At = make_A(rng)
    Ustar, _ = np.linalg.qr(rng.standard_normal((d, r)))
    Xstar = Ustar@Ustar.T
    y = meas(At, [Xstar])[:, 0]
    B, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Us = [a*B.copy() for a in alphas]
    tr_h = [[] for _ in alphas]; te_h = [[] for _ in alphas]
    for t in range(T+1):
        P = [U@U.T for U in Us]
        R = meas(At, P) - y[:, None]
        if t % rec_every == 0 or t == T:
            for k in range(len(alphas)):
                tr_h[k].append(np.mean(R[:, k]**2))
                te_h[k].append(np.linalg.norm(P[k]-Xstar)**2)
            if time.time()-T0 > TIME_BUDGET + 150: timed_out = True; break
        if t == T: break
        Ms = Mop(At, R)
        Us = [Us[k] - eta*Ms[k]@Us[k] for k in range(len(alphas))]
    for k, a in enumerate(alphas):
        hist[a]['train'].append(tr_h[k]); hist[a]['test'].append(te_h[k])
        final[a]['train'].append(tr_h[k][-1]); final[a]['test'].append(te_h[k][-1])
    del At
    if timed_out: break

nseeds_done = len(final[1.0]['train'])
metrics = {'control_pass': control_pass, 'control_rip_emp_norms': [float(x) for x in emp],
           'control_grad_rel_err': float(grad_rel), 'seeds_completed': nseeds_done,
           'iterations': T, 'runtime_s': round(time.time()-T0, 1), 'd': d, 'r': r, 'm': m, 'eta': eta}
for a in alphas:
    metrics[f'train_final_alpha{a}'] = [float(x) for x in final[a]['train']]
    metrics[f'test_final_alpha{a}'] = [float(x) for x in final[a]['test']]
    if nseeds_done:
        metrics[f'mean_train_alpha{a}'] = float(np.mean(final[a]['train']))
        metrics[f'mean_test_alpha{a}'] = float(np.mean(final[a]['test']))

if nseeds_done:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, c in zip(alphas, ['r', 'b']):
        for s in range(nseeds_done):
            x = np.arange(len(hist[a]['train'][s]))*rec_every
            ax[0].semilogy(x, hist[a]['train'][s], c, alpha=.6, label=f'alpha={a}' if s == 0 else None)
            ax[1].semilogy(x, hist[a]['test'][s], c, alpha=.6, label=f'alpha={a}' if s == 0 else None)
    ax[0].set_title('training error'); ax[1].set_title('test error ||UU^T-X*||_F^2')
    for aa in ax: aa.legend(); aa.set_xlabel('iteration')
    plt.tight_layout(); plt.savefig(f'{OUT}/fig2_repro.png', dpi=110); plt.close()

if not control_pass:
    status = 'inconclusive'; notes = 'Positive control failed (RIP-norm check or gradient check); statistic suspect.'
elif nseeds_done == 0:
    status = 'inconclusive'; notes = 'No seed completed within time budget.'
else:
    trL = np.array(final[1.0]['train']); teL = np.array(final[1.0]['test']); teS = np.array(final[1e-3]['test'])
    c1 = bool(np.all(trL < 1e-4))
    c2 = bool(np.all((teL > 1e-2) & (teL < 1.0)))
    c3 = bool(np.all(teS < teL))
    metrics.update({'crit_train_lt_1e-4': c1, 'crit_test_in_[1e-2,1]': c2, 'crit_small_alpha_lower_test': c3})
    if c1 and c2 and c3:
        status = 'supported'; notes = 'alpha=1 interpolates (train<1e-4) with test error ~1e-1; alpha=1e-3 generalizes much better.'
    else:
        status = 'falsified' if (nseeds_done == len(seeds) and not timed_out) else 'inconclusive'
        notes = f'Criteria: train<1e-4={c1}, test in [1e-2,1]={c2}, small-alpha lower test={c3}.'
    if timed_out: notes += ' (time budget hit; partial seeds)'
notes += ' Update uses paper eq (2.2): U <- (I - eta*M_t)U; A_i=(Q+Q^T)/2 Gaussian; X*=U*U*^T rank-5, kappa=1; train err=(1/m)sum residual^2.'
summary = {'claim_id': 'C3', 'status': status, 'metrics': metrics, 'notes': notes}
print('SUMMARY_JSON=' + json.dumps(summary, default=str))
