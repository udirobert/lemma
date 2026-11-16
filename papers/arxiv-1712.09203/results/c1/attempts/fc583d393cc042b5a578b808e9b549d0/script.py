import json, os, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'results/c1'
os.makedirs(OUT, exist_ok=True)

d = 100
r = 5
m = 5 * d * r
eta = 0.0025
alphas = [1e-1, 1e-2, 1e-3, 1e-4]
seeds = [0, 1, 2]
T_MIN = 10000
T_MAX = 25000
REC = 100


def tri(dd):
    i0, i1 = np.triu_indices(dd)
    ww = np.where(i0 == i1, 1.0, 2.0)
    return i0, i1, ww


def grad_and_loss(U, P, y, i0, i1, ww, mm):
    # U: (B, dd, dd); P: (mm, n) rows = upper-tri of A_i weighted by ww
    X = U @ np.transpose(U, (0, 2, 1))
    Xu = X[:, i0, i1]
    pred = (Xu.astype(P.dtype) @ P.T).astype(np.float64)
    res = pred - y[None, :]
    loss = 0.5 * np.mean(res ** 2, axis=1)
    g = (res.astype(P.dtype) @ P).astype(np.float64) / mm / ww[None, :]
    dd = U.shape[1]
    M = np.zeros((U.shape[0], dd, dd))
    M[:, i0, i1] = g
    M[:, i1, i0] = g
    G = 2.0 * (M @ U)  # grad of 1/(2m) sum (y_i - <A_i,UU^T>)^2 for symmetric A_i
    return G, loss, X


# ---------------- positive control ----------------
# (a) gradient implementation vs finite differences on tiny problem using full A matrices
rng = np.random.default_rng(123)
ds, ms = 6, 40
j0, j1, ws = tri(ds)
Q = rng.standard_normal((ms, ds, ds))
Ad = 0.5 * (Q + np.transpose(Q, (0, 2, 1)))
Ps = Ad[:, j0, j1] * ws[None, :]
Xt = rng.standard_normal((ds, 2))
Xt = Xt @ Xt.T
ys = np.einsum('ijk,jk->i', Ad, Xt)
fwd_err = float(np.max(np.abs(Ps @ Xt[j0, j1] - ys)))
U1 = 0.3 * rng.standard_normal((ds, ds))


def fsmall(Uu):
    Xx = Uu @ Uu.T
    return 0.5 * np.mean((np.einsum('ijk,jk->i', Ad, Xx) - ys) ** 2)


Gan, _, _ = grad_and_loss(U1[None], Ps, ys, j0, j1, ws, ms)
Gan = Gan[0]
Gfd = np.zeros_like(U1)
hh = 1e-6
for a in range(ds):
    for b in range(ds):
        E = np.zeros_like(U1)
        E[a, b] = hh
        Gfd[a, b] = (fsmall(U1 + E) - fsmall(U1 - E)) / (2 * hh)
grad_rel_err = float(np.linalg.norm(Gan - Gfd) / np.linalg.norm(Gfd))
impl_ok = bool(fwd_err < 1e-9 and grad_rel_err < 1e-5)

# (b) regression check: synthetic err = C*alpha with small multiplicative noise -> slope ~ 1
rc = np.random.default_rng(7)
syn = 0.5 * np.array(alphas) * np.exp(0.05 * rc.standard_normal(len(alphas)))
syn_slope = float(np.polyfit(np.log(alphas), np.log(syn), 1)[0])
reg_ok = bool(abs(syn_slope - 1.0) < 0.15)
control_pass = bool(impl_ok and reg_ok)

# ---------------- main experiment ----------------
i0, i1, w = tri(d)
n = len(i0)


def make_problem(seed):
    rg = np.random.default_rng(1000 + seed)
    Us = rg.standard_normal((d, r))
    Us = Us / np.linalg.norm(Us, axis=0, keepdims=True)
    Xs = Us @ Us.T
    sd = np.where(i0 == i1, 1.0, np.sqrt(0.5))  # entry std of A=(Q+Q^T)/2
    P = np.empty((m, n), dtype=np.float32)
    for st in range(0, m, 250):
        k = min(250, m - st)
        P[st:st + k] = (rg.standard_normal((k, n)) * sd * w).astype(np.float32)
    y = (P @ Xs[i0, i1].astype(np.float32)).astype(np.float64)
    return Xs, P, y


def run_gd(Xs, P, y):
    U = np.stack([a * np.eye(d) for a in alphas])
    ts, errs, losses = [], [], []
    t = 0
    while True:
        G, loss, X = grad_and_loss(U, P, y, i0, i1, w, m)
        if t % REC == 0:
            err = np.sum((X - Xs[None]) ** 2, axis=(1, 2))
            ts.append(t)
            errs.append(err)
            losses.append(loss)
            if not np.all(np.isfinite(err)):
                break
            if t >= T_MIN and np.all(loss < 1e-9):
                break
            if t >= T_MAX:
                break
        U = U - eta * G
        t += 1
    return np.array(ts), np.array(errs), np.array(losses)


t0 = time.time()
all_err, all_loss, trajs, steps = [], [], [], []
for s in seeds:
    Xs, P, y = make_problem(s)
    ts, errs, losses = run_gd(Xs, P, y)
    all_err.append(errs[-1])
    all_loss.append(losses[-1])
    trajs.append((ts, errs, losses))
    steps.append(int(ts[-1]))
    del P
all_err = np.array(all_err)
all_loss = np.array(all_loss)
mean_err = all_err.mean(axis=0)
mean_loss = all_loss.mean(axis=0)
runtime = time.time() - t0

finite = bool(np.all(np.isfinite(mean_err)) and np.all(mean_err > 0))
if finite:
    slope, icpt = np.polyfit(np.log(alphas), np.log(mean_err), 1)
else:
    slope, icpt = np.nan, np.nan
monotone = bool(np.all(np.diff(mean_err) < 0))
small_alpha_err = float(mean_err[-1])
small_alpha_loss = float(mean_loss[-1])
ratio_to_bound = (mean_err / (np.array(alphas) * np.sqrt(d))).tolist()

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].loglog(alphas, mean_err, 'o-', label='mean final ||UU^T-X*||_F^2')
ax[0].loglog(alphas, np.array(alphas) * np.sqrt(d), 'k--', label='alpha*sqrt(d)')
ax[0].set_xlabel('alpha')
ax[0].set_title('slope=%.2f' % slope)
ax[0].legend()
ts, errs, losses = trajs[0]
for k, a in enumerate(alphas):
    ax[1].semilogy(ts, errs[:, k], label='err a=%g' % a)
    ax[1].semilogy(ts, np.maximum(losses[:, k], 1e-16), ':', label='loss a=%g' % a)
ax[1].set_xlabel('iteration (seed 0)')
ax[1].legend(fontsize=6)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'c1_error_vs_alpha.png'), dpi=110)
plt.close()

converged = bool(small_alpha_loss < 1e-6)
if not control_pass:
    status = 'inconclusive'
    note = 'Positive control failed; implementation/statistic unreliable.'
elif (not finite) or (not converged):
    status = 'inconclusive'
    note = 'GD did not reach near-zero training loss at small alpha within budget (or diverged).'
elif slope > 0 and monotone and small_alpha_err < 1e-2:
    status = 'supported'
    note = 'Recovery error decreases monotonically as alpha decreases (log-log slope %.2f); err(1e-4)=%.3g with train loss %.2g.' % (slope, small_alpha_err, small_alpha_loss)
else:
    status = 'falsified'
    note = 'Training loss ~0 but recovery error does not decrease with alpha as claimed (slope %.2f, monotone=%s, err(1e-4)=%.3g).' % (slope, monotone, small_alpha_err)

summary = {
    'claim_id': 'C1',
    'status': status,
    'metrics': {
        'loglog_slope': float(slope),
        'monotone_decrease': monotone,
        'mean_final_err': [float(v) for v in mean_err],
        'mean_final_train_loss': [float(v) for v in mean_loss],
        'err_over_alpha_sqrt_d': [float(v) for v in ratio_to_bound],
        'per_seed_err': all_err.tolist(),
        'steps_run_per_seed': steps,
        'alphas': alphas, 'd': d, 'r': r, 'm': m, 'eta': eta,
        'control_pass': control_pass,
        'ctrl_fwd_err': fwd_err,
        'ctrl_grad_rel_err': grad_rel_err,
        'ctrl_syn_slope': syn_slope,
        'runtime_s': float(runtime),
    },
    'notes': note + ' Update uses true gradient 2*M(UU^T-X*)U with eta=0.0025, U0=alpha*I, A_i=(Q+Q^T)/2, m=5dr, 3 seeds.',
}
print('SUMMARY_JSON=' + json.dumps(summary, default=str))
