import numpy as np, json, time, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

T0 = time.time()
rng = np.random.default_rng(0)
d, r = 100, 5
m = 5 * d * r
out = 'results/c6'
os.makedirs(out, exist_ok=True)

# ---- measurements: A_i = (Q+Q^T)/2, Q iid N(0,1) (isotropic on symmetric matrices) ----
A = np.empty((m, d * d))
for s in range(0, m, 250):
    Q = rng.standard_normal((250, d, d))
    A[s:s + 250] = ((Q + Q.transpose(0, 2, 1)) / 2).reshape(250, -1)
    del Q
Ustar = np.linalg.qr(rng.standard_normal((d, r)))[0]
Xstar = Ustar @ Ustar.T  # rank r, spectral norm 1
y = A @ Xstar.ravel()
Xs_norm = np.linalg.norm(Xstar)

def errs(U):
    X = U @ U.T
    res = A @ X.ravel() - y
    train = 0.5 * np.mean(res ** 2)  # f(U) as in eq (1.1)
    test = np.linalg.norm(X - Xstar) ** 2  # ||UU^T - X*||_F^2 (eq 3.1)
    reltest = np.sqrt(test) / Xs_norm
    return train, test, reltest

# ---- POSITIVE CONTROL: explicit interpolant X* + N with N in null space of measurement operator ----
G = rng.standard_normal((d, d)); N = (G + G.T) / 2
nv = N.ravel()
coef = np.linalg.solve(A @ A.T, A @ nv)
nv = nv - A.T @ coef
N = nv.reshape(d, d); N = (N + N.T) / 2
Xc = Xstar + N
res_c = A @ Xc.ravel() - y
ctrl_train = 0.5 * np.mean(res_c ** 2)
ctrl_test = np.linalg.norm(Xc - Xstar) ** 2
ctrl_expected = np.linalg.norm(N) ** 2
ctrl_rel = np.sqrt(ctrl_test) / Xs_norm
control_pass = bool(ctrl_train < 1e-12 and abs(ctrl_test - ctrl_expected) / ctrl_expected < 1e-8 and ctrl_rel > 0.1)

# ---- step size for GD: power iteration on M(.) = (1/m) sum <A_i,.> A_i ----
v = rng.standard_normal(d * d)
for _ in range(50):
    v = A.T @ (A @ v) / m; lam = np.linalg.norm(v); v /= lam
eta_gd = 0.3 / lam

def run_gd(U, eta, max_it, tol):
    hist = []
    for t in range(max_it):
        X = U @ U.T
        res = A @ X.ravel() - y
        if t % 50 == 0:
            tr = 0.5 * np.mean(res ** 2); te = np.linalg.norm(X - Xstar) / Xs_norm
            hist.append((t, tr, te))
            if tr <= tol or not np.isfinite(tr):
                break
        M = (A.T @ res).reshape(d, d) / m
        U = U - eta * M @ U
    return U, hist

# ---- GD from identity ----
U_gd, hist_gd = run_gd(np.eye(d), eta_gd, 60000, 1e-8)
gd_train, gd_test, gd_rel = errs(U_gd)
t_gd = time.time() - T0

# ---- contrast: GD from small init (paper's theorem regime) ----
U_si, hist_si = run_gd(1e-3 * np.eye(d), eta_gd, 5000, 1e-10)
si_train, si_test, si_rel = errs(U_si)

# ---- SGD from identity, step 8e-5, one sample per step, reduced budget ----
eta_sgd = 8e-5
max_steps = 4_000_000
time_cap = 230.0
U = np.eye(d).copy()
hist_sgd = []
t_start = time.time(); step = 0; chunk = 20000
tr, te, rel = errs(U); hist_sgd.append((0, tr, rel))
diverged = False
while step < max_steps:
    idx = rng.integers(0, m, size=chunk)
    for i in idx:
        Ai = A[i].reshape(d, d)
        W = Ai @ U
        g = np.sum(W * U) - y[i]
        U -= (eta_sgd * g) * W
    step += chunk
    tr, te, rel = errs(U); hist_sgd.append((step, tr, rel))
    if not np.isfinite(tr):
        diverged = True; break
    if tr <= 1e-8 or time.time() - t_start > time_cap:
        break
sgd_train, sgd_test, sgd_rel = errs(U)
sgd_steps = step

# ---- plot ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
h = np.array(hist_gd)
ax[0].semilogy(h[:, 0], h[:, 1], label='GD train f(U)'); ax[0].semilogy(h[:, 0], h[:, 2], label='GD rel test')
h2 = np.array(hist_si)
ax[0].semilogy(h2[:, 0], h2[:, 2], '--', label='GD small-init rel test')
ax[0].set_xlabel('iteration'); ax[0].legend(); ax[0].set_title('GD, U0=I')
h = np.array(hist_sgd)
ax[1].semilogy(h[:, 0], h[:, 1], label='SGD train f(U)'); ax[1].semilogy(h[:, 0], h[:, 2], label='SGD rel test')
ax[1].set_xlabel('SGD step'); ax[1].legend(); ax[1].set_title('SGD, U0=I, eta=8e-5')
plt.tight_layout(); plt.savefig(out + '/fig5_repro.png', dpi=110)

# ---- verdict ----
TR_TOL, TE_MIN = 1e-6, 0.1
gd_fit = gd_train <= TR_TOL; sgd_fit = sgd_train <= TR_TOL
gd_ok = gd_fit and gd_rel >= TE_MIN
sgd_ok = sgd_fit and sgd_rel >= TE_MIN
if not control_pass:
    status = 'inconclusive'; note = 'positive control failed; statistic unreliable.'
elif gd_ok and sgd_ok:
    status = 'supported'; note = 'Both GD and SGD from U0=I reach training error <=1e-6 while relative test error stays >=0.1.'
elif (gd_fit and gd_rel < TE_MIN) or (sgd_fit and sgd_rel < TE_MIN):
    status = 'falsified'; note = 'An algorithm fit training data but generalized (rel test < 0.1).'
else:
    status = 'inconclusive'
    note = 'Training error did not reach <=1e-6 for ' + ('GD ' if not gd_fit else '') + ('SGD (reduced budget %d steps vs paper 4e6)' % sgd_steps if not sgd_fit else '') + '; test error behaviour: GD rel %.3g, SGD rel %.3g.' % (gd_rel, sgd_rel)
note += ' Small-init GD contrast rel test=%.3g. Train error = f(U)=1/(2m)sum r_i^2; test = ||UU^T-X*||_F^2.' % si_rel

summary = dict(claim_id='C6', status=status, metrics=dict(
    d=d, r=r, m=m, eta_gd=float(eta_gd), gd_iters=int(hist_gd[-1][0]), gd_train=float(gd_train), gd_test_sq=float(gd_test), gd_rel_test=float(gd_rel),
    sgd_eta=eta_sgd, sgd_steps=int(sgd_steps), sgd_diverged=diverged, sgd_train=float(sgd_train), sgd_test_sq=float(sgd_test), sgd_rel_test=float(sgd_rel),
    smallinit_train=float(si_train), smallinit_rel_test=float(si_rel),
    ctrl_train=float(ctrl_train), ctrl_test=float(ctrl_test), ctrl_expected=float(ctrl_expected), control_pass=control_pass,
    runtime_s=float(time.time() - T0)), notes=note)
print('SUMMARY_JSON=' + json.dumps(summary, default=str))
