import json, os, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out = 'results/c1'
os.makedirs(out, exist_ok=True)
t0 = time.time()

d = 100
r = 5
m = 5 * d * r
eta = 0.0025
T = 10000
alphas = np.array([1e-1, 1e-2, 1e-3, 1e-4], dtype=np.float64)
seeds = [0, 1, 2]

iu0, iu1 = np.triu_indices(d)
p = len(iu0)
is_diag = (iu0 == iu1)
c = np.where(is_diag, 1.0, 2.0)
diag_idx = np.arange(d)

def make_problem(seed):
    rng = np.random.default_rng(seed)
    Us = rng.standard_normal((d, r))
    Us /= np.linalg.norm(Us, axis=0, keepdims=True)
    Xs = Us @ Us.T
    Amat = np.empty((m, p), dtype=np.float64)
    Amat[:, is_diag] = rng.standard_normal((m, int(is_diag.sum())))
    Amat[:, ~is_diag] = rng.standard_normal((m, int((~is_diag).sum()))) / np.sqrt(2.0)
    Xstar_upper = Xs[iu0, iu1]
    y = Amat @ (c * Xstar_upper)
    return Xs, Amat, y, Xstar_upper

def run_gd(Amat, y, Xs, alpha, T, population=False):
    U = alpha * np.eye(d, dtype=np.float64)
    for t in range(T):
        X = U @ U.T
        Xu = X[iu0, iu1]
        if population:
            M = X - Xs
        else:
            meas = Amat @ (c * Xu)
            resid = meas - y
            Mu = (resid @ Amat) / m
            M = np.zeros((d, d), dtype=np.float64)
            M[iu0, iu1] = Mu
            M = M + M.T
            M[diag_idx, diag_idx] *= 0.5
        U = U - eta * (M @ U)
        if not np.all(np.isfinite(U)):
            break
    X = U @ U.T
    err = float(np.sum((X - Xs) ** 2))
    if population:
        loss = float(0.5 * np.sum((X - Xs) ** 2))
    else:
        meas = Amat @ (c * X[iu0, iu1])
        resid = meas - y
        loss = float(0.5 * np.mean(resid ** 2))
    return err, loss

def aggregate(alphas, seed_results):
    errs = np.array([[r[a][0] for a in range(len(alphas))] for r in seed_results])
    losses = np.array([[r[a][1] for a in range(len(alphas))] for r in seed_results])
    return errs.mean(axis=0), losses.mean(axis=0)

# Main experiment
main_seed_results = []
for seed in seeds:
    Xs, Amat, y, _ = make_problem(seed)
    res = []
    for a in alphas:
        err, loss = run_gd(Amat, y, Xs, float(a), T, population=False)
        res.append((err, loss))
    main_seed_results.append(res)
    del Amat

main_err, main_loss = aggregate(alphas, main_seed_results)
la = np.log(alphas)
le = np.log(np.maximum(main_err, 1e-300))
slope_main, intercept_main = np.polyfit(la, le, 1)
order = np.argsort(alphas)  # ascending alpha
errs_sorted = main_err[order]
mono_main = bool(np.all(np.diff(errs_sorted) > 0))  # err increases with alpha
small_err_main = float(main_err[np.argmin(alphas)])
small_loss_main = float(main_loss[np.argmin(alphas)])

# Positive control: population gradient (exact expectation / perfect RIP)
ctrl_seed_results = []
for seed in seeds:
    rng = np.random.default_rng(seed + 1000)
    Us = rng.standard_normal((d, r))
    Us /= np.linalg.norm(Us, axis=0, keepdims=True)
    Xs = Us @ Us.T
    res = []
    for a in alphas:
        err, loss = run_gd(None, None, Xs, float(a), T, population=True)
        res.append((err, loss))
    ctrl_seed_results.append(res)

crtl_err, crtl_loss = aggregate(alphas, ctrl_seed_results)
le_c = np.log(np.maximum(crtl_err, 1e-300))
slope_ctrl, intercept_ctrl = np.polyfit(la, le_c, 1)
small_err_ctrl = float(crtl_err[np.argmin(alphas)])
small_loss_ctrl = float(crtl_loss[np.argmin(alphas)])
control_pass = bool(
    np.all(np.isfinite(crtl_err)) and slope_ctrl > 0 and small_err_ctrl < 1e-1
)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].loglog(alphas, main_err, 'o-', label='empirical')
axes[0].loglog(alphas, crtl_err, 's--', label='population control')
axes[0].set_xlabel('alpha')
axes[0].set_ylabel('final ||UU^T-X*||_F^2')
axes[0].set_title('Recovery error vs initialization')
axes[0].legend()
axes[0].grid(True, which='both', alpha=0.3)
axes[1].loglog(alphas, main_loss, 'o-', label='empirical loss')
axes[1].loglog(alphas, crtl_loss, 's--', label='population loss')
axes[1].set_xlabel('alpha')
axes[1].set_ylabel('final training loss')
axes[1].set_title('Training loss vs initialization')
axes[1].legend()
axes[1].grid(True, which='both', alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(out, 'fig.png'), dpi=150)
plt.close(fig)

pass_main = bool(
    np.all(np.isfinite(main_err))
    and slope_main > 0
    and mono_main
    and small_err_main < 1e-1
    and small_loss_main < 1e-2
)
if not control_pass:
    status = 'inconclusive'
elif pass_main:
    status = 'supported'
else:
    status = 'falsified'

summary = {
    'claim_id': 'C1',
    'status': status,
    'metrics': {
        'slope_main': float(slope_main),
        'intercept_main': float(intercept_main),
        'err_alpha_1e-1': float(main_err[0]),
        'err_alpha_1e-2': float(main_err[1]),
        'err_alpha_1e-3': float(main_err[2]),
        'err_alpha_1e-4': float(main_err[3]),
        'loss_small_alpha': float(small_loss_main),
        'mono_main': bool(mono_main),
        'slope_control': float(slope_ctrl),
        'err_control_small_alpha': float(small_err_ctrl),
        'control_pass': bool(control_pass),
        'runtime_sec': float(time.time() - t0),
    },
    'notes': 'GD with U0=alpha I on d=100, r=5, m=5dr symmetric Gaussian measurements; 10000 iterations, 3 seeds. Population-gradient run is positive control.',
}
print('SUMMARY_JSON=' + json.dumps(summary, default=str))
