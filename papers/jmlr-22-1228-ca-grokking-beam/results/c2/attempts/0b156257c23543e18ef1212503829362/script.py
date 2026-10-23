import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brentq

CLAIM_ID = 'C2'
OUT_DIR = 'results/c2'
os.makedirs(OUT_DIR, exist_ok=True)

EPSILON = 2.0
N = 20000
SEED = 42


def sample_ball(n, d, rng):
    X = rng.normal(size=(n, d))
    norms = np.linalg.norm(X, axis=1)
    norms[norms == 0] = 1.0
    X = X / norms[:, None]
    r = rng.random(n) ** (1.0 / d)
    return X * r[:, None]


def fit_exponent(deltas, errors):
    deltas = np.asarray(deltas, dtype=float)
    errors = np.asarray(errors, dtype=float)
    mask = (deltas > 0) & (errors > 0)
    x = np.log(deltas[mask])
    y = np.log(errors[mask])
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def make_shift(D):
    s = np.zeros(D)
    s[0] = 1.5
    if D > 1:
        s[1] = 1.0
    return s


def cap_cdf(D, z0):
    """P(Z <= z0) for projection of uniform unit ball in R^D."""
    def dens(z):
        return (1.0 - z * z) ** ((D - 1) / 2.0)
    num, _ = quad(dens, -1.0, z0, limit=300)
    den, _ = quad(dens, -1.0, 1.0, limit=300)
    return num / den


rng = np.random.default_rng(SEED)
metrics = {}
all_supported = True
control_pass = True

# Positive control for the fitting routine.
ctrl_delta = np.geomspace(1e-3, 5.0, 50)
ctrl_error = 2.3 * ctrl_delta ** 2.7
nu_ctrl = fit_exponent(ctrl_delta, ctrl_error)
control_pass = bool(abs(nu_ctrl - 2.7) < 1e-9)
metrics['control_pass'] = control_pass
metrics['control_nu'] = nu_ctrl

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, D in zip(axes, [2, 5, 10]):
    s = make_shift(D)
    e1 = np.zeros(D)
    e1[0] = 1.0

    pos = sample_ball(N, D, rng) + EPSILON * e1[None, :]
    neg = sample_ball(N, D, rng) - EPSILON * e1[None, :]
    X = np.vstack([pos, neg])
    y = np.concatenate([np.ones(N), -np.ones(N)])
    Xtilde = X - s[None, :]

    A = (Xtilde.T @ Xtilde) / (2.0 * N)
    c = np.mean(y[:, None] * Xtilde, axis=0)

    # Symmetric eigendecomposition for matrix exponential dynamics.
    w_evals, w_vecs = np.linalg.eigh(A)
    c_proj = w_vecs.T @ c

    def w_at(t):
        if t <= 0:
            return np.zeros(D)
        factors = (1.0 - np.exp(-w_evals * t)) / w_evals
        return w_vecs @ (factors * c_proj)

    def a_at(t):
        wt = w_at(t)
        nrm = np.linalg.norm(wt)
        if nrm < 1e-14:
            return 0.5
        v = EPSILON * e1 - s
        return float(wt @ v / nrm)

    ts = np.linspace(0.0, 30.0, 3001)
    try:
        t_eps = float(brentq(lambda t: a_at(t) - 1.0, 1e-8, 30.0, xtol=1e-12, rtol=1e-12, maxiter=200))
    except Exception:
        t_eps = float('nan')
        metrics[f'D{D}_t_eps'] = t_eps
        metrics[f'D{D}_nu'] = float('nan')
        metrics[f'D{D}_n_points'] = 0
        metrics[f'D{D}_E_final'] = float('nan')
        all_supported = False
        continue

    # Fit over the requested scaling window.
    deltas = []
    errors = []
    for t in ts[1:]:
        if not (t < t_eps):
            continue
        delta = t_eps - t
        if delta < 1e-3 or delta > 5.0:
            continue
        a = a_at(t)
        if a >= 1.0:
            continue
        E = 0.5 * cap_cdf(D, -a)
        if E <= 0:
            continue
        deltas.append(delta)
        errors.append(E)

    deltas = np.array(deltas, dtype=float)
    errors = np.array(errors, dtype=float)
    nu = fit_exponent(deltas, errors) if len(deltas) >= 5 else float('nan')

    # Final classifier error after transition.
    a_final = a_at(30.0)
    if a_final >= 1.0:
        E_final = 0.0
    else:
        E_final = 0.5 * cap_cdf(D, -a_final)

    pred = (D + 1) / 2.0
    metrics[f'D{D}_t_eps'] = t_eps
    metrics[f'D{D}_nu'] = nu
    metrics[f'D{D}_nu_pred'] = pred
    metrics[f'D{D}_nu_abs_err'] = abs(nu - pred)
    metrics[f'D{D}_n_points'] = int(len(deltas))
    metrics[f'D{D}_E_final'] = float(E_final)

    if (not np.isfinite(nu)) or abs(nu - pred) > 0.1 or E_final > 1e-3 or len(deltas) < 20:
        all_supported = False

    ax.loglog(deltas, errors, '.', ms=2, label='data')
    if np.isfinite(nu):
        fit_line = np.exp(np.polyval([nu, np.log(errors[0]) - nu * np.log(deltas[0])], np.log(deltas)))
        ax.loglog(deltas, fit_line, '-', lw=1.5, label=f'fit nu={nu:.3f}')
    ax.set_title(f'D={D}')
    ax.set_xlabel(r'$t_\epsilon - t$')
    ax.set_ylabel(r'$E_D(t)$')
    ax.legend(fontsize=8)
    ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig.png'), dpi=150)
plt.close()

if not control_pass:
    status = 'inconclusive'
elif all_supported:
    status = 'supported'
else:
    # Distinguish a genuine miss from an obvious setup failure.
    setup_failed = any(metrics.get(f'D{D}_E_final', 1.0) > 1e-3 or metrics.get(f'D{D}_n_points', 0) < 20 for D in [2, 5, 10])
    status = 'inconclusive' if setup_failed else 'falsified'

summary = {
    'claim_id': CLAIM_ID,
    'status': status,
    'metrics': metrics,
    'notes': 'D-dimensional uniform ball model with nontrivial shift; population-trained perceptron dynamics; test error from spherical-cap volume; log-log exponent fit.'
}
print('SUMMARY_JSON=' + json.dumps(summary, default=str))
