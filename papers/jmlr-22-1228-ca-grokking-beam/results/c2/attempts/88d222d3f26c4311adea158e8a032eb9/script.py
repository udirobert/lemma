import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import betainc

OUT = 'results/c2'
os.makedirs(OUT, exist_ok=True)


def sample_ball(N, D, rng):
    X = rng.normal(size=(N, D))
    norms = np.linalg.norm(X, axis=1)
    X = X / norms[:, None]
    r = rng.random(N) ** (1.0 / D)
    return X * r[:, None]


def cap_cdf(delta, D):
    """P(n . Y < delta) for Y uniform in unit ball in R^D."""
    delta = float(delta)
    if delta <= -1.0:
        return 0.0
    if delta >= 1.0:
        return 1.0
    a = 0.5
    b = (D + 1) / 2.0
    return float(0.5 + 0.5 * np.sign(delta) * betainc(a, b, delta * delta))


def fit_exponent(ts, E, t_eps):
    dt = t_eps - ts
    mask = (E > 1e-15) & (dt > 0)
    sel = mask & (dt >= 1e-3) & (dt <= 2.0)
    if sel.sum() < 10:
        idx = np.where(mask)[0]
        sel = np.zeros_like(mask, dtype=bool)
        sel[idx[-20:]] = True
    x = np.log(dt[sel])
    y = np.log(E[sel])
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept), sel


# Positive control: exact power law with known exponent.
def control_fit(alpha):
    ts = np.linspace(0.0, 0.999, 2000)
    t_eps = 1.0
    E = (t_eps - ts) ** alpha
    nu, _, _ = fit_exponent(ts, E, t_eps)
    return nu


Ds = [2, 5, 10]
N = 500
epsilon = 2.0
rng = np.random.default_rng(42)

results = {}
all_ts = {}
all_E = {}
all_fits = {}

for D in Ds:
    e1 = np.zeros(D)
    e1[0] = 1.0
    s = np.zeros(D)
    if D > 1:
        s[1] = 1.0

    pos = sample_ball(N, D, rng) + epsilon * e1[None, :]
    neg = sample_ball(N, D, rng) - epsilon * e1[None, :]
    X = np.vstack([pos, neg])
    y = np.array([1] * N + [-1] * N, dtype=float)
    Xtilde = X - s[None, :]
    M = 2 * N
    A = (Xtilde.T @ Xtilde) / M
    c = (Xtilde.T @ y) / M

    w0 = np.zeros(D)
    lam, Q = np.linalg.eigh(A)
    c_rot = Q.T @ c
    # avoid division by zero
    lam = np.where(lam < 1e-12, 1e-12, lam)

    ts = np.linspace(0.0, 30.0, 3001)
    E = np.zeros_like(ts)
    for i, t in enumerate(ts):
        coeff = (1.0 - np.exp(-lam * t)) / lam
        w = Q @ (coeff * c_rot)
        nw = np.linalg.norm(w)
        if nw < 1e-15:
            # no separation yet; error ~ 0.5
            E[i] = 0.5
            continue
        n = w / nw
        mu_pos = epsilon * e1
        mu_neg = -epsilon * e1
        delta_pos = float(n @ (s - mu_pos))
        delta_neg = float(n @ (s - mu_neg))
        p_pos = cap_cdf(delta_pos, D)
        p_neg = cap_cdf(-delta_neg, D)
        E[i] = 0.5 * p_pos + 0.5 * p_neg

    # find transition time: first zero test error
    zero_mask = E <= 1e-15
    if zero_mask.any():
        t_eps = float(ts[np.argmax(zero_mask)])
    else:
        t_eps = float(ts[-1])

    nu, intercept, sel = fit_exponent(ts, E, t_eps)
    target = (D + 1) / 2.0
    results[D] = {
        'nu': nu,
        'target': target,
        'abs_err': abs(nu - target),
        't_eps': t_eps,
        'n_fit_points': int(sel.sum()),
    }
    all_ts[D] = ts
    all_E[D] = E
    all_fits[D] = (nu, intercept, sel, t_eps)

# Positive control for each D's target exponent.
control_passes = []
for D in Ds:
    alpha = (D + 1) / 2.0
    nu_c = control_fit(alpha)
    control_passes.append(abs(nu_c - alpha) < 0.1)
control_pass = bool(all(control_passes))

# Plot log-log fits.
fig, axes = plt.subplots(1, len(Ds), figsize=(12, 4))
if len(Ds) == 1:
    axes = [axes]
for ax, D in zip(axes, Ds):
    ts = all_ts[D]
    E = all_E[D]
    nu, intercept, sel, t_eps = all_fits[D]
    dt = t_eps - ts
    mask = (E > 1e-15) & (dt > 0)
    ax.scatter(np.log(dt[mask]), np.log(E[mask]), s=4, alpha=0.4, label='simulation')
    xs = np.linspace(np.log(dt[sel].min()), np.log(dt[sel].max()), 100)
    ax.plot(xs, nu * xs + intercept, 'r-', lw=2, label=f'fit nu={nu:.3f}')
    ax.set_title(f'D={D}, target={(D+1)/2:.1f}')
    ax.set_xlabel('log(t_eps - t)')
    ax.set_ylabel('log E_D(t)')
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig.png'), dpi=150)
plt.close(fig)

max_abs_err = max(results[D]['abs_err'] for D in Ds)
if (not control_pass) or max_abs_err >= 0.1:
    if not control_pass:
        status = 'inconclusive'
    else:
        status = 'falsified'
else:
    status = 'supported'

metrics = {
    'control_pass': control_pass,
    'max_abs_err': float(max_abs_err),
}
for D in Ds:
    metrics[f'D{D}_nu'] = float(results[D]['nu'])
    metrics[f'D{D}_target'] = float(results[D]['target'])
    metrics[f'D{D}_abs_err'] = float(results[D]['abs_err'])
    metrics[f'D{D}_t_eps'] = float(results[D]['t_eps'])
    metrics[f'D{D}_n_fit_points'] = int(results[D]['n_fit_points'])

summary = {
    'claim_id': 'C2',
    'status': status,
    'metrics': metrics,
    'notes': 'Least-squares perceptron gradient flow on shifted uniform-ball data; exact spherical-cap test error; power-law fit near transition.',
}
print('SUMMARY_JSON=' + json.dumps(summary, default=str))
