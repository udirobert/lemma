import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Shifted 1D exponential model: training center mu = epsilon + s, s = epsilon.
epsilon = 1.0
s = epsilon
mu = epsilon + s
lambda1 = 0.5
lambda2 = 0.0
a = mu * mu + 2.0

def ode(t, y):
    w, b = y
    sw = 0.0 if w == 0 else float(np.sign(w))
    dw = -a * w - mu * b + 1.0 - lambda1 * sw
    db = -mu * w - (1.0 + lambda2) * b
    return [dw, db]

def z_of(y):
    return y[1] + epsilon * y[0]

def event(t, y):
    return z_of(y)
event.terminal = True
event.direction = -1

sol = solve_ivp(
    ode,
    [0.0, 30.0],
    [0.0, 0.0],
    method='RK45',
    events=event,
    dense_output=True,
    rtol=1e-10,
    atol=1e-12,
    max_step=0.05,
)

t_eps = None
if sol.t_events is not None and len(sol.t_events) > 0 and len(sol.t_events[0]) > 0:
    t_eps = float(sol.t_events[0][0])

if t_eps is None:
    t_eps = float(sol.t[-1])

# Sample the pre-transition trajectory.
ts = np.linspace(0.0, t_eps, 50001)
W, B = sol.sol(ts)
Z = B + epsilon * W
absW = np.abs(W)
ratio = np.where(absW > 1e-12, np.abs(Z) / absW, epsilon)
E = 0.5 * (1.0 - np.exp(-ratio))

delta = t_eps - ts
mask = (delta <= 0.1) & (delta >= 1e-3) & (ts >= 0.0)
valid = mask & (E > 0.0) & np.isfinite(E) & np.isfinite(delta)

if np.count_nonzero(valid) >= 2:
    logd = np.log(delta[valid])
    loge = np.log(E[valid])
    coef = np.polyfit(logd, loge, 1)
    nu = float(coef[0])
else:
    nu = float('nan')

n_fit_points = int(np.count_nonzero(valid))
delta_min = float(np.min(delta[valid])) if n_fit_points > 0 else float('nan')
delta_max = float(np.max(delta[valid])) if n_fit_points > 0 else float('nan')

w_cross = float(sol.sol(t_eps)[0])
b_cross = float(sol.sol(t_eps)[1])
signw_cross = 0.0 if w_cross == 0 else float(np.sign(w_cross))
dzdt_cross = (
    -mu * w_cross - (1.0 + lambda2) * b_cross
    + epsilon * (-a * w_cross - mu * b_cross + 1.0 - lambda1 * signw_cross)
)

# Positive control: exact linear data E = A (t_eps - t)^1.
t_eps_c = 5.0
tc = np.linspace(t_eps_c - 0.1, t_eps_c - 1e-3, 100)
Ec = 2.3 * (t_eps_c - tc) ** 1.0
control_coef = np.polyfit(np.log(t_eps_c - tc), np.log(Ec), 1)
control_slope = float(control_coef[0])
control_pass = bool(abs(control_slope - 1.0) <= 0.05)

if not control_pass:
    status = 'inconclusive'
elif n_fit_points < 5 or not np.isfinite(nu):
    status = 'inconclusive'
elif abs(nu - 1.0) <= 0.05:
    status = 'supported'
else:
    status = 'falsified'

# Plot.
os.makedirs('results/c1', exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ax = axes[0]
ax.loglog(delta[valid], E[valid], '.', ms=2, label='model')
if np.count_nonzero(valid) >= 2:
    dd = np.array([delta[valid].min(), delta[valid].max()])
    ax.loglog(dd, np.exp(coef[1]) * dd ** coef[0], 'r-', label=f'fit nu={nu:.4f}')
ax.set_xlabel(r'$t_\epsilon - t$')
ax.set_ylabel(r'$E(t)$')
ax.set_title('1D exponential model')
ax.legend()

ax2 = axes[1]
ax2.loglog(t_eps_c - tc, Ec, '.', ms=2, label='control')
ax2.loglog(
    t_eps_c - tc,
    np.exp(control_coef[1]) * (t_eps_c - tc) ** control_slope,
    'r-',
    label=f'slope={control_slope:.6f}',
)
ax2.set_xlabel(r'$t_\epsilon - t$')
ax2.set_ylabel(r'$E(t)$')
ax2.set_title('Positive control')
ax2.legend()

plt.tight_layout()
plt.savefig('results/c1/fig.png', dpi=150, bbox_inches='tight')
plt.close()

summary = {
    'claim_id': 'C1',
    'status': status,
    'metrics': {
        'nu': nu,
        't_eps': t_eps,
        'n_fit_points': n_fit_points,
        'delta_min': delta_min,
        'delta_max': delta_max,
        'w_at_crossing': w_cross,
        'dzdt_at_crossing': float(dzdt_cross),
        'mu': float(mu),
        'lambda1': float(lambda1),
        'lambda2': float(lambda2),
        'epsilon': float(epsilon),
        'control_slope': control_slope,
        'control_pass': control_pass,
    },
    'notes': 'Shifted training distribution (mu=2*epsilon) with local fit over 1e-3 <= t_eps-t <= 0.1.',
}
print('SUMMARY_JSON=' + json.dumps(summary, default=str))
