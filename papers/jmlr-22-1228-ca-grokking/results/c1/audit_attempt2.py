import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

np.random.seed(0)
os.makedirs('results/c1', exist_ok=True)

# ---- 1D exponential model (Sec 3.2, Zunkovic & Ilievski 2024) ----
# P+(x) = exp(-(x-eps)) Theta(x-eps), P-(x) = P+(-x)
eps = 1.0
N = 100
lam1 = 0.05
lam2 = 0.0

pos = eps + np.random.exponential(1.0, N)   # positive class
neg = -eps - np.random.exponential(1.0, N)  # negative class
x = np.concatenate([pos, neg])
xbar = float(np.mean(x))                    # training mean (small by symmetry)

max_neg = float(np.max(neg))
min_pos = float(np.min(pos))

# Closed-form dynamics (Eq. 5-6): db/dt = xbar - sgn(b)*lam1 - (1+lam2)*b
# solution b(t) = xbar_l - (xbar_l - b0) exp(-(1+lam2) t)
xbar_l = (xbar - lam1) / (1.0 + lam2)  # assume b >= 0 branch checked below
b0 = max_neg - 2.0                     # start below all negative samples

# b starts negative; while b<0 the fixed point is (xbar+lam1)/(1+lam2).
# For simplicity integrate the closed form piecewise: since db/dt>0 throughout
# and the crossing of b=0 is smooth, use exact ODE solution piecewise.
def b_of_t(t):
    # piecewise exact integration of db/dt = xbar - sgn(b)*lam1 - (1+lam2)*b
    # region 1: b<0, fixed point xbar_m = (xbar+lam1)/(1+lam2)
    xbar_m = (xbar + lam1) / (1.0 + lam2)
    xbar_p = (xbar - lam1) / (1.0 + lam2)
    rate = 1.0 + lam2
    if b0 < 0 and xbar_m > 0:
        t_cross = np.log((xbar_m - b0) / xbar_m) / rate  # time b reaches 0
    else:
        t_cross = np.inf
    t = np.asarray(t, dtype=float)
    b = np.empty_like(t)
    m1 = t <= t_cross
    b[m1] = xbar_m - (xbar_m - b0) * np.exp(-rate * t[m1])
    if np.any(~m1):
        b[~m1] = xbar_p - xbar_p * np.exp(-rate * (t[~m1] - t_cross))
    return b

def test_error(b):
    # analytic: E = 0.5*P+(x<b) + 0.5*P-(x>b)
    b = np.asarray(b, dtype=float)
    Ep = np.where(b > eps, 1.0 - np.exp(-(b - eps)), 0.0)
    Em = np.where(b < -eps, 1.0 - np.exp(-(-eps - b)), 0.0)
    return 0.5 * (Ep + Em)

def train_error(b):
    b = np.asarray(b, dtype=float)
    # fraction of training points misclassified by sgn(x-b)
    return ((pos[None, :] <= b[:, None]).sum(axis=1) + (neg[None, :] > b[:, None]).sum(axis=1)) / (2.0 * N)

# grokking condition: fixed point inside the gap -> E -> 0
gap_ok = (-eps < (xbar + lam1) / (1 + lam2)) and ((xbar - lam1) / (1 + lam2) < eps)

# time grid
t_max = 12.0
t = np.linspace(0, t_max, 20001)
b = b_of_t(t)
E = test_error(b)

# transition time t_eps: first time test error reaches 0 (b enters the gap).
# (Train error reaches zero earlier, at b=max_neg; the test-error critical
#  exponent of Eq. 9/10 is defined w.r.t. the test-error transition.)
idxE = np.argmax(E <= 1e-15)
t_eps = float(t[idxE]) if E[idxE] <= 1e-15 else np.nan
# train error zero time for reference
tr = train_error(b)
idxT = np.argmax(tr <= 0.0)
t_train = float(t[idxT]) if tr[idxT] == 0.0 else np.nan

Emax = float(np.max(E))

# ---- fit window: E in [1%, 50%] of Emax, t < t_eps ----
mask = (t < t_eps) & (E > 0.01 * Emax) & (E < 0.5 * Emax)
nu = np.nan
if mask.sum() >= 5:
    lx = np.log(t_eps - t[mask])
    ly = np.log(E[mask])
    A = np.polyfit(lx, ly, 1)
    nu = float(A[0])

# ---- POSITIVE CONTROL: exact synthetic power law with exponent 1 ----
# same window-selection and fitting machinery on data known to be E ~ (t_eps - t)^1
A_true = 0.7
E_ctrl = A_true * np.maximum(t_eps - t, 0.0)
E_ctrl = np.minimum(E_ctrl, Emax)  # mimic saturation away from transition
mctrl = (t < t_eps) & (E_ctrl > 0.01 * Emax) & (E_ctrl < 0.5 * Emax)
nu_ctrl = float(np.polyfit(np.log(t_eps - t[mctrl]), np.log(E_ctrl[mctrl]), 1)[0])
control_pass = bool(abs(nu_ctrl - 1.0) < 0.02)

# ---- plot ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].plot(t, E, label='test error E(t)')
ax[0].plot(t, tr, label='train error')
ax[0].axvline(t_eps, color='k', ls='--', label='t_eps (E->0)')
ax[0].set_xlabel('t'); ax[0].set_ylabel('error'); ax[0].legend(); ax[0].set_title('1D exponential model dynamics')
if mask.sum() >= 5:
    ax[1].loglog(t_eps - t[mask], E[mask], 'o', ms=3, label='E(t) in window')
    ax[1].loglog(t_eps - t[mask], np.exp(np.polyval(A, np.log(t_eps - t[mask]))), '-', label='fit: nu=%.3f' % nu)
ax[1].set_xlabel('t_eps - t'); ax[1].set_ylabel('E(t)'); ax[1].legend(); ax[1].set_title('log-log critical fit')
plt.tight_layout()
plt.savefig('results/c1/fig.png', dpi=120)

# ---- verdict ----
if not gap_ok:
    status = 'inconclusive'
    notes = 'Fixed point not inside gap; grokking does not occur for this draw.'
elif not control_pass:
    status = 'inconclusive'
    notes = 'Positive control failed (nu_ctrl=%.3f); fitting statistic is buggy.' % nu_ctrl
elif abs(nu - 1.0) <= 0.05:
    status = 'supported'
    notes = 'Fitted critical exponent nu=%.3f within 0.05 of 1.0 (Eq. 9/10).' % nu
else:
    status = 'falsified'
    notes = 'Fitted critical exponent nu=%.3f deviates from 1.0 by more than 0.05.' % nu

summary = {
    'claim_id': 'C1',
    'status': status,
    'metrics': {
        'nu_fitted': nu,
        'nu_control': nu_ctrl,
        'control_pass': control_pass,
        't_eps': t_eps,
        't_train_zero': t_train,
        'xbar': xbar,
        'xbar_fixed_point': xbar_l,
        'Emax': Emax,
        'n_fit_points': int(mask.sum())
    },
    'notes': notes
}
print('SUMMARY_JSON=' + json.dumps(summary, default=str))
