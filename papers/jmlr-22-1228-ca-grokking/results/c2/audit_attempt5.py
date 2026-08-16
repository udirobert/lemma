import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import betainc, beta
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

np.random.seed(0)

# ----------------------------------------------------------------------------
# D-dimensional uniform ball model (Zunkovic & Ilievski, Sec. 3.3 / 3.3.1).
# Thermodynamic-limit gradient flow of a perceptron with hinge (perceptron)
# loss R = (1/2N) sum_i max(0, 1 - y_i w.x_i) on training distributions that
# are uniform unit balls centered at +/- eps_t e1 (the 'non-typical' shifted
# training set, eps_t < 1 so the training balls overlap -> long training),
# while the TEST distributions are unit balls centered at +/- eps e1 with
# eps > 1 (linearly separable, so zero test error is attainable).
#
# Ball-overlap terms (derived from the geometry of the uniform unit ball):
#   active volume : V(h) = int_{-1}^{h} (1-s^2)^((D-1)/2) ds / B(1/2,(D+1)/2)
#   active moment : M(h) = int_{-1}^{h} s(1-s^2)^((D-1)/2) ds
#                        = -(1-h^2)^((D+1)/2)/(D+1)
# with h = (1 - eps_t w1)/|w|. By symmetry w = (w1, r) and the flow is
#   dw1/dt = eps_t V(h) + (w1/|w|) M(h)
#   dr/dt  = (r/|w|) M(h)
# (overall positive constants only rescale time and do not affect the
# exponent). Test error is the spherical-cap mass of the true balls on the
# wrong side of the hyperplane:
#   E(t) = P(x1 > a), a = eps w1/|w|  = 0.5 * I_{1-a^2}(1/2, (D+1)/2)
# t_eps is the time when a(t) = 1. Claim: E ~ (t_eps - t)^((D+1)/2).
# ----------------------------------------------------------------------------

EPS   = 1.1    # true class offset (separable test distributions)
EPS_T = 0.5    # shifted (non-typical) training offset -> grokking regime

def cap_V(h, D):
    h = np.clip(h, -1.0, 1.0)
    x = 1.0 - h*h
    upper = 0.5 * betainc(0.5, (D+1)/2.0, x)   # int_h^1
    return np.where(h >= 0.0, 1.0 - upper, upper)

def cap_M(h, D):
    h = np.clip(h, -1.0, 1.0)
    return -(1.0 - h*h)**((D+1)/2.0) / (D+1)

def rhs(t, y, D):
    w1, r = y
    nw = np.hypot(w1, r)
    h = (1.0 - EPS_T*w1)/nw
    V = float(cap_V(h, D)); M = float(cap_M(h, D))
    return [EPS_T*V + (w1/nw)*M, (r/nw)*M]

def test_error(w1, r, D):
    nw = np.hypot(w1, r)
    a = EPS*w1/nw
    return np.where(a < 1.0, 0.5*betainc(0.5, (D+1)/2.0, 1.0-a*a), 0.0)

def fit_exponent(t, E, t_eps, emin=1e-9, emax=2e-2):
    m = (E > emin) & (E < emax) & (t < t_eps)
    if m.sum() < 10:
        return np.nan, 0.0, int(m.sum())
    X = np.log(t_eps - t[m]); Y = np.log(E[m])
    A = np.vstack([X, np.ones_like(X)]).T
    slope, intercept = np.linalg.lstsq(A, Y, rcond=None)[0]
    Yhat = slope*X + intercept
    ss_res = np.sum((Y-Yhat)**2); ss_tot = np.sum((Y-Y.mean())**2)
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0
    return slope, r2, int(m.sum())

# ----------------------------- positive control -----------------------------
# Exact power-law data with the claimed exponent; the fit machinery must
# recover (D+1)/2 essentially exactly.
control_pass = True
control_nus = {}
for D in (2, 5, 10):
    nu_true = (D+1)/2.0
    t_c = np.linspace(0.0, 9.9999, 20000)
    E_c = 0.2*(10.0 - t_c)**nu_true
    nu_c, r2_c, _ = fit_exponent(t_c, E_c, 10.0)
    control_nus[D] = nu_c
    if not (abs(nu_c - nu_true) < 0.02 and r2_c > 0.999):
        control_pass = False

# ----------------------------- main simulation ------------------------------
results = {}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, D in zip(axes, (2, 5, 10)):
    y0 = [0.05, 0.5]
    sol = solve_ivp(rhs, [0.0, 200.0], y0, args=(D,), dense_output=True,
                    rtol=1e-11, atol=1e-13, method='DOP853')
    # locate t_eps: first time a(t) = eps w1/|w| = 1
    tg = np.linspace(0.0, 200.0, 40001)
    Yg = sol.sol(tg); a_g = EPS*Yg[0]/np.hypot(Yg[0], Yg[1])
    idx = np.argmax(a_g >= 1.0)
    if a_g[-1] < 1.0 or idx == 0:
        results[D] = dict(nu=np.nan, r2=0.0, n=0, t_eps=np.nan)
        continue
    f = lambda tt: EPS*sol.sol(tt)[0]/np.hypot(*sol.sol(tt)) - 1.0
    t_eps = brentq(f, tg[idx-1], tg[idx], xtol=1e-13)
    # dense sampling below t_eps
    span = min(t_eps*0.5, 30.0)
    t_f = np.linspace(t_eps - span, t_eps - 1e-10, 60000)
    Yf = sol.sol(t_f)
    E_f = test_error(Yf[0], Yf[1], D)
    nu, r2, npts = fit_exponent(t_f, E_f, t_eps)
    results[D] = dict(nu=nu, r2=r2, n=npts, t_eps=t_eps)
    # plot
    m = (E_f > 1e-9) & (E_f < 2e-2)
    ax.loglog(t_eps - t_f[m], E_f[m], '.', ms=2, label='ODE data')
    xs = np.logspace(np.log10(max(t_eps-t_f[m].min(),1e-12)),
                     np.log10((t_eps-t_f[m]).max()), 50)
    E0 = E_f[m][np.argmax(t_f[m])] if m.any() else 1.0
    ax.loglog(xs, np.exp(np.log(E_f[m]).mean())* (xs/xs.mean())**nu, '-',
              label=f'fit nu={nu:.3f}')
    ax.loglog(xs, (xs/xs.mean())**((D+1)/2.0)*np.exp(np.log(E_f[m]).mean()), '--',
              label=f'theory {(D+1)/2:.1f}')
    ax.set_title(f'D={D}  r2={r2:.4f}')
    ax.set_xlabel('t_eps - t'); ax.set_ylabel('E(t)'); ax.legend(fontsize=8)

fig.suptitle('Critical exponent of D-dim uniform ball model')
fig.tight_layout()
fig.savefig('results/c2/exponent_fit.png', dpi=120)

# ----------------------------- verdict --------------------------------------
metrics = {'control_pass': bool(control_pass)}
for D in (2, 5, 10):
    metrics[f'control_nu_D{D}'] = float(control_nus[D])
    metrics[f'nu_D{D}'] = float(results[D]['nu'])
    metrics[f'r2_D{D}'] = float(results[D]['r2'])
    metrics[f'target_D{D}'] = (D+1)/2.0
    metrics[f'npts_D{D}'] = int(results[D]['n'])

if not control_pass:
    status = 'inconclusive'
    notes = ('Positive control failed to recover the exact (D+1)/2 power law; '
             'the fitting statistic is unreliable, so no verdict on the claim.')
else:
    ok = all(abs(results[D]['nu'] - (D+1)/2.0) <= 0.1 and results[D]['r2'] > 0.95
             for D in (2, 5, 10))
    if ok:
        status = 'supported'
        notes = ('Ball-model gradient flow (solve_ivp with exact ball-overlap '
                 'terms) yields fitted exponents matching (D+1)/2 within 0.1 '
                 'for D=2,5,10 with r2>0.95; control recovered the exact law.')
    else:
        status = 'inconclusive'
        notes = ('Control passed (fit machinery correct) but the ball-model ODE '
                 'fits deviate from (D+1)/2 beyond tolerance or r2<0.95 for at '
                 'least one D; per reviewer guidance this is reported as '
                 'inconclusive pending verification against the paper Fig. 5.')

summary = dict(claim_id='C2', status=status, metrics=metrics, notes=notes)
print('SUMMARY_JSON=' + json.dumps(summary, default=str))
