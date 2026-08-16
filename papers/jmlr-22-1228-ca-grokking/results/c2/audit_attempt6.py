import numpy as np, json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import gamma as gammafn, hyp2f1

np.random.seed(0)
os.makedirs('results/c2', exist_ok=True)

LAM2 = 0.01
N = 500
R0 = 0.5
SHIFTS = [0.5, 1.0, 1.5, 2.0]

def make_dataset(D, s):
    # non-typical grokking dataset: unit-normalized Gaussian scaled by sqrt(U[r0,1])
    Xp = np.random.randn(N, D); Xp /= np.linalg.norm(Xp, axis=1, keepdims=True)
    Xp *= np.sqrt(np.random.uniform(R0, 1.0, (N, 1)))
    Xn = np.random.randn(N, D); Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
    Xn *= np.sqrt(np.random.uniform(R0, 1.0, (N, 1)))
    eps_vec = np.zeros(D); eps_vec[0] = s/np.sqrt(2.0)
    if D > 1: eps_vec[1] = s/np.sqrt(2.0)
    Xp = Xp + eps_vec
    Xn = Xn - eps_vec
    X = np.vstack([Xp, Xn]); y = np.concatenate([np.ones(N), -np.ones(N)])
    return X, y, eps_vec

def E_D_of_h(h, D):
    h = np.clip(h, 0.0, 1.0)
    c = D*gammafn(D/2.0)/(2.0*np.sqrt(np.pi)*gammafn((D+1)/2.0))
    return 0.5 - c*hyp2f1(0.5, 1.0-D/2.0, 1.5, h**2)*h

def simulate(D, s):
    X, y, eps_vec = make_dataset(D, s)
    eps = np.linalg.norm(eps_vec)
    G = (X.T @ X)/(2*N) + np.outer(eps_vec, eps_vec) + LAM2*np.eye(D)
    a = (y[:,None]*X).mean(axis=0)
    lam, Q = np.linalg.eigh(G)
    w_lam = Q @ ((Q.T @ a)/lam)
    w0 = np.zeros(D)
    d = Q.T @ (w_lam - w0)
    ts = np.exp(np.linspace(np.log(1e-2), np.log(1e6), 4000))
    # w(t) = w_lam - Q (exp(-lam t) d)
    W = w_lam[None,:] - (np.exp(-np.outer(ts, lam)) * d[None,:]) @ Q.T
    w1 = W[:,0]
    nrm2 = np.sum(W**2, axis=1)
    h = eps*w1/nrm2
    return ts, h

def fit_exponent(ts, E, t_eps):
    mask = (ts < t_eps) & (E > 0.01*E.max()) & (E < 0.5*E.max()) & (E > 0)
    if mask.sum() < 5:
        return None, None, 0
    x = np.log(t_eps - ts[mask]); yy = np.log(E[mask])
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, yy, rcond=None)[0]
    pred = A @ np.array([slope, intercept])
    ss_res = np.sum((yy-pred)**2); ss_tot = np.sum((yy-yy.mean())**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0
    return slope, r2, int(mask.sum())

results = {}
control_pass_all = True
main_pass_all = True
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for ax, D in zip(axes, [2, 5, 10]):
    nu_theory = (D+1)/2.0
    chosen = None
    for s in SHIFTS:
        ts, h = simulate(D, s)
        idx = np.where(h >= 1.0)[0]
        if len(idx) == 0:
            continue
        t_eps = ts[idx[0]]
        E = E_D_of_h(h, D)
        # grokking window check: test error still > 0.1 at some point before t_eps
        if np.any((ts < t_eps) & (E > 0.1)):
            chosen = (s, ts, h, E, t_eps)
            break
    if chosen is None:
        # fall back: use last shift if it reaches h=1 at all
        ts, h = simulate(D, SHIFTS[-1])
        idx = np.where(h >= 1.0)[0]
        if len(idx) == 0:
            results[D] = {'nu': None, 'r2': None, 'shift': SHIFTS[-1], 'note': 'h never reached 1'}
            main_pass_all = False
            control_pass_all = False
            continue
        t_eps = ts[idx[0]]
        E = E_D_of_h(h, D)
        chosen = (SHIFTS[-1], ts, h, E, t_eps)
    s, ts, h, E, t_eps = chosen
    nu, r2, npts = fit_exponent(ts, E, t_eps)
    # POSITIVE CONTROL: exact power-law E=(1-h)^((D+1)/2) through same h(t)
    E_ctrl = np.clip(1.0 - h, 1e-15, None)**nu_theory
    E_ctrl = np.where(h >= 1.0, 0.0, E_ctrl)
    nu_c, r2_c, _ = fit_exponent(ts, E_ctrl, t_eps)
    ctrl_ok = (nu_c is not None) and (abs(nu_c - nu_theory) <= 0.1) and (r2_c is not None and r2_c > 0.95)
    main_ok = (nu is not None) and (abs(nu - nu_theory) <= 0.1) and (r2 is not None and r2 > 0.95)
    control_pass_all = control_pass_all and ctrl_ok
    main_pass_all = main_pass_all and main_ok
    results[D] = {'nu': nu, 'r2': r2, 'nu_control': nu_c, 'r2_control': r2_c,
                  'nu_theory': nu_theory, 'shift_used': s, 'n_fit_pts': npts,
                  'control_pass': bool(ctrl_ok), 'main_pass': bool(main_ok)}
    mask = (ts < t_eps) & (E > 0)
    ax.loglog(t_eps - ts[mask], E[mask], '.', ms=2, label='sim E(t)')
    mc = (ts < t_eps) & (E_ctrl > 0)
    ax.loglog(t_eps - ts[mc], E_ctrl[mc], '-', lw=1, alpha=0.6, label='control (1-h)^nu')
    ax.set_title(f'D={D}: nu={nu if nu is None else round(nu,3)} (theory {nu_theory})')
    ax.set_xlabel('t_eps - t'); ax.set_ylabel('E'); ax.legend(fontsize=7)

fig.tight_layout()
fig.savefig('results/c2/exponent_fits.png', dpi=100)

metrics = {}
for D in [2, 5, 10]:
    r = results[D]
    metrics[f'D{D}_nu'] = r.get('nu')
    metrics[f'D{D}_r2'] = r.get('r2')
    metrics[f'D{D}_nu_theory'] = r.get('nu_theory')
    metrics[f'D{D}_shift'] = r.get('shift_used')
    metrics[f'D{D}_nu_control'] = r.get('nu_control')
metrics['control_pass'] = bool(control_pass_all)

if not control_pass_all:
    status = 'inconclusive'
    notes = 'Positive control failed: fitting statistic buggy; cannot judge claim.'
elif main_pass_all:
    status = 'supported'
    notes = 'Fitted exponents match (D+1)/2 within 0.1 with r2>0.95 for D=2,5,10 using paper closed-form dynamics (Eqs 20-23).'
else:
    bad = {D: results[D] for D in [2,5,10] if not results[D].get('main_pass', False)}
    status = 'falsified'
    notes = f'Control passes but fitted exponents deviate from (D+1)/2 beyond tolerance: {bad}'

summary = {'claim_id': 'C2', 'status': status, 'metrics': metrics, 'notes': notes}
print('SUMMARY_JSON=' + json.dumps(summary, default=str))
