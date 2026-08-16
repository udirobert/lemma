import numpy as np, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import betainc

# ------------------------------------------------------------
# Audit C2: critical exponent nu = (D+1)/2 in the D-dim uniform ball model
# Model (Sec 3.3): classes uniform in unit balls centered at +/- eps*e1.
# Student: perceptron f(x)=sgn(w.x), no bias (Eq. 18), gradient flow on
# hinge loss + L1 (lam1) + small L2 (lam2).
# Non-typical dataset (Sec 3.3.1): training samples shifted along e2
# (opposite shifts per class) so the zero-train-error classifier is tilted
# away from e1 and test error vanishes only later (grokking).
# Test error = exact spherical-cap fraction misclassified (statistical
# average over P+/-, i.e. infinite test set as in the paper).
# ------------------------------------------------------------

def sample_ball(n, D, rng):
    x = rng.normal(size=(n, D))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    r = rng.random(n) ** (1.0 / D)
    return x * r[:, None]

def test_error(w, eps, D):
    """Exact test error: fraction of the two unit balls (centers +/-eps*e1)
    misclassified by the homogeneous plane normal to w."""
    nrm = np.linalg.norm(w)
    if nrm == 0:
        return 0.5
    d = eps * w[0] / nrm  # eps*cos(theta), theta = angle(w, e1)
    if d >= 1.0:
        return 0.0
    if d <= -1.0:
        return 1.0
    a = (D + 1) / 2.0
    c = 0.5 * betainc(a, 0.5, 1.0 - d * d)
    return float(c if d >= 0 else 1.0 - c)

def simulate(D, eps, s, N=100, lam1=0.01, lam2=1e-4, dt=0.05,
             max_steps=60000, seed=0):
    rng = np.random.default_rng(seed)
    Xp = sample_ball(N, D, rng); Xp[:, 0] += eps; Xp[:, 1] += s
    Xn = sample_ball(N, D, rng); Xn[:, 0] -= eps; Xn[:, 1] -= s
    X = np.vstack([Xp, Xn])
    y = np.concatenate([np.ones(N), -np.ones(N)])
    w = 0.05 * rng.normal(size=D)
    ts, Es, trs = [], [], []
    t = 0.0
    t_train0, t_eps = None, None
    for k in range(max_steps):
        m = y * (X @ w)
        tr = float(np.mean(m <= 0))
        E = test_error(w, eps, D)
        ts.append(t); Es.append(E); trs.append(tr)
        if t_train0 is None and tr == 0.0:
            t_train0 = t
            if E == 0.0:  # generalized before/at train-zero: no grokking
                return None
        if t_train0 is not None and E == 0.0:
            t_eps = t
            break
        bad = m <= 0
        if bad.any():
            g = -(X[bad] * y[bad, None]).mean(axis=0)
        else:
            g = np.zeros(D)
        g = g + lam1 * np.sign(w) + lam2 * w
        w = w - dt * g
        t += dt
    if t_eps is None or t_train0 is None:
        return None
    return np.array(ts), np.array(Es), np.array(trs), t_train0, t_eps

def fit_exponent(ts, Es, t_eps, t_train0):
    base = (Es > 1e-9) & (ts < t_eps - 1e-12) & (ts >= t_train0)
    m = None
    for emax in [5e-2, 2e-1, 0.5, 1.0]:
        m = base & (Es < emax)
        if m.sum() >= 30:
            break
    if m is None or m.sum() < 10:
        return None, None, 0
    x = np.log(t_eps - ts[m])
    yv = np.log(Es[m])
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, yv, rcond=None)[0]
    pred = A @ np.array([slope, intercept])
    ss_res = float(np.sum((yv - pred) ** 2))
    ss_tot = float(np.sum((yv - yv.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(slope), float(r2), int(m.sum())

# ---------------- positive control: synthetic exact power laws ----------
control_pass = True
control_nus = {}
for D in [2, 5, 10]:
    nu_true = (D + 1) / 2.0
    t_eps_c = 1.0
    tgrid = np.linspace(0.0, t_eps_c - 1e-4, 2000)
    Ec = 0.3 * (t_eps_c - tgrid) ** nu_true
    nu_c, r2_c, n_c = fit_exponent(tgrid, Ec, t_eps_c, 0.0)
    control_nus[str(D)] = nu_c
    if nu_c is None or abs(nu_c - nu_true) > 0.05:
        control_pass = False

# ---------------- actual simulations ------------------------------------
EPS = 1.05  # just above critical separation eps=1 (balls radius 1)
results = {}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, D in zip(axes, [2, 5, 10]):
    nu_exp = (D + 1) / 2.0
    got = None
    for s_mult in [0.9, 1.3, 1.8, 2.5]:
        for seed in [0, 1, 2]:
            out = simulate(D, EPS, s_mult * EPS, seed=seed)
            if out is None:
                continue
            ts, Es, trs, t_tr0, t_ep = out
            nu, r2, nfit = fit_exponent(ts, Es, t_ep, t_tr0)
            if nu is not None and nfit >= 30:
                got = (nu, r2, nfit, ts, Es, trs, t_tr0, t_ep, s_mult, seed)
                break
        if got is not None:
            break
    if got is None:
        results[str(D)] = dict(nu=None, note="no grokking transition found")
        ax.text(0.5, 0.5, "no transition", ha="center", transform=ax.transAxes)
        continue
    nu, r2, nfit, ts, Es, trs, t_tr0, t_ep, s_mult, seed = got
    results[str(D)] = dict(nu=nu, expected=nu_exp, err=abs(nu - nu_exp),
                           r2=r2, n_fit=nfit, t_train0=float(t_tr0),
                           t_eps=float(t_ep), shift=s_mult, seed=seed)
    m = (Es > 1e-9) & (ts < t_ep) & (ts >= t_tr0) & (Es < 5e-1)
    ax.loglog(t_ep - ts[m], Es[m], '.', ms=2, label=f"D={D} data")
    xx = np.logspace(np.log10(max(t_ep - ts[m]).__float__()),
                     np.log10(min(t_ep - ts[m]).__float__()), 50)
    # fit line
    mm = (Es > 1e-9) & (ts < t_ep - 1e-12) & (ts >= t_tr0) & (Es < 5e-2)
    if mm.sum() < 30:
        mm = m
    cf = np.polyfit(np.log(t_ep - ts[mm]), np.log(Es[mm]), 1)
    ax.loglog(xx, np.exp(cf[1]) * xx ** cf[0], '-',
              label=f"fit nu={nu:.3f} (exp {nu_exp})")
    ax.set_xlabel("t_eps - t"); ax.set_ylabel("E_D(t)")
    ax.set_title(f"D={D}")
    ax.legend(fontsize=8)
fig.tight_layout()
os.makedirs("results/c2", exist_ok=True)
fig.savefig("results/c2/exponent_fit.png", dpi=120)

# ---------------- verdict ------------------------------------------------
tol = 0.1
per_dim = {}
all_ok = True
any_fit = False
for D in [2, 5, 10]:
    r = results[str(D)]
    if r.get("nu") is None:
        per_dim[f"D{D}"] = "no_transition"
        all_ok = False
    else:
        any_fit = True
        ok = abs(r["nu"] - r["expected"]) <= tol
        per_dim[f"D{D}"] = "pass" if ok else "fail"
        if not ok:
            all_ok = False

if not control_pass:
    status = "inconclusive"
    note = "Positive control failed; fitting machinery unreliable."
elif all_ok and any_fit:
    status = "supported"
    note = "Fitted exponents match (D+1)/2 within 0.1 for all tested D."
elif any_fit:
    status = "falsified"
    note = "Exponents obtained but deviate from (D+1)/2 beyond tolerance."
else:
    status = "inconclusive"
    note = "No grokking transition could be produced within budget."

metrics = {
    "control_pass": bool(control_pass),
    "control_nu_D2": control_nus.get("2"),
    "control_nu_D5": control_nus.get("5"),
    "control_nu_D10": control_nus.get("10"),
    "nu_D2": results["2"].get("nu"),
    "nu_D5": results["5"].get("nu"),
    "nu_D10": results["10"].get("nu"),
    "expected_D2": 1.5, "expected_D5": 3.0, "expected_D10": 5.5,
    "err_D2": results["2"].get("err"),
    "err_D5": results["5"].get("err"),
    "err_D10": results["10"].get("err"),
    "r2_D2": results["2"].get("r2"),
    "r2_D5": results["5"].get("r2"),
    "r2_D10": results["10"].get("r2"),
    "per_dim": per_dim,
}
summary = {"claim_id": "C2", "status": status, "metrics": metrics, "notes": note}
print("SUMMARY_JSON=" + json.dumps(summary, default=str))
