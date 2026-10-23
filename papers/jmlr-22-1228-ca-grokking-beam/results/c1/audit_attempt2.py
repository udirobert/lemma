import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

# 1D exponential model in the grokking regime.
# Use the statistical (thermodynamic-limit) training dynamics: expectations over P^\pm.
# L2 on the bias is set to zero so that the threshold can reach epsilon in finite time.
eps = 1.0
lam1 = 0.5
lam2 = 0.0
a = eps**2 + 2.0

dt = 0.001
T = 10.0
steps = int(T / dt)

ts = []
Es = []
ws = []
bs = []
zs = []

w = 0.0
b = 0.0
t_eps = None

for i in range(1, steps + 1):
    t = i * dt

    def deriv(ww, bb):
        if ww >= 0:
            dw = -a * ww - eps * bb + 1.0 - lam1
        else:
            dw = -a * ww - eps * bb + 1.0 + lam1
        db = -eps * ww - (1.0 + lam2) * bb
        return dw, db

    # RK4
    k1w, k1b = deriv(w, b)
    k2w, k2b = deriv(w + 0.5*dt*k1w, b + 0.5*dt*k1b)
    k3w, k3b = deriv(w + 0.5*dt*k2w, b + 0.5*dt*k2b)
    k4w, k4b = deriv(w + dt*k3w, b + dt*k3b)
    w += (dt/6.0)*(k1w + 2*k2w + 2*k3w + k4w)
    b += (dt/6.0)*(k1b + 2*k2b + 2*k3b + k4b)

    z = b + eps * w  # zero test error occurs when z = 0
    if w != 0:
        E = 0.5 * (1.0 - np.exp(-abs(z) / abs(w)))
    else:
        E = 0.5

    ts.append(t)
    ws.append(w)
    bs.append(b)
    zs.append(z)
    Es.append(E)

    if t_eps is None and len(zs) >= 2 and zs[-2] > 0 and z <= 0:
        # linear interpolation for the crossing time
        z_prev = zs[-2]
        t_prev = ts[-2]
        frac = z_prev / (z_prev - z)
        t_eps = t_prev + frac * (t - t_prev)
        break

ts = np.array(ts)
Es = np.array(Es)
zs = np.array(zs)

if t_eps is None:
    t_eps = float(ts[-1])

# Fit log E vs log(t_eps - t) for t < t_eps.
mask = (ts < t_eps) & (Es > 0) & ((t_eps - ts) > 1e-4)
if mask.sum() >= 5:
    delta = t_eps - ts[mask]
    log_delta = np.log(delta)
    log_E = np.log(Es[mask])
    slope, intercept = np.polyfit(log_delta, log_E, 1)
    nu = float(slope)
else:
    nu = float('nan')

# Positive control: exact linear power law E = A (t_eps - t)^1.
tc = np.linspace(0.0, 0.9, 200)
t_eps_c = 1.0
Ec = 2.0 * (t_eps_c - tc) ** 1.0
slope_c, _ = np.polyfit(np.log(t_eps_c - tc), np.log(Ec), 1)
control_pass = bool(abs(slope_c - 1.0) < 0.05)

if not control_pass:
    status = "inconclusive"
    notes = "Positive control failed; statistic is unreliable."
elif not np.isfinite(nu):
    status = "inconclusive"
    notes = "No usable fit points or finite transition time."
elif abs(nu - 1.0) <= 0.05:
    status = "supported"
    notes = "Fitted exponent is within 0.05 of 1.0."
else:
    status = "falsified"
    notes = "Fitted exponent differs from 1.0 by more than 0.05."

os.makedirs('results/c1', exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(ts, Es, label='E(t)')
ax[0].axvline(t_eps, color='r', linestyle='--', label='t_eps')
ax[0].set_xlabel('t')
ax[0].set_ylabel('E(t)')
ax[0].legend()
ax[1].scatter(np.log(t_eps - ts[mask]), np.log(Es[mask]), s=10, label='data')
if mask.sum() >= 5:
    ax[1].plot(np.log(t_eps - ts[mask]), slope * np.log(t_eps - ts[mask]) + intercept, 'r', label='fit')
ax[1].set_xlabel('log(t_eps - t)')
ax[1].set_ylabel('log E(t)')
ax[1].legend()
fig.tight_layout()
fig.savefig('results/c1/fig.png')
plt.close(fig)

summary = {
    "claim_id": "C1",
    "status": status,
    "metrics": {
        "nu": nu,
        "t_eps": float(t_eps),
        "control_slope": float(slope_c),
        "control_pass": control_pass,
        "n_fit_points": int(mask.sum()),
        "lambda1": float(lam1),
        "lambda2": float(lam2),
        "epsilon": float(eps)
    },
    "notes": notes
}
print("SUMMARY_JSON=" + json.dumps(summary, default=str))
