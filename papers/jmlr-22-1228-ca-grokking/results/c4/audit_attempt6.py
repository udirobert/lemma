import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rng = np.random.default_rng(12345)

# ---------------- Model / analytic grokking condition (Appendix B.1, lambda_1=0) ----------------
LAMBDA2 = 0.1
N = 10          # N positive and N negative samples
M = 2000        # Monte-Carlo datasets per (D, eps)
DS = [2, 5, 10, 20]
EPSS = [1.005, 1.01, 1.02, 1.05, 1.1, 1.2]

def sample_x1(n, a, p, rng):
    # sample from f(x) ~ (1-x^2)^p on [a, 1] via numerical inverse CDF
    xs = np.linspace(a, 1.0, 20001)
    pdf = np.maximum(1.0 - xs**2, 0.0)**p
    cdf = np.concatenate([[0.0], np.cumsum(0.5*(pdf[1:]+pdf[:-1])*np.diff(xs))])
    cdf /= cdf[-1]
    u = rng.random(n)
    return np.interp(u, cdf, xs)

def sample_datasets(D, eps, M, rng):
    # P+ : uniform in unit D-ball with x1 >= a ; P- : x1 <= -a, a = eps-1
    a = eps - 1.0
    p = (D - 1) / 2.0
    X = np.zeros((M, 2*N, D))
    y = np.concatenate([np.ones(N), -np.ones(N)])
    for cls_sign, sl in [(1, slice(0, N)), (-1, slice(N, 2*N))]:
        x1 = sample_x1(M*N, a, p, rng).reshape(M, N)
        r = np.sqrt(np.maximum(1.0 - x1**2, 0.0))
        if D > 1:
            g = rng.standard_normal((M, N, D-1))
            gn = np.linalg.norm(g, axis=2, keepdims=True)
            gn[gn == 0] = 1.0
            u = rng.random((M, N, 1))**(1.0/(D-1))
            rest = r[:, :, None] * u * g / gn
            X[:, sl, 1:] = rest
        X[:, sl, 0] = cls_sign * x1
    return X, y

def grok_pass(X, y, D, eps):
    lam_D = 1.0/(D+2) + LAMBDA2
    beta  = eps/(lam_D+eps**2) + eps/((lam_D+eps**2)**2 * (D+2))
    a1 = 1.0/(lam_D+eps**2) - 2*eps**2/(lam_D+eps**2)**2
    a2 = -eps/(lam_D+eps**2)**2
    a3 = 1.0/lam_D - eps/(lam_D*(lam_D+eps**2))
    a4 = -eps/(lam_D*(lam_D+eps**2))
    x1 = X[:, :, 0]
    xbar1 = (y[None, :]*x1).mean(axis=1)
    x1sq  = (x1**2).mean(axis=1)
    w1 = beta + a1*xbar1 + a2*x1sq
    if D > 1:
        xj = X[:, :, 1:]
        xbarj = (y[None, :, None]*xj).mean(axis=1)      # (M, D-1)
        x1xj  = (x1[:, :, None]*xj).mean(axis=1)
        wj = a3*xbarj + a4*x1xj
        sumwj2 = (wj**2).sum(axis=1)
    else:
        sumwj2 = np.zeros(X.shape[0])
    return (w1 > 0) & ((eps**2 - 1.0)*w1**2 >= sumwj2)

# ---------------- Monte-Carlo table P_grok(D, eps) ----------------
P = np.zeros((len(DS), len(EPSS)))
for i, D in enumerate(DS):
    for j, eps in enumerate(EPSS):
        X, y = sample_datasets(D, eps, M, rng)
        P[i, j] = grok_pass(X, y, D, eps).mean()
        print(f"D={D:3d} eps={eps:.3f}  P_grok={P[i,j]:.4f}")

table = {f"D={D}" : {f"eps={eps}": float(P[i, j]) for j, eps in enumerate(EPSS)} for i, D in enumerate(DS)}

# ---------------- Verdict logic ----------------
def verdict(Ps):
    Ps = np.asarray(Ps, dtype=float)
    strictly_dec = bool(np.all(np.diff(Ps) < 0))
    drop = float(Ps[0] - Ps[-1])
    return strictly_dec and drop >= 0.2 and len(Ps) >= 3, strictly_dec, drop

# Positive control: known decreasing sequence must be detected; increasing must not
ctrl_dec = [0.9, 0.5, 0.2, 0.05]
ctrl_inc = [0.05, 0.2, 0.5, 0.9]
ctrl_pass_dec, _, _ = verdict(ctrl_dec)
ctrl_pass_inc, _, _ = verdict(ctrl_inc)
control_pass = bool(ctrl_pass_dec and not ctrl_pass_inc)

# Selection: prefer a COMMON eps with 0<P<1 for all D; else per-D window picks
chosen_eps = {}
chosen_P = None
mode = None
common_ok = [eps for j, eps in enumerate(EPSS) if np.all((P[:, j] > 0) & (P[:, j] < 1))]
for j, eps in enumerate(EPSS):
    if np.all((P[:, j] > 0) & (P[:, j] < 1)):
        chosen_P = P[:, j]
        chosen_eps = {str(D): eps for D in DS}
        mode = f"common eps={eps}"
        break
if chosen_P is None:
    picks = []
    ok = True
    for i, D in enumerate(DS):
        window = [(eps, P[i, j]) for j, eps in enumerate(EPSS) if 0 < P[i, j] < 1]
        if not window:
            ok = False
            break
        e, pv = min(window, key=lambda t: abs(t[1]-0.5))
        chosen_eps[str(D)] = e
        picks.append(pv)
    if ok:
        chosen_P = np.array(picks)
        mode = "per-D window eps"

supported = False
strictly_dec = False
drop = 0.0
slope = np.nan
r2 = np.nan
if chosen_P is not None:
    supported, strictly_dec, drop = verdict(chosen_P)
    if np.all(chosen_P > 0):
        coef = np.polyfit(np.array(DS, dtype=float), np.log(chosen_P), 1)
        slope = float(coef[0])
        pred = np.polyval(coef, np.array(DS, dtype=float))
        ss_res = float(np.sum((np.log(chosen_P)-pred)**2))
        ss_tot = float(np.sum((np.log(chosen_P)-np.log(chosen_P).mean())**2))
        r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else 1.0

if not control_pass:
    status = "inconclusive"
    notes = "Positive control failed: verdict logic is buggy."
elif chosen_P is None:
    status = "inconclusive"
    notes = "No eps window with 0<P<1 found for every D; N>>1 limit may be too sharp for this sweep. Not falsifying on saturated/empty window."
else:
    status = "supported" if supported else "falsified"
    notes = (f"Mode: {mode}. P_grok(D) at chosen eps: {np.round(chosen_P,4).tolist()} over D={DS}. "
             f"Strictly decreasing: {strictly_dec}, drop={drop:.3f}, log-linear slope={slope}, R2={r2}.")

# ---------------- Plots ----------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
im = axes[0].imshow(P, aspect='auto', cmap='viridis', vmin=0, vmax=1)
axes[0].set_xticks(range(len(EPSS)), [str(e) for e in EPSS])
axes[0].set_yticks(range(len(DS)), [str(d) for d in DS])
axes[0].set_xlabel('eps'); axes[0].set_ylabel('D')
axes[0].set_title('P_grok(D, eps)')
for i in range(len(DS)):
    for j in range(len(EPSS)):
        axes[0].text(j, i, f"{P[i,j]:.2f}", ha='center', va='center', color='w', fontsize=8)
fig.colorbar(im, ax=axes[0])
if chosen_P is not None:
    axes[1].semilogy(DS, chosen_P, 'o-', label='chosen window')
    axes[1].set_xlabel('D'); axes[1].set_ylabel('P_grok')
    axes[1].set_title(f'P_grok vs D ({mode})')
    axes[1].grid(True, alpha=0.3)
for j, eps in enumerate(EPSS):
    axes[1].semilogy(DS, np.maximum(P[:, j], 1e-4), '.--', alpha=0.4, label=f'eps={eps}')
axes[1].legend(fontsize=7)
plt.tight_layout()
plt.savefig('results/c4/pgrok_vs_D.png', dpi=120)

summary = {
    "claim_id": "C4",
    "status": status,
    "metrics": {
        "P_table": table,
        "chosen_eps_per_D": chosen_eps,
        "chosen_P": [float(v) for v in chosen_P] if chosen_P is not None else "none",
        "strictly_decreasing": bool(strictly_dec),
        "overall_drop": float(drop),
        "log_linear_slope": slope if chosen_P is not None else "n/a",
        "log_linear_R2": r2 if chosen_P is not None else "n/a",
        "control_pass": control_pass,
        "selection_mode": mode if mode else "none"
    },
    "notes": notes
}
print("SUMMARY_JSON=" + json.dumps(summary, default=str))
