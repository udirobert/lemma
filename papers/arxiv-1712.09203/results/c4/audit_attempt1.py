import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('results/c4', exist_ok=True)


def run_experiment(d, r, T, log_every, eta, alpha, seed, Xstar=None):
    rng = np.random.default_rng(seed)
    if Xstar is None:
        V = rng.standard_normal((d, r))
        Q, _ = np.linalg.qr(V)
        Xstar = (Q[:, :r] @ Q[:, :r].T).astype(np.float32)
    else:
        Xstar = Xstar.astype(np.float32)

    m = 5 * d * r if r > 0 else 5 * d
    n_tri = d * (d + 1) // 2
    triu_rows, triu_cols = np.triu_indices(d)
    offdiag_mask = triu_rows != triu_cols

    Atri = np.empty((m, n_tri), dtype=np.float32)
    for i in range(m):
        G = rng.standard_normal((d, d)).astype(np.float32)
        A = 0.5 * (G + G.T)
        Atri[i] = A[triu_rows, triu_cols]

    bstar = Xstar[triu_rows, triu_cols].copy()
    bstar[offdiag_mask] *= 2.0
    y = Atri @ bstar

    U = (alpha * np.eye(d, dtype=np.float32))
    normX = float(np.linalg.norm(Xstar, 'fro') ** 2)
    if normX == 0.0:
        normX = 1.0

    logs = []
    for t in range(1, T + 1):
        X = U @ U.T
        b = X[triu_rows, triu_cols].copy()
        b[offdiag_mask] *= 2.0
        r = Atri @ b - y
        Mtri = (Atri.T @ r) / np.float32(m)
        M = np.zeros((d, d), dtype=np.float32)
        M[triu_rows, triu_cols] = Mtri
        M[triu_cols, triu_rows] = Mtri
        grad = 2.0 * (M @ U)
        U -= np.float32(eta) * grad

        if t % log_every == 0:
            Xcur = U @ U.T
            err = float(np.linalg.norm(Xcur - Xstar, 'fro') ** 2 / normX)
            logs.append((t, err))

    return logs, Xstar


# Main experiments for the claim.
logs_100, Xstar_100 = run_experiment(100, 5, 100000, 1000, 0.0025, 0.01, 12345)
logs_200, Xstar_200 = run_experiment(200, 5, 100000, 1000, 0.0025, 0.01, 67890)

# Positive control: known-decreasing case (zero target).
control_logs, _ = run_experiment(10, 1, 2000, 200, 0.0025, 0.01, 999, Xstar=np.zeros((10, 10), dtype=np.float32))
control_initial = control_logs[0][1]
control_final = control_logs[-1][1]
control_pass = bool(np.isfinite(control_final) and control_final < 0.5 * control_initial)


def summarize(logs):
    ts = np.array([x[0] for x in logs])
    es = np.array([x[1] for x in logs])
    idx_20k = int(np.argmin(np.abs(ts - 20000)))
    err_20k = float(es[idx_20k])
    final = float(es[-1])
    after_20k = es[ts >= 20000]
    min_after_20k = float(np.min(after_20k)) if len(after_20k) else float('nan')
    last_10k = es[ts >= 90000]
    max_last_10k = float(np.max(last_10k)) if len(last_10k) else float('nan')
    no_div = bool(np.all(np.isfinite(es)) and max_last_10k <= 0.5)
    decreasing = bool(final <= err_20k)
    return {
        'err_at_20000': err_20k,
        'final': final,
        'min_after_20000': min_after_20k,
        'max_last_10000': max_last_10k,
        'no_divergence': no_div,
        'final_le_20000': decreasing,
    }

s100 = summarize(logs_100)
s200 = summarize(logs_200)

# Plot.
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot([x[0] for x in logs_100], [x[1] for x in logs_100], label='d=100')
ax.plot([x[0] for x in logs_200], [x[1] for x in logs_200], label='d=200')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('GD iteration')
ax.set_ylabel('Normalized test error')
ax.set_title('C4: test error vs iterations')
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig('results/c4/fig.png', dpi=150)
plt.close(fig)

pass_100 = (control_pass and s100['final_le_20000'] and s100['final'] <= 0.05 and s100['no_divergence'])
pass_200 = (control_pass and s200['final_le_20000'] and s200['final'] <= 0.05 and s200['no_divergence'])

if not control_pass:
    status = 'inconclusive'
elif pass_100 and pass_200:
    status = 'supported'
else:
    status = 'falsified'

summary = {
    'claim_id': 'C4',
    'status': status,
    'metrics': {
        'control_pass': bool(control_pass),
        'control_initial_error': float(control_initial),
        'control_final_error': float(control_final),
        'd100_err_at_20000': s100['err_at_20000'],
        'd100_final': s100['final'],
        'd100_min_after_20000': s100['min_after_20000'],
        'd100_max_last_10000': s100['max_last_10000'],
        'd100_no_divergence': bool(s100['no_divergence']),
        'd100_final_le_20000': bool(s100['final_le_20000']),
        'd200_err_at_20000': s200['err_at_20000'],
        'd200_final': s200['final'],
        'd200_min_after_20000': s200['min_after_20000'],
        'd200_max_last_10000': s200['max_last_10000'],
        'd200_no_divergence': bool(s200['no_divergence']),
        'd200_final_le_20000': bool(s200['final_le_20000']),
        'iterations_run': 100000,
    },
    'notes': 'GD on over-parameterized matrix sensing with Gaussian measurements; normalized Frobenius test error logged every 1000 iterations.',
}
print('SUMMARY_JSON=' + json.dumps(summary, default=str))
