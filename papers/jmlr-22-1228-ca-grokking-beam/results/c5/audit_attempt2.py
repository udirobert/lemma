import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 0
rng = np.random.default_rng(SEED)


def rule30_bit(left, center, right):
    # Rule 30: 111->0, 110->0, 101->0, 100->1, 011->1, 010->1, 001->1, 000->0
    idx = (left << 2) | (center << 1) | right
    table = {7: 0, 6: 0, 5: 0, 4: 1, 3: 1, 2: 1, 1: 1, 0: 0}
    return table[idx]


def make_data(n, M, task, rng):
    X = rng.integers(0, 2, size=(n, M), dtype=np.int64)
    if task == 'rule30':
        i = M // 2
        left = X[:, (i - 1) % M]
        center = X[:, i]
        right = X[:, (i + 1) % M]
        bits = np.array([rule30_bit(int(l), int(c), int(r))
                         for l, c, r in zip(left, center, right)], dtype=np.int64)
    elif task == 'parity':
        bits = (X.sum(axis=1) % 2).astype(np.int64)
    else:
        raise ValueError(task)
    y = (2 * bits - 1).astype(np.float64)  # +/-1
    return X, y


class MPS:
    def __init__(self, chi, M, rng, wd):
        scale = 1.0 / np.sqrt(chi)
        self.chi = chi
        self.M = M
        self.wd = wd
        self.A = rng.normal(0.0, scale, size=(2, chi, chi)).astype(np.float64)
        self.L = rng.normal(0.0, scale, size=(chi,)).astype(np.float64)
        self.R = rng.normal(0.0, scale, size=(chi,)).astype(np.float64)

    def forward(self, X):
        N = X.shape[0]
        h = np.broadcast_to(self.L, (N, self.chi)).copy()
        states = [h.copy()]
        for i in range(self.M):
            Abatch = self.A[X[:, i]]  # (N, chi, chi)
            h = np.einsum('na,nak->nk', h, Abatch)
            states.append(h.copy())
        z = h @ self.R
        return z, states

    def gradients(self, X, y):
        z, states = self.forward(X)
        N = X.shape[0]
        grad_z = (z - y) / N

        grad_R = states[-1].T @ grad_z
        grad_h = grad_z[:, None] * self.R[None, :]

        grad_A = np.zeros_like(self.A)
        for i in range(self.M - 1, -1, -1):
            h_prev = states[i]
            Abatch = self.A[X[:, i]]
            for s in (0, 1):
                mask = X[:, i] == s
                if np.any(mask):
                    grad_A[s] += h_prev[mask].T @ grad_h[mask]
            grad_h = np.einsum('nk,nak->na', grad_h, Abatch)

        grad_L = grad_h.sum(axis=0)

        # L2 weight decay
        grad_A = grad_A + self.wd * self.A
        grad_L = grad_L + self.wd * self.L
        grad_R = grad_R + self.wd * self.R

        # Global gradient clipping
        total_norm = np.sqrt(
            np.sum(grad_A ** 2) + np.sum(grad_L ** 2) + np.sum(grad_R ** 2)
        )
        if total_norm > 10.0:
            scale = 10.0 / total_norm
            grad_A *= scale
            grad_L *= scale
            grad_R *= scale

        return grad_A, grad_L, grad_R, z


def compute_error(model, X, y):
    z, _ = model.forward(X)
    pred = np.where(z >= 0.0, 1.0, -1.0)
    return float(np.mean(pred != y))


def train_model(chi, M, Xtr, ytr, Xte, yte, epochs, lr, wd, eval_every, rng):
    model = MPS(chi, M, rng, wd)
    epochs_rec = []
    train_errs = []
    test_errs = []
    for epoch in range(1, epochs + 1):
        gA, gL, gR, _ = model.gradients(Xtr, ytr)
        model.A -= lr * gA
        model.L -= lr * gL
        model.R -= lr * gR

        if epoch == 1 or epoch % eval_every == 0:
            tr_err = compute_error(model, Xtr, ytr)
            te_err = compute_error(model, Xte, yte)
            epochs_rec.append(epoch)
            train_errs.append(tr_err)
            test_errs.append(te_err)
    return model, epochs_rec, train_errs, test_errs


def analyze(epochs_rec, train_errs, test_errs):
    phase_epochs = [e for e, tr, te in zip(epochs_rec, train_errs, test_errs)
                    if tr <= 0.05 and te > 0.5]
    first_phase = int(phase_epochs[0]) if phase_epochs else None
    num_phase = len(phase_epochs)
    transition = None
    if phase_epochs:
        first = phase_epochs[0]
        for e, te in zip(epochs_rec, test_errs):
            if e > first and te < 0.05:
                transition = int(e)
                break
    max_test_phase = float(max((te for e, tr, te in zip(epochs_rec, train_errs, test_errs)
                               if tr <= 0.05 and te > 0.5), default=np.nan))
    return {
        'first_phase_epoch': first_phase,
        'num_phase_checkpoints': num_phase,
        'transition_epoch': transition,
        'min_train_error': float(np.min(train_errs)),
        'max_test_error_during_phase': max_test_phase,
        'min_test_error': float(np.min(test_errs)),
        'final_train_error': float(train_errs[-1]),
        'final_test_error': float(test_errs[-1]),
    }


# Positive control: parity
Xtr_c, ytr_c = make_data(200, 100, 'parity', rng)
Xte_c, yte_c = make_data(500, 100, 'parity', rng)
model_c, ep_c, tr_c, te_c = train_model(
    4, 100, Xtr_c, ytr_c, Xte_c, yte_c, 1500, 0.05, 0.01, 10, rng
)
ctrl_metrics = analyze(ep_c, tr_c, te_c)
control_pass = bool(ctrl_metrics['min_test_error'] < 0.05)

# Main experiment: Rule-30
Xtr_m, ytr_m = make_data(300, 100, 'rule30', rng)
Xte_m, yte_m = make_data(1000, 100, 'rule30', rng)
model_m, ep_m, tr_m, te_m = train_model(
    8, 100, Xtr_m, ytr_m, Xte_m, yte_m, 3000, 0.05, 0.01, 10, rng
)
main_metrics = analyze(ep_m, tr_m, te_m)
main_pass = bool(
    main_metrics['num_phase_checkpoints'] > 0
    and main_metrics['transition_epoch'] is not None
)

if not control_pass:
    status = 'inconclusive'
elif main_pass:
    status = 'supported'
else:
    status = 'falsified'

metrics = {
    'control_pass': control_pass,
    'main_pass': main_pass,
    **{f'main_{k}': v for k, v in main_metrics.items()},
    **{f'control_{k}': v for k, v in ctrl_metrics.items()},
}

os.makedirs('results/c5', exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(ep_m, tr_m, label='train error')
axes[0].plot(ep_m, te_m, label='test error')
axes[0].set_title('Rule-30 (main)')
axes[0].set_xlabel('epoch')
axes[0].set_ylabel('error')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(ep_c, tr_c, label='train error')
axes[1].plot(ep_c, te_c, label='test error')
axes[1].set_title('Parity (control)')
axes[1].set_xlabel('epoch')
axes[1].set_ylabel('error')
axes[1].legend()
axes[1].grid(True)

fig.tight_layout()
fig.savefig('results/c5/fig.png', dpi=150)
plt.close(fig)

summary = {
    'claim_id': 'C5',
    'status': status,
    'metrics': metrics,
    'notes': 'Uniform MPS tensor-network student trained with full-batch GD and L2 weight decay. '
             'Control is parity learning; main is Rule-30 local-rule learning.',
}
print('SUMMARY_JSON=' + json.dumps(summary, default=str))
