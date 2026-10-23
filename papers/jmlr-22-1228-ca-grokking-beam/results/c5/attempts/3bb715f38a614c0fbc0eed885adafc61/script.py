import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = 'results/c5'
os.makedirs(OUT_DIR, exist_ok=True)


def rule30(a, b, c):
    # elementary rule 30 table
    idx = (a << 2) | (b << 1) | c
    table = {7: 0, 6: 0, 5: 0, 4: 1, 3: 1, 2: 1, 1: 1, 0: 0}
    return table[idx]


def make_data(M, n, label_fn, rng):
    X = rng.integers(0, 2, size=(n, M), dtype=np.int64)
    y = np.array([label_fn(x) for x in X], dtype=np.float64)
    return X, y


def center_rule30(x):
    M = len(x)
    c = M // 2
    return rule30(int(x[c-1]), int(x[c]), int(x[c+1]))


def center_parity(x):
    M = len(x)
    c = M // 2
    return int(x[c-1] ^ x[c] ^ x[c+1])


class MPSModel:
    def __init__(self, M, chi, rng, scale=0.1):
        self.M = M
        self.chi = chi
        self.params = []
        # first tensor: (1, 2, chi)
        self.params.append(rng.normal(0, scale, size=(1, 2, chi)))
        for _ in range(M - 2):
            self.params.append(rng.normal(0, scale, size=(chi, 2, chi)))
        self.params.append(rng.normal(0, scale, size=(chi, 2, 1)))

    def forward(self, X):
        N, M = X.shape
        s = np.ones((N, 1), dtype=np.float64)
        for i in range(M):
            Ti = self.params[i]
            Tsel = np.transpose(Ti[:, X[:, i], :], (1, 0, 2))  # (N, chi_prev, chi_next)
            s = np.einsum('na,nab->nb', s, Tsel, optimize=True)
        return s[:, 0]

    def gradients(self, X, y):
        N, M = X.shape
        states = [None] * (M + 1)
        states[0] = np.ones((N, 1), dtype=np.float64)
        for i in range(M):
            Ti = self.params[i]
            Tsel = np.transpose(Ti[:, X[:, i], :], (1, 0, 2))
            states[i + 1] = np.einsum('na,nab->nb', states[i], Tsel, optimize=True)
        pred = states[M][:, 0]
        # MSE gradient
        g_state = ((2.0 / N) * (pred - y))[:, None]  # (N, 1)
        grads = [np.zeros_like(p) for p in self.params]
        for i in range(M - 1, -1, -1):
            Ti = self.params[i]
            Tsel = np.transpose(Ti[:, X[:, i], :], (1, 0, 2))
            outer = states[i][:, :, None] * g_state[:, None, :]  # (N, chi_prev, chi_next)
            for bit in (0, 1):
                mask = (X[:, i] == bit)
                if np.any(mask):
                    grads[i][:, bit, :] += outer[mask].sum(axis=0)
            g_state = np.einsum('nb,nab->na', g_state, Tsel, optimize=True)
        return grads, pred

    def train_step(self, X, y, lr, wd, clip=10.0):
        grads, pred = self.gradients(X, y)
        # add weight decay
        for i in range(len(self.params)):
            grads[i] = grads[i] + wd * self.params[i]
            gnorm = np.linalg.norm(grads[i])
            if gnorm > clip:
                grads[i] = grads[i] * (clip / gnorm)
            self.params[i] -= lr * grads[i]
        return pred

    def error(self, X, y):
        pred = self.forward(X)
        pred_bin = (pred >= 0.5).astype(np.int64)
        return float(np.mean(pred_bin != y.astype(np.int64)))


def run_experiment(M, chi, n_train, n_test, epochs, lr, wd, label_fn, seed):
    rng = np.random.default_rng(seed)
    Xtr, ytr = make_data(M, n_train, label_fn, rng)
    Xte, yte = make_data(M, n_test, label_fn, rng)
    model = MPSModel(M, chi, np.random.default_rng(seed + 1), scale=0.05)
    rec_epochs = []
    rec_train = []
    rec_test = []
    for ep in range(1, epochs + 1):
        model.train_step(Xtr, ytr, lr, wd)
        if ep % 5 == 0 or ep == 1:
            tr = model.error(Xtr, ytr)
            te = model.error(Xte, yte)
            rec_epochs.append(ep)
            rec_train.append(tr)
            rec_test.append(te)
    rec_epochs = np.array(rec_epochs)
    rec_train = np.array(rec_train)
    rec_test = np.array(rec_test)
    # detect phase: train ~0 and test >0.5, followed by test <0.05
    phase = False
    for i in range(len(rec_epochs)):
        if rec_train[i] <= 0.05 and rec_test[i] > 0.5:
            if np.any(rec_test[i + 1:] < 0.05):
                phase = True
                break
    return rec_epochs, rec_train, rec_test, phase


# Main experiment: Rule-30 CA, M=100
M = 100
main_epochs = 600
me, mtr, mte, main_phase = run_experiment(
    M=M, chi=6, n_train=300, n_test=800, epochs=main_epochs,
    lr=0.01, wd=0.001, label_fn=center_rule30, seed=42
)

# Positive control: simple parity local rule (known learnable)
c_epochs = 300
ce, ctr, cte, control_phase = run_experiment(
    M=M, chi=4, n_train=200, n_test=500, epochs=c_epochs,
    lr=0.01, wd=0.001, label_fn=center_parity, seed=7
)
control_pass = bool(np.min(cte) < 0.05)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(me, mtr, label='train')
axes[0].plot(me, mte, label='test')
axes[0].set_title('Rule-30 (main)')
axes[0].set_xlabel('epoch')
axes[0].set_ylabel('error')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[1].plot(ce, ctr, label='train')
axes[1].plot(ce, cte, label='test')
axes[1].set_title('Parity control')
axes[1].set_xlabel('epoch')
axes[1].set_ylabel('error')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig.png'), dpi=120)
plt.close(fig)

# Determine status
if not control_pass:
    status = 'inconclusive'
    notes = 'Control failed; MPS training/evaluation statistic is unreliable.'
elif main_phase:
    status = 'supported'
    notes = 'Observed interval with train error ~0 and test error >0.5 followed by test error <0.05.'
else:
    status = 'inconclusive'
    notes = 'No grokking phase found in simplified MPS audit; exact tensor-network attention model not reproduced.'

summary = {
    'claim_id': 'C5',
    'status': status,
    'metrics': {
        'control_pass': control_pass,
        'control_min_test_error': float(np.min(cte)),
        'main_phase_found': bool(main_phase),
        'main_min_train_error': float(np.min(mtr)),
        'main_max_test_error': float(np.max(mte)),
        'main_min_test_error': float(np.min(mte)),
        'main_final_train_error': float(mtr[-1]),
        'main_final_test_error': float(mte[-1]),
        'main_epochs': int(main_epochs),
        'M': int(M),
        'chi_main': 6,
    },
    'notes': notes,
}
print('SUMMARY_JSON=' + json.dumps(summary, default=str))
