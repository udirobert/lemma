import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(0)
os.makedirs('results/c5', exist_ok=True)

# Rule-30 lookup over the 8 possible triplets (K=1 local rule).
# The Sec-4.2 perceptron map over triplet features sees ONLY these 8 patterns,
# so train error 0 implies test error 0 by construction (no overfitting regime).
RULE30 = {(1,1,1):0,(1,1,0):0,(1,0,1):0,(1,0,0):1,
          (0,1,1):1,(0,1,0):1,(0,0,1):1,(0,0,0):0}
triplets = np.array(list(RULE30.keys()), dtype=float)
labels = np.array([RULE30[tuple(map(int,t))] for t in triplets], dtype=float)
X = np.concatenate([triplets, np.ones((8,1))], axis=1)  # bias
y = 2*labels - 1  # +/-1

# Positive control (tiny, instant): perceptron-style logistic GD on the 8 triplets.
# Known answer: Rule-30 is linearly separable over triplet features? Verify by training.
w = np.zeros(4)
lr = 0.5
for t in range(20000):
    z = y * (X @ w)
    grad = -X.T @ (y / (1.0 + np.exp(z))) / 8.0
    w -= lr * grad
pred = np.sign(X @ w)
control_final_train_err = float(np.mean(pred != y))
# Test distribution over triplets = same 8 patterns (uniform): test err == train err.
control_final_test_err = control_final_train_err
control_pass = bool(control_final_train_err == 0.0 and control_final_test_err == 0.0)

# Demonstrate the trivialization: input space cardinality.
n_unique = len({tuple(map(int,t)) for t in triplets})

fig, ax = plt.subplots(figsize=(5,4))
ax.bar(['train err','test err'], [control_final_train_err, control_final_test_err])
ax.set_ylim(0,1); ax.set_title('Perceptron-map proxy: 8-triplet lookup (no gen gap possible)')
fig.savefig('results/c5/control.png', bbox_inches='tight')

summary = {
  'claim_id': 'C5',
  'status': 'inconclusive',
  'metrics': {
    'control_pass': control_pass,
    'control_final_train_err': control_final_train_err,
    'control_final_test_err': control_final_test_err,
    'n_unique_triplet_patterns': n_unique,
    'generalization_gap_possible': False
  },
  'notes': ('The perceptron-map proxy trivializes Rule-30 (8-triplet lookup: zero '
            'generalization gap by construction). The paper\'s claim concerns the '
            'over-parameterized tensor-network regime (gpu-small, out of CPU-audit '
            'scope). Verdict: inconclusive by compute-scope limitation - reproducing '
            'it requires tensor-network training on long sequences.')
}
print('SUMMARY_JSON=' + json.dumps(summary, default=str))
