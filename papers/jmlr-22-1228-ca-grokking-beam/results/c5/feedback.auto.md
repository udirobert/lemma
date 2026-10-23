# Reviewer notes for next audit attempt

## Diagnosis

The previous run did **not** provide evidence against the claim. The MPS never learned: both train and test error stayed near 0.5, and the positive control also failed. The most likely root cause is **numerical vanishing in the M=100 tensor contraction**, not the claim itself.

With random cores scaled by `0.05` and bond dimension `χ≈6`, the variance of a length-100 contraction is roughly `(χ σ²)^M`. For `σ=0.05`, this is astronomically small, so activations and gradients vanish. The model therefore cannot move away from its initial all-zero-ish predictions. The control failure confirms an implementation/optimization issue rather than a falsification.

A secondary issue is that the audit used a generic MPS, while the test plan asks for the **tensor-network attention model from Section 4.2**. If that model has attention, positional, or multi-layer structure beyond an MPS, the next attempt must implement that exact architecture rather than assuming an MPS is equivalent.

## Required fixes

1. **Fix initialization scale.**
   Initialize every tensor core with scale `1/sqrt(χ)` (e.g. `0.408` for `χ=6`, `0.354` for `χ=8`), not `0.05`. This gives O(1) forward values and usable gradients for `M=100`.

2. **Use a stronger but still faithful training setup.**
   - Full-batch gradient descent with weight decay, as required.
   - `lr = 0.05`
   - `wd = 0.01`
   - `epochs = 3000` for the main experiment
   - `epochs = 1500` for the control
   - Evaluate every 10 epochs, not every 5, to reduce overhead.
   - Clip the **global** gradient norm to 10, not each tensor independently.

3. **Use sufficient capacity for memorization.**
   - Main: `χ = 8`, `n_train = 300`, `n_test = 1000`
   - Control: `χ = 4`, `n_train = 200`, `n_test = 500`
   This keeps the model able to memorize the training set while still needing weight decay to generalize.

4. **Implement the exact Section 4.2 model if it is not an MPS.**
   If Section 4.2 specifies tensor-network attention, positional encodings, multiple layers, or a particular contraction order, implement those details. Do not silently substitute a generic MPS and call it the claimed model.

## Evaluation protocol

Keep the claim's own criterion unchanged:

- A checkpoint qualifies as the grokking phase only if:
  - `train_error <= 0.05`
  - `test_error > 0.5`
  at the same recorded epoch.
- The experiment passes only if some later recorded checkpoint has:
  - `test_error < 0.05`.

Also record:

- `first_phase_epoch`
- `num_phase_checkpoints`
- `transition_epoch` (first epoch after a phase checkpoint with test error `< 0.05`)
- `min_train_error`
- `max_test_error_during_phase`
- `min_test_error`
- final train/test errors

This makes the audit auditable rather than relying on a single boolean.

## Positive control

The parity control must reach `test_error < 0.05`. If it does not after the initialization and training fixes above, then the issue is still in the model, gradient, or optimization code. Debug that before drawing any conclusion about Rule-30 grokking.

Do not skip or weaken the control. A failed control means the audit is inconclusive, not that the claim is false.

## Verdict handling

- If the control passes and the main run shows train error near zero with test error above 0.5, followed by test error below 0.05, report `supported`.
- If the control passes but no such phase occurs after a reasonable training budget, report `refuted` or `inconclusive` with the recorded curves and metrics. A correct falsification demonstration should show that training error can reach zero while test error remains high, or that test error never drops below 0.05 despite adequate optimization—not merely that the model failed to learn.
- Do not fabricate metrics, shorten the criterion, or relabel a non-grokking curve as grokking.
