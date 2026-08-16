# Reviewer resolution for C5 (Round 4)

Round 3 produced an honest negative that must be re-scoped: the
Sec-4.2 perceptron map over triplet features TRIVIALIZES Rule-30 — the
input space is exactly 8 triplet patterns, so train error 0 implies
test error 0 with no overfitting regime for grokking to occur in. The
paper's grokking result lives in the over-parameterized tensor-network
regime on long CA sequences, which is gpu-small and out of CPU-audit
budget. The round-2 feedback anticipated exactly this outcome.

Required for the next attempt:

1. Emit status=inconclusive (NOT falsified). A proxy that cannot
   exhibit the phenomenon by construction does not refute the paper's
   claim about a different (tensor-network) model class.
2. Include in notes, precisely: "The perceptron-map proxy trivializes
   Rule-30 (8-triplet lookup: zero generalization gap by construction).
   The paper's claim concerns the over-parameterized tensor-network
   regime (gpu-small, out of CPU-audit scope). Verdict: inconclusive by
   compute-scope limitation — reproducing it requires tensor-network
   training on long sequences."
3. Keep the round-3 control metrics (control_final_train_err,
   control_final_test_err) in the summary and control_pass=true since
   the control trained correctly.
4. Do NOT run further training loops; the conclusion is settled by the
   scope argument. Script should be tiny and fast.
