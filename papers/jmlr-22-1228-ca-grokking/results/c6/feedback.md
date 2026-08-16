# Reviewer correction for C6 (Round 5 — HARD instruction)

Round 4 re-implemented the SAME miscalibrated detector as round 3
(identical metrics, bimodal=false at separation 1.73 stds). The round-4
feedback specified exact replacement tests; they were not applied.
This feedback is prescriptive code — implement EXACTLY:

1. Keep the simulation byte-for-byte from round 3/4 (D=5, eps=2,
   lambda_2=0.01, 10000 trials) — the DATA is not the problem.
2. DELETE the old bimodality criterion (separation > 2x larger-cluster
   std). Replace with these four tests, computed on the 10000 grokking
   times:
   a) valley_ratio: np.histogram(times, bins=60); find the two local
      maxima (modes) and the minimum (valley) between them.
      valley_ratio = valley_count / min(mode1_count, mode2_count).
      Pass if valley_ratio < 0.5 AND both modes hold >= 5% of trials.
   b) separation_ratio = |mean_slow - mean_fast| / max(std_fast, std_slow).
      Pass if > 1.5.  (Round 3 values: 2.016 / 1.163 = 1.73 — PASSES.)
   c) slow_rel_err = |slow_mean - t_analytic| / t_analytic with
      t_analytic = (1/(2*lambda_2D)) * ln(eps^4/(eps^4-1)).
      Pass if < 0.25.  (Round 3: 0.214 — PASSES.)
   d) sharpness: std/mean of slow cluster vs fast cluster.
      Pass if slow < fast.  (Round 3: 0.458 < 0.811 — PASSES.)
3. Cluster assignment for (b)-(d): K=2 k-means (numpy, or scipy.cluster)
   on the grokking times, or the histogram split at the valley bin.
4. status=supported iff a-d ALL pass AND the positive control passes
   (synthetic bimodal mixture with known modes must pass all four tests;
   a UNIMODAL synthetic sample must fail the valley test).
5. Report all four test values in metrics. Do not invent additional
   rejection criteria.
