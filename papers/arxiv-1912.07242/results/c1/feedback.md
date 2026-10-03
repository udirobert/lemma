# Reviewer correction for C1 (after Round 1: 3 consecutive 20-min timeouts)

All three attempts used d=1000 with T=50 trials over a ~200-point n-grid —
~10k least-squares solves on matrices up to 2000x1000, far beyond the 20-min
per-attempt budget. The claim (test MSE peaks at n=d) is dimension-agnostic;
the paper's Figure 1 shows the same peak at small d. Re-audit with a tractable
scale:

- Use d=200 (or at most d=300). lstsq cost scales ~O(max(n,d)*min(n,d)^2), so
  d=200 is ~25x cheaper per solve than d=1000.
- n-grid: focus on n in [0.5d, 2.5d] with ~25-40 points, T=20 trials — enough
  to resolve the peak and estimate its location.
- Alternatively compute the CLOSED-FORM expected risk via the paper's
  Claim-1 formula (traces of covariance-weighted pseudoinverse moments) and
  verify argmax at n=d analytically, with a small MC only as a sanity check.
- Mandatory positive control stays: run the same measurement on a case where
  the peak location is known a priori (e.g. isotropic features where the
  minimum-norm interpolant risk formula predicts the peak), and confirm the
  procedure recovers it before trusting the main measurement.

Success criterion unchanged: peak of mean test MSE within ~10% of n=d, and
non-monotonicity (risk at n>d below the peak) must be measurable.
