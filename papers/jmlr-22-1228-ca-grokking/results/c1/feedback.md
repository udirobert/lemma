# Reviewer feedback — round 2

Round 1 result: fitted exponent nu = 1.8e-14 (degenerate), control passed.
The fit collapsed because the implementation did not use the paper's closed-form
dynamics and/or the fit window was wrong.

Corrections:
1. Use the CLOSED-FORM dynamics, not sample-level gradient descent. The 1D model
   (Eq. 5-6): db/dt = xbar - sgn(b)*lambda1 - (1+lambda2)*b, with solution
   b(t) = xbar_l - (xbar_l - b0)*exp(-(1+lambda2)*t), where
   xbar_l = (xbar - lambda1)/(1+lambda2) for b>=0, (xbar + lambda1)/(1+lambda2)
   for b<0. xbar is the mean of the training inputs (2N samples, N from each
   class, drawn once and fixed).
2. Data: P+(x) = exp(-(x-eps))*Theta(x-eps), P-(x) = P+(-x). To induce grokking,
   shift the data so the training mean xbar is small (as in Fig 3) — draw the
   2N samples, then compute xbar directly from them.
3. Test error E(t): fraction of misclassified TEST points (fresh draws from the
   same distributions) under threshold b(t). Compute exactly: for a threshold b,
   E = 0.5*(P+ mass below b) + 0.5*(P- mass above b), which is analytic for the
   exponential distributions — no Monte Carlo needed for E itself.
4. t_epsilon = time when train error first reaches 0 (all training points
   correctly classified by threshold b(t)).
5. Fit log(E(t)) vs log(t_epsilon - t) ONLY in a window close to the transition
   (e.g., E between ~1% and ~50% of its max, t < t_epsilon). Including the long
   flat tail away from the transition is what produced nu ~ 0.
6. Expected: nu = 1.0 +/- 0.05 (Eq. 9/10, second-order transition).
