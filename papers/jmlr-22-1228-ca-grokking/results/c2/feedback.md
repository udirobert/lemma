# Reviewer feedback — round 2

Round 1 result: all fitted exponents "N/A" — the fit never produced values.
Control fit worked (nu=1.499, r2=0.9999), so the machinery is fine but the
actual D-dimensional simulations never yielded usable E_D(t) curves.

Corrections:
1. Model (Sec 3.3): D-dim inputs, positive/negative samples uniform in unit
   balls shifted by +/-eps along axis 1. Student perceptron f(x)=sgn(w.x),
   bias zero (Eq. 18). Gradient flow on w with L1+L2 regularization.
2. To make this tractable and correct, use the paper's reduced dynamics where
   possible: by symmetry the dynamics reduces to the norm/angle of w; if you
   simulate the full perceptron, use moderate N (e.g. N=50-200 per class) and
   enough steps to observe grokking (train error 0 first, test error 0 later).
3. Grokking requires a NON-TYPICAL dataset (Sec 3.3.1): shift samples so the
   initial margin structure delays test-error collapse. If a plain draw does
   not grok within the step budget, increase eps slightly toward the
   critical separation and retry.
4. Test error E_D(t): fraction of misclassified fresh test points.
5. Fit log(E_D) vs log(t_eps - t) in a window near the transition.
6. Expected exponents (Eq. 23): nu = (D+1)/2 -> D=2: 1.5, D=5: 3.0, D=10: 5.5.
   Tolerance 0.1 per dimension.
7. If a dimension's simulation cannot reach grokking within budget, report its
   status individually rather than N/A — partial results are fine.
