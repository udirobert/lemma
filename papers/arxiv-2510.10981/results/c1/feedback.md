# Reviewer correction for C1 (Round 6)

A reviewer-provided reference implementation exists at
results/c1/reviewer_reference.py and has been verified by execution
(rel_diff = 0.004, RBG_main = 0.288 nonzero, control_pass = True,
status = supported). The pipeline now executes it automatically if the
LLM-generated attempts in this round fail; the trace will record a
reviewer_reference_executed event.

For the LLM attempts: use the reference VERBATIM (copy it; it is
self-contained numpy + matplotlib and prints SUMMARY_JSON). Do not
re-derive the estimator — 8 prior attempts failed doing so. The identity
verified is Proposition 3.1 (R = RBG + RPV) with MAIN predictor = the
task-1-only oracle (nonzero Bayes Gap) and CONTROL = the Bayes predictor.
