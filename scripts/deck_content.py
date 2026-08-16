"""Slide content specs for the re:AGENT submission deck (Lemma).

Each slide: title, kind, and content. Kept separate from the API glue so the
copy can be reviewed without touching Google calls.
"""

# Public figure URLs from the published HF evidence datasets
CA_DS = "https://huggingface.co/datasets/Papajams/repro-evidence-grokking-ca-local-rules/resolve/main"
ICL_DS = "https://huggingface.co/datasets/Papajams/repro-evidence-icl-provably-bayesian/resolve/main"

LINKS = {
    "repo": "https://github.com/udirobert/lemma",
    "ca_space": "https://huggingface.co/spaces/Papajams/repro-grokking-ca-local-rules",
    "icl_space": "https://huggingface.co/spaces/Papajams/repro-icl-provably-bayesian",
    "ca_data": "https://huggingface.co/datasets/Papajams/repro-evidence-grokking-ca-local-rules",
    "icl_data": "https://huggingface.co/datasets/Papajams/repro-evidence-icl-provably-bayesian",
}

SLIDES = [
    # 1 — Title
    {
        "kind": "title",
        "title": "Lemma",
        "subtitle": "An AI scientist that distrusts itself\n\n"
        "re:AGENT — End to End Agentic Science · Track A (Co-Scientist) · Founders Inc SF\n"
        "August 15–16, 2026 · papa",
    },
    # 2 — Problem
    {
        "kind": "bullets",
        "title": "The bottleneck is not generating claims. It is trusting them.",
        "bullets": [
            "AI agents increasingly *assert* that a paper reproduces — but an assertion is not evidence",
            "A trustworthy audit needs the unglamorous parts: controls, error bars, negative results, and a trail a human can inspect",
            "So we built the auditor, not the author: Lemma extracts claims, writes and runs its own audit scripts, keeps every failure, and scores itself",
        ],
    },
    # 3 — Pipeline
    {
        "kind": "bullets",
        "title": "Four stages. Every step logged to an append-only trace.",
        "bullets": [
            "1. Extract — paper PDF → testable claims, each with a success criterion taken from the claim itself",
            "2. Audit — the agent writes a self-contained numerical script, runs it, and iterates; a positive control is mandatory on every claim",
            "3. Evidence — scripts, metrics, figures and feedback assemble into a Trackio logbook",
            "4. Judge — an automated rubric scores whether the trail is trustworthy: PASS / CONDITIONAL PASS / FAIL",
        ],
    },
    # 4 — Honesty machinery
    {
        "kind": "bullets",
        "title": "The differentiator: honesty machinery",
        "bullets": [
            "Verdict validator — a claim is rejected if its conclusion contradicts its own metrics (failed control, NaN, zero measurable points)",
            "Failures are preserved — buggy attempts and their numbers ship in the public trace (15 failed attempts in the ICL logbook)",
            "Escalation — when the LLM's implementations stay broken, a hand-verified reviewer reference runs and its verdict wins",
            "An inconclusive is a result. A patched false-positive is not.",
        ],
    },
    # 5 — Scoreboard
    {
        "kind": "table",
        "title": "Two papers audited end-to-end. Both evidence trails judged PASS 5/5.",
        "table": {
            "headers": [
                "Paper",
                "Claims",
                "Supported",
                "Falsified",
                "Inconclusive",
                "Judge",
            ],
            "rows": [
                [
                    "CA grokking (JMLR 22-1228)",
                    "6",
                    "5",
                    "0",
                    "1 (compute scope)",
                    "PASS 5/5",
                ],
                [
                    "ICL is Bayesian inference (arXiv 2510.10981)",
                    "6",
                    "3",
                    "0",
                    "3",
                    "PASS 5/5",
                ],
            ],
        },
    },
    # 6 — CA deep dive
    {
        "kind": "figure",
        "title": "Deep dive: grokking phase transitions (JMLR 22-1228)",
        "image": f"{CA_DS}/results/c2/exponent_fits.png",
        "image_dims": (1500, 450),
        "caption": (
            "Five claims closed with closed-form verification: critical exponents "
            "ν = 1 (1D) and ν = (D+1)/2 in D-ball geometry (fitted: 1.500 / 2.9998 / 5.4993), "
            "P_grok strictly decreasing with rule complexity, L1 > L2 grokking probability, "
            "and a bimodal grokking-time distribution. "
            "One claim left inconclusive — the CPU-auditable proxy trivializes Rule-30; that is a scope statement, not a failure."
        ),
    },
    # 7 — ICL deep dive
    {
        "kind": "figure",
        "title": "Deep dive: ICL is provably Bayesian inference (arXiv 2510.10981)",
        "image": f"{ICL_DS}/results/c4/fig_pN.png",
        "image_dims": (1000, 600),
        "caption": (
            "Three claims supported: the risk-decomposition identity (relative error 0.4%, nonzero Bayes gap), "
            "posterior concentration as the mechanism, and the Bayes-gap upper bound scaling as m/(p·N) — "
            "measured slope −0.826 at r² = 0.995 against the predicted −1. "
            "Three claims remain inconclusive; their audit traces show exactly why."
        ),
    },
    # 8 — Capability arc / Modal
    {
        "kind": "bullets",
        "title": "The capability arc: the boundary moved with a config change",
        "bullets": [
            "Round 1–2 on free 27B models: every physics-sensitive claim resisted — a bake-off gave ν ≈ 6000 against the paper's 1.0",
            "The model swap was gated: Kimi K3 (hosted by Modal on our endpoint) had to pass the same closed-form bake-off first — it scored ν = 0.9911",
            "Round 3 on Kimi K3: the headline exponent came back at ν = 0.9983 — claim reproduced",
            "Lesson: capability is a tier, and a good pipeline detects the tier boundary instead of faking past it",
        ],
    },
    # 9 — Built with
    {
        "kind": "bullets",
        "title": "Built with",
        "bullets": [
            "Modal — hosted the Kimi K3 endpoint that powered the final CA rounds (the capability-arc swap)",
            "Paperclip / GXL — corpus API powers the literature-context cells on every claim page in both logbooks",
            "Hugging Face — free Qwen3.8-27B inference, plus hosting for both logbook Spaces and both evidence datasets",
            "Anthropic — wired into the provider stack as the fallback tier",
        ],
    },
    # 10 — Artifacts
    {
        "kind": "links",
        "title": "Everything is public. Go find the failed attempts.",
        "links": [
            ("Repository (full trace, scripts, feedback files)", LINKS["repo"]),
            ("Logbook: CA grokking — live Trackio space", LINKS["ca_space"]),
            (
                "Logbook: ICL Bayesian inference — live Trackio space",
                LINKS["icl_space"],
            ),
            ("Evidence dataset: CA (90 files, failures included)", LINKS["ca_data"]),
            ("Evidence dataset: ICL (81 files, failures included)", LINKS["icl_data"]),
        ],
    },
    # 11 — Roadmap + team
    {
        "kind": "bullets",
        "title": "What's next",
        "bullets": [
            "Phase A — static dashboard generated from the judge reports: the portfolio view across papers",
            "Phase B — interactive demo space: paste an arXiv id, watch the audit stream with the live trace visible",
            "Phase C — the product: a claim ledger and verifier API over audited evidence trails",
            "Built solo over the weekend by papa · github.com/udirobert/lemma",
        ],
    },
]
