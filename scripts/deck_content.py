"""Slide content specs for the re:AGENT submission deck (Lemma).

Design system: dark navy + spectrum glass-xylophone accents (hero art
generated via Runware/FLUX; art direction Codrops "Xylophone" — floating
colored glass bars evoking a DNA ladder; also the Phase-A frontend motif).

Kinds: hero, bullets, cards, stats, figure, timeline, links, closer.
Layout is implemented in make_submission_deck.py.
"""

# --- artwork (Runware FLUX.2, user-picked v1s) ----------------------------
ART = {
    "hero": "https://im.runware.ai/image/os/a05d22/ws/3/ii/d17fe52d-007b-4386-9f00-b29ff2706e72.jpg",
    "arc": "https://im.runware.ai/image/os/a09dlim3/ws/3/ii/e09f1ff1-6a4c-410d-aa1d-92090c03906a.jpg",
    "artifacts": "https://im.runware.ai/image/os/a09dlim3/ws/3/ii/9a3237da-1259-4618-8d27-e5c0ecdbb1ba.jpg",
}

# --- public figure URLs from the HF evidence datasets ---------------------
CA_DS = "https://huggingface.co/datasets/Papajams/repro-evidence-grokking-ca-local-rules/resolve/main"
ICL_DS = "https://huggingface.co/datasets/Papajams/repro-evidence-icl-provably-bayesian/resolve/main"

LINKS = {
    "repo": "https://github.com/udirobert/lemma",
    "ca_space": "https://huggingface.co/spaces/Papajams/repro-grokking-ca-local-rules",
    "icl_space": "https://huggingface.co/spaces/Papajams/repro-icl-provably-bayesian",
    "ca_data": "https://huggingface.co/datasets/Papajams/repro-evidence-grokking-ca-local-rules",
    "icl_data": "https://huggingface.co/datasets/Papajams/repro-evidence-icl-provably-bayesian",
}

# spectrum accent order used across slides
BLUE, MAGENTA, TEAL, GOLD, VIOLET = "blue", "magenta", "teal", "gold", "violet"

SLIDES = [
    # 1 — Title (full-bleed art + scrim)
    {
        "kind": "hero",
        "art": ART["hero"],
        "kicker": "re:AGENT · Founders Inc SF · Aug 15–16, 2026 · Track A",
        "title": "Lemma",
        "subtitle": "An AI scientist that distrusts itself.\nIt audits claims. It keeps the failures. It scores itself.",
        "footer": "papa  ·  github.com/udirobert/lemma",
    },
    # 2 — Problem
    {
        "kind": "bullets",
        "title": "The bottleneck is not generating claims. It is trusting them.",
        "bullets": [
            (
                BLUE,
                "Agents now *assert* that papers reproduce — but an assertion is not evidence.",
            ),
            (
                MAGENTA,
                "Trustworthy audits need the unglamorous parts: controls, error bars, negative results, and a trail a human can inspect.",
            ),
            (TEAL, "So we built the auditor, not the author."),
        ],
    },
    # 3 — Pipeline (4 numbered cards)
    {
        "kind": "cards",
        "title": "The pipeline — every step lands in an append-only trace",
        "cards": [
            (
                "01",
                BLUE,
                "Extract",
                "paper PDF → testable claims, each with a success criterion taken from the claim itself",
            ),
            (
                "02",
                MAGENTA,
                "Audit",
                "writes a self-contained numerical script, runs it, iterates; a positive control is mandatory",
            ),
            (
                "03",
                TEAL,
                "Evidence",
                "scripts, metrics, figures and feedback assemble into a Trackio logbook",
            ),
            (
                "04",
                GOLD,
                "Judge",
                "an automated rubric scores whether the trail is trustworthy: PASS / CONDITIONAL / FAIL",
            ),
        ],
    },
    # 4 — Honesty machinery (2×2 cards)
    {
        "kind": "cards",
        "title": "The differentiator: honesty machinery",
        "grid": 2,
        "cards": [
            (
                "✓",
                BLUE,
                "Verdict validator",
                "a claim is rejected if its conclusion contradicts its own metrics — failed control, NaN, zero measurable points",
            ),
            (
                "!",
                MAGENTA,
                "Failures are preserved",
                "buggy attempts and their numbers ship in the public trace: 15 failed attempts in the ICL logbook",
            ),
            (
                "↑",
                TEAL,
                "Escalation",
                "when LLM implementations stay broken, a hand-verified reviewer reference runs — and its verdict wins",
            ),
            (
                "≠",
                GOLD,
                "Honest scoping",
                "an inconclusive is a result. A patched false-positive is not.",
            ),
        ],
    },
    # 5 — Scoreboard (two paper stat cards)
    {
        "kind": "stats",
        "title": "Two papers audited end-to-end. Both evidence trails judged PASS 5/5.",
        "panels": [
            {
                "name": "Grokking phase transitions in learning local rules",
                "src": "JMLR 22-1228 · cellular automata · statistical physics",
                "accent": BLUE,
                "big": "5 / 6",
                "big_label": "claims supported",
                "metrics": [
                    ("ν = 1.00, 1.50, 3.00, 5.50", "critical exponents, reproduced"),
                    ("P_grok(D) ↓", "grokking probability vs rule complexity"),
                    ("bimodal PDF", "fast vs slow grokking-time branches"),
                ],
                "judge": "Judge: PASS 5/5 · 353 trace events",
            },
            {
                "name": "In-Context Learning Is Provably Bayesian Inference",
                "src": "arXiv 2510.10981 · learning theory",
                "accent": MAGENTA,
                "big": "3 / 6",
                "big_label": "claims supported",
                "metrics": [
                    ("rel. diff 0.4%", "risk-decomposition identity"),
                    ("slope −0.83, r² = 0.995", "Bayes-gap m/(p·N) coupling rate"),
                    ("3 inconclusive", "trace explains exactly why"),
                ],
                "judge": "Judge: PASS 5/5 · 267 trace events",
            },
        ],
    },
    # 6 — CA deep dive (figure on white card)
    {
        "kind": "figure",
        "title": "Deep dive — CA grokking: five claims closed with closed-form verification",
        "image": f"{CA_DS}/results/c2/exponent_fits.png",
        "image_dims": (1500, 450),
        "chips": [
            ("ν = (D+1)/2", TEAL, "fitted exponents: 1.500 / 2.9998 / 5.4993"),
            (
                "P_grok ↓ with D",
                BLUE,
                "Eq 86 integral + 20k-draw Monte Carlo cross-check",
            ),
            ("bimodal t_G", GOLD, "fast branch + Dirac slow branch reproduced"),
        ],
        "caption": (
            "One claim left inconclusive by design: the CPU-auditable proxy trivializes Rule-30 — "
            "a compute-scope statement preserved in the trace, not a silent skip."
        ),
    },
    # 7 — ICL deep dive
    {
        "kind": "figure",
        "title": "Deep dive — ICL: three theory claims checked numerically",
        "image": f"{ICL_DS}/results/c4/fig_pN.png",
        "image_dims": (1000, 600),
        "chips": [
            (
                "Prop 3.1",
                BLUE,
                "risk identity holds: rel. error 0.4%, Bayes gap nonzero",
            ),
            ("Thm 3.3", MAGENTA, "posterior concentration drives ICL task inference"),
            ("m/(p·N)", TEAL, "measured slope −0.826 vs predicted −1 at r² = 0.995"),
        ],
        "caption": (
            "Three claims remain inconclusive (stability bound, minimax variance link, coupling at scale) — "
            "their audit traces show exactly where the CPU budget ran out, not a verdict fudge."
        ),
    },
    # 8 — Capability arc (art backdrop + timeline steps)
    {
        "kind": "timeline",
        "art": ART["arc"],
        "title": "The capability arc — the boundary moved with a config change",
        "steps": [
            (
                "Rounds 1–2 · free 27B models",
                "every physics-sensitive claim resisted. Bake-off: ν ≈ 6000 against the paper's 1.0",
                MAGENTA,
            ),
            (
                "The gate",
                "any new model must pass the same closed-form bake-off before touching claims",
                TEAL,
            ),
            (
                "Swap · Kimi K3, hosted by Modal",
                "bake-off score: ν = 0.9911 — gate PASS. One config change, zero code changes",
                BLUE,
            ),
            (
                "Round 3",
                "headline exponent returns at ν = 0.9983. Claim reproduced",
                GOLD,
            ),
        ],
        "punchline": "capability is a tier — a good pipeline detects the boundary instead of faking past it",
    },
    # 9 — Built with (sponsor cards)
    {
        "kind": "cards",
        "title": "Built with",
        "grid": 2,
        "cards": [
            (
                "M",
                BLUE,
                "Modal",
                "hosted the Kimi K3 endpoint that powered the final CA rounds — the capability-arc swap",
            ),
            (
                "P",
                MAGENTA,
                "Paperclip / GXL",
                "corpus API powers the literature-context cells on every claim page in both logbooks",
            ),
            (
                "H",
                TEAL,
                "Hugging Face",
                "free Qwen3.8-27B inference + hosting for both logbook Spaces and both evidence datasets",
            ),
            (
                "A",
                GOLD,
                "Anthropic",
                "Claude wired into the provider stack as the fallback tier — used for the judge stage design",
            ),
        ],
    },
    # 10 — Artifacts (art backdrop + link list)
    {
        "kind": "links",
        "art": ART["artifacts"],
        "title": "Everything is public. Go find the failed attempts.",
        "links": [
            ("Repository — full trace, scripts, feedback files", LINKS["repo"]),
            ("Logbook · CA grokking (live Trackio space)", LINKS["ca_space"]),
            (
                "Logbook · ICL Bayesian inference (live Trackio space)",
                LINKS["icl_space"],
            ),
            ("Evidence dataset · CA — 90 files, failures included", LINKS["ca_data"]),
            ("Evidence dataset · ICL — 81 files, failures included", LINKS["icl_data"]),
        ],
    },
    # 11 — Closer (roadmap + thanks)
    {
        "kind": "closer",
        "title": "What's next",
        "lines": [
            (
                "A",
                BLUE,
                "static dashboard generated from the judge reports — the portfolio view across papers",
            ),
            (
                "B",
                MAGENTA,
                "interactive demo space — paste an arXiv id, watch the audit stream with the live trace",
            ),
            (
                "C",
                TEAL,
                "the product: a claim ledger and verifier API over audited evidence trails",
            ),
        ],
        "thanks": "Built solo over the weekend by papa — the trace is the demo.",
    },
]
