"""Tool schemas for the lemma plugin — name/description is how the model
decides when to call us, so the descriptions say what comes back."""

LEMMA_QUICK_AUDIT = {
    "name": "lemma_quick_audit",
    "description": (
        "Audit a paper's first testable claims end to end: extract claims "
        "from an arXiv id / OpenReview id / PDF path, write and run "
        "numerical audit scripts (positive control mandatory), then judge "
        "the evidence trail. Returns per-claim verdicts "
        "(supported/falsified/inconclusive) plus paths to the preserved "
        "evidence folder. Bounded to a few claims and ~10 min — use "
        "lemma_audit for a full-paper run."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "arXiv id, OpenReview id, arXiv URL, or local PDF path.",
            },
            "max_claims": {
                "type": "integer",
                "description": "Cap on claims audited (default 2, max 5).",
                "default": 2,
            },
        },
        "required": ["source"],
    },
}

LEMMA_AUDIT = {
    "name": "lemma_audit",
    "description": (
        "Start a full lemma audit of a paper (all testable claims, all "
        "pipeline stages) in the background — this takes tens of minutes. "
        "Returns a run/workdir path immediately; poll with lemma_status."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "source": {
                "type": "string",
                "description": "arXiv id, OpenReview id, arXiv URL, or local PDF path.",
            },
        },
        "required": ["source"],
    },
}

LEMMA_STATUS = {
    "name": "lemma_status",
    "description": (
        "List audited papers and their per-claim verdicts, plus judge "
        "verdicts and evidence-trail paths, for every workdir under the "
        "lemma plugin data dir. Use after lemma_audit to check progress."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "paper": {
                "type": "string",
                "description": "Optional: restrict to one workdir name.",
            }
        },
    },
}

LEMMA_JUDGE = {
    "name": "lemma_judge",
    "description": (
        "Re-run lemma's automated evidence-trustworthiness judge on an "
        "existing audit workdir and return the verdict + rubric "
        "(structure, evidence, integrity, cost, trace)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "paper": {
                "type": "string",
                "description": "Workdir name under the lemma papers dir.",
            },
        },
        "required": ["paper"],
    },
}
