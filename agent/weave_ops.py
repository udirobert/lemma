"""Optional W&B Weave tracing shim (CoreWeave Hacks build).

With WANDB_API_KEY set (and `weave` installed), init_weave() activates and
@op wraps pipeline functions in weave ops so every stage — extract, per-claim
audit attempts, script tool runs, judge — lands in the Weave UI next to the
auto-patched OpenAI/Anthropic LLM calls.

Without a key the shim is a pass-through: the append-only JSONL trace remains
the only record and nothing breaks. `LEMMA_WEAVE=off` forces the shim off.
"""

from __future__ import annotations

import functools
import os

_ACTIVE = False
_DISABLED_REASON: str | None = None


def init_weave(trace=None) -> bool:
    """Initialise Weave if configured. Returns True when tracing is live."""
    global _ACTIVE, _DISABLED_REASON
    if _ACTIVE:
        return True
    from dotenv import load_dotenv

    load_dotenv(override=True)
    if os.environ.get("LEMMA_WEAVE", "").strip().lower() in {"0", "off", "false"}:
        _DISABLED_REASON = "disabled via LEMMA_WEAVE"
        return False
    if not os.environ.get("WANDB_API_KEY", "").strip():
        _DISABLED_REASON = "WANDB_API_KEY not set"
        return False
    try:
        import weave
    except ImportError:
        _DISABLED_REASON = "weave package not installed"
        return False
    project = os.environ.get("LEMMA_WEAVE_PROJECT", "lemma-audits").strip()
    try:
        weave.init(project)
    except Exception as e:
        _DISABLED_REASON = f"weave.init failed: {e}"
        if trace is not None:
            trace.note("weave", f"weave.init failed ({e}); JSONL trace only")
        return False
    _ACTIVE = True
    if trace is not None:
        trace.note("weave", f"weave tracing live -> project {project!r}")
    return True


def active() -> bool:
    return _ACTIVE


def disabled_reason() -> str | None:
    return _DISABLED_REASON


def op(fn):
    """weave.op when active, pass-through otherwise. Resolved per call so
    decorators applied at import time still pick up a later init_weave()."""
    cached = None

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        nonlocal cached
        if _ACTIVE:
            if cached is None:
                import weave

                cached = weave.op(fn)
            return cached(*args, **kwargs)
        return fn(*args, **kwargs)

    return wrapper
