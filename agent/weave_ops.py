"""Optional W&B Weave tracing shim (CoreWeave Hacks build).

Off by default: init_weave() activates only with LEMMA_WEAVE=1 (on/true/yes)
AND WANDB_API_KEY set — a key's presence alone never turns on third-party
tracing. When active, @op wraps pipeline functions in weave ops so every
stage — extract, per-claim audit attempts, script tool runs, judge — lands
in the Weave UI next to the auto-patched OpenAI/Anthropic LLM calls.

When off the shim is a pass-through: the append-only JSONL trace remains
the only record and nothing breaks.
"""

from __future__ import annotations

import functools
import os

_ACTIVE = False
_DISABLED_REASON: str | None = None


def init_weave(trace=None) -> bool:
    """Initialise Weave only when explicitly enabled. Returns True when
    tracing is live. Weave is OFF unless LEMMA_WEAVE=1 (or on/true/yes)
    AND WANDB_API_KEY is set — tracing to a third-party service is never
    implied by a key's presence."""
    global _ACTIVE, _DISABLED_REASON
    if _ACTIVE:
        return True
    from dotenv import load_dotenv

    if not os.environ.get("LEMMA_NO_DOTENV"):
        load_dotenv(override=True)
    if os.environ.get("LEMMA_WEAVE", "").strip().lower() not in {
        "1",
        "on",
        "true",
        "yes",
    }:
        _DISABLED_REASON = "weave off (set LEMMA_WEAVE=1 to enable)"
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


def current_call_url() -> str | None:
    if not _ACTIVE:
        return None
    import weave

    call = weave.get_current_call()
    if call is None or not call.id:
        return None
    try:
        return call.ui_url
    except ValueError:
        return None


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
