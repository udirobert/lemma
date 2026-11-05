"""lemma — the agent that audits what agents read.

Register() must be import-clean: no network, no heavy imports — every
handler lazily shells out to the lemma repo at call time.
"""

import logging
from functools import partial

from . import schemas, tools

logger = logging.getLogger(__name__)


def register(ctx):
    ctx.register_tool(
        name="lemma_quick_audit",
        toolset="lemma",
        schema=schemas.LEMMA_QUICK_AUDIT,
        handler=partial(tools.lemma_quick_audit, ctx=ctx),
    )
    ctx.register_tool(
        name="lemma_audit",
        toolset="lemma",
        schema=schemas.LEMMA_AUDIT,
        handler=partial(tools.lemma_audit, ctx=ctx),
    )
    ctx.register_tool(
        name="lemma_status",
        toolset="lemma",
        schema=schemas.LEMMA_STATUS,
        handler=partial(tools.lemma_status, ctx=ctx),
    )
    ctx.register_tool(
        name="lemma_judge",
        toolset="lemma",
        schema=schemas.LEMMA_JUDGE,
        handler=partial(tools.lemma_judge, ctx=ctx),
    )
    ctx.register_tool(
        name="lemma_improve",
        toolset="lemma",
        schema=schemas.LEMMA_IMPROVE,
        handler=partial(tools.lemma_improve, ctx=ctx),
    )

    # Ambient: notice paper links in user turns, offer (never run) an audit
    ctx.register_hook("pre_llm_call", partial(tools.paper_link_hint, ctx=ctx))

    # /audit <source> — zero-latency demo path: dispatch our own quick audit
    ctx.register_command(
        "audit",
        lambda raw: ctx.dispatch_tool("lemma_quick_audit", {"source": raw.strip()}),
        description="Lemma quick audit (arXiv/OpenReview id or PDF path)",
        args_hint="<paper>",
    )
