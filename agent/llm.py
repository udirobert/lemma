"""LLM backend with provider fallback.

Provider order (override with LEMMA_PROVIDER=anthropic|openai in .env):
  1. anthropic — needs ANTHROPIC_API_KEY (model: LEMMA_MODEL or Claude default)
  2. openai-compatible — needs OPENAI_API_KEY (optionally OPENAI_BASE_URL,
     OPENAI_API_BASE; model: LEMMA_MODEL or OPENAI_MODEL or gpt-4o default)

Every call is logged to the run trace, including provider fallbacks.
"""

from __future__ import annotations

import json
import os
import time

from dotenv import load_dotenv

from agent.traces import Trace

ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
OPENAI_DEFAULT_MODEL = "gpt-4o"
MAX_TOKENS = 8192

_clients: dict = {}


def _providers() -> list[str]:
    load_dotenv(override=True)
    explicit = os.environ.get("LEMMA_PROVIDER", "").strip().lower()
    if explicit in ("anthropic", "openai"):
        others = [p for p in ("anthropic", "openai") if p != explicit]
        return [explicit, *others]
    order = []
    if os.environ.get("ANTHROPIC_API_KEY"):
        order.append("anthropic")
    if os.environ.get("OPENAI_API_KEY"):
        order.append("openai")
    if not order:
        raise RuntimeError(
            "No LLM provider configured: set ANTHROPIC_API_KEY or OPENAI_API_KEY "
            "in .env (see .env.example)."
        )
    return order


def model_name(provider: str | None = None) -> str:
    load_dotenv(override=True)
    if os.environ.get("LEMMA_MODEL"):
        return os.environ["LEMMA_MODEL"]
    if provider == "openai":
        return os.environ.get("OPENAI_MODEL", OPENAI_DEFAULT_MODEL)
    if provider == "anthropic":
        return ANTHROPIC_DEFAULT_MODEL
    try:
        return model_name(_providers()[0])
    except RuntimeError:
        return ANTHROPIC_DEFAULT_MODEL


def complete(
    trace: Trace,
    stage: str,
    system: str,
    user: str,
    *,
    max_tokens: int = MAX_TOKENS,
    temperature: float = 0.2,
) -> str:
    """Single-turn completion with provider fallback."""
    last_err: Exception | None = None
    for provider in _providers():
        try:
            text, model, duration = _complete_one(
                provider, system, user, max_tokens, temperature
            )
            trace.llm_call(
                stage,
                model=f"{provider}:{model}",
                prompt_chars=len(system) + len(user),
                response_chars=len(text),
                duration_s=duration,
            )
            return text
        except Exception as e:
            last_err = e
            trace.note(
                stage,
                f"provider {provider} failed ({type(e).__name__}: "
                f"{str(e)[:200]}); trying fallback",
            )
    raise RuntimeError(
        f"all LLM providers failed; last error: {last_err}"
    ) from last_err


def _complete_one(
    provider: str, system: str, user: str, max_tokens: int, temperature: float
) -> tuple[str, str, float]:
    t0 = time.time()
    if provider == "anthropic":
        import anthropic

        if "anthropic" not in _clients:
            _clients["anthropic"] = anthropic.Anthropic()
        model = model_name("anthropic")
        resp = _clients["anthropic"].messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
    elif provider == "openai":
        from openai import OpenAI

        if "openai" not in _clients:
            load_dotenv(override=True)
            base_url = (
                os.environ.get("OPENAI_BASE_URL")
                or os.environ.get("OPENAI_API_BASE")
                or None
            )
            kwargs = {"api_key": os.environ["OPENAI_API_KEY"]}
            if base_url:
                kwargs["base_url"] = base_url
            _clients["openai"] = OpenAI(**kwargs)
        model = model_name("openai")
        resp = _clients["openai"].chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = resp.choices[0].message.content or ""
    else:
        raise ValueError(f"unknown provider {provider!r}")
    return text, model, time.time() - t0


def extract_json(text: str) -> object:
    """Pull the first JSON object/array out of an LLM response."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    start_obj, start_arr = text.find("{"), text.find("[")
    starts = [s for s in (start_obj, start_arr) if s != -1]
    if not starts:
        return json.loads(text)
    start = min(starts)
    depth, quote, esc = 0, "", False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
        elif c == "\\":
            esc = True
        elif c == '"':
            quote = "" if quote else '"'
        elif not quote:
            if c in "{[":
                depth += 1
            elif c in "}]":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : i + 1])
    return json.loads(text[start:])
