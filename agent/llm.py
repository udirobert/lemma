"""LLM backend with a multi-endpoint provider stack and fallback.

Providers are tried in order (override with LEMMA_PROVIDER=<name>):
  1. named OpenAI-compatible endpoints from .env:
       LEMMA_ENDPOINTS=hf,orca            (order = priority)
       LEMMA_<NAME>_API_KEY / _BASE_URL / _MODEL   (e.g. LEMMA_HF_API_KEY)
  2. anthropic — ANTHROPIC_API_KEY (model: LEMMA_MODEL or Claude default)
  3. openai — OPENAI_API_KEY (+ OPENAI_BASE_URL/OPENAI_API_BASE,
     OPENAI_MODEL; model override: LEMMA_MODEL)

Rate-limit aware: 429s honour Retry-After (up to ~40 s, 3 retries), and a
60 s error window triggers a fallback to the next provider. Every call and
fallback is logged to the run trace.
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
MAX_RETRIES = 3
MAX_RETRY_WAIT_S = 40.0
FALLBACK_ERROR_WINDOW_S = 60.0

_clients: dict = {}
_provider_fail_at: dict[str, float] = {}


def _endpoint_names() -> list[str]:
    load_dotenv(override=True)
    raw = os.environ.get("LEMMA_ENDPOINTS", "").strip()
    return [n.strip().upper() for n in raw.split(",") if n.strip()]


def _endpoint_cfg(name: str) -> dict | None:
    api_key = os.environ.get(f"LEMMA_{name}_API_KEY", "").strip()
    base_url = (os.environ.get(f"LEMMA_{name}_BASE_URL") or "").strip() or None
    model = (
        os.environ.get(f"LEMMA_{name}_MODEL")
        or os.environ.get("LEMMA_MODEL")
        or OPENAI_DEFAULT_MODEL
    ).strip()
    if not api_key:
        return None
    extra_body = None
    raw_extra = (os.environ.get(f"LEMMA_{name}_EXTRA_BODY") or "").strip()
    if raw_extra:
        try:
            extra_body = json.loads(raw_extra)
        except json.JSONDecodeError:
            extra_body = None
    extra_headers = None
    raw_headers = (os.environ.get(f"LEMMA_{name}_HEADERS") or "").strip()
    if raw_headers:
        try:
            extra_headers = json.loads(raw_headers)
        except json.JSONDecodeError:
            extra_headers = None
    return {
        "name": name,
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "extra_body": extra_body,
        "extra_headers": extra_headers,
    }


def _providers() -> list[str]:
    load_dotenv(override=True)
    explicit = os.environ.get("LEMMA_PROVIDER", "").strip().lower()
    if explicit:
        others = [
            p
            for p in (*_endpoint_names(), "anthropic", "openai")
            if p.lower() != explicit
        ]
        return [explicit.upper(), *[o for o in others if o.lower() != explicit]]
    order: list[str] = []
    for name in _endpoint_names():
        if _endpoint_cfg(name):
            order.append(name)
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        order.append("anthropic")
    if os.environ.get("OPENAI_API_KEY", "").strip():
        order.append("openai")
    if not order:
        raise RuntimeError(
            "No LLM provider configured: set LEMMA_ENDPOINTS (+ per-endpoint "
            "LEMMA_<NAME>_* vars), ANTHROPIC_API_KEY, or OPENAI_API_KEY in .env "
            "(see .env.example)."
        )
    return order


def model_name(provider: str | None = None) -> str:
    load_dotenv(override=True)
    p = (provider or "").upper()
    if p in _endpoint_names():
        cfg = _endpoint_cfg(p)
        if cfg:
            return cfg["model"]
    if os.environ.get("LEMMA_MODEL"):
        return os.environ["LEMMA_MODEL"]
    if p == "OPENAI":
        return os.environ.get("OPENAI_MODEL", OPENAI_DEFAULT_MODEL)
    if p == "ANTHROPIC":
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
    """Single-turn completion with per-provider retries and fallback."""
    last_err: Exception | None = None
    for provider in _providers():
        # skip providers that are in a recent-error window
        if time.time() - _provider_fail_at.get(provider, 0.0) < FALLBACK_ERROR_WINDOW_S:
            trace.note(stage, f"provider {provider} in error window; skipping")
            continue
        try:
            text, model, duration = _complete_with_retry(
                provider, system, user, max_tokens, temperature
            )
            trace.llm_call(
                stage,
                model=f"{provider.lower()}:{model}",
                prompt_chars=len(system) + len(user),
                response_chars=len(text),
                duration_s=duration,
            )
            return text
        except Exception as e:
            last_err = e
            _provider_fail_at[provider] = time.time()
            trace.note(
                stage,
                f"provider {provider} failed ({type(e).__name__}: "
                f"{str(e)[:200]}); trying fallback",
            )
    raise RuntimeError(
        f"all LLM providers failed; last error: {last_err}"
    ) from last_err


def _complete_with_retry(
    provider: str, system: str, user: str, max_tokens: int, temperature: float
) -> tuple[str, str, float]:
    from openai import APIStatusError, RateLimitError

    attempt = 0
    t0 = time.time()
    while True:
        attempt += 1
        try:
            text, model = _complete_one(provider, system, user, max_tokens, temperature)
            return text, model, time.time() - t0
        except RateLimitError as e:
            if attempt > MAX_RETRIES:
                raise
            wait = min(
                float(e.response.headers.get("retry-after", "5") or "5"),
                MAX_RETRY_WAIT_S,
            )
            time.sleep(wait)
        except APIStatusError as e:
            # transient server-side errors: retry a couple of times
            if e.status_code >= 500 and attempt <= MAX_RETRIES:
                time.sleep(min(2.0**attempt, 8.0))
                continue
            raise


def _complete_one(
    provider: str, system: str, user: str, max_tokens: int, temperature: float
) -> tuple[str, str]:
    p = provider.upper()
    if p == "ANTHROPIC":
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
        return text, model

    # OpenAI-compatible (named endpoints + the generic openai provider)
    from openai import OpenAI

    if p == "OPENAI":
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
        client = _clients["openai"]
    else:
        cfg = _endpoint_cfg(p)
        if not cfg:
            raise RuntimeError(f"endpoint {p} not configured")
        if p not in _clients:
            kwargs = {"api_key": cfg["api_key"]}
            if cfg["base_url"]:
                kwargs["base_url"] = cfg["base_url"]
            _clients[p] = OpenAI(**kwargs)
        client = _clients[p]

    model = model_name(provider)
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    cfg = _endpoint_cfg(p) if p != "OPENAI" else None
    if cfg and cfg.get("extra_body"):
        kwargs["extra_body"] = cfg["extra_body"]
    if cfg and cfg.get("extra_headers"):
        # fresh uuid per request for "{uuid}" placeholders (idempotency ids)
        import uuid as _uuid

        kwargs["extra_headers"] = {
            k: (v.format(uuid=_uuid.uuid4()) if "{uuid}" in v else v)
            for k, v in cfg["extra_headers"].items()
        }
    resp = client.chat.completions.create(**kwargs)
    text = resp.choices[0].message.content or ""
    if not text.strip():
        raise RuntimeError(
            "empty content (reasoning budget likely exhausted max_tokens)"
        )
    return text, model


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
