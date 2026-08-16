#!/usr/bin/env python3
"""Debug: print what the model actually generated for the bakeoff task."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dotenv import load_dotenv  # noqa: E402

from agent.llm import _endpoint_cfg  # noqa: E402
from scripts.bakeoff_codegen import SYSTEM, TASK  # noqa: E402

load_dotenv(REPO / ".env", override=True)

load_dotenv(REPO / ".env", override=True)


def main() -> None:
    name = sys.argv[1].upper()
    cfg = _endpoint_cfg(name)
    from openai import OpenAI

    kwargs = {"api_key": cfg["api_key"]}
    if cfg["base_url"]:
        kwargs["base_url"] = cfg["base_url"]
    client = OpenAI(**kwargs)
    extra = {}
    if cfg.get("extra_body"):
        extra["extra_body"] = cfg["extra_body"]
    if cfg.get("extra_headers"):
        import uuid

        extra["extra_headers"] = {
            k: (v.format(uuid=uuid.uuid4()) if "{uuid}" in v else v)
            for k, v in cfg["extra_headers"].items()
        }
    resp = client.chat.completions.create(
        model=cfg["model"],
        max_tokens=6000,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": TASK},
        ],
        **extra,
    )
    text = resp.choices[0].message.content or ""
    print(f"=== {name} ({cfg['model']}) raw response: {len(text)} chars ===")
    print(text[:2500])
    print("...")
    print(text[-500:])


if __name__ == "__main__":
    main()
