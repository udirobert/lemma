"""Reproduction entry point for 'To Grok Grokking: Provable Grokking in Ridge Regression'.

Runs a single-claim numerical audit. Closed-form for C1-C3 (numpy), small
simulation for C4, two-layer ReLU for C5 (torch, CPU is fine).

Usage:
    python papers/5nNNVY8NW4-grokking/reproduce.py --claim=1
    uv run papers/5nNNVY8NW4-grokking/reproduce.py --claim=5

Theory papers don't require an HF GPU Job per the ICML 2026 judge docs,
so we keep this CPU-only. If you want to escalate one claim to an HF
Job as a polish step:

    hf jobs run --flavor t4 --timeout 30m \\
        -v .:/work python:3.12 \\
        bash -lc "cd /work && uv sync && uv run python papers/5nNNVY8NW4-grokking/reproduce.py --claim=5"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--claim",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Which claim number from the paper to reproduce (1-5).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/",
        help="Output directory for figures/tables.",
    )
    return parser.parse_args()


def run_claim(claim: int, out_dir: Path) -> dict:
    """Placeholder; real implementations land here as Trackio pages are written."""
    raise NotImplementedError(f"claim {claim}: implement the audit first")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = run_claim(args.claim, out_dir)
    (out_dir / f"claim{args.claim}.json").write_text(json.dumps(summary, indent=2))
    print(f"[reproduce] claim {args.claim} -> {summary}")


if __name__ == "__main__":
    main()
