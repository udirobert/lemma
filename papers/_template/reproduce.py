"""Reproduction entry point for <paper title>.

Runs the numerical audit / experiment for a single claim from the paper.

Usage:
    python papers/<paper-id>/reproduce.py --claim=1
    uv run papers/<paper-id>/reproduce.py --claim=1
    # For empirical claims, prefer an HF GPU Job under your own namespace:
    hf jobs run --flavor t4 --timeout 1h \\
        -v .:/work python:3.12 \\
        bash -lc "cd /work && uv sync && uv run python papers/<paper-id>/reproduce.py --claim=1"
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--claim",
        type=int,
        default=1,
        help="Which claim number from the paper to reproduce.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/",
        help="Output directory for figures/tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[reproduce] Claim {args.claim} — TODO: implement")


if __name__ == "__main__":
    main()
