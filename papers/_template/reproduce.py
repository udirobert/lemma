"""Reproduction entry point for <paper title>.

Usage:
    modal run papers/<paper-id>/reproduce.py --claim=1
    # or locally:
    python papers/<paper-id>/reproduce.py --claim=1
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--claim",
        type=int,
        default=1,
        help="Which claim number from the paper to reproduce",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[reproduce] Claim {args.claim} — TODO: implement")


if __name__ == "__main__":
    main()
