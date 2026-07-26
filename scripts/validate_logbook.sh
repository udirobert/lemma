#!/usr/bin/env bash
# Run the ICML 2026 validator against a draft publish slug.
#
# Usage:
#   ./scripts/validate_logbook.sh <owner>/<slug>
# Example:
#   ./scripts/validate_logbook.sh Papajams/repro-grokking-ridge-regression
#
# The validator script is mirrored from the org Space under
# scripts/validate_icml_logbook.py so we don't depend on the
# canonical script staying at the same URL (and we can edit it).

set -euo pipefail

slug="${1:-}"
if [[ -z "$slug" ]]; then
    echo "usage: $0 <owner>/repro-<slug>" >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

exec python3 scripts/validate_icml_logbook.py --space "$slug"
