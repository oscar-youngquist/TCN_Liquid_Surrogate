#!/usr/bin/env bash
# Use an activated environment inside an existing allocation. Preserve GPU visibility.
set -euo pipefail
PYTHON="${PYTHON:-python}"
if [[ "${1:-}" == "--python" ]]; then
    if [[ $# -lt 2 ]]; then echo '--python requires an executable' >&2; exit 2; fi
    PYTHON="$2"
    shift 2
fi
export PYTHONUNBUFFERED=1
exec "$PYTHON" -u -m liquid_tcn.experiment "$@"
