#!/usr/bin/env bash
# Finish full-validation shortlist selection, then assess the selected model.
# Optional leading --python selects the activated environment's interpreter.
set -euo pipefail
PYTHON="${PYTHON:-python}"
if [[ "${1:-}" == "--python" ]]; then
    if [[ $# -lt 2 ]]; then echo '--python requires an executable' >&2; exit 2; fi
    PYTHON="$2"
    shift 2
fi
export PYTHONUNBUFFERED=1
exec "$PYTHON" -u -m liquid_tcn.assessment "$@"
