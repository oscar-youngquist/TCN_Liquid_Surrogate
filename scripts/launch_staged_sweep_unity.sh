#!/usr/bin/env bash
# Run inside an existing allocation and activated environment. GPU indices are
# logical indices within CUDA_VISIBLE_DEVICES, which this script never changes.
set -euo pipefail
PYTHON="${PYTHON:-python}"
if [[ "${1:-}" == "--python" ]]; then
    if [[ $# -lt 2 ]]; then
        echo '--python requires an executable' >&2
        exit 2
    fi
    PYTHON="$2"
    shift 2
fi
export PYTHONUNBUFFERED=1
# exec preserves signals and the coordinator's exit code without a shell parent.
exec "$PYTHON" -u -m liquid_tcn.staged_sweep "$@"
