#!/usr/bin/env bash
# All model/data/training settings live in the selected JSON.
# Examples:
#   GPU=0 CONFIG=configs/combined.json bash launch_train.sh
#   GPU=1 CONFIG=configs/short.json bash launch_train.sh
#   GPU=cpu CONFIG=configs/supervised.json bash launch_train.sh --dry-run
set -euo pipefail
GPU="${GPU:-0}"
CONFIG="${CONFIG:-configs/combined.json}"
PYTHON="${PYTHON:-python}"

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ "${1:-}" == "--help" ]]; then
    echo 'Usage: GPU=0 CONFIG=configs/combined.json bash launch_train.sh [--dry-run]'
    echo 'Use GPU=cpu for CPU. All other settings, paths, and resume controls are in JSON.'
    exit 0
fi
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--dry-run" ) ]]; then
    echo 'Expected --dry-run or --help' >&2; exit 2
fi
if [[ "$GPU" == "cpu" ]]; then
    device='cpu'
elif [[ "$GPU" =~ ^[0-9]+$ ]]; then
    device="cuda:$GPU"
else
    echo 'GPU must be a device index (0, 1, ...) or cpu' >&2; exit 2
fi
exec "$PYTHON" -m liquid_tcn train --config "$CONFIG" --device "$device" "$@"
