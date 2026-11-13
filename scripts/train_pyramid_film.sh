#!/usr/bin/env bash
# Train with subset monitoring, then automatically select on full validation.
# Run from the repository root in the activated Python environment.
set -euo pipefail

PYTHON="${PYTHON:-python}"
DATA="${DATA:-data/prepared_expanded}"
OUTPUT="${OUTPUT:-runs/pyramid_film}"
DEVICE="${DEVICE:-cuda:1}"
export PYTHONUNBUFFERED=1

# Additional train arguments support --resume, --dry-run and logging overrides.
# The JSON owns all learning, architecture and shortlist settings.
exec "$PYTHON" -u -m liquid_tcn train \
    --config configs/pyramid_film.json \
    --data "$DATA" \
    --output "$OUTPUT" \
    --device "$DEVICE" \
    --logging-mode plain \
    "$@"
