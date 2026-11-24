#!/usr/bin/env bash
set -euo pipefail
# RTX 5070 defaults: cuda:0, one worker, microbatch 2. CLI overrides include
# --concurrency N --micro-batch N --device cuda:0 --output runs/<new-search>.
# No arguments creates the candidate plan; --mode run explicitly starts/resumes.
cd "$(dirname "$0")/.."
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
exec "${PYTHON:-python}" -u -m liquid_tcn.architecture_search "$@"
