# Optional Mamba memory

`configs/pyramid_film_mamba.json` adds one recurrent residual after the FiLM
pyramid's attention fusion and before its five existing heads:

```text
h = attention_features + Mamba(LayerNorm(attention_features))
```

The implementation uses the official [`mamba_ssm.Mamba`](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py),
with `d_model=channels`, `d_state=16`, `d_conv=4`, and `expand=2`.
Its initialization is untouched. AdamW excludes parameters marked
`_no_weight_decay` by Mamba, including after resume. Attention and Mamba have no
dropout in this configuration. Mamba is lazily imported only when selected.

`model.temporal_memory` defaults to `"none"`. Existing TCN and pyramid checkpoints
keep their original parameter layout; no partial state-dict loading is used.
The default FiLM pyramid has 291,664 parameters; Mamba increases this to 324,432
(+32,768). This includes the pre-normalization layer.

## Installation

The ordinary installation does **not** require Mamba. Use a separate environment
for this optional CUDA model if the current PyTorch build and local CUDA compiler
do not match. A tested combination is Python 3.11, PyTorch 2.7.0+cu128,
CUDA toolkit 12.8, Mamba 2.2.6.post3, Triton 3.3.0, and Transformers 4.57.6.

From the repository root, create either a Conda environment:

```bash
conda create -n liquid-tcn-mamba python=3.11 pip -y
conda activate liquid-tcn-mamba
```

or a virtual environment using a Python 3.11 interpreter:

```bash
python3.11 -m venv .venv-mamba
source .venv-mamba/bin/activate
```

Then install the tested CUDA stack and package:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install ninja packaging einops==0.8.2 transformers==4.57.6
python -m pip install -e '.[test]'

# Use the installed toolkit path; on a cluster, load its CUDA module first.
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
MAX_JOBS=2 python -m pip install mamba-ssm==2.2.6.post3 --no-build-isolation
python -m pip install -e '.[mamba]'
```

The Mamba installer may download a matching binary wheel or build CUDA kernels.
A GPU is needed for the official CUDA execution checks; some Triton versions also
initialize the driver during import. On Unity, perform that import check in a GPU
allocation, not on a GPU-less login node:

```bash
python - <<'PY'
import torch
from mamba_ssm import Mamba
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('Mamba:', Mamba)
PY
```

The existing local `liquid-tcn` environment uses PyTorch 2.15.0+cu132, while its
available compiler is CUDA 12.9. Mamba's source build failed with that mismatch.
Validation therefore used the existing CUDA-12.8 interpreter and an isolated
`/tmp` dependency install; it did not alter either environment. The `/tmp`
installation is not a permanent setup. No runtime precision or hardware admission
gate is added by this package.

## Independent training clips

No re-preparation or new recordings are needed when the prepared data has the
same TCN/attention history contract (249 samples by default). A compact
per-segment index selects independent contiguous clips directly from existing
mmap shards. Sampling uses the existing seeded, resumable `EpochSampler`.

The example uses:

- `training.batch_size=8` independent clips per update.
- `training.sequence_burn_in=200` contextual recurrent features (1 second).
- `training.sequence_scored_intervals=400`: 401 scored endpoints (2 seconds).
- 248 additional raw history samples before the first contextual feature.

A default pass therefore consumes **849 input samples** and produces 601 contextual
features. The first 200 are burn-in; only the last 401 have scored losses.
Burn-in stays in autograd. Every clip starts with zero recurrent state; no state
is carried between shuffled clips or stored in training checkpoints. Clips never
cross episode, gap, invalid-history, or series/payload boundaries. Target validity
masks still control supervision independently of causal-input validity.

Clip geometry is separate from `model.input_length=289`, which continues to
describe the prepared 41-endpoint physics/window contract. `receptive_field=249`
describes the finite backbone context, not the recurrent model's entire memory.
Sequence batch size is explicit; there is no automatic capacity adjustment.

Supervision counts each scored endpoint once. Short physics counts each adjacent
scored interval once. Long physics uses all rolling **41-endpoint / 40-interval**
slices from that same forward pass: 361 long constraints for a 401-endpoint
sequence. Each long slice uses its own recorded initial base origin and the
ported discrete formulation. Losses reduce by valid counts; neither short physics
nor supervision is repeated for each long slice. The horizon remains 200 ms.

## Validation, cached inference and assessment

Monitoring samples a reproducible, stratified 5% of complete validation segments
(rounded to a whole segment, minimum one), persisted in `monitor_subset.json`.
All eligible prepared windows in selected segments contribute to monitoring;
segment and window coverage are recorded separately. Recovery checkpoints remain
every 200 updates, monitoring every 1,000, and patience is three eligible checks.
The best three subset checkpoints plus stopping checkpoint form the shortlist.
Full-validation normalized supervised MSE selects the final model before testing.

Evaluation first reconstructs chronological predictions with persistent recurrent
state. Chunking overlaps only raw TCN history, never recurrent feature updates.
The first chunk uses the official scan; continuation chunks use the official
one-step state update. Chunk memory is bounded by `block_rows`; the current
implementation processes segments serially rather than batching whole episodes.
Mmap predictions are reused for window metrics and offline analyses. Monitoring
uses temporary caches for the current weights; assessment caches persist for
resume. Validation windows remain overlap-weighted for selection; offline metrics
retain their unique-interval weighting.

The one-step cache retains convolution/SSM states per environment. Its first
persistent recurrent update is at the first fully contextual TCN feature;
startup features never enter its memory. Selective resets, changed properties,
weight changes, and cache invalidation clear affected recurrent states as well as
TCN history. Deployment returns valid outputs after TCN warmup; the 200-step
training burn-in controls loss scoring, not an extra deployment-validity delay.
`step()` still returns 12 values, or all five three-vector heads with
`return_all=True`. Autograd training never uses this in-place inference cache.

## Commands

Train from scratch with progress output (40K maximum updates in the example):

```bash
python -u -m liquid_tcn train --config configs/pyramid_film_mamba.json \
  --data data/prepared_expanded --output runs/pyramid_film_mamba \
  --device cuda:1 --logging-mode progress
```

Full-validation shortlist selection runs automatically at stopping. Then run both
held-out inference modes and the existing offline analyses:

```bash
bash scripts/evaluate_local_full.sh --run runs/pyramid_film_mamba \
  --data data/prepared_expanded --device cuda:1 --threads 2
```

Resume an interrupted clip-training run:

```bash
python -u -m liquid_tcn train --data data/prepared_expanded \
  --output runs/pyramid_film_mamba --resume runs/pyramid_film_mamba/latest.pt \
  --device cuda:1 --logging-mode progress
```

The parallel experiment launcher accepts `--config configs/pyramid_film_mamba.json`
and preserves the sequence settings across its seeds and physics ablations. Set
concurrency for this model's measured memory needs; the old batch-1024 window
capacity measurements do not apply. Slurm remains the wall-time enforcer.

## Focused validation and measurement

`tests/test_mamba_memory.py` checks compact clip indexing, reset boundaries,
fixed 200 ms long slices, causality, physics gradients into official Mamba,
selective cache reset, payload changes, invalidation, optimizer no-decay groups,
and a two-update train/resume/full-validation/test assessment. CUDA checks skip
when CUDA or Mamba is unavailable; they do not substitute a mock SSM.

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m pytest -q tests/test_mamba_memory.py
python tools/benchmark_mamba_memory.py --device cuda:1 --batch-size 4096 \
  --output artifacts/mamba_memory_benchmark.json
```

The recorded CUDA:1 check used an RTX 4090, FP32, TF32 disabled, no compilation,
200 recurrent burn-in samples after TCN warmup, eight additional warmup steps,
and five repetitions of 20 timed steps. CUDA synchronization occurs outside each
20-step loop. Each step returns 12 values for 4096 environments.

| Cached model | Mean / median / p95, ms | Peak allocated MiB | Persistent cache MiB |
|---|---:|---:|---:|
| FiLM pyramid | 25.54 / 24.92 / 30.85 | 1627.6 | 1016.8 |
| FiLM pyramid + Mamba | 26.42 / 25.80 / 32.41 | 1667.7 | 1056.8 |

Mamba adds **40 MiB of persistent state at batch 4096**. The measured mean latency
ratio was 1.034, but another process occupied approximately 17 GiB on CUDA:1.
These timings are **contended measurements**, not isolated throughput or a reliable
estimate of intrinsic overhead. Existing jobs were left running. The
[raw benchmark](mamba_memory_benchmark_cuda1.json) includes versions, initialization
cost, peak reserved memory, and throughput. No full training campaign or full-data
assessment was launched for this change.

## Reset-conditioned stateful training

Use `configs/pyramid_film_mamba_stateful.json` to train on ordered episode
streams. Each full chunk contains 400 new samples. At a recorded reset the
backbone gets 248 copies of the full recorded 15-channel motion input, including
gravity orientation, velocities, and held accelerations. Mamba's convolution
state starts at zero. A zero-initialized output projection in the learned
initializer conditions the SSM state on normalized properties and reset motion.
Mamba processes real samples from the reset onward. State and the last 40
predictions are detached between chunks and serialized with the stream position.
Physics windows use this history to score each valid interval or window at its
ending sample.

The stateful implementation uses PyTorch Mamba-1 equations with a differentiable
scan. It runs in the standard environment without the optional Mamba package.
It reads the existing prepared shards plus the lightweight reset records from
the original collection paths referenced by the manifest. It builds the stream
index on launch. Splits and train-only normalization come from the manifest.
A missing or nonfinite input ends an episode's usable prefix.

```bash
python -u -m liquid_tcn train \
  --config configs/pyramid_film_mamba_stateful.json \
  --data data/prepared_mamba --device cuda:0 --logging-mode progress
```

For deployment, set payload properties and call
`cache.initialize_reset(reset_motion, environments)` with the recorded reset
inputs in the selected environment order. The next `cache.step(...)` predicts at the
first real sample. Alternatively, pass `reset_motion=` and the selected
`reset=` environments to the same step call.
