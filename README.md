# Liquid-wrench TCN

Standalone PyTorch pipeline for completed **5 ms liquid-on-robot wrench intervals**. It reads collection attempts directly, prepares mmap shards once, trains all five heads, evaluates supervised and discrete conservation errors, and exports a cached 12-value inference interface. No Genesis, robot policy, QP package, or collection-repository `PYTHONPATH` is required.

The [architecture guide](docs/architecture.md) defines the network, input channels, output alignment, and inference interface. The [physics reference](docs/liquid_discrete_constraints.md) derives the training losses, and the [recording format](docs/liquid_surrogate_collection.md) describes the input data. These documents contain the specifications used by this package.

The data and physics utilities originate from the HCR Genesis collection project; [source provenance and licensing](docs/provenance.md) identify the imported revision and local adaptations. [Validation results](docs/validation.md) record regression, equivalence, and smoke-test coverage. Recordings and trained weights are not bundled.

See the **[field-by-field configuration reference](docs/configuration.md)** for every JSON value, its default, units, scheduling semantics, and resume behavior.

## Codebase layout

| Path | Responsibility |
|---|---|
| `src/liquid_tcn/ported/` | Collection-format readers, discrete losses, storage/audit, transforms, and diagnostic helpers. |
| `data.py`, `metadata.py`, `discovery.py`, `preprocess.py` | Causal input retention, recorded property/schema parsing, group splits, bounded preprocessing and provenance. |
| `dataset.py`, `normalization.py` | Lazy mmap windows, seeded resumable sampling, worker-local caches, training-only statistics. |
| `model.py`, `objectives.py` | Causal residual TCN, streaming caches, five-head supervision and discrete physics objectives. |
| `train.py`, `checkpoint.py`, `evaluate.py` | Optimizer phases, progress/TensorBoard, atomic resume, fixed validation and physical metrics. |
| `cli.py`, `inference.py`, `launch_train.sh` | Commands, frozen export loading, and GPU/config selection. |
| `configs/` | Complete ablation training JSONs and a source-preparation example. |
| `tests/`, `tools/`, `docs/` | Imported and integration regressions, comparison/profiling tools, provenance and executed results. |

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
```

Python 3.10+; NumPy, PyTorch, TensorBoard, and tqdm. Install the appropriate PyTorch build for your CPU/CUDA environment. An existing PyTorch environment can also install this package directly. See [validation results](docs/validation.md) for packaging checks.

## Prepare data once

Point preparation at the **top-level output folder of the collection script**. It recursively discovers attempts across robots, rounds, and retries:

```bash
liquid-tcn prepare --root /path/to/liquid_collection/campaign --output data/prepared
```

For configurable split fractions, model/window geometry, or shard sizing, copy `configs/prepare.example.json` to `configs/prepare.local.json`, set `collection_root`, and run:

```bash
liquid-tcn prepare --config configs/prepare.local.json --output data/prepared
# --root can also override collection_root in that config.
```

This works while collection is running. Attempts without a complete `result.json`, including unfinished auxiliary streams, are skipped and listed in `discovery.json`. Completed attempts must pass metadata and checksum verification; corruption is reported and stops preparation. Traversal stops at each attempt so saved source trees and `events/`/`boundaries/` are not mistaken for separate attempts. All verified complete retries are retained.

Automatic collection groups hash the recorded round seed, round number, and payload identity. Matching robots and retries share a group; directory names and the chosen scan root do not affect identity. Seeded whole-group hashing uses `fractions` (default 80%/10%/10%) before windowing. Existing groups retain their assignments as new recordings arrive. `held_out_payloads` lists payload SHA-256 IDs to force into test. Small collections can have empty splits; training requires nonempty training and validation data.

Each preparation is a **fixed snapshot**, not a live dataset: candidate membership and completion decisions are scanned before checksum verification. Attempts completed after those decisions are picked up by a later preparation into a **new output directory**. `discovery.json` records included/skipped/error paths and scan times; `manifest.json` also records the report, exact source hashes and split assignments. If no complete attempts exist yet, preparation reports that and leaves no completed dataset manifest.

For manual split assignments or legacy metadata, `configs/prepare.sources.example.json` retains the explicit `sources` workflow. Each entry supplies an attempt `path`, a collection-family `group`, and optionally `split` (`train`, `validation`, or `test`). Do not combine `sources` and `collection_root`. Related retries/robots must share a group; conflicting assignments and duplicate paths are rejected.

Preparation verifies completion, reconciles chunk indices with result manifests, and checks primary and auxiliary checksums. It reads three streams one chunk at a time, deduplicates by native tick and record sequence, and invokes the episode/interval functions on one 5 ms bin at a time. Old terminal states precede same-time resets; asynchronous resets cannot mix episode histories. Split pieces use a common world moment reference. Native debug recordings are also supported using their stored world impulses, but native-rate data is **not required**.

Recorded `z` and held inputs are retained. Missing `z` is reconstructed only from current gravity, base velocity/orientation, and recorded held estimates; missing held values can be recovered from `z`. Neither upcoming acceleration nor instantaneous acceleration reconstruction is used. Unsupported/unselected states, missing grid transitions, incomplete histories, and mass/gravity inconsistencies cannot produce training windows. Episode-wide mass/gravity violations invalidate already indexed windows too.

Rows are stored once in fixed-size `.npy` shards; 16-byte indices identify window endpoints, including windows crossing shard boundaries. The loader never decodes NPZ files. Memory is bounded by three decoded source chunks, one grid bin, and one `shard_rows` buffer per active environment. Normalization uses valid **training** rows from eligible series, without duplicating overlapping windows. Constant channels use unit scale. The completion manifest records source and artifact hashes, source/prepared schemas, splits, normalization counts, rejection counts, and profiling information. A failed preparation has no completed manifest and requires a fresh output path.

## Select a config and GPU

All training settings are in JSON. The four complete configurations retain direct supervision and differ only in physics weights:

| Config | Short weight | Long weight |
|---|---:|---:|
| `configs/supervised.json` | 0 | 0 |
| `configs/short.json` | 1 | 0 |
| `configs/long.json` | 0 | 1 |
| `configs/combined.json` | 1 | 1 |

Edit `data`, `output`, `training`, and optional `resume` / `max_updates` in the selected JSON. `{timestamp}` in `output` creates a unique run directory. Keep the model section consistent with the preparation config; training rejects mismatched prepared window geometry. Changing model receptive field/horizon requires new preparation. Machine-specific `configs/*.local.json` files are ignored by Git.

```bash
# Inspect without creating data/log directories or starting training.
GPU=cpu CONFIG=configs/combined.json bash launch_train.sh --dry-run

# Train on CUDA device 1; all other settings come from JSON.
GPU=1 CONFIG=configs/combined.json bash launch_train.sh

# CPU is supported, including num_workers=0.
GPU=cpu CONFIG=configs/supervised.json bash launch_train.sh
```

The launcher only selects GPU/config and invokes the CLI. `PYTHON=/path/to/python` optionally selects the interpreter. It does not prepare data or launch collection. The CLI is also available as `python -m liquid_tcn`:

```bash
liquid-tcn train --config configs/combined.json --device cpu
# Optional CLI path/stop overrides for a short run:
liquid-tcn train --config configs/combined.json --data data/prepared \
  --output runs/smoke --max-updates 10 --device cpu
```

A progress bar displays updates, phase, losses, learning rate, and validation metric. Set `training.progress` to false for quiet batch jobs. `max_updates` stops at an absolute update without changing the configured scheduler horizon. It must be large enough to reach a scheduled validation if a best checkpoint is desired.

Set `training.epochs` to a positive count for complete dataset passes, or leave it null and use `training.updates`. Phase/ramp/validation/checkpoint intervals remain measured in optimizer updates; resolved logs record the effective update budget.

Default training uses FP32 AdamW, `1e-3` supervised pretraining then `1e-4` fine-tuning, weight decay `1e-4`, cosine decay, gradient clipping at 1, and a linear ramp of both physics weights. Supervision remains active throughout. Physics runs outside autocast in float64 by default. Optional BF16 `amp` is gated by startup loss/gradient comparisons against FP32 and float64. Nonfinite predictions/gradients and empty objective sets fail explicitly. Valid-count-weighted normalized MSE or Huber supervision covers all five heads. Validation always selects checkpoints using a fixed normalized MSE over all five heads, independent of physics ramps or the chosen supervision loss. `patience` counts validation events.

Training uses conflict-only one-way PCGrad by default. Supervised losses from all
five heads form the primary objective. The secondary objective is
`lambda_short * L_short + lambda_long * L_long`, with the current ramped weights.
If the two gradients conflict, only the secondary gradient is projected; aligned
physics gradients are retained. The merged gradient is clipped before AdamW.
Set `training.gradient_strategy` to `"sum"` for ordinary summed-loss gradients.
See [gradient combination](docs/liquid_discrete_constraints.md#gradient-combination)
for the formula and numerical conventions.

Worker-local mmap caches have a configurable map count; prefetching is only enabled for workers. Spawned workers and sampling are seeded. Sampling uses an O(1)-state affine permutation, visiting every window once per epoch rather than allocating a dataset-sized permutation. Resume saves the **consumed** sampler position, independent of prefetch progress. Pinned transfers are enabled when CUDA is available. For very small cached datasets, zero workers can be faster than multiprocessing.

## Run logs, TensorBoard, and resume

Every `output` directory contains:

```text
launch_config.json       selected JSON (when invoked through the CLI)
requested_config.json    requested training settings, including device override
config.json              resolved training/model config, normalization, data identity
latest.pt                atomic resumable checkpoint
best.pt                  atomic best fixed-validation-metric checkpoint
tensorboard/             TensorBoard event files
metrics.jsonl            per-update losses, valid counts, weights and gradient norm
validation.json          latest physical-unit and conservation metrics
precision.json           startup FP32/float64 and optional AMP checks
profile.json             loader/training time and peak process/device memory
```

TensorBoard logs training losses/weights/counts, PCGrad projection decisions and shared-gradient diagnostics, clipping, learning rate, and per-head validation errors and normalized conservation residuals. Checkpoint writes flush the event writer; resume purges events beyond the restored update and continues the same step numbering.

```bash
tensorboard --logdir runs
liquid-tcn train --config configs/combined.json --data data/prepared \
  --output runs/my-run --resume runs/my-run/latest.pt --device cpu
```

For the launcher, set JSON `output` to the existing run directory and `resume` to its `latest.pt`. Keep training settings/device and prepared data unchanged. Checkpoints include optimizer/scheduler, phase/update, RNG state, sampler state, normalization, config, and split/source provenance. Early stopping uses the same validation metric on every evaluation. Training refuses to overwrite an existing run without resume. Checkpoints are trusted local PyTorch artifacts.

## Evaluate and export

```bash
liquid-tcn evaluate --data data/prepared --checkpoint runs/my-run/best.pt \
  --split test --output runs/my-run/test.json
liquid-tcn export --checkpoint runs/my-run/best.pt --output exports/liquid.pt
```

Evaluation verifies prepared artifact hashes and checkpoint data identity. It reports per-axis RMSE/MAE/bias in physical units, maximum vector errors, errors at target force/moment peaks, within-window lag, short/long conservation RMS/max residuals, ground-truth conservation floors, valid counts/coverage, and per-robot/payload results. Lag is demeaned vector cross-correlation within ±50 ms; constant signals have no lag estimate. Metrics are explicitly window-weighted, so overlapping endpoints are counted per window. Empty splits are `inconclusive` with a null selection metric. `--max-batches` permits a bounded smoke evaluation and reports its coverage.

```python
from liquid_tcn.inference import load_export

surrogate = load_export('exports/liquid.pt', environments=2, device='cpu')
# motion: [2, 15], properties: [2, 24], physical units, float32 tensors
values, valid = surrogate.step(motion, properties)  # values [2, 12]
surrogate.reset([0])  # reset only this environment on resets/gaps/payload changes
```

The output order is `[force, moment, r_B, v_B]`; `S_B` remains training-only in the public interface. Cached inference stores each convolution's causal context. Warmup returns `valid=False` and NaN values until the full receptive field exists; it cannot expose padded/reset-crossing predictions as usable values. Properties follow `PROPERTY_FIELDS` in `metadata.py`: 24 recorded quantities embedded into 32 channels, including actual mass/volume, geometry, and mounting. Genesis `mu` stays **kinematic viscosity in m²/s**, and `gamma` stays the recorded raw coefficient.

Default architecture: 15 motion channels, 32-dimensional property embedding, 64-channel projection, five two-convolution residual blocks with kernel 5/dilations `[1,2,4,8,16]`, SiLU, no dropout or batch/time normalization, and five independent `64→64→3` heads. The receptive field is 249; 289 inputs yield 41 fully contextual endpoint predictions. Physics uses endpoint states at `0:41` and completed-interval wrench predictions at `1:41`, summing all 40 impulses directly.

**Wrench convention:** completed previous 5 ms interval, expressed in its endpoint base frame, moment about the fixed world location of that endpoint base origin. It includes liquid weight. Export does not convert it to an instantaneous or upcoming-interval forecast. The package does not include a closed-loop controller.

## Tests and profiling tools

```bash
python -m pytest -q
python tools/compare_port.py --source /read-only/baseline-snapshot \
  --attempt /recorded/attempt --output artifacts/port-equivalence.json
python tools/profile_loader.py --data data/prepared --workers 0 --output artifacts/loader.json
python tools/check_precision.py --data data/prepared --amp --output artifacts/precision.json
```

The reference snapshot is needed only by the optional comparison tool; [provenance](docs/provenance.md) describes how to obtain it. The regular test suite uses synthetic fixtures generated in this repository. [Validation results](docs/validation.md) distinguish reproducible tests from historical checks on recordings that are not bundled. Source recordings, prepared data, weights, TensorBoard logs, profiler traces, and local configs are ignored by Git; reproducible source/tests/configs and compact validation summaries remain tracked.
