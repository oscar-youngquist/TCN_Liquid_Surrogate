# Configuration reference

All paths in CLI/JSON commands resolve relative to the current working directory. The top-level launcher changes to the repository root first. JSON uses `true`, `false`, and `null` rather than Python spellings. Keep reusable defaults in the four checked-in training configurations; keep local dataset paths and experiments in `configs/*.local.json`.

## Training file structure

`supervised.json`, `short.json`, `long.json`, and `combined.json` contain the same full schema. Their physics weights are respectively `(0,0)`, `(1,0)`, `(0,1)`, and `(1,1)`. All retain five-head supervision.

| Key | Default/example | Meaning |
|---|---|---|
| `data` | `data/prepared` | Completed prepared dataset directory containing `manifest.json`; source NPZ files are not read during training. |
| `output` | `runs/combined/{timestamp}` | Run log directory. `{timestamp}` expands to `YYYY_MM_DD_HH_MM_SS_ffffffZ` (UTC, with microseconds). Contains selected/resolved configs, best/latest checkpoints, TensorBoard, metrics and profiles. |
| `resume` | `null` | Trusted checkpoint path to restore, usually an existing run's `latest.pt`. Set `output` to that same concrete run directory to continue its logs. |
| `max_updates` | `null` | Optional absolute stop update for a short/interrupted run, without changing the scheduler horizon or checkpoint configuration. Resume with this removed or increased. It does not mean “additional updates.” |
| `model` | object below | Architecture and window geometry; must match the prepared dataset's model section. |
| `training` | object below | Optimizer, losses, scheduling, loader, precision and logging settings. |

CLI `--data`, `--output`, `--resume`, `--max-updates`, and `--device` override corresponding values. The launcher selects `GPU` and `CONFIG` only; `GPU=cpu` selects CPU, and `GPU=1` selects `cuda:1` under the current `CUDA_VISIBLE_DEVICES` mapping. `PYTHON` can select an existing interpreter. `--dry-run` prints resolved settings without preparing data or training.

```bash
GPU=0 CONFIG=configs/combined.json bash launch_train.sh
GPU=cpu CONFIG=configs/combined.json bash launch_train.sh --dry-run
```

## Model and window geometry

The same `model` object appears in the preparation configuration. The [architecture guide](architecture.md) defines the default network. Changes affecting receptive field or horizon require preparing a matching dataset. A model mismatch fails instead of silently using a different architecture.

| Key | Default | Meaning |
|---|---:|---|
| `property_dim` | 24 | Fixed metadata feature schema below. Production preparation requires 24; a new dimension requires implementing a new property schema. |
| `embedding_dim` | 32 | Output width of the static property MLP. |
| `channels` | 64 | Input projection and residual convolution width; also each head's hidden width. |
| `kernel_size` | 5 | Causal convolution kernel width. |
| `dilations` | `[1,2,4,8,16]` | One dilation per residual block, with two convolutions per block. All entries must be positive integers. |
| `long_intervals` | 40 | Number of adjacent 5 ms intervals in the long objective. Default: 200 ms and 41 predicted endpoints. A different positive count is an explicit horizon experiment, not equivalent to the default acceptance contract. |

`receptive_field = 1 + 2*(kernel_size-1)*sum(dilations)`; `input_length = receptive_field + long_intervals`. Defaults are 249 and 289. Motion input always has 15 channels in the order `[g_B, u_O_B, omega_B, held_a_B, held_alpha_B]`. Five independent heads predict force, moment, `r_B`, `v_B`, and `S_B`. There is no configurable dropout or temporal/batch normalization because neither is present in this initial architecture.

The 24 physical property values, in order, are density, Genesis kinematic viscosity, raw Genesis surface-tension coefficient, actual mass, actual rest-density volume, fill-box volume, three interior dimensions, three fill dimensions, three mount translations, mounting quaternion in wxyz order, three geometry scale factors, fill offset, and particle size. `metadata.PROPERTY_FIELDS` names their exact units. Normalization is fitted only on training data. In particular, viscosity is **m²/s**, not Pa·s; no unsupported conversion is performed.

## Training duration and optimization

| `training` key | Default | Meaning |
|---|---:|---|
| `epochs` | `null` | Optional positive number of complete passes over training windows. If set, it takes precedence over `updates`: effective updates = `epochs * ceil(training_windows / batch_size)`. |
| `updates` | 20000 | Optimizer-update budget when `epochs` is null. It also defines the fine-tuning scheduler horizon. |
| `batch_size` | 16 | Number of contiguous windows per optimizer update; final batches may be smaller. |
| `pretrain_updates` | 5000 | Number of supervised updates before switching learning rates and starting the physics ramp. Counts updates even when epoch-based duration is used. |
| `ramp_updates` | 5000 | Ramp duration in optimizer updates. For the zero-based update index `s`, ramp = `clip((s-pretrain_updates)/max(1,ramp_updates),0,1)`. |
| `pretrain_lr` | 0.001 | AdamW learning rate during supervised pretraining. |
| `physics_lr` | 0.0001 | Starting AdamW learning rate after pretraining. The phase transition occurs even in the supervised-only ablation, keeping schedules matched. |
| `weight_decay` | 0.0001 | AdamW decoupled weight decay. |
| `scheduler` | `"cosine"` | `"cosine"` decays from `physics_lr` toward its floor over the remaining budget; `"constant"` retains `physics_lr`. Pretraining is constant-rate. |
| `min_lr_ratio` | 0.1 | Final cosine learning rate as a fraction of `physics_lr`. |
| `gradient_clip` | 1.0 | Maximum global gradient norm. Pre-clipping norm and clipping indicator/fraction are logged. Nonfinite gradients raise. |
| `seed` | 1 | Seeds Python, NumPy, Torch, loader workers and epoch sampling. Deterministic Torch algorithms are enabled. |

One epoch visits every eligible training window exactly once using a seeded, O(1)-state affine permutation. This avoids a dataset-sized shuffle allocation; it is not a uniformly random permutation of all possible orderings. Sampler positions in checkpoints count consumed examples, not prefetched examples. The resolved `config.json` and checkpoints record `effective_updates`, including the result of epoch-based sizing.

For a 20-epoch run, set `epochs: 20` and choose `pretrain_updates`/`ramp_updates` appropriate to the number of windows. Setting too short a total budget can leave the run entirely in pretraining; the configured phase and ramp lengths are preserved. For a smoke run of an otherwise unchanged experiment, set top-level `max_updates`, not the scheduler budget.

## Objectives and precision

| `training` key | Default | Meaning |
|---|---:|---|
| `lambda_short` | 1.0 | Final multiplier on the adjacent-interval discrete loss, after the ramp. |
| `lambda_long` | 1.0 | Final multiplier on the whole-window discrete loss, after the ramp. |
| `supervised` | `"mse"` | `"mse"` or `"huber"`, computed on normalized targets with equal weight per valid scalar across the five heads. |
| `huber_delta` | 1.0 | Huber transition in normalized target units. Only used for Huber supervision. |
| `gradient_strategy` | `"pcgrad"` | Conflict-only one-way projection of the summed physics gradient against supervision. `"sum"` uses ordinary gradient addition. |
| `pcgrad_eps` | 1e-12 | Positive finite additive epsilon in the projection denominator `||g_supervised||² + eps`. |
| `physics_dtype` | `"float64"` | `"float32"` or `"float64"` for de-normalization and discrete calculations outside autocast. |
| `amp` | false | Enable BF16 network autocast after a startup precision gate. Parameters and optimizer state remain FP32. No FP16 mode is provided. |
| `precision_rtol` | 0.005 | Maximum relative loss and global parameter-gradient-norm difference for FP32 versus float64 startup checks. |
| `amp_rtol` | 0.05 | Corresponding tolerance for optional BF16 versus float64 checks. A failure stops the run. |

The reported total loss is supervised loss plus ramped short and long losses. With `gradient_strategy="pcgrad"`, the optimizer uses the [projected gradient combination](liquid_discrete_constraints.md#gradient-combination), rather than the derivative of that reported total. The weighted physics terms are summed before projection; they are not projected against one another. Every supervised head remains active in all four ablations. Physics uses physical-unit predictions at both endpoints, recorded base motion/mass/gravity, and the [physical scales and fixed-reference conventions](liquid_discrete_constraints.md). A valid default window supplies 41 endpoint states and 40 completed-interval wrenches at predicted indices `1:41`; impulse sums are direct. There is no JVP or instantaneous/next-interval reinterpretation. An empty valid set raises during training; it is never a successful zero-loss batch.

## Loading and execution

| `training` key | Default | Meaning |
|---|---:|---|
| `device` | `"cpu"` | Torch device for direct CLI use. The launcher's GPU choice overrides it and is captured in the resolved run config. |
| `threads` | 4 | Torch CPU thread count. |
| `num_workers` | 0 | Number of spawned DataLoader workers. Zero is fully supported. |
| `prefetch_factor` | 2 | Prefetched batches per worker; omitted from DataLoader when workers are zero. |
| `pin_memory` | true | Pin collated batches when CUDA is available; transfers request nonblocking operation. |
| `cache_size` | 8 | Maximum open shard mappings per dataset instance/process, excluding its compact index. |
| `progress` | true | Show a CLI progress bar with phase, losses, learning rate and latest validation metric. |

Each worker lazily opens its own mappings. Data are copied only for the requested batch windows; source NPZ decoding happens only during preparation. Large worker/prefetch/batch settings can still increase host memory, so profile the intended setup. On the small acceptance dataset, worker startup made two workers slower than zero workers.

## Validation, logs and checkpoints

| `training` key | Default | Meaning |
|---|---:|---|
| `validation_every` | 250 | Validate every this many optimizer updates, and at the configured final update. |
| `validation_batch_size` | `null` | Validation windows per batch; null inherits training `batch_size`. Increase if inference memory allows. |
| `validation_num_workers` | `null` | Validation DataLoader workers; null inherits training `num_workers`. Uses the configured prefetching and pinned memory settings. |
| `validation_batches` | `null` | Null evaluates the complete validation split. A positive value uses the same fixed initial batches every time; useful for smoke tests, with coverage reported. |
| `checkpoint_every` | 250 | Atomic latest-checkpoint interval. Latest is also saved when the requested run finishes/stops. Best is saved on validation improvement. |
| `patience` | 20 | Stop after this many validation events without sufficient improvement. |
| `min_delta` | 0.0 | Minimum decrease of the fixed validation selection metric required to count as improvement. |

The selection metric is **normalized supervised MSE over all five heads**, even when Huber training is selected. It never changes with physics weights or training phase. Checkpoints preserve optimizer, scheduler, update/phase, Python/NumPy/Torch RNG, sampler, normalizers, configuration, and data/split provenance. Resume requires identical resolved training settings and dataset manifest. Top-level stop/output/resume paths are not optimizer settings.

The run saves `launch_config.json` (selected source JSON), `requested_config.json` (training options/device override), and `config.json` (resolved settings, effective duration, architecture, normalization and data identity). `tensorboard/` logs scalar losses, weights, valid counts, learning rate, gradient norm/clipping, per-head validation RMSE/max errors and conservation residuals. TensorBoard is flushed on checkpoint saves, and resume purges future steps. `metrics.jsonl`, `validation.json`, `precision.json`, and `profile.json` provide non-TensorBoard records.

## Preparation configuration

`configs/prepare.example.json` uses automatic collection-root discovery. The launcher never starts preparation implicitly. Point it at the currently running collector's top-level output folder:

```bash
liquid-tcn prepare --root /path/to/collection/output --output data/prepared
# Or set collection_root in the preparation JSON:
liquid-tcn prepare --config configs/prepare.local.json --output data/prepared
```

`--root` overrides JSON `collection_root`; `--config` is optional when `--root` is supplied. Other settings use their defaults when omitted. `configs/prepare.sources.example.json` demonstrates manual attempt lists instead.

| Key | Default/example | Meaning |
|---|---|---|
| `num_workers` | 0 | Independent attempt-processing processes; zero runs in the main process. Also bounds concurrent verification/output-hash threads. Example files use 4. CLI override: `--workers`. |
| `progress` | true | Show stage progress and active grid-bin counts. CLI `--no-progress` disables all preparation bars. |
| `collection_root` | `/data/liquid_collection/campaign` | Recursively discover completed attempts under this directory. Supports a campaign folder, its parent containing multiple campaigns, or one attempt. Mutually exclusive with a nonempty `sources` list. |
| `sources` | list (manual mode only) | Explicit attempt directories and group/split identities, defined below. Used instead of `collection_root`. |
| `seed` | 1 | Seed used by deterministic whole-group hash splitting. |
| `shard_rows` | 8192 | Maximum rows per series shard and per-active-environment output buffer. Shards never store duplicated overlapping windows. |
| `model` | model object above | Defines causal-history and long-window lengths at indexing time; train config must match. |
| `fractions` | `[0.8,0.1,0.1]` | Train/validation/test hash thresholds for groups without explicit splits. Must be nonnegative and sum to one. Small datasets may have empty splits; no forced reassignment or leakage is performed. |
| `held_out_payloads` | `[]` | Exact recorded payload SHA-256 IDs that must be in test; takes precedence over implicit hash assignments and rejects conflicting explicit splits. |

Discovery recognizes `attempt-*` directories (including not-yet-initialized attempts) or directories with `job.json` and liquid/resolved metadata. It does not descend inside attempts or follow child directory symlinks. Completed attempts require valid supported metadata and matching chunk checksums. Incomplete/non-complete attempts are skipped; corrupt completed attempts fail with details in `discovery.json`. No collector files or locks are modified.

All verified complete attempts are included, including successful retries. Automatic group IDs hash `[job.selected.seed, job.selected.round, payload_id]`, excluding robot/policy/attempt names. The collector shares these values across matching robot runs. Identical identities across repeated campaigns are conservatively kept together. This makes assignments independent of directory ordering and stable as new attempts arrive. Discovery requires the recorded seed/round fields; use manual sources for legacy metadata without them.

A scan freezes candidate membership and observed completion decisions before hash verification; it does not watch for new results. `discovery.json` and the final `manifest.json` report paths, reasons, counts and scan times. Later completions require a new preparation destination. A dataset with no eligible training windows still fails explicitly; discovery does not force small groups across splits to make them nonempty.

Each manual `sources` entry has:

| Key | Required | Meaning |
|---|---|---|
| `path` | yes | Source attempt directory, containing recorded metadata, completion manifest, and primary/auxiliary chunks. |
| `group` | yes | Stable collection family ID shared by related retries and trajectories/robots from that round. Split assignment happens before windowing. |
| `split` | no | Explicit `"train"`, `"validation"`, or `"test"`; all entries in a group must agree. |

Metadata comes from strict JSON parsing of `job.json`, `resolved.json`, and `liquid.json`, with source schema versions 2 and 3 accepted. Saved Python policy configs are not executed. Actual mass/volume come from solver-recorded metadata and mass is cross-checked with endpoint records. Other source schema versions fail explicitly and require a schema adapter.

Preparation fails if there are no eligible training windows. Training additionally requires nonempty validation. Failed output directories lack `manifest.json`; use a fresh destination after correcting inputs. Do not put raw recordings, prepared shards, learned checkpoints, or machine-local configs under version control.

## Gradient projection logs and resume

`metrics.jsonl` stores the weighted `secondary` loss and a `projection` object.
TensorBoard exposes these as `train/secondary` and `train/pcgrad/*`:

| Field | Meaning |
|---|---|
| `projected` | 1 when the secondary gradient was projected because of conflict; otherwise 0 |
| `dot_before`, `dot_after` | Inner product of the primary and secondary gradients before/after projection |
| `cosine_before` | Cosine similarity before projection, with a stabilized denominator |
| `primary_norm`, `secondary_norm` | Gradient norms before projection on parameters used by both objectives |

Diagnostics are computed before global clipping and AdamW. They are zero when
there is no active secondary objective or no shared parameters. During supervised
pretraining, the secondary backward pass is skipped. The precision gate compares
the actual merged gradients under the selected strategy against float64, using
the configured final physics weights.

Projection has no persistent optimizer state. Its strategy and epsilon are saved
with the training configuration, so resume checks enforce the same behavior.
Checkpoints created before these fields existed are interpreted as `"sum"` with
`pcgrad_eps=1e-12`; select those values to resume them. Resuming such a checkpoint
with `"pcgrad"` is rejected rather than silently changing the training rule.

## Individual training losses

Every optimizer update logs the following scalars to TensorBoard. The same values
are stored under `loss_components` in each `metrics.jsonl` record, using the tag
without its `train/` prefix.

| TensorBoard tag | Meaning |
|---|---|
| `train/supervised/{head}` | Mean normalized MSE or Huber loss for that head, divided by its own valid scalar count |
| `train/supervised_contribution/{head}` | Head's contribution to the aggregate supervised loss, using the total valid scalar count across all heads |
| `train/short/{channel}` | Unweighted normalized physics loss for an adjacent interval equation |
| `train/long/{channel}` | Unweighted normalized physics loss for a whole-window equation |
| `train/short_weighted/{channel}` | Short equation loss multiplied by the current ramped short lambda |
| `train/long_weighted/{channel}` | Long equation loss multiplied by the current ramped long lambda |
| `train/short_weighted`, `train/long_weighted` | Weighted totals for each physics horizon |

Heads are `force`, `moment`, `r_B`, `v_B`, and `S_B`. Physics channels are `r`
(position consistency), `P` (linear momentum), and `H` (angular momentum). Physics
losses are mean squared normalized vector norms; supervised head losses are means
over valid scalar coordinates. The supervised choice (`mse` or `huber`) applies
to the head logs as well as their aggregate.

The five `supervised_contribution` values sum to `train/supervised`; the per-head
means generally do not. Each horizon's three unweighted channels sum to its
existing `train/short` or `train/long` total. Weighted channels sum to their
weighted horizon total, and both weighted horizon totals sum to `train/secondary`.
Adding `train/supervised` gives `train/loss`. Raw physics channels remain visible
when their weights are zero. These are objective values before PCGrad; projection
changes gradients, not the loss values. Validation metrics retain their existing
per-head error and per-equation residual tags.

## Preprocessing throughput and memory

Parallelism is across attempts. Each attempt retains sequential stream ordering,
per-environment resets, split intervals, and complete-history checks. Whole-group
splitting still happens before processing. Workers stage shards under
`.prepare-work`; the parent assigns series IDs in source order, renames shards
without copying their contents, and remaps compact indices in bounded blocks.
At most one attempt per worker is submitted at a time. Waiting for an earlier
source can leave some workers idle; this bounds staging and keeps output ordering
deterministic. A single attempt does not use multiple processing workers.

Checksum verification and final output hashing use bounded thread queues when
workers are enabled. Source chunks are checked again as they are read. Normalization
is reduced in a fixed serial order, preserving identical statistics across worker
counts. The manifest is written only after all processing and verification succeeds.
Failures leave an incomplete output directory, potentially including staging files;
use a fresh output directory for a retry. Worker errors trigger cancellation of
other active attempts at their next grid-bin boundary.

The processing bar counts completed attempts and shows bins processed by active
attempts. Serial processing also shows a grid-bin bar. Bins are 5 ms grid intervals,
not raw rows or individual environment endpoints. Stages without a known total show
counts and elapsed time rather than an estimated completion percentage.

`manifest.json.profile` includes total and processing seconds, per-attempt elapsed
time/RSS, main-process peak RSS, maximum individual-worker peak RSS, and processed
grid endpoints per second. RSS is a process lifetime high-water mark (KiB on Linux),
not a sum across workers. Memory grows with workers and simultaneous environments;
a process also imports PyTorch even though preprocessing runs on CPU. Start with
2–4 workers, then measure the actual storage system and collection.

`tools/profile_prepare.py` prepares the configured dataset once per worker count
and compares source identities, artifact hashes, window counts, series metadata,
and normalization. It writes a fresh dataset directory for each run. Use a fixed
explicit source list when benchmarking an actively growing collection.

## Terminal progress layout

Training displays update progress on the first row and wraps `phase`, `loss`,
`sup`, `short`, `long`, `lr`, and `val` onto separate rows as needed. Width is
checked on each update, so terminal resizing takes effect without restarting.
Below 65 columns the header omits elapsed time and update rate; below 40 columns
it shows only percentage and update counts. Metric values remain on the rows
below the header. Metrics redraw at most five times per second. Set
`training.progress=false` to disable the display, including for redirected logs
that do not support terminal cursor movement.

## Validation losses and performance

Validation retains full coverage by default. Peak and lag calculations are
vectorized over windows and wrench heads, and their results are reused in
per-robot/payload reports. Batch predictions and physics residuals are transferred
to CPU once for those reports. Ground-truth evaluation computes conservation
floors without repeating peak/lag analysis against itself.

TensorBoard records `validation/loss`, `validation/supervised`, `validation/short`,
`validation/long`, and `validation/secondary`, plus the current `lambda_short` and
`lambda_long`. Individual values use the same suffixes as training:

- `validation/supervised/{head}` and `validation/supervised_contribution/{head}`
- `validation/short/{r,P,H}` and `validation/long/{r,P,H}`
- `validation/short_weighted/{r,P,H}` and `validation/long_weighted/{r,P,H}`
- `validation/short_weighted` and `validation/long_weighted`

The `losses` object in `validation.json` includes these values, validity counts,
and supervision kind. Supervised losses use the configured MSE or Huber objective;
physics contributions use the weights from the most recent training update.
Standalone checkpoint evaluation uses that checkpoint's last-update weights.
Numerators and counts are accumulated across the whole evaluated set before
forming losses, so partial batches and unequal label coverage are weighted
correctly. Empty evaluations have `losses: null`. Physical-unit error and
conservation metrics remain available under their existing names.

`validation/selection_metric` remains normalized supervised MSE, including when
training uses Huber. Weighted validation loss changes during the physics ramp
and is not used for checkpoint selection.

For a new run, `validation_batch_size` can be larger than the training batch size
because validation does not retain gradients. Tune it to available GPU memory.
Try validation workers when loading is a bottleneck; worker overhead can hurt
small cached datasets. `validation_batches` reduces evaluation coverage, while
`validation_every` changes frequency; neither speeds up the same complete pass.
Resume still requires matching settings. Older checkpoints without the new loader
fields inherit their original training batch size and worker count via null defaults.

## Sweep and batch logging controls

See [Hyperparameter sweeps](sweeps.md) for the fixed 16-trial design, common
validation overrides, launch arguments, and resume rules. Sweep configs inherit
the packaged `5c7792d` baseline rather than the current editable combined config.

Training CLI options `--logging-mode auto|plain|progress` and `--trial-id` are
runtime controls, not training hyperparameters. They do not participate in
checkpoint configuration equality. With no options, older checkpoints use
`auto` and `standalone`. Auto mode enables the progress display only on a TTY,
outside Slurm, and when `training.progress` is true. Otherwise it emits plain
batch events; setting `training.progress=false` no longer makes the console silent.
Explicit `--logging-mode progress` takes precedence over that legacy boolean.


## Staged search configuration

The [staged sweep guide](staged_sweeps.md) documents its launcher options,
dependent hyperparameter generation, fixed 20,000-update schedule, resource
preflight and persistent deadline. It derives candidates from the packaged
baseline rather than the editable local `combined.json`. Exact generated configs
are saved under each trial's `resolved.json`; runtime logging, verification receipts
and timing hooks do not become checkpoint-equality configuration fields.


Staged recovery/monitoring/selection is a runtime policy, separate from optimizer
parameter identity: recovery every 200 updates, fixed 5% subset MSE every 1,000,
exact endpoint full validation for stages 1–5, and final-stage patience 3 eligible
at `pretrain_updates + ramp_updates`. `min_delta` affects patience only; the three
lowest raw subset scores plus stopping update form the full-validation shortlist.
See [the staged schedule and resume rules](staged_sweeps.md#configuration-and-comparisons).

Subset monitoring also computes and logs both physics objectives with per-channel
losses and valid counts. These do not affect patience or checkpoint ranking. Full
validation retains physics, wrench, lag and group diagnostics.
