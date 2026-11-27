# Reset-backfilled shuffled-window training

`configs/pyramid_film_reset_backfill.json` keeps the architecture and every existing
training/monitoring setting from `pyramid_film.json`. The additions are
`startup_fraction: 0.20`, `startup_validation_ages: 16`, and the companion index
path. Existing configurations default to zero startup sampling.

The model remains a nonrecurrent, sparse-pyramid-attention + FiLM TCN. Its receptive
field is 249 samples: 248 preceding 5 ms steps, or 1.24 s. Each training sample has
289 motion inputs and 41 real supervised endpoints spanning 40 physics intervals.
For a first supervised endpoint at reset age `s < 248`, motion indices
`s-248 .. s+40` are gathered; negative indices repeat the recorded reset motion.
Repetition happens in physical units, before the existing training normalization.
Motion includes the recorded gravity orientation, velocity, angular velocity, and
held accelerations. Payload properties are unchanged. No targets or physical
states/intervals are padded; undefined reset-time wrench labels remain invalid.

Startup eligibility requires an explicit collection reset event at the first
prepared endpoint. Recording starts, shard boundaries, and gaps never create
resets. Index construction stops startup context at the first missing input/tick.
The source recordings supply reset metadata for older prepared manifests; new
preprocessing saves the small `recorded_reset` metadata record directly. Existing
shards, normalization, ordinary-window indices, and collection splits are reused.
The loader accepts inert Mamba metadata only when the prepared manifest explicitly
specifies `temporal_memory: none`; it does not substitute recurrent models.

The sampler shuffles startup slots within each batch, balances startup assignments
across episodes, and draws early-history ages uniformly within each episode.
Every ordinary window is still visited once per sampling epoch; startup mixing
lengthens an epoch rather than dropping ordinary windows. Update-based budgets
are unchanged. Integer batch rounding makes the realized fraction very close to
20%; `metrics.jsonl` and TensorBoard record actual fractions and an eight-bin
reset-age histogram. Checkpoints acknowledge optimizer-consumed samples only,
including sampler position and counters. Worker prefetch does not advance resume
state. Resume with the same batch size, startup fraction, and companion index.

`input_context_valid` certifies the permitted backfill separately from the original
`history_valid`. The objective continues to require actual state validity, interval
validity, episode continuity, and valid physical durations. Missing head labels and
empty physics constraints remain masked and have explicit zero counts; they do not
become fabricated supervision. Legacy batches retain the existing empty-coverage
checks.

## Commands on this machine

The implementation is in a separate worktree on branch
`tcn-reset-backfill-training`, based on `origin/main` at `8d03add`. The original
worktree, recordings, and previous runs are untouched.

```bash
cd /home/robo/Research/TCN_Liquid_Surrogate_reset_backfill
conda activate liquid-tcn
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export RESET_DATA=/home/robo/Research/TCN_Liquid_Surrogate/data/prepared_mamba
```

Prepare the companion index (already completed locally; rerunning reuses it):

```bash
python -m liquid_tcn.reset_backfill prepare \
  --data "$RESET_DATA" --output data/reset_backfill_index
```

The existing data supplies 4,060/400/540 verified-reset episodes and
1,001,955/98,709/133,391 startup windows in train/validation/test. No raw-data
reprocessing was required. Only small reset records and index entries are added;
there are no padded arrays on disk.

Start training with the original batch size 1024, learning rates 0.001, 40,000-update
horizon, 500-update pretraining, and 1,500-update physics ramp:

```bash
bash scripts/launch_reset_backfill_training.sh \
  --data "$RESET_DATA" \
  --output runs/pyramid_film_reset_backfill/run_1 \
  --device cuda:0 --logging-mode plain
```

Resume that run:

```bash
bash scripts/launch_reset_backfill_training.sh \
  --data "$RESET_DATA" \
  --output runs/pyramid_film_reset_backfill/run_1 \
  --resume runs/pyramid_film_reset_backfill/run_1/latest.pt \
  --device cuda:0 --logging-mode plain
```

After training and validation selection finish, evaluate the selected checkpoint
on the held-out test split:

```bash
python -m liquid_tcn.reset_backfill evaluate \
  --data "$RESET_DATA" \
  --checkpoint runs/pyramid_film_reset_backfill/run_1/selection/best.pt \
  --startup-index data/reset_backfill_index \
  --split test --device cuda:0 --batch-size 64 \
  --output runs/pyramid_film_reset_backfill/run_1/reset_backfill_test.json
```

This command performs no selection or training. It reports existing mature-window
metrics, deterministic startup/mature diagnostic windows, and unique-endpoint
chronological episode metrics. It can also evaluate an older compatible selected
TCN checkpoint under the same backfill and episode protocol.

If moving to a machine whose prepared data lacks reset metadata, retain access to
the original source paths recorded in its manifest when building the index.
`index.json` reports every excluded series and the reason. If early reset rows
are genuinely absent, create a separate prepared directory from the original
preparation configuration (including its explicit source/group/split assignments),
then rebuild the companion index:

```bash
python -m liquid_tcn prepare --config ORIGINAL_PREPARATION_CONFIG.json \
  --output data/prepared_reset_backfill
python -m liquid_tcn.reset_backfill prepare --data data/prepared_reset_backfill \
  --output data/reset_backfill_reprocessed_index
```

Use that data path and set `training.startup_index` to the new index path in a copy
of the training config. Do not rediscover/reassign groups to repair missing rows.

## Validation and interpretation

The original monitor/shortlist machinery is unchanged: a fixed 5% mature validation
monitor every 1,000 updates, recovery checkpoints every 200 updates, original
patience, and full mature validation of the shortlist plus final checkpoint.
Checkpoint selection continues to use normalized all-head reconstruction MSE on
mature validation windows. Test data never selects a model or checkpoint.

At the same cadence, startup diagnostics evaluate up to 16 deterministic uniform
startup ages and 16 mature window positions per verified-reset validation episode.
Results are written to `reset-validation-UPDATE.json`, TensorBoard
`reset_validation/*`, and full shortlist validation reports. Reconstruction for
all five heads and 5 ms/200 ms physics residuals are reported for `[0,0.2)`,
`[0.2,history)`, and `[history,infinity)` seconds, where `history=(R-1)*dt`.
Reports distinguish physical per-axis RMS from 3D vector RMS and include valid
counts, episodes, and collection groups. Constraints belong to their completed
endpoint's time band; a 200 ms constraint ending at 200 ms is therefore in the
second band. Empty bands/horizons have zero counts and null metrics.

Diagnostic windows are explicitly window-weighted and may repeat endpoints.
The standalone episode report instead scores each real endpoint once, with
nonoverlapping 5 ms intervals and 200 ms physics windows (a partial tail contributes
short intervals but no partial 200 ms constraint). Bounded causal-forward blocks
overlap only raw context; this is equivalent to persistent cached stepping for
this nonrecurrent model. Both use exactly the same reset backfill. Gaps invalidate
context until real history has recovered; they do not receive reset padding.
The original full-history evaluation remains visible separately, so startup gains
cannot conceal mature regressions.

## Deployment

Legacy `CachedInference` warmup remains the default. To opt in:

```python
cache = CachedInference(model, normalization, environments)
cache.set_properties(payload_properties)
cache.initialize_reset(first_recorded_motion)  # [environments, 15], raw units
prediction, valid = cache.step(first_recorded_motion, return_all=True)
```

The initializer processes exactly `R-1` synthetic input rows. `step` then processes
the first real row once and returns valid predictions immediately. Independent
resets leave other environments' caches and the shared cursor intact:

```python
cache.initialize_reset(reset_motion_for_selected_envs, environments=[1, 4])
prediction, valid = cache.step(current_motion_for_all_envs, return_all=True)
```

Set changed payload properties before initializing the reset. The same operation
can be expressed as `step(..., reset=[1,4], reset_motion=selected_reset_motion)`.
Only invoke this path for known recorded resets, not gaps or arbitrary segments.

## Verification

Focused tests cover raw-input/real-target alignment across shards, invalid masks,
gap rejection, unchanged ordinary windows and recipe, consumed-position sampling
with worker prefetch, exact interrupted/uninterrupted weights, staged selection,
and cached/batched predictions with nonzero reset motion and asynchronous resets.

```bash
PYTHONPATH=src python -m pytest -q tests/test_reset_backfill.py \
  tests/test_film_streaming.py tests/test_standalone_monitoring.py
```

A five-update real-data smoke run on the RTX 5070 used the unchanged full TCN
architecture, a smoke-only batch size of 16, and a smoke-only immediate physics
ramp. All losses and gradient norms were finite, including active 5 ms and 200 ms
losses. It consumed 80 samples, 16 startup (20%). Training-loop wall time including
checkpoint/logging was 1.018 s (about 78.6 samples/s); peak allocated GPU memory was
104.7 MiB. These brief startup-inclusive measurements are not steady-state
benchmarks for the configured batch size 1024. Evidence is in
`runs/reset_backfill_smoke/{config.json,metrics.jsonl,profile.json}`.
No full training run or held-out assessment of an unselected smoke model was run.
