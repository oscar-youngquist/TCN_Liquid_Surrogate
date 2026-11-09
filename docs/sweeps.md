# Two-stage hyperparameter sweep

For dependent search, matched-seed ablations and resumable allocation sessions,
use the separate [staged sweep](staged_sweeps.md). This guide describes the original
fixed 16-trial workflow, whose configurations and promotion rules are unchanged.

The sweep screens 16 independent configurations, then continues the baseline and
the three best alternatives. It uses the prepared training and validation splits;
the test split is never used to rank or promote trials. The coordinator starts
ordinary Python subprocesses inside an existing allocation. It neither requests
resources nor submits jobs.

## Baseline and trials

The baseline is the exact `configs/combined.json` from commit
`5c7792d0840ddfc039f48877d077cacd1da2038e`. A packaged copy lives at
[`src/liquid_tcn/baselines/combined_5c7792d.json`](../src/liquid_tcn/baselines/combined_5c7792d.json),
with SHA-256 `ade364a795b42c258a90e4927ff605e088e08b46bedfa05e6a322b2003675155`.
The current `configs/combined.json` is not modified or used to regenerate the
baseline. Each trial starts from a separate copy of that snapshot.

| ID | Training overrides relative to the baseline |
|---|---|
| 00 | None |
| 01 | Huber, delta 0.25 |
| 02 | Huber, delta 0.5 |
| 03 | Huber, delta 1.0 |
| 04 | Huber, delta 2.0 |
| 05 | Physics learning rate 0.0003 |
| 06 | Physics learning rate 0.0001 |
| 07 | Short weight 0.3; long weight 1 |
| 08 | Short weight 1; long weight 3 |
| 09 | Short weight 0.3; long weight 3 |
| 10 | Pretrain 500 updates; ramp 1500 updates |
| 11 | Short weight 0; long weight 0 |
| 12 | Short weight 1; long weight 0 |
| 13 | Short weight 0; long weight 1 |
| 14 | Huber, delta 0.5; both physics weights 0.5 |
| 15 | Huber, delta 1.0; both physics weights 0.5 |

Huber uses a half-quadratic term near zero. Trials 14–15 test the corresponding
change in supervised/physics balance; they do not rescale the implementation of
Huber itself. PCGrad continues to treat supervision as primary and the weighted
sum of physics losses as secondary. No gradient-enhanced losses are added.

All trials use seed 1, batch size 1024, a 20,000-update schedule, the baseline
architecture, FP32 networks, float64 physics, and PCGrad. Common changes are
validation every 1000 updates, patience 5 validation events, checkpointing every
500 updates, full validation coverage, plain console logging, and zero training
and validation DataLoader workers. Paths, logical GPU assignment, and CPU threads
are supplied at launch. All remaining baseline hyperparameters are retained,
including the baseline's **0.001 physics learning rate**.

## Commands inside an existing allocation

Activate the environment where `liquid-tcn` is installed. Run the following from
the repository root. The paths below match a scratch workspace with the repository
and prepared data in separate directories:

```bash
export LIQUID_WS="/scratch3/workspace/${USER}-liquid-tcn"
export SWEEP_DATA="$LIQUID_WS/data/prepared"
export SWEEP_OUTPUT="$LIQUID_WS/runs/sweep_5c7792d"
```

Dry-run: display all resolved configs without creating files, checking GPU
availability, or starting workers. The prepared data may still be transferring.

```bash
bash scripts/launch_sweep_unity.sh \
    --data "$SWEEP_DATA" --output "$SWEEP_OUTPUT" \
    --stage screen --concurrency 4 --devices 0 --threads 2 --dry-run
```

Screen all trials through **absolute update 5000**:

```bash
bash scripts/launch_sweep_unity.sh \
    --data "$SWEEP_DATA" --output "$SWEEP_OUTPUT" \
    --stage screen --concurrency 4 --devices 0 --threads 2
```

After successful screening, promote the baseline plus the best three other
trials and continue them from `run/latest.pt` to **20,000 total updates**:

```bash
bash scripts/launch_sweep_unity.sh \
    --data "$SWEEP_DATA" --output "$SWEEP_OUTPUT" \
    --stage continue --resume --concurrency 4 --devices 0 --threads 2
```

For automatic screening followed by continuation, use a fresh output directory:

```bash
bash scripts/launch_sweep_unity.sh \
    --data "$SWEEP_DATA" --output "$LIQUID_WS/runs/sweep_5c7792d_auto" \
    --stage auto --concurrency 4 --devices 0 --threads 2
```

Resume the same sweep after interruption, or retry failed trials:

```bash
bash scripts/launch_sweep_unity.sh \
    --data "$SWEEP_DATA" --output "$SWEEP_OUTPUT" \
    --stage auto --resume --concurrency 4 --devices 0 --threads 2
```

Use `--stage screen --resume` to retry screening without automatically promoting.
Completed trials are skipped. A failed process is not retried in a tight loop:
inspect its console file, resolve the operational problem, and restart explicitly.
Nonzero exit codes, missing checkpoints, incomplete screening, invalid metrics,
and partial validation all block promotion. A failure in one worker does not
prevent the other screening workers from finishing.

Refresh reports without starting workers or requiring a GPU:

```bash
bash scripts/launch_sweep_unity.sh \
    --data "$SWEEP_DATA" --output "$SWEEP_OUTPUT" --stage report
```

An explicit interpreter can be supplied as the first option:

```bash
bash scripts/launch_sweep_unity.sh --python "$LIQUID_WS/env/bin/python" \
    --data "$SWEEP_DATA" --output "$SWEEP_OUTPUT" \
    --stage screen --devices 0 --threads 2 --dry-run
```

The wrapper uses `exec`, so signals and exit codes reach the coordinator directly.
It leaves `CUDA_VISIBLE_DEVICES` unchanged. With `--devices 0` (the default), all
four processes share logical GPU 0. With `--devices 0,1`, trial IDs alternate
between logical GPUs 0 and 1. Assignment stays fixed on resume, so devices can
have uneven utilization when trials finish at different times. Physical GPU
numbers from a workstation or scheduler allocation are not logical device indices.

`--concurrency` defaults to 4 and accepts positive counts. `--threads` defaults to
4 and bounds math-library threads per worker; DataLoader workers remain zero.
The requested settings are honored without allocation admission checks. Devices,
threads and concurrency can change on resume. The coordinator also uses CPU and
memory for orchestration.

Four batch-1024 workers may exceed a GPU's memory. OOM is reported as a failure;
the coordinator never changes batch size, loss weights, precision, or learning
rates. Reduce concurrency when retrying if memory pressure was caused by sharing.

## Selection and continuation

Screening passes `--max-updates 5000`; it does **not** change `training.updates`.
The cosine schedule therefore retains its 20,000-update horizon. Continuation
restores model weights, optimizer, scheduler, RNG, consumed sampler position,
patience, and the best-so-far score. It does not restart pretraining or the ramp.
Patience remains active: a continuation can finish early after five validation
checks without improvement. This is reported explicitly in its profile.

Ranking uses each trial's best full-validation checkpoint and the fixed
`normalized_supervised_mse_all_heads` metric, including for Huber trials. Ties
are broken by trial ID. The baseline is always included among the four promoted
trials. All 16 screening trials must complete successfully before selection is
written to `selection.json`; failed or incomplete trials are never silently
excluded. Continuation never uses a best checkpoint as its starting point:
`latest.pt` retains the actual optimization/sampler trajectory.

Seed repetitions are a separate, optional follow-up experiment. This coordinator
runs only seed 1 and does not automatically launch repetitions. Use separate
configs and run directories for any later seed study; do not edit a sweep's
resolved configs or use test results to choose seeds or hyperparameters.

## Files, provenance, and recovery

```text
sweep_5c7792d/
├── sweep.json                    # Baseline/code/data hashes and trial IDs
├── baseline.json                 # Exact reviewed baseline
├── data_manifest.json            # Prepared schema, splits, normalization, hashes
├── state.json                    # Stage status, attempts, exit codes, runtime
├── selection.json                # Frozen screening results and promoted IDs
├── report.json                   # Metrics, resources, paths, and attempt history
├── report.md                     # Compact status/metric table and report links
└── trials/00/                    # One independent directory per trial
    ├── resolved.json             # Immutable fully resolved launch config
    ├── console/0001_screen.log   # Combined stdout/stderr, one file per attempt
    ├── screen_best_validation.json
    ├── continue_best_validation.json
    ├── screen/{best,latest}.pt   # Archived screening checkpoints
    ├── continue/{best,latest}.pt # Archived continuation checkpoints
    └── run/
        ├── best.pt
        ├── latest.pt
        ├── tensorboard/
        ├── metrics.jsonl
        ├── best_validation.json
        ├── validation.json
        └── profile.json
```

Each stage report includes normalized MSE, physical-unit per-axis RMSE for all
heads, peak wrench errors, temporal lag, conservation residuals, coverage, and
per-robot/payload metrics. `report.json` retains paths to those reports and both
checkpoint versions. Attempt records contain console paths, exit codes, and wall
runtime. Successful attempt resources include peak process RSS, peak GPU tensor
allocation, training/validation/loading times, and throughput. These measurements
are per process, not node-wide GPU utilization. Abruptly killed workers may have
no resource profile; their exit status remains visible rather than reporting zero
usage as a measurement.

Prepared data is opened read-only without automatic audits. Resume uses the
supplied paths and requested devices/threads, restores saved training state and
skips completed stages. Source, runtime, configuration and checksum metadata are
not compared. Unknown output directories are not overwritten; coordinator and
worker locks prevent concurrent writers.

SIGTERM/SIGINT stops dispatch and requests checkpoint-and-exit from active workers.
Workers finish the current update and any validation already in progress, then
write `latest.pt` atomically. Allow time for validation/checkpointing in the
allocation's termination policy. SIGKILL cannot be handled; recovery then uses
the last atomic checkpoint. A surviving worker holds its run lock, blocking a
retry from overwriting it. Once the old allocation has ended, `--resume` retries
interrupted attempts. JSONL entries beyond the resumed checkpoint are removed,
and TensorBoard purges events beyond that update.

## Console logging

`python -m liquid_tcn train` accepts `--logging-mode auto|plain|progress` and
`--trial-id`. These are runtime options, outside the checkpoint training config;
older checkpoints default to `auto` and trial ID `standalone` without a migration.
`auto` chooses plain output under Slurm, redirected output, or when
`training.progress=false`. Explicit `progress` enables the terminal display.

The sweep always selects plain logging. Worker files contain timestamped JSON
lines at startup, phase transitions, every 250 updates, validation start/end,
checkpoint writes, completion, and failure. Fields include trial ID, absolute
update/budget, phase, supervised/short/long losses, LR, the latest validation
metric, and average updates/second. Unavailable startup values are null.
Tracebacks follow failure events when applicable. There are no progress bars,
carriage-return refreshes, or per-update console lines. Detailed TensorBoard and
JSONL metrics remain at their existing per-update cadence.

Only the coordinator writes lifecycle events to the parent console. Its heartbeat
runs approximately every 60 seconds and identifies active trials and
completed/failed counts.

## Validation

`tests/test_sweep.py` uses mock Python workers to check the four-process limit,
failure/retry handling, promotion, stage modes, immutable configs, logical GPU
mapping, and non-mutating dry runs. Tiny CPU fixtures check signal checkpointing,
exact model/optimizer/scheduler/sampler continuation, plain output, full-validation
eligibility, and stage checkpoint archives. Existing regression tests cover the
physics, data pipeline, PCGrad, and older checkpoint defaults.

```bash
python -m pytest -q tests/test_sweep.py
python -m pytest -q
```

No production sweep or scheduler job is needed to run these tests. Unity GPU
sharing, OOM behavior on the chosen allocation, and scheduler termination timing
must be measured in that environment; CPU fixtures do not establish those limits.

Implementation checks (2026-11-06): 12 focused sweep tests passed. The regression
run passed 85 tests with two multiworker DataLoader cases excluded; both excluded
cases then passed outside the filesystem/network sandbox (87 total). Within the
sandbox the existing multiworker loader test stalled waiting for its worker queue.
The Bash syntax check, quoted-path launcher dry run, and wheel build also passed;
the wheel includes the reviewed baseline snapshot. No real sweep, production
training run, or scheduler submission was performed.

## Measuring process concurrency

Use the standalone [single-versus-four-process benchmark](benchmark.md) before
choosing how many trials to share on one GPU. It derives all five workers from
trial `00`, measures updates 201–500 after a coordinated warmup, and separately
measures full validation on the final checkpoint. It does not run the sweep or
change its promotion rules. The guide documents the launcher, timing definitions,
memory telemetry limits, and failure handling.
