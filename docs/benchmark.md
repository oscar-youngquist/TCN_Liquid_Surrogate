# Single-versus-four-process throughput benchmark

This benchmark compares one training worker with four independent workers sharing
one logical GPU. It runs the single-worker phase first and waits for that process
to exit before launching the four-worker phase. Each worker has its own model,
optimizer, output directory, TensorBoard events, and console log. All workers start
fresh with seed 1 and the same data order. This is process-level concurrency, not
distributed training or a larger effective batch size.

## Launch

Use an activated environment with this repository installed, inside an existing
allocation. From the repository root:

```bash
bash scripts/benchmark_tcn_concurrency.sh \
  --data /path/to/prepared \
  --output /path/to/benchmark_results \
  --devices 0 \
  --threads 2
```

The launcher does not create or submit scheduler jobs. It uses unbuffered Python,
`exec`, and the existing `CUDA_VISIBLE_DEVICES` unchanged. Device `0` means the
first logical GPU visible inside the allocation. Exactly one device must be
selected for both phases. `--devices cpu` is available for CPU tests.

To inspect all five resolved configurations without creating files or launching
workers:

```bash
bash scripts/benchmark_tcn_concurrency.sh \
  --data /path/to/prepared \
  --output /path/to/benchmark_results \
  --devices 0 --threads 2 --dry-run
```

An explicit interpreter can be supplied as the first option:

```bash
bash scripts/benchmark_tcn_concurrency.sh --python /path/to/env/bin/python \
  --data /path/to/prepared --output /path/to/benchmark_results \
  --devices 0 --threads 2
```

| Option | Default | Meaning |
|---|---:|---|
| `--data` | Required | Completed prepared dataset, opened read-only |
| `--output` | Required | New, empty benchmark directory |
| `--devices` | `0` | One visible logical GPU index, or `cpu` |
| `--threads` | `2` | Same math-library/PyTorch intra-op thread limit per worker in both phases |
| `--warmup-updates` | `200` | Fresh-run updates excluded from throughput measurements |
| `--measured-updates` | `300` | Consecutive updates measured after warmup |
| `--barrier-timeout` | `3600` | Maximum seconds to reach/release each cohort barrier; includes work leading to the next barrier |
| `--shutdown-timeout` | `300` | Grace period for worker checkpoint-and-exit before a forced kill |
| `--dry-run` | Off | Print resolved configuration and budgets without running |

Counts must be positive and warmup plus measured updates cannot exceed 20,000.
Four times `--threads` must fit `SLURM_CPUS_PER_TASK`, when set, and the process's
CPU affinity. The examples use eight worker math threads in the four-process
phase. DataLoader workers remain zero, and each worker uses one PyTorch inter-op
thread. TensorBoard and the coordinator also have some CPU overhead.

Never place the benchmark inside a production sweep or prepared-data directory.
Existing output directories containing prior results are rejected. Benchmarks
are not resumed: each comparison needs fresh, equivalently warmed workers. After
an interruption, use a new output directory. Checkpoints are retained for diagnosis.

## Configuration and execution

Configurations come from `sweep.generate_trials()['00']`, which verifies the
packaged `combined_5c7792d.json` baseline hash. Only worker output paths differ
between the five configs. The supplied data path, logical device, and thread
count are resolved by the existing sweep generator. The benchmark preserves:

- Batch size 1024, baseline architecture and prepared normalization.
- Seed 1, FP32 network, float64 physics, PCGrad, AdamW, and loader settings.
- The existing pretraining/physics ramp and cosine schedule.
- `training.updates=20000`; the absolute update limit stops the default run at 500.

The existing trainer executes every update. Optional runtime hooks observe timing
boundaries; they are not training configuration fields or checkpoint state.
Ordinary training and sweep checkpoint compatibility rules are unchanged.

For this measurement only, hooks defer scheduled validation and checkpoint
serialization until measured training has finished. The initial and final
`latest.pt` checkpoints are still written. This avoids holes in the timing
interval, including when custom update counts cross a scheduled maintenance
boundary. It does not change losses, optimizer updates, scheduler progression,
sampling, or detailed per-update logging. Sweep configs, stage budgets, and
promotion logic remain unchanged.

Each phase proceeds through four bounded rendezvous points:

1. **Warmup done:** every worker has completed its warmup and synchronized CUDA.
   The coordinator releases all workers into measured training.
2. **Measured training done:** every worker has completed its measured updates.
   Only then may workers serialize final checkpoints. A faster worker cannot
   write checkpoints while another worker is still being timed.
3. **Validation ready:** every worker has saved `latest.pt`, loaded its final
   model, and prepared the validation loader. The coordinator releases concurrent
   full validation, with batch size 1024 and zero DataLoader workers.
4. **Validation done:** all evaluations have finished before workers serialize
   validation reports and exit.

Validation uses the existing evaluator on the **validation split** and final
`latest.pt`, never `best.pt` or the test split. Default stopping at update 500 does
not reach the sweep's scheduled validation at update 1000. Consequently, a
benchmark run need not contain `best.pt`.

## Timing definitions

Default measured updates are **201–500 inclusive**. Each worker's monotonic
measurement starts after the warmup barrier, before fetching the next batch. It
ends after update 500's normal metrics logging, a TensorBoard flush, and CUDA
synchronization. Individual update latency includes DataLoader work, device
transfers, forward/backward/optimizer work, and ordinary training logs. The last
latency also includes the final metrics flush. Warmup, barrier waiting, checkpoint
serialization, and validation are outside these intervals.

Per-worker throughput is measured updates divided by that worker's measured
seconds. Median and p95 latency use the individual update intervals, not an
estimate from the progress logger. The secondary `compute_section_seconds`
measurement covers the trainer's host-timed transfer/forward/backward/optimizer/
scheduler/CUDA-sync section; it excludes loading and per-update metric logging.
It is **not** a pure GPU-kernel timing measurement.

Aggregate training throughput is:

```text
sum(completed measured updates) / (latest worker end - earliest worker start)
```

Thus the four-worker default numerator is 1200 updates. The denominator includes
release skew and the slowest worker's tail. Summing individual worker rates would
overstate useful throughput when workers finish at different times; that sum is
not used. All workers run on the coordinator's host, sharing a monotonic clock.

Validation has its own synchronized start/end timestamps and aggregate formula:
all evaluated windows divided by the shared validation interval. Its timing
includes validation loading and metric computation, but excludes model/checkpoint
loading, rendezvous waits, and report serialization. Each worker separately
reports validation setup and evaluation duration.

Results also include startup, warmup, checkpoint, barrier-wait, and total elapsed
times. `startup_seconds` excludes initial checkpoint serialization;
`startup_including_initial_checkpoint_seconds` retains the full elapsed startup
interval, including interpreter imports, checksum verification, model/loader setup,
and precision checks. These fields and phase totals have overlapping scopes and
should not simply be added. Whole-run `profile.json` and
`BatchLogger.updates_per_second` are not used as benchmark throughput measurements.

The comparison table shows aggregate speedup, the range of per-trial slowdowns
relative to the single worker, validation windows/second, and maximum per-worker
measured-training peak allocated/reserved memory. The JSON retains each worker's
latencies and exact timing intervals.

## Memory, provenance, and interpretation

Every worker captures CUDA allocator peak allocated and reserved bytes for
startup **before any reset**, then warmup, measured training, post-training /
validation setup, and validation. Peak counters are reset between these scopes;
the allocator cache is not emptied. Resetting peak statistics therefore does not
remove retained cache memory from the experiment. Startup peaks remain in the
results even if the measured-phase peak is smaller.

CUDA-visible total capacity and sampled free memory are recorded when available,
along with logical GPU index, GPU name, UUID when exposed by PyTorch, compute
capability, and hostname. Capacity/free-memory readings are snapshots, not
physical-device peak utilization. Allocated/reserved peaks are per process;
summing them does not establish a simultaneous physical-device peak. Physical GPU
and MIG attribution/telemetry are explicitly marked unavailable because NVML is
not collected. Under MIG, CUDA capacity refers to the visible instance. CPU runs
record unavailable GPU measurements as null, not zero.

`benchmark.json` records baseline, package source and checkout commit hashes,
prepared-manifest hash, Python/PyTorch/NumPy/CUDA versions, GPU identity, CPU
allocation, and runtime measurement options. The exact baseline and prepared
manifest are copied alongside it. Each worker's resolved config is hashed and
checked before execution. The prepared manifest retains normalization, split,
schema, source, and artifact provenance; the trainer verifies dataset artifacts.

This benchmark measures the selected GPU and storage under current contention.
Filesystem caches are not cleared between phases, and no clocks, GPU partitions,
or scheduler resources are changed. It does not guarantee that four workers will
fit memory or improve throughput. OOM and other failures are reported; concurrency,
batch size, and precision are never silently reduced.

## Outputs and failure handling

```text
benchmark_results/
├── benchmark.json
├── baseline.json
├── data_manifest.json
├── summary.json
├── comparison.md
├── single/
│   ├── state.json
│   ├── barriers/
│   └── single_00/
│       ├── resolved.json
│       ├── worker_spec.json
│       ├── console.log
│       ├── result.json
│       ├── validation.json
│       └── run/                 # latest.pt, TensorBoard, metrics.jsonl, config, profile
└── four/
    ├── state.json
    ├── barriers/
    └── four_00/ ... four_03/     # Same independent worker layout
```

Workers reuse plain timestamped training logging, with separate console files.
The coordinator uses the same logger for lifecycle events, barrier releases, and
approximately 60-second heartbeats. There are no progress bars or per-update
console messages. Detailed loss metrics retain their normal JSONL/TensorBoard
cadence.

A failed worker, OOM, barrier timeout, or SIGTERM/SIGINT aborts the phase, stops
further dispatch, and signals remaining workers. Training writes a checkpoint at
a completed-update boundary. Workers waiting at barriers observe the abort;
validation observes it between batches. After the shutdown grace period, a stuck
worker is forcibly killed and its status is recorded. The launcher returns
nonzero and no successful comparison is produced. A failed single-worker phase
never starts the four-worker phase. `summary.json`, phase `state.json`, worker
`failure.json`, and console paths retain diagnostics.

## Verification

`tests/test_benchmark.py` covers trial-00 inheritance, independent configuration
copies, quoted-path dry runs, shared-interval arithmetic, barrier ordering,
concurrent validation, failure/timeout/interruption cleanup, startup peak
preservation, and measured timing that excludes barrier waits. Tiny CPU fixtures
exercise the actual worker subprocess and final-checkpoint evaluator. With hooks
present, model, optimizer, scheduler, sampler, config, and RNG state match ordinary
training exactly after the same updates.

```bash
python -m pytest -q tests/test_benchmark.py tests/test_sweep.py
bash -n scripts/benchmark_tcn_concurrency.sh scripts/launch_sweep_unity.sh
```

These tests do not launch the production GPU benchmark. GPU contention, allocator
peaks, and throughput must be measured in the intended allocation.

Implementation verification (2026-11-06): all 101 tests passed across the regression
runs, including 14 benchmark tests. The main CPU run passed 99 tests; two existing
multiworker DataLoader cases passed separately outside the sandbox, where worker
communication is available. Both Bash launchers passed syntax checks. No production
GPU benchmark or scheduler job was launched.
