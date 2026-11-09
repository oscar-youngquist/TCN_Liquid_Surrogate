# Resumable staged sweeps

`launch_staged_sweep_unity.sh` runs a dependent search, followed by matched-seed
physics ablations and held-out test evaluation. It runs independent Python
processes inside an existing allocation. It does not submit jobs, request GPUs,
or change `CUDA_VISIBLE_DEVICES`. The [fixed 16-trial sweep](sweeps.md) remains a
separate command with its original candidates and promotion rules.

## Launch and resume

Activate the environment containing the installed package. Use an output folder
separate from prepared data, old sweeps, and concurrency benchmarks:

```bash
bash scripts/launch_staged_sweep_unity.sh \
  --data /path/to/prepared \
  --output /path/to/runs/staged \
  --devices 0 \
  --concurrency 12 \
  --threads 2
```

`--devices 0,1` uses two logical GPUs. `--devices cpu` is available for development.
These are device indices inside the scheduler's existing visibility mask.
`--concurrency` is a maximum, from 1 through 12; it is not a promise that twelve
batch-1024 models fit. CPU affinity, `SLURM_CPUS_PER_TASK`, available host/cgroup
memory and measured CUDA memory can reduce concurrency. Each process retains its
own optimizer, sampler and model. There is no DDP or cross-process gradient sharing.

Append `--resume` with the original data and output paths to continue an interrupted
run. Slurm supplies each allocation's wall-time allowance; historical deadline and
budget fields are ignored. `--budget-hours` is accepted only for compatibility,
with a deprecation warning. No forecast or application timer cancels work or prunes
candidates. Update limits and validation patience remain active.

Use `--resume --allow-code-change` to acknowledge an implementation update. Resume
archives metadata under `metadata_history/` before migration and records old/new
source fingerprints and allocation history in `staged_sweep.json`. Original
provenance is retained. Dataset identity, normalization, architecture, exact saved
training configuration, checkpoint integrity and runtime/serialization compatibility
remain strict; the flag does not waive those checks. A runtime-version mismatch
requires a compatible environment.

Each allocation reassesses capacity and records current devices, thread counts and
execution assignments. Valid assignments are retained where possible; execution
can move without rewriting immutable trial configurations. Model, optimizer,
scheduler, RNG, consumed sampler position, monitoring/patience state and shortlist
resume from the checkpoint. Different hardware need not reproduce bitwise results.
Verified completed stages and matching cached evaluations are reused.

To recover the existing Unity run after updating the installed package, run this
inside a new allocation with the environment activated:

```bash
SWEEP_OUTPUT=/scratch3/workspace/oyoungquist_umass_edu-liquid-tcn/runs/sweep_a100_66862094
SWEEP_DATA=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["data"])' "$SWEEP_OUTPUT/staged_sweep.json")
bash scripts/launch_staged_sweep_unity.sh \
  --data "$SWEEP_DATA" \
  --output "$SWEEP_OUTPUT" \
  --devices 0 --concurrency 12 --threads 2 \
  --resume --allow-code-change
```

Keep this original output path, rather than creating one from the new
`${SLURM_JOB_ID}`. The new Slurm submission supplies the new wall-time allowance.
For the interrupted stage-2 run, verified `t005`/`t006` results are retained and
`t000` resumes its saved stage-1 endpoint toward 5,000 updates before promotion.
No successful results are fabricated from incomplete work.

Optional leading `--python /path/to/env/bin/python` selects an interpreter. The
launcher otherwise uses `PYTHON`, or `python` from the activated environment.
It uses unbuffered output and `exec`, preserving signals and exit status.

A dry run prints the stage plan and five initial resolved configurations without
creating files, requiring prepared data to exist, or initializing CUDA:

```bash
bash scripts/launch_staged_sweep_unity.sh \
  --data /path/to/prepared --output /path/to/runs/staged \
  --devices 0,1 --concurrency 12 --threads 2 --dry-run
```

The stage-plan portion is available in [staged_plan.example.json](staged_plan.example.json).
Later configurations remain unresolved until the preceding validation results
exist. A dry run does not invent winners or estimate hardware throughput.

## Configuration and comparisons

The starting point is the packaged, hash-checked `combined_5c7792d.json` described
in [sweeps.md](sweeps.md). Baseline batch size 1024, model configuration,
normalization, FP32 network, float64 physics, AdamW and one-way PCGrad are retained.
The source fingerprint records the actual installed implementation, including
working-tree changes; the baseline JSON alone does not specify Python code.
The staged runtime policy separates recovery, monitoring and selection:

| Operation | Schedule |
|---|---|
| Resumable `latest.pt` | Every 200 updates, monitoring-state commits, and stopping/interruption |
| Fixed validation subset, supervised MSE + physics objectives | Every 1,000 updates |
| Full validation, stages 1–5 | Exact stage endpoint |
| Full validation, stage 6 | After stopping, at most four shortlisted updates |
| Test | Once per final selected run, after all twelve selections are frozen |

The shared subset is approximately 5% of validation windows (nearest integer,
at least one): 12,533 of 250,658 windows. Sampling uses seed 1729 and proportional
collection/robot/payload strata with largest-remainder allocation. It does not
select the first batches. `monitor_subset.json` stores indices, seed, group counts,
coverage and dataset fingerprint; its hash is recorded in the sweep manifest and
checkpoint monitoring state. Every candidate and seed uses the same windows.
Monitoring computes normalized supervised MSE and the short/long physics objectives,
with the same masks, physics precision and valid-count weighting as evaluation,
including partial batches. It logs per-channel physics losses, valid counts and
the ramp-weighted physics/total objectives. **Only supervised MSE controls patience
and selection**; physics remains diagnostic. Group, peak and lag diagnostics are
reserved for full-validation reports.

| Stage | Comparison | Total updates per candidate |
|---|---|---:|
| 1 | MSE; Huber delta 0.25, 0.5, 1, 2 | 3,000 |
| 2 | Physics-phase LR 0.001, 0.0003, 0.0001 | 5,000 |
| 3 | Pretraining/ramp updates 100/900, 500/1500, 300/1200 | 6,000 |
| 4 | Short/long weights (1,1), (0.3,1), (1,3), (0.3,3) | 8,000 |
| 5 | Eight local interactions described below | 10,000 |
| 6 | Full, supervised-only, short-only, long-only × seeds 1,2,3 | Up to 20,000 |

Every configuration keeps `training.updates=20000`. Stage budgets use the
trainer's absolute `max_updates` limit, preserving the 20,000-update scheduler
horizon. Stage 2 inherits the loss winner; stage 3 also inherits LR; stage 4 also
inherits activation. Each comparison retains its incumbent. Search seed 1 and
its window ordering are common to all candidates.

Identical configurations reuse `latest.pt`, including optimizer, scheduler, RNG,
and acknowledged sampler position. Saved device configuration stays fixed after registration; allocation-specific execution placement is recorded separately.
Changed training settings or seeds start fresh. Output paths and placement are
excluded from the registry's scientific identity; exact saved training settings,
including the assigned device, are still checked by the trainer on continuation.

Stage 5 starts with the incumbent, runner-up loss, lower LR, higher LR,
alternative weights, runner-up loss + alternative weights, runner-up activation
+ alternative weights, and lower LR + alternative weights. LRs use geometric
midpoints toward the nearest earlier tested LR, bounded by 0.0001–0.001. Weight
pairs use geometric midpoints toward the stage-4 runner-up. At a boundary, a
proposed LR can duplicate the incumbent; stable Cartesian combinations of those
nearby settings replace duplicates. Generation rules, parent rankings and exact
resolved configurations are saved. There are eight distinct candidates at 10,000 updates; forecasts do not remove candidates.

Stages 1–5 disable early stopping and rank the full-validation
`normalized_supervised_mse_all_heads` score of the **exact endpoint checkpoint**.
Historical best and subset scores never promote a search candidate. Ties use
stable trial ID. Incomplete endpoints or nonfinite/partial validation block
promotion. The endpoint checkpoint, report and selection record are archived;
continuation restores its latest optimizer trajectory.

Stage 6 starts fresh monitoring/patience state on first entry, at the current
resumed update. Checks from earlier stages do not count. Patience is three
non-improving eligible monitoring checks, using the existing `min_delta`.
Eligibility starts at `pretrain_updates + ramp_updates`, identically for all
ablations. The first eligible check establishes the patience baseline. The
maximum remains 20,000 updates. Parameter identity and the optimizer/LR schedule
are unchanged by this runtime policy.

Retain the three lowest finite **raw** monitoring scores, regardless of
`min_delta`, breaking ties by earlier update. Add the stopping checkpoint and
deduplicate by update. After maximum-update or valid patience completion, evaluate
all retained checkpoints on full validation and choose the lowest full MSE,
breaking ties by earlier update. `selection.json` records stopping reason/update,
selected update, checkpoint hashes, subset coverage and full-validation score.
Interruption cannot qualify as patience completion.

Monitoring state, patience baseline/stale count and shortlist hashes are embedded
in resumable `latest.pt`, separately from training configuration equality.
Superseded shortlist files are removed only after the new latest checkpoint is
committed. Recovery does not retain every 200-update snapshot. Completed full
validation passes are cached by checkpoint/subset identity and reused if selection
is interrupted; selection/report writes and selected-checkpoint replacement are
atomic. Keep latest, shortlist, selected best and stage archives when moving runs.
Standalone and legacy training retain their existing validation/patience behavior.
Code changes require explicit acknowledgement as described above; existing
results still undergo integrity and selection-policy checks.

Stage 6 freezes the chosen settings and changes only seed and ablation weights:
full retains both, supervised-only zeros both, short-only zeros long, and long-only
zeros short. All other settings, including pretraining/ramp and LR schedule,
remain fixed. Matching seeds therefore match initial weights and sample order.
Disabling physics does not change the learning-rate trajectory.

## Preflight and allocation limits

Preflight runs the actual trainer for 200 warmup + 50 measured updates, then full
validation (batch 1024, zero loader workers). It first probes each logical device
individually to obtain host RSS and CUDA peaks across startup, training and
validation. It then measures the **actual selected concurrent cohort** using the
benchmark's failure-aware barriers, with no elapsed-time expiry in staged preflight. Twelve-process throughput is never
extrapolated from a four-process measurement.

The CPU limit reserves one coordinator CPU when possible. Per-worker math-library
thread pools are bounded. Capacity uses measured peak host RSS with 25% headroom,
80% of available host memory, and conservative CUDA reserved-memory/free-memory
headroom. These estimates do not reserve memory from other processes. If the
concurrent preflight fails or OOMs, the sweep stops and records the failure; it
does not silently shrink batches, change precision or retry at a different
concurrency. Startup GPU peaks are retained before resetting counters.
Physical-device/MIG utilization telemetry is explicitly unavailable; CUDA reports
capacity for the visible device or instance.

A bounded loader matrix measures 0/1/2 workers with prefetch factors 2/4 when the
CPU and estimated buffer-memory budgets permit them. Each setting uses at most
16 batches, including startup. Results and skipped settings are saved. These
measurements do not automatically change the baseline's zero-loader-worker
configuration or invalidate checkpoints.

Forecasts use the slowest measured concurrent worker, full validation duration,
startup overhead, 200-update recovery writes and monitoring snapshot writes,
per-device scheduling waves and a 25% time margin. Subset-check cost is conservatively
estimated at 5% of the measured detailed full-validation pass; it is not an
independent throughput measurement. Forecasts separately count subset checks,
one endpoint pass per search candidate, up to four shortlist passes per final run,
and twelve test passes. Final-stage estimates assume the maximum 20,000 updates.
Known completed updates are deducted for current-stage continuations; unknown
future candidates are conservatively costed as fresh runs. Forecasts are
informational only: there is no shutdown reserve, launch refusal, or candidate
pruning based on available time. Preflight barriers and loader profiling have no
application time deadline; worker failures and termination signals still abort them.

SIGTERM/SIGINT stops dispatch and requests checkpointed worker exits. Unresponsive
children are killed after a bounded 300-second shutdown grace, which starts only
after a termination request; the last atomic checkpoint remains recoverable.
Interrupted stages cannot promote. Slurm may kill a process before checkpointing
finishes, so recovery always uses the last completed atomic checkpoint.

## Test gate and reports

Only validation drives search and checkpoint selection. After all twelve final
runs finish, `final_selections.json` freezes their validation-selected checkpoints,
SHA-256 identities, config hashes, seeds and validation metrics. The coordinator
stores its hash before dispatching any test worker. Every test worker verifies
that all twelve final results and checkpoint identities still match that file.
Test results cannot enter a ranking or trigger further search.

`report.json` retains individual test reports and per-stage validation reports:

- Per-head, per-axis MAE and RMSE: force N, moment N m, position m, velocity m/s,
  and internal angular momentum kg m²/s.
- Short/long vector RMS and maximum residuals for displacement (m), linear momentum
  (N s), and angular momentum (N m s), plus normalized residuals. Normalization
  divides by duration times the existing reference speed/force/moment scales.
- Vector wrench error at the target magnitude peak of each contextual window,
  mean/maximum peak errors, and demeaned cross-correlation lag over valid,
  nonconstant windows, with offsets up to ±50 ms.
- Recorded robot/payload groups, and maneuver groups only if the prepared series
  actually contains a `maneuver` field. Missing maneuver labels are not inferred.
- Individual seeds, mean and sample SD (`ddof=1`) across the three seeds, and
  full-minus-ablation differences paired by seed. Overlapping-window variation is
  not used as a substitute for variation across seeds.

`report.md` gives completion status, stage winners and a compact final comparison;
JSON carries all axes, units, coverage, checkpoint paths, hashes and provenance.
An incomplete run has no purported twelve-run aggregate.

```text
staged/
  staged_sweep.json          # Original provenance, session history and execution assignments
  metadata_history/         # Metadata snapshots before each resume migration
  state.json                # Candidate registry, parents, rankings, attempts, resources
  verification.json         # Trusted local prepared-data verification receipt
  monitor_subset.json       # Shared fixed stratified validation indices
  baseline.json
  data_manifest.json
  probe_*/                  # Per-device startup/memory probes
  concurrent_preflight/     # Actual concurrency measurement and worker console files
  final_selections.json     # Written only after all 12 final runs finish
  report.json
  report.md
  trials/t000/
    resolved.json
    console/0001_1.log
    1/{best,latest}.pt       # Both identify the exact search endpoint
    1/selection.json        # Endpoint score, hashes and completion reason
    6/validation-*.json     # Cached full-validation shortlist reports
    6/selection.json        # Final full-validation selection
    1_best_validation.json
    test.json               # Only final selected runs
    run/{latest.pt,best.pt,metrics.jsonl,profile.json,tensorboard/,...}
```

Worker consoles contain timestamped phase, update (every 250), validation,
checkpoint and failure events, with stage/trial/seed/device in the identifier.
Coordinator heartbeats report active assignments, pending/completed counts and
elapsed time. Detailed per-update JSONL and TensorBoard tags remain available;
there are no terminal progress bars.

## Loader, logging and timing changes

Batched window fetches group reads by mmap shard and restore sampler order. Motion
uses the complete causal history; only contextual endpoints are copied for label
and physics fields. Static metadata and shard maps have bounded process-local
caches. Sampler progress is acknowledged only after the optimizer consumes a batch.
There is no change to masks, episode boundaries, physics precision or normalization.

Coordinated runs hash prepared artifacts once, then workers validate a local
receipt containing the manifest hash and every artifact's inode, size, modification
and change timestamps. A mismatch aborts. Keep prepared data immutable throughout
a run. The receipt is trusted coordinator output, not a mechanism for authenticating
an untrusted dataset or concurrent writers. Standalone training still hashes all
artifacts. Resume checks the receipt, code, runtime, configs and archived checkpoints.

Training JSONL uses a persistent buffered stream, flushed at checkpoints, shutdown,
100 records, or a 10-second interval checked on writes. TensorBoard retains every
scalar tag at every update but batches them into one event. Scalar transfers and
nonfinite checks are consolidated without changing the PCGrad arithmetic.
Ordinary training no longer explicitly synchronizes CUDA after every optimizer
step; required finite checks and metric transfers can still synchronize it.

`profile.json` separates loader/collation, host transfer dispatch, training,
logging, validation and checkpoint durations. Ordinary asynchronous CUDA timings
are host sections, not pure kernel times. Benchmark hooks synchronize transfer
and training boundaries, preserve warmup-excluded latency/throughput, and include
ordinary loading/logging overhead. Their extra synchronization can make these
measurements conservative relative to ordinary training.

## Local verification

The tests include shard-crossing/reordered/repeated samples, receipt invalidation,
bitwise checkpoint/optimizer parity against ordinary sample collation, exact resume,
dependent inheritance/deduplication, equal-budget promotion, matched seeds,
resource caps, wave arithmetic, ignored legacy deadlines, and the test-selection gate.
A tiny CPU integration test exercises the full six-stage lifecycle, twelve final
selections, test reports and completed-run resume with reduced test-only budgets. Focused monitoring tests
check subset reuse, agreement with the existing evaluator, final patience state
across resume, shortlist selection and reuse of completed full-validation passes.
The existing sweep/benchmark tests exercise subprocess failures, barriers, signals
and cleanup. No production sweep is required:

```bash
python -m pytest -q tests/test_precision_resume.py tests/test_staged_sweep.py tests/test_sweep.py tests/test_benchmark.py
bash -n scripts/launch_staged_sweep_unity.sh
```

A small local synthetic loader comparison (724 default-length windows, 64-window
batches, 71-row shards, zero workers) measured 9,067 windows/s before versus 21,592
after the batched-fetch change, using the fastest of four in-memory-cache passes.
This roughly 2.38× result measures loader work, not end-to-end GPU training or disk
throughput on Unity. See [performance verification](performance_verification.md)
for training parity, timings and regression results. Twelve-process GPU capacity,
MIG telemetry, long-run memory behavior and production completion require measurements
inside the target allocation; they have not been established by CPU tests.

Focused schedule-policy verification (2026-11-06): 27 directly affected staged,
legacy-sweep and trainer-hook tests passed. The subset test matches both supervised
MSE and short/long physics losses/counts against the existing evaluator on identical
windows. The tiny CPU lifecycle covers endpoint promotion, final patience,
shortlist selection, persisted resume state, cached full-validation reuse and the
frozen test gate. Bash syntax and diff-whitespace checks passed. No GPU benchmark
or production sweep was run; production timing and GPU behavior remain unverified.

Recovery verification (2026-11-09): 42 directly affected precision, staged-sweep,
legacy-sweep and benchmark tests passed. A tiny CPU recovery test verifies an
expired historical deadline, explicit code-change acknowledgement, preserved
completed stage-2 results, checkpoint continuation and monitoring state. Precision
tests cover finite FP32 warnings, nonfinite failures, AMP rejection and unchanged
model/optimizer/RNG state. Bash syntax and whitespace checks passed. The Unity run
was not accessed or restarted; allocation remapping on GPU remains unverified.
