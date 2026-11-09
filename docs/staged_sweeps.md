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
`--concurrency` defaults to 12 and accepts any positive count. It limits active
workers; it does not promise the models fit memory. No resource estimate reduces
this setting. Each process retains its
own optimizer, sampler and model. There is no DDP or cross-process gradient sharing.

Append `--resume` with the original data and output paths to continue an interrupted
run. Slurm supplies each allocation's wall-time allowance; historical deadline and
budget fields are ignored. `--budget-hours` is accepted only for compatibility,
with a deprecation warning. No forecast or application timer cancels work or prunes
candidates. Update limits and validation patience remain active.

Resume loads saved progress using the supplied data/output paths and requested
execution resources. No code-change acknowledgement, provenance comparison,
checksum audit, precision probe, or hardware preflight runs automatically.
`--allow-code-change` remains accepted as an ignored legacy argument.
Saved configuration is the resume default; explicitly supplied settings override
it. Tensor shape and file-loading errors surface normally. Completed stages,
rankings, monitoring state, shortlist evaluations and reports remain available.
Moving a sweep directory rebases its run-owned paths to the supplied output root.

To recover the existing Unity run after updating the installed package, run this
inside a new allocation with the environment activated:

```bash
SWEEP_OUTPUT=/scratch3/workspace/oyoungquist_umass_edu-liquid-tcn/runs/sweep_a100_66862094
SWEEP_DATA=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["data"])' "$SWEEP_OUTPUT/staged_sweep.json")
bash scripts/launch_staged_sweep_unity.sh \
  --data "$SWEEP_DATA" \
  --output "$SWEEP_OUTPUT" \
  --devices 0 --concurrency 12 --threads 2 \
  --resume
```

Keep this original output path, rather than creating one from the new
`${SLURM_JOB_ID}`. The new Slurm submission supplies the new wall-time allowance.
For the interrupted stage-2 run, completed `t005`/`t006` results are retained and
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

The starting point is the packaged `combined_5c7792d.json` described
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

Continuations restore `latest.pt`, including optimizer, scheduler, RNG and consumed
sampler position. New scientific candidates or seeds start fresh according to the
stage plan. Current devices and thread counts control execution. Historical hashes
and device metadata do not gate loading.

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
validation passes are cached by shortlisted update and reused if selection
is interrupted; selection/report writes and selected-checkpoint replacement are
atomic. Keep latest, shortlist, selected best and stage archives when moving runs.
Standalone and legacy training retain their existing validation/patience behavior.
Code and allocation changes require no compatibility acknowledgement.

Stage 6 freezes the chosen settings and changes only seed and ablation weights:
full retains both, supervised-only zeros both, short-only zeros long, and long-only
zeros short. All other settings, including pretraining/ramp and LR schedule,
remain fixed. Matching seeds therefore match initial weights and sample order.
Disabling physics does not change the learning-rate trajectory.

## Execution resources and interruption

The coordinator honors the requested devices, concurrency and per-worker threads.
It does not probe GPUs, estimate capacity, profile loaders, or run a preflight
benchmark. Choose resources for the allocation; actual CUDA or allocation errors
surface through worker logs and failed attempt records. The separate
[concurrency benchmark](benchmark.md) remains available when explicitly requested.

Slurm controls wall time. SIGTERM/SIGINT stops dispatch and requests checkpointed
worker exits. Unresponsive children are killed after a 300-second shutdown grace;
the last completed atomic checkpoint remains recoverable. Interrupted stages
cannot promote. Update budgets and validation-based patience remain unchanged.

## Test gate and reports

Only validation drives search and checkpoint selection. After all twelve final
runs finish, `final_selections.json` records their validation-selected checkpoints
before any test worker starts. Existing selections and reports are reused without
hash or configuration comparisons. Test scores never enter rankings or trigger
further search.

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
  state.json                # Candidate registry, parents, rankings, attempts, resources
  monitor_subset.json       # Shared fixed stratified validation indices
  baseline.json
  data_manifest.json
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

Prepared data is opened directly. Historical receipts, hashes, source versions and
allocation metadata are informational and do not gate training or evaluation.
Explicit audit helpers in `integrity.py` remain available separately. Process
locks and atomic writes still prevent concurrent writers and incomplete commits.

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

Focused CPU tests cover direct checkpoint continuation, changed historical
provenance/devices, completed-trial reuse, monitoring-state restoration, worker
failure and signal handling. Explicit audit and precision tools remain opt-in;
training and evaluation do not call them. No production sweep or GPU benchmark
is needed to exercise these loading paths.

The loading simplification was checked with synthetic CPU resumes and directly
affected trainer, sweep, benchmark-orchestration and loader tests. The checks
include relocated paths, ignored historical hashes/runtime/device metadata,
restored update/sampler/optimizer state, and skipped completed trials. Shell syntax
and whitespace checks passed. No production sweep or GPU benchmark was launched.
