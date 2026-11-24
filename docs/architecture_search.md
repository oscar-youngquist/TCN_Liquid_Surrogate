# Focused short-history architecture search

This search compares history plus capacity using six TCN–pyramid-attention–FiLM
architectures. Each has the existing five prediction heads. The paired hybrid
adds the differentiable Mamba layer and learned reset initializer.

| Requested history | Realized history | Blocks | Dilations | Attention tokens | TCN parameters | Hybrid parameters |
|---|---|---|---|---|---|---|
| 200 ms | 200 ms | 3 | 1, 1, 3 | 5 | 200464 | 302096 |
| 500 ms | 520 ms | 4 | 1, 2, 3, 7 | 11 | 246160 | 347792 |
| 1.24 s | 1.24 s | 5 | 1, 2, 4, 8, 16 | 28 | 291664 | 393296 |

Each residual block has two causal kernel-5 convolutions. Context therefore
advances on a 40 ms grid, rounding the requested 500 ms up to 520 ms. History is
measured between the oldest input and current endpoint, excluding the endpoint
itself. The short-history helper removes unnecessary deep blocks and redistributes
dilations. It no longer substitutes the baseline for shorter requests.

Width 64, attention embedding dimension 64, four attention heads, FiLM and
prediction heads stay fixed. Each plain/hybrid pair has identical TCN front ends.
Attention budgets derive from the baseline configuration's actual duration and
28 tokens, rounded in proportion to requested history, with at least one token
per layer. Each layer includes its current feature; additional tokens include
its oldest fully contextual sample and then bisect gaps deterministically,
weighted by inverse square root of the layer receptive field. The deepest current
feature remains the single query. A token's age plus its convolution context
never exceeds the realized history. There is no temporal self-attention.

## Launch and resume on the RTX 5070

From the repository root, with the CUDA-enabled environment activated:

```bash
# Generate the plan without starting training (also the default mode).
bash scripts/launch_architecture_search.sh --mode plan

# Small hardware pilot: four optimizer updates per architecture, then
# cached inference/reset measurements at batch 4096. No screening runs.
bash scripts/launch_architecture_search.sh --mode pilot

# Explicitly start or resume the 24-candidate search.
bash scripts/launch_architecture_search.sh --mode run \
  --device cuda:0 --concurrency 1 --micro-batch 2
```

Set `PYTHON=/home/robo/miniconda3/envs/liquid-tcn/bin/python` before the command
if that environment is not activated. `--concurrency N` launches up to N
independent workers on the selected GPU. `--micro-batch N` controls gradient
accumulation independently of the candidate's effective batch of 16 or 32.
Default concurrency is one on the 12 GB RTX 5070. Higher concurrency shares
GPU compute and memory and does not imply proportional throughput gains.
`--threads`, `--log-every`, `--data`, `--output` and `--config` are configurable.
No automatic batch-size, history, attention, precision or LR reductions occur.

Outputs default to `runs/focused_architecture_search`. Previous results under
`runs/architecture_search` and their exported snapshots are preserved. Changed
scientific settings require a new output directory; device, concurrency,
microbatch and logging settings may change on resume. This is a configuration
compatibility check, with no code/data provenance or precision gate.

Checkpoints are saved every 200 optimizer steps, at monitor evaluations and on
orderly interruption. SIGINT/SIGTERM finishes the current optimizer operation
and saves consumed stream positions, recurrent states, prediction tails, RNG and
optimizer state. Restart the same command to resume. Completed stages and workers
are reused. Worker failures stop the coordinator and checkpoint the other active
workers. No wall-time limit is enforced. With concurrency greater than one,
per-trial line-based logs are in `screening/trials/<id>/worker.log` and
`final/trials/<id>/worker.log`; the coordinator reports worker starts/completions.
With one worker, progress appears directly on standard output. Progress records
explicitly label `reference_updates` and `optimizer_steps`.

## Training and selection

Every architecture tests LR `{1e-4, 3e-4}` crossed with effective batch `{16, 32}`
using seed 1: exactly 24 screening trials. Each receives 10000 reference updates,
or 64000000 target-bearing endpoints. The final optimizer batch masks excess
endpoints, giving identical exposure budgets across batch sizes. Microbatch
numerators and valid counts accumulate before the aggregate gradient projection
and optimizer step. LR is not rescaled with batch size.

All candidates retain five-head normalized reconstruction plus the 5 ms and
200 ms physics objectives at their existing weights (1 each before ramping).
The optional extra one-second physics objective is disabled throughout this
search. Train-only normalization is reused. AdamW uses weight decay 1e-4,
conflict-only physics-gradient projection, and gradient clipping at 1. The
cosine schedule has a common 40000-reference horizon and a minimum LR ratio of
0.1, including screening. Physics starts after 500 reference updates and ramps
for 1500; hybrid recurrent residuals then ramp for 1000.

Episodes remain chronological within parallel streams, with up to 400 new
5 ms steps per chunk. Mamba states and previous physics predictions carry across
chunks and detach at chunk boundaries. Episode assignments shuffle. The first
recorded motion input (including velocities and accelerations) and payload
properties supply the same reset backfill in training, evaluation and cached
inference. Synthetic history fills TCN/attention caches without advancing Mamba.
The learned SSM initializer is conditioned on normalized properties and initial
motion, with a zero convolution state. First-chunk losses reach the initializer.
Existing numerical arrays, group splits, masks and reset metadata are reused.

A fixed monitor subset targets 20% of validation endpoints and retains whole
episodes, stratified by robot. Its membership is shared across candidates and
seeds. Monitoring occurs every 1000 reference updates. All screening endpoints
receive full validation. Screening ranks strictly by full-validation normalized
wrench MSE; trial ID breaks exact ties. The best TCN and best hybrid may have
different histories, LRs and effective batches.

Each winner then trains from scratch for seeds 1, 2, 3, in `final/`, up to 40000
reference updates. Existing patience is three successive monitor evaluations
without improvement after the physics ramp. After stopping, full validation
compares the top three monitor checkpoints and stopping checkpoint. Minimum
full-validation reconstruction MSE selects the checkpoint. Test data are evaluated
once per final run's selected checkpoint, after all validation selections finish.
Test reports never influence model or checkpoint selection. A saved test report
is reused on resume, including if the subsequent efficiency benchmark failed.

There are no confirmation, selective tuning, recurrent-only, or long-history
candidate stages in this search. Other model/training modes remain available
through the ordinary training infrastructure.

## Reports

- `candidate_manifest.json`: all 24 resolved candidates, parameters and shared
  settings. A repository copy is in `docs/focused_architecture_candidates.json`.
- `architectures.json`: requested/realized histories, dilations and per-layer ages.
- `screening/trials/<id>/`: trial configuration, monitor and endpoint reports,
  training profile, and resumable checkpoints.
- `decisions.json`: reconstruction rankings and family winners.
- `final/trials/<id>/`: seed runs, shortlist-validation reports, selected checkpoint,
  one test report and cached-inference efficiency results.
- `final_validation_comparison.json` and `final_comparison.json`: validation/test
  three-seed means and sample standard deviations, plus paired collection-group
  bootstrap intervals. Seed variation and bootstrap uncertainty are separate.
- `pilot.json` and `runtime_estimate.json`: short hardware pilots and preliminary
  serial runtime estimates. Startup costs make short-pilot throughput approximate;
  the estimate excludes full endpoint validation/final assessment overhead.

Accuracy is reported overall and for `[0, 0.2)`, `[0.2, 1.24)`, and `[1.24, end]`
seconds. The cumulative first-1.24-second metric is also retained. All ten elapsed
one-second bins use the same cohort with a continuous complete 0–10 second window,
with episode IDs/counts recorded. Short episodes do not enter those bins. Target
validity masks still determine usable labels within each bin.

`force_moment_axis_rms` contains the separate xyz RMS values for force (N) and
moment (Nm). `force_moment_component_rms` pools xyz squared errors per component.
`force_moment_vector_rms` is the RMS 3D error magnitude, equal to sqrt(3) times the
pooled component value. `force_moment_rms` is a compatibility alias for the pooled
component RMS. Normalized wrench MSE combines force/moment after training-only
per-axis standardization. The legacy mixed-unit `rms` field is not a vector RMS.

Existing accumulated impulse and kinematic/linear/angular consistency metrics
are retained, including their validity counts and unavailable horizons. They
explain model behavior but do not determine ranking. Efficiency reports give
parameter count, training peak allocation, cached persistent/peak memory,
mean/p95 step latency and full/partial reset costs at 4096 environments. If this
benchmark fails, the report records the failure without shrinking its batch.

To export completed reports into a standalone review directory:

```bash
python tools/export_search_results.py
```

## RTX 5070 smoke measurements (2026-11-20)

All six four-update pilots and batch-4096 cached benchmarks completed. No raw
recording reprocessing was needed. Detailed measurements are saved in
`docs/focused_architecture_pilot.json` as well as the run directory.

| Model | Train reference updates/s | Train peak MiB | Persistent cache MiB | Cached mean / p95 ms | Full / partial reset ms |
|---|---:|---:|---:|---:|---:|
| TCN 200 ms | 2.185 | 81.00 | 115.77 | 1.146 / 1.187 | 61.78 / 20.61 |
| Hybrid 200 ms | 1.160 | 87.83 | 155.77 | 2.314 / 2.584 | 39.27 / 19.50 |
| TCN 520 ms | 5.747 | 89.29 | 364.78 | 1.726 / 1.863 | 158.29 / 53.25 |
| Hybrid 520 ms | 1.155 | 96.12 | 404.78 | 2.978 / 3.254 | 159.30 / 53.62 |
| TCN 1.24 s | 6.429 | 107.48 | 1016.82 | 3.189 / 3.484 | 730.37 / 145.62 |
| Hybrid 1.24 s | 1.144 | 114.31 | 1056.82 | 4.322 / 4.595 | 735.45 / 146.64 |

Training uses effective batch 16 and microbatch 2 with physics and recurrence
active. Cached timing is per batched step over 4096 environments, with five
warm-up and 20 measured steps; partial reset affects 256 environments. Memory is
PyTorch allocated memory, not the process's full driver/reserved memory. Cached
peak allocation, including reset workspaces, is also in the JSON report.

These very short pilots include first-use effects (especially the first TCN),
so their throughput ordering is not a steady-state model comparison. Using
these rates gives approximately 38 hours for screening and 72–82 hours total if
all six final runs reach 40000 reference updates, before endpoint/full final
validation and assessment overhead. Early stopping can reduce that estimate.
No concurrency speedup has been measured. The full search has not been started.
