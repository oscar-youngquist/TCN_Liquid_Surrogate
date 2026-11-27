# Three-seed reset-backfill physics ablation

The launcher reads the actual `runs/pyramid_film_reset_backfill/run_1/config.json`
as the baseline. It creates fresh model/optimizer/sampler states for each of these
12 runs; it never resumes the old `run_1` weights:

| Variant | Short weight | Long weight | Seeds |
|---|---:|---:|---|
| supervised | 0 | 0 | 1, 2, 3 |
| short | 1 | 0 | 1, 2, 3 |
| long | 0 | 1 | 1, 2, 3 |
| full | 1 | 1 | 1, 2, 3 |

The nonrecurrent TCN architecture, FiLM, sparse single-query attention, training
normalization, batch size 1024, 20% startup mix, optimizer/gradient handling,
learning rates 0.001, 500-update pretraining, 1,500-update physics ramp, and
40,000-update maximum are retained. Initialization and shuffled sample ordering
are identical between variants with the same seed. The sampler has its own
seeded consumed-position state, independent of physics-gradient computation.

## Launch on this machine

```bash
cd /home/robo/Research/TCN_Liquid_Surrogate_reset_backfill
conda activate liquid-tcn
bash scripts/launch_reset_backfill_ablation.sh \
  --output runs/reset_backfill_physics_ablation/experiment_1 \
  --device cuda:0 --concurrency 2 --threads 2
```

Resume the same experiment, including interrupted training, checkpoint selection,
and episode evaluation, while skipping completed runs:

```bash
cd /home/robo/Research/TCN_Liquid_Surrogate_reset_backfill
conda activate liquid-tcn
bash scripts/launch_reset_backfill_ablation.sh \
  --output runs/reset_backfill_physics_ablation/experiment_1 \
  --device cuda:0 --concurrency 2 --threads 2 --resume
```

The absolute output directory is
`/home/robo/Research/TCN_Liquid_Surrogate_reset_backfill/runs/reset_backfill_physics_ablation/experiment_1`.
A fresh launch requires a new directory; `--resume` uses the persisted experiment
settings. Device, worker concurrency, and CPU threads can change at resume.
The CLI does not silently lower batch size, change precision, or rescale LR.
No full experiment was launched during implementation.

To inspect the 12 planned configurations without creating runs or training:

```bash
bash scripts/launch_reset_backfill_ablation.sh \
  --output runs/reset_backfill_physics_ablation/experiment_1 --dry-run
```

Paths and evaluation/report settings are in
`configs/reset_backfill_physics_ablation.json`. The default data and reset index
reuse the already prepared arrays and splits; no reprocessing is needed. `PYTHON`
can select an interpreter; the launcher sets `PYTHONPATH` to this worktree.

## Live progress display

From this worktree with `liquid-tcn` activated:

```bash
watch -n 10 'bash scripts/monitor_reset_backfill_ablation.sh --output runs/reset_backfill_physics_ablation/experiment_1'
```

This read-only command displays completed runs, active/pending/failed workers,
per-run training updates and latest logged loss, full-validation/evaluation phase,
completed test episodes, and the validation-selected update. Omit `watch -n 10`
for a single snapshot. It reads bounded log tails and JSON metadata, without
loading checkpoints, importing Torch, or accessing the GPU. Updates may lag
buffered training logs; the maximum-update total is not an ETA because early
stopping can shorten runs.

## Monitoring and selection

The existing monitor/shortlist policy now accepts explicit patience and cadence
parameters, with legacy defaults unchanged. This experiment uses:

- One fixed group-stratified seeded monitor of **20%** of eligible mature
  validation windows, shared by all runs: **127,053 of 635,265** locally.
- Monitor every 1,000 updates; resumable checkpoints every 200 updates.
- **Six stale eligible monitor checks** rather than the previous effective three.
  This changes the actual shortlist policy, not the ignored `training.patience`.
- The same eligibility threshold (`pretrain_updates + ramp_updates = 2,000`) and
  strict improvement criterion `score < best - min_delta`.
- Separate deterministic startup diagnostics at the existing monitor cadence.
- The best three monitor checkpoints plus the final checkpoint receive full
  mature validation; the lowest normalized all-head reconstruction MSE selects
  that run's checkpoint. Ties retain the existing update-order rule.

Monitor size, effective patience, selected update, and stopping reason appear in
worker logs and selection JSON. Test evaluation begins only after that worker's
validation selection succeeds. Test results never choose models/checkpoints.

The existing process coordinator owns the worker slots, state file, run locks,
per-attempt logs, and resume handling. Each worker runs training, selection, and
then evaluation sequentially in its same slot: at concurrency 2 there are at
most two training/evaluation processes total. Ctrl-C/SIGTERM requests a training
checkpoint; evaluation resumes completed episodes. There is no internal wall-time
limit or forced shutdown deadline. Timestamped JSON lines replace progress bars.

## Matched episode evaluation

`evaluation_protocol/` freezes the test episode IDs, native timestamps, valid
head/interval masks, profile cohort, and a single training-derived evaluation
scale. Historical models keep their own checkpoint normalization internally;
all scoring uses the same frozen baseline target scales. Missing normalization
or weights is an error, not a reason to substitute current defaults.

Batched `CachedInference` processes recorded motion and payload properties only.
At each genuine recorded reset it repeats the recorded reset input for `R-1`
steps before processing the first real step exactly once. Caches persist across
all inference chunks and throughout the episode. An input gap invalidates that
environment's context; real history must recover, without reset backfill. Other
environments retain their caches. SPH targets enter scoring only.

Expected predictions at common scoring timestamps must be finite. A failure is
recorded explicitly in `failure.json` and fails the worker; predictions are not
silently dropped. Interrupted inference batches are retried from genuine resets;
completed per-episode reports are reused. Prediction scratch arrays are removed
after their episode is scored.

The report includes all five heads with per-axis and 3D vector RMS, normalized
wrench MSE, and valid counts for whole episodes, 0–200 ms, 200 ms–history, and
after history. The history boundary is `(R-1)*dt`, 1.24 s for this architecture.
One-second profiles use exactly the same cohort of recorded episodes with finite,
continuous inputs reaching 10 seconds; shorter episodes remain in overall and
period results. Bins are left-closed/right-open, with the final bin also including
the endpoint at 10 s. Cohort IDs and exclusions are persisted.

Physics uses the existing discrete Newton–Euler interval equations and frame/
reference conventions. Every eligible sliding 5 ms, 200 ms, 1 s, 2 s, and 5 s
horizon is scored once per ending timestamp. No horizon crosses a missing or
invalid interval, reset, or episode boundary. Predicted kinematic/linear-momentum/
angular-momentum consistency, SPH consistency floors, and predicted-versus-SPH
force/angular impulse errors are separate metrics. Moment impulses use endpoint
rotations and shift to each window's recorded starting origin. Normalization
uses the existing duration × gravitational/container reference scales.

Three physics scopes are explicit: reset-inclusive; startup (window ends before
history); after-history (the entire window starts at or after history). The same
physics scoring runs for supervised-only checkpoints. Unavailable long horizons
have zero counts and null values. Blocks overlap only what is needed for sliding
physics; endpoints/intervals are not duplicated in reductions.

## Reports

- `trials/<variant>-s<seed>/resolved.json`: resolved training settings.
- `trials/<variant>-s<seed>/console/*.log`: per-worker attempt logs.
- `trials/<variant>-s<seed>/run/`: checkpoints, training logs, startup diagnostics,
  and `selection/` full-validation reports.
- `trials/<variant>-s<seed>/evaluation/`: resumable per-episode reports and summary,
  including per-collection-group sums/counts and exclusions.
- `report/per_run_metrics.csv`, `seed_summary.csv`, `paired_bootstrap.csv`,
  `selections.csv`, `summary.json`, and `assessment.md`.
- Three plot sets, each PNG and PDF: `reconstruction_vs_episode_age`,
  `impulse_error_vs_horizon` (200 ms marked), and `startup_vs_mature`.

Three-seed summaries use sample SD (`ddof=1`). Paired full-minus-supervised,
full-minus-short, and full-minus-long confidence intervals resample entire
collection groups with replacement (2,000 seeded draws). Each draw is shared by
both variants and all training seeds. Count-weighted metrics are recomputed for
each seed before averaging paired differences. Groups with no eligible samples
for a metric are explicitly counted. Seed variability and bootstrap uncertainty
conditional on the fixed seeds are reported separately. These are pointwise
percentile intervals, not simultaneous multiple-comparison intervals.

Optional historical references can be supplied on the initial launch:

```bash
bash scripts/launch_reset_backfill_ablation.sh \
  --output runs/reset_backfill_physics_ablation/with_references \
  --device cuda:0 --concurrency 2 --threads 2 \
  --historical-checkpoint /absolute/path/to/selected_old_tcn.pt
```

Repeat the flag for additional already-selected checkpoints, or list paths in the
JSON configuration. References consume the same evaluation worker slots, get the
same masks/backfill/scoring, and appear separately from the three-seed ablations.
If a reference has a longer receptive field, the common post-gap input eligibility
uses the maximum receptive field; the period boundary remains the baseline model's
history for comparability. Historical references are not required to train.

## Minimal verification

Focused tests cover matrix weights and seed-matched initialization/order, fixed
20% monitor and six-check effective patience/eligibility, persistent cached
predictions across chunk boundaries/gaps, all sliding horizons and SPH floors,
missing-prediction failures, bootstrap pairing, and orchestration/selection/resume.
The tiny end-to-end smoke uses four seed-1 two-update CPU runs on synthetic fixture
data, with two concurrent workers. It checks generated JSON/CSV/Markdown/PNG/PDF
outputs and resuming after evaluation but before completion acknowledgement.

```bash
PYTHONPATH=src python -m pytest -q tests/test_physics_ablation.py \
  tests/test_standalone_monitoring.py tests/test_film_streaming.py
```

`--smoke` is explicitly a separate tiny diagnostic mode; the production launch
above does not use it. No production training or production test inference is
performed by these verification tests.
