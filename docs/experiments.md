# Selected-model experiment on an expanded dataset

This runner trains twelve independent models: full, supervised-only, short-only
and long-only, each with seeds 1, 2 and 3. It uses the existing trainer,
`StageValidation` policy, parallel sweep coordinator and full evaluator. It does
not repeat the staged hyperparameter search or train reduced-data models.

## Start, resume, and assess

Inside an existing Unity allocation, activate the workspace environment and
install the updated package, including Matplotlib for report plots:

```bash
conda activate /scratch3/workspace/oyoungquist_umass_edu-liquid-tcn/env
python -m pip install -e .

DATA=/scratch3/workspace/oyoungquist_umass_edu-liquid-tcn/data/prepared_expanded
OUTPUT=/scratch3/workspace/oyoungquist_umass_edu-liquid-tcn/runs/expanded_ablation_experiment

bash scripts/launch_experiment_unity.sh \
  --data "$DATA" --output "$OUTPUT" \
  --devices 0 --concurrency 12 --threads 2 --max-updates 20000
```

Use the same output directory in the next allocation:

```bash
bash scripts/launch_experiment_unity.sh \
  --data "$DATA" --output "$OUTPUT" \
  --devices 0 --concurrency 12 --threads 2 --resume
```

To continue assessments without training, or regenerate the final tables/plots
from completed assessments:

```bash
bash scripts/launch_experiment_unity.sh \
  --data "$DATA" --output "$OUTPUT" \
  --devices 0 --concurrency 12 --threads 2 --assess-only
```

`--assess-only` requires all twelve completed validation selections. It reuses
completed window reports, trajectory caches and per-series metric files. It does
not select a new checkpoint. Use a new experiment directory for a new scientific
comparison. Completed trials remain complete even if a larger `--max-updates` is
supplied on resume; that override only affects unfinished candidates.

The launcher accepts optional leading `--python /path/to/python`, uses unbuffered
output and `exec`, and leaves `CUDA_VISIBLE_DEVICES` unchanged. `--devices 0,1`
distributes workers across logical devices; `--devices cpu` supports development.
Concurrency and threads are honored without capacity estimates, preflight runs,
precision checks or provenance requirements. Slurm controls wall time. Locks,
atomic checkpoint/report writes, signal forwarding, worker-exit detection and
plain timestamped coordinator heartbeats remain in use. Individual console logs
are under `trials/<variant>-s<seed>/console/`.

`--dry-run` prints twelve resolved configs without starting work. `--config`
accepts a selected run's resolved JSON to retain its architecture/optimizer
settings. Without it, the packaged sweep baseline supplies those settings.
`--block-rows` defaults to 2048 for bounded chronological inference/analysis.
Change it to tune memory; it does not change interval eligibility or horizons.
The dataset manifest supplies window counts, splits, properties and normalization.
No collection group is reassigned. Do not replace the dataset midway through an
experiment if the goal is a matched scientific comparison.

## Training and checkpoint selection

All variants use MSE, physics-phase LR 0.001, 500 pretraining updates, a 1,500-update
ramp, and enabled physics coefficients of 1. Full enables both coefficients,
supervised sets both to zero, short zeros long, and long zeros short. All other
settings, initialization seeds, sampler seeds and LR schedules match. The default
maximum is 20,000 updates; `--max-updates` sets the common scheduler horizon.

The existing runtime policy writes recovery checkpoints every 200 updates,
monitors one persisted stratified 5% validation subset every 1,000, and applies
patience 3 after update 2,000 for every variant. Monitoring includes diagnostic
physics objectives, but normalized supervised MSE alone controls patience.
The three lowest subset-scoring checkpoints plus the stopping checkpoint are
retained, deduplicated by update. After stopping, full validation chooses the
lowest MSE among this shortlist. Full-validation results are cached by update.

The coordinator waits for all twelve validation selections and records
`selected.json` before dispatching any test assessment. Test errors never influence
training, patience, selection or promotion. `assessment/window_test.json` preserves
the existing full-test assessment with **overlapping-window weighting**. These
metrics and their seed summaries are separate from chronological results.

## Chronological eligibility and caches

`trajectories.py` reads prepared mmap shards directly. Each series retains its
collection group, attempt/source, environment, episode and native tick. These
identities prevent unrelated retries or environments from being collapsed merely
because they have the same timestamp. No overlapping training windows are expanded.

A valid chronological segment requires the recorded complete causal-history mask,
state and interval validity, all five target masks, finite inputs/labels and 5 ms
intervals. Unsupported physical-assumption series are excluded. Episode changes,
native-tick gaps and invalid intervals break segments. The first endpoint anchors
a segment; integration begins with the following completed interval. New history
must be built after a reset/gap. Nonfinite model predictions fail assessment and
are never removed to improve a model's score.

Older manifests do not retain native `dt`. For these, physical timestamps use the
exact clock ratio `h / delta_native_tick` from two consecutive history-certified
endpoints. If no such pair exists, that series has no eligible trajectory and its
clock is unavailable. No acceleration is reconstructed from future observations.

Inference overlaps only RF-minus-one inputs at block boundaries. Contextual
outputs are disjoint and cached in atomic per-series NPY files. Labels and recorded
motion remain in the prepared dataset. Analyses reuse these predictions. A killed
worker may recompute its unfinished series, but completed prediction and metric
files are reused. Caches are tied operationally to the selected model; they are not
automatically hash-verified or invalidated after manual file changes.

## Analyses and definitions

**Accumulated disturbance.** Every eligible ending timestamp contributes one
rolling horizon of 50, 200, 500, 1,000 or 2,000 ms when enough uninterrupted history
exists. Sum interval-average force times actual duration after endpoint rotation
to world. Shift endpoint-origin moments to a fixed world reference before summing;
reported horizon moments use the recorded base origin at the horizon's start.
RMS vector errors, signed component biases and tail estimates are reported in
N s and N m s. Horizons themselves overlap, but no interval is duplicated within
a horizon, and overlapping horizons are not treated as independent bootstrap units.

**Events.** A bounded uniform sample of eligible **training** intervals sets the
95th percentile of held linear/angular acceleration changes, 10th speed percentile
for downward stop crossings, 95th turn-rate percentile, and 99th SPH force/moment
magnitude percentiles. All detections use strict exceedances except stop crossings.
A turn means high recorded angular speed, not an inferred maneuver annotation.
Changes combine the six recorded held acceleration channels in their recorded
units; the threshold is empirical, not a dimensionless physical acceleration.

Detection windows extend 200 ms before and 400 ms after a trigger and merge when
overlapping. For merged events, pre ends at the first trigger, during ends 200 ms
after the last trigger, and post uses the remaining 200 ms. Segment boundaries
clip these windows. Peak errors/timing search the full merged event, with signed
prediction-minus-SPH values. Impulse errors integrate the merged event at a fixed
reference. Settling means the first sustained 100 ms of post-event wrench error
below 5% of training's 95th-percentile wrench magnitude, separately for force and
moment. Missing/clipped/nonsettling observations are censored and counted, not
reported as zero. `shared_assessment/events.sqlite` freezes all event intervals
before model assessment. The representative plot uses the first event in the
lowest series ID for seed 1, for every variant, regardless of prediction error.

**Conditions and tails.** Wrench errors are broken down by robot, recorded payload
identity, individual numeric payload properties, held-motion intensity and training
reference sparsity. Property bins use training-sample thirds; intensity uses
training median/90th-percentile bins. Sparsity is Euclidean distance in 39
training-standardized motion/property features to the nearest of up to 1,024
sampled training references. Its bins use training-reference leave-self-out
median/90th percentiles. It is an approximate reference-density descriptor, not a
claim that held-out episodes are extrapolation. Genesis viscosity retains its
kinematic m²/s convention. All numeric property names/values remain in per-series
reports and the shared reference. Counts and collection-group counts accompany
condition summaries. Missing labels are reported as unavailable.

**Longer physical consistency.** The existing differentiable discrete formulation
supplies adjacent kinematic, linear momentum and angular momentum residuals. Their
sums telescope over 500 ms, 1 s and 2 s. Angular residuals are shifted consistently
to each horizon's fixed initial base origin. Raw units are m, N s and N m s;
normalization uses duration times gravitational speed, weight and weight times
container diagonal. Identical calculations on SPH targets give residual floors.
The paired full-minus-short results and consistency plots explicitly compare the
full and short-only models. These are offline diagnostics, not new training losses.

**Mechanical work.** Endpoint quadrature uses
`power = F_W · u_O_W + moment_B · omega_B` and `work = sum(h * power)` for both
predictions and SPH. Moments are about the base origin associated with those
velocities. Reports include signed bias, absolute/RMS errors and horizon curves.
This is an **interval-data work approximation**, not exact fluid-energy
conservation or a passivity guarantee.

## Statistics, outputs and limitations

Counts, sums and squared sums use all eligible data. Median, p95, p99 and worst-1%
mean estimates use bounded uniform priority reservoirs of at most 2,048 values per
metric. Their sampling method/counts are explicit; extreme-tail precision is limited.
Inference and analysis arrays are bounded by block size plus at most 400 history
endpoints; one series' intermediate statistics are committed at a time. Persistent
cache size grows with data and the twelve models. Group-level moments and report
metadata grow with the number of collection groups/categories, not the number of
overlapping windows. Shared feature sampling is capped at 4,096 intervals.

`report.json` contains individual results, mean ± sample SD across three seeds,
paired full-minus-ablation comparisons and 95% bootstrap intervals for key RMS
errors. The bootstrap jointly resamples whole collection groups across models and
seeds, recomputes count-weighted RMS, then averages paired seed differences. It
conditions on the three trained seeds; fewer than two groups yields an unavailable
interval. No window-level IID bootstrap is used.

Outputs include `metrics.csv`, `conditions.csv`, `report.md`, and `plots/` with
horizon, consistency, tail and representative transient figures. Per-series files
retain episode identity, physical properties, exclusion counts, event censoring
and group contributions. No continuous 2 s segment means the 2 s analysis is
unavailable, not a zero residual. The chronological test report is cached at
`trials/<trial>/assessment/summary.json`.

The tiny synthetic smoke test covers unique chronological outputs versus batch
inference, constant-wrench impulse/work and moment transport, reset boundaries,
event processing, cache reuse and resumed dispatch that skips completed trials.
No production training or GPU benchmark is launched by the test.

Local verification: the synthetic CPU smoke test passed, including a two-update
training/shortlist selection, chronological cached/batch agreement, reset splits,
event analysis, constant impulse/work and moment-reference calculations, report
aggregation, paired group bootstrap, and resumed dispatch. CLI dry-run and Bash
syntax checks passed. GPU execution and plot rendering were not exercised in the
local environment; Matplotlib is installed by the package dependency update.
