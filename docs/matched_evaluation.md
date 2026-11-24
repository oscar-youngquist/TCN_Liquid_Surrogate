# Matched sequential evaluation

Run from the repository root in the CUDA-enabled `liquid-tcn` environment:

```bash
bash scripts/launch_matched_evaluation.sh \
  --legacy-zip runs/pyramid_film_full_selected_checkpoints.zip \
  --older-dir runs/imported_pyramid_film_selected \
  --focused-root runs/focused_architecture_search \
  --data data/prepared_mamba \
  --output runs/matched_sequential_evaluation \
  --device cuda:0 --batch-size 64 --chunk-size 400
```

Set `PYTHON=/home/robo/miniconda3/envs/liquid-tcn/bin/python` before the command
if the environment is not activated. `--smoke-only` performs only the small
cached-versus-causal/chronological check. `--threads` and `--bootstrap` are
configurable. Repeating the command resumes completed episode evaluations.
An unfinished episode restarts from its real recorded reset. There is no training,
checkpoint selection, wall-time limit, provenance audit, or precision gate.

Extraction validates ZIP member paths/types and installs an entirely new directory
atomically. Existing runs and checkpoints are never written. All old `best.pt`
files with adjacent validation-selection reports are evaluated. New checkpoints
come from `decisions.json`'s finalists and their existing `best.pt` files.
Strict model state-dictionary loading and checkpoint-specific motion/property/
output normalization are used. Missing normalization/weights raise a concrete
loading error rather than using current defaults. The original format-1 contract
is motion `[g_B,u_O_B,omega_B,held_a_B,held_alpha_B]`, 24 named properties from
`metadata.PROPERTY_FIELDS`, and heads `[force,moment,r_B,v_B,S_B]`.

The common cohort is the focused search's test split. A valid recorded reset is
required. Every episode uses the full recorded-motion reset prefill, including
linear/angular velocity and held accelerations. Synthetic history fills only
TCN/attention caches; Mamba starts from its saved learned initializer with zero
convolution state. All caches and recurrent states persist across 400-step I/O
chunks. Predictions consume only recorded motion and constant payload properties.
Targets enter scoring only. Finished batch slots are never scored or reused.

Input gaps/nonfinite motion end the common evaluable prefix and are reported;
a mid-episode gap is never treated as a reset. State/target/interval masks are
shared and saved alongside exact native ticks and times. Force/moment scoring
requires a valid completed interval and state; the three state heads require
valid state/target data. Prediction failures never shrink these masks. Affected
aggregate scores become unavailable and missing predictions are explicitly counted.

Reports contain unique-timestamp reset-inclusive, first-200-ms, 200-ms-to-1.24-s,
and post-1.24-s reconstruction. All five heads have physical per-axis RMS and
3D vector RMS. Wrench MSE uses the focused prepared training-only target standard
deviations for *every* checkpoint, independent of its internal normalization.
The ten elapsed bins all use the same episodes reaching a continuous 10 seconds.
They are half-open `[i,i+1)` intervals. The exact endpoint at 10 seconds still
contributes to overall/post-1.24-second scoring.

The old eligibility diagnostic reproduces the original `SeriesWriter` RF+40
valid-window rule. `legacy_eligible` scores its unique endpoint union;
`legacy_window_weighted` uses each endpoint's original window multiplicity.
Both shared-scale wrench MSE and checkpoint-own-scale all-five-head MSE are
reported. The old archived selection score was a *validation* all-head score,
so this diagnostic is not an exact re-evaluation of that validation population.

Physics reuses `offline.horizon_errors` and the implemented discrete constraints
at 5 ms, 200 ms, 1 s, 2 s and 5 s. Every horizon requires uninterrupted valid
recorded intervals and states. Chunks overlap raw/scoring context by up to
1000 steps; only new ending timestamps are counted. Angular impulses/residuals
are shifted to each horizon's fixed starting base origin. State consistency
residuals, SPH consistency floor, and prediction impulse errors against SPH are
reported separately, in physical and normalized per-axis/vector RMS. The
post-1.24-second physics view requires the *whole* horizon to start after 1.24 s.
Unsupported horizons have zero counts and unavailable metrics.

Outputs:

- `assessment.md`: compact tables and interpretation, including startup/late error.
- `comparison.json/csv`: mean and sample seed SD for all reported metrics.
- `bootstrap.json/csv`: paired collection-group differences and 95% intervals.
- `reconstruction_vs_time.png/pdf`, `physics_vs_horizon.png/pdf`: compact plots.
- `<family>_seed<N>/report.json/csv`: per-checkpoint/per-seed aggregate results.
- `<family>_seed<N>/episodes/*.json`: resumable per-episode sufficient statistics.
- `<family>_seed<N>/checkpoint.json`: architecture, normalization, ordering and
  selected step loaded from that checkpoint (no copied weights).
- `common/cohort.json` and `common/series-*.npz`: exact common cohort and masks.
- `smoke.json`: cache/causal agreement, state carry and timestamp-alignment check.

The bootstrap samples entire collection groups with replacement, preserving all
robots/environments/episodes and fixed trained seeds within each sampled group.
All models use the same resampled groups. Squared-error totals and eligible counts
are pooled before calculating each metric and model difference. Percentile
2.5/97.5 limits use 2000 draws by default and seed 1729. These intervals quantify
held-out-group variability conditional on the trained seeds; seed SD is separate.
