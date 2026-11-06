# Executed acceptance checks

These results validate pipeline behavior, not a production-trained surrogate. Only short acceptance training was run. Numerical details and provenance are in [validation_results.json](validation_results.json); source attribution is in [provenance.md](provenance.md).

## Environment and source

The initial checks used Python 3.10.19, PyTorch 2.8.0+cu126, and CPU computation.
Multiprocessing tests require local IPC sockets. Source equivalence was checked
against a read-only snapshot of commit `300d3288b9bce83ccdc31b38bd7926f9484787e7`;
[provenance](provenance.md) identifies the source and snapshot procedure.

The synthetic test suite and its fixtures are included in this repository.
The recorded-data results below describe a historical smoke run: its recordings,
prepared shards, checkpoints, and profiler outputs are not bundled. The compact
results are retained in [validation_results.json](validation_results.json).
Reproducing those exact measurements requires the same recordings and environment.

## Regression and equivalence results

The initial suite passed **34 tests in 9.92 seconds**; the collection-discovery extension below increased this to 39 tests. The suite includes 18 imported discrete regression tests and integration checks covering:

- Static equilibrium, free fall, translating/rotating frames, fixed endpoint moment reference, short/long residual summation, float32 duration tolerance, invalid sets and double-precision gradient checks.
- Stream deduplication/conflicts, repeated-time terminal/reset ordering, asynchronous environment resets, split intervals, gaps, selected/supported masks and incomplete causal histories.
- Training-only normalization, whole-group splits and held-out payload assignment, mass/gravity rejection, chunk corruption/completion checks, cross-shard windows and zero/two-worker equivalence.
- Causality/output alignment, batch/cache equivalence with per-environment resets, all-head and both-endpoint gradients, explicit failure on nonfinite/empty objective sets.
- Exact uninterrupted-versus-resumed weights and metrics with **both zero and two workers**, including consumed sampler progress and TensorBoard step continuation.
- Frozen export round-trip, epoch-derived update budgets, safe launcher dry-run, and an 80-update two-window overfit test using a small configured TCN. The overfit loss fell below 3% of its initial value.

`tools/compare_port.py` compared the imported module against the unmodified baseline snapshot. Random float64 discrete losses, all returned tensors and all five head gradients were bitwise equal. On a recorded 12,603-row attempt, merged rows, per-environment episodes, grid intervals, and eligible windows also matched exactly. Each of two selected episodes had 949 complete histories, 948 short constraints and 909 long windows. Reset-only following episodes contributed none. Bounded preparation independently produced the matching 1,818 long windows.

Synthetic compact data directly exercises the no-native-data path, including checksum/completion verification and split impulse assembly against the source analyzer. Native debug attempts exercise recorded-world-impulse assembly because their original labels use coupling-phase poses rather than endpoint poses.

## Recorded-data smoke run

Three complete **Go1** attempts from the source campaign `liquid_validation_planes10_4_20261103` were used, with separate payload groups:

| Payload case | Split | Eligible windows |
|---|---|---:|
| representative | train | 1,818 |
| viscous | validation | 1,818 |
| high_fill | test | 1,818 |

Preparation retained 7,206 selected grid endpoints, rejected 24 invalid grid endpoints/full intervals and 6 unselected grid records, and assembled 7,200 split intervals from native pieces. There were no unmerged partial pieces or episode-assumption rejected windows. Derived artifacts were written separately from the source recordings.

The **default 249-sample/64-channel TCN** ran six CPU updates, batch size four, with two pretraining updates and a two-update physics ramp. All five heads remained supervised; each full batch reported 160 short and four long constraints. The short/long weights reached one. Validation selected best/latest checkpoints, and the JSON launcher generated config copies, a progress bar and TensorBoard events. The run ended after six updates.

Evaluation covered all 1,818 held-out test windows: 74,538 window-weighted endpoint labels per head, 72,720 short constraints and 1,818 long constraints. The fixed normalized supervised MSE was **1.56450**. This is a smoke-run score, not an accuracy claim.

| Head | Physical-unit RMSE by x / y / z |
|---|---|
| Force (N) | 16.7544 / 26.3321 / 29.3083 |
| Moment (N m) | 5.4510 / 3.6466 / 0.8201 |
| `r_B` (m) | 0.005728 / 0.005149 / 0.016325 |
| `v_B` (m/s) | 0.056970 / 0.053074 / 0.020550 |
| `S_B` (kg m²/s) | 0.020093 / 0.025129 / 0.016012 |

The report also includes peak-wrench errors, within-window temporal lag, maximum errors, dimensional conservation residuals, ground-truth conservation floors and per-robot/payload results. Normalized predicted short RMS `(r,P,H)` was `(0.02867, 0.41922, 0.11213)`; long RMS was `(0.008919, 0.23306, 0.04414)`. Ground-truth long RMS was approximately `(1.02e-4, 7.97e-5, 1.39e-4)`, showing that low source residuals do not imply the barely trained model is accurate.

A frozen export was checked on an actual 289-input window. Its 41 cached 12-value outputs matched batch inference within `7.63e-6` maximum absolute physical-unit difference. Synthetic tests independently exercise per-environment resets and invalid warmup.

## Precision and profiles

Startup checks compare full parameter gradients, not only output values. For a two-window real batch, FP32 versus float64 relative loss error was `1.61e-9` and relative gradient error `5.19e-7`. Optional CPU BF16 autocast passed the configured gate: loss error `7.62e-5`, gradient error `0.00503`, versus a `0.05` limit. Physics remained float64 outside autocast. This does not establish CUDA AMP equivalence.

| Check | Measured result |
|---|---|
| Preparation, three attempts | 6.59 s; peak main RSS 453,912 KiB |
| Loader, zero workers | 1,600 windows in 0.109 s; about 14,642 windows/s |
| Loader, two workers | 1,600 windows in 0.972 s; about 1,646 windows/s, including 0.699 s startup |
| Six training updates | 24 windows in 0.0673 s of training work; about 357 windows/s |
| Training main-process peak RSS | 613,380 KiB; no GPU allocation measured |

These are small warm-cache acceptance profiles. Training throughput excludes startup precision checks, validation and checkpoint serialization; loader benchmarks include worker startup. Main-process RSS includes imported Torch libraries and excludes spawned-worker RSS. Each dataset instance caps its shard cache at eight mappings and each worker prefetches two batches. The profiling tool bounds the sampler and closes persistent workers before exit.

A wheel was built without resolving/installing collection dependencies, installed into a separate `/tmp` target, and imported from outside the repository. The launch script passed Bash syntax validation and the dry-run regression. Git ignore checks cover recordings, prepared data, weights, TensorBoard files, local configs and temporary artifacts.

## Reproduction and remaining scope

The synthetic suite needs no external recordings:

```bash
python -m pytest -q
```

For a smoke run on available recordings, first prepare them using the
[README instructions](../README.md#prepare-data-once). The prepared dataset must
have nonempty train and validation splits. Create a local smoke configuration:

```bash
python - <<'PYCONFIG'
import json
from pathlib import Path
config = json.loads(Path('configs/combined.json').read_text())
config.update(data='data/prepared', output='runs/smoke', resume=None, max_updates=None)
config['training'].update(epochs=None, updates=6, pretrain_updates=2,
    ramp_updates=2, batch_size=4, validation_every=2, checkpoint_every=2,
    validation_batches=2)
Path('configs/smoke.local.json').write_text(json.dumps(config, indent=2) + '\n')
PYCONFIG
GPU=cpu CONFIG=configs/smoke.local.json bash launch_train.sh
liquid-tcn evaluate --data data/prepared --checkpoint runs/smoke/best.pt \
  --split test --output runs/smoke/test.json
liquid-tcn export --checkpoint runs/smoke/best.pt --output exports/liquid.pt
python tools/check_export.py --data data/prepared --export exports/liquid.pt \
  --output artifacts/export_check.json
python tools/profile_loader.py --data data/prepared --workers 2 --output artifacts/loader.json
python tools/check_precision.py --data data/prepared --amp --output artifacts/precision.json
```

Use a fresh run directory. An empty test split produces an inconclusive evaluation.
The [source-equivalence procedure](provenance.md#optional-source-equivalence-check)
is separate from model training. Source hashes and collection-relative attempt
paths in `validation_results.json` identify the historical recordings; they are
not download links or paths expected on another machine.

Not exercised: full model convergence, trained ablation comparisons, real Go2 smoke training, CUDA throughput/AMP/pinned-transfer profiling, large multi-robot campaigns, long-duration generalization, closed-loop stability or HardPACT deployment. Those are not implied by the regression results. The package does not include closed-loop integration. The 200 Hz completed-interval convention is retained at export; it is not a validated next-step controller forecast.

## Collection-root discovery tests

After adding automatic discovery for live collection roots, the full regression suite passed **39 tests in 10.82 seconds**. Five new tests cover recursive candidate discovery, pruning saved-source trees, matching retry/robot groups, stable assignments as recordings are added, incomplete auxiliary streams, completion arriving during checksum verification, corrupt completed recordings, empty collections, conflicting input modes, and the root-based CLI. Automatic preparation produced exactly the same shard/index hashes, normalization, and window counts as manual source selection on the matched fixture. Discovery never starts collection or training and leaves unfinished recordings unchanged.

## One-way PCGrad checks

The PCGrad integration passed the full 59-test suite in 11.35 seconds. A subsequent
legacy-checkpoint compatibility test passed separately (1 test in 0.71 seconds),
bringing the suite to 60 tests. Checks cover conflict, alignment, orthogonality,
zero/tiny primary gradients, both float32 and float64, projection of the weighted
physics sum, global shared-parameter projection, unused parameter ownership,
nonfinite gradients, and invalid epsilon values. The imported helper and reference
fixture have identical function syntax trees to the pinned upstream source.

Synthetic training checks exercise PCGrad and ordinary summed gradients, exact
resume with zero and two workers, TensorBoard projection logs, and the supervised
to physics transition. A checkpoint without gradient-strategy metadata resumes
only under the legacy `sum` setting. The startup precision gate compares merged
update gradients as well as scalar losses.

The recorded-data smoke results earlier in this document used ordinary summed
gradients before PCGrad was added. They do not measure PCGrad convergence or
accuracy. No new recorded-data or full training run was performed for this change.

## Parallel preprocessing

The full suite passed **66 tests in 14.53 seconds** after adding preprocessing
progress and parallel workers. New tests compare serial/two-worker artifact hashes,
normalization, splits, counters, episode identities, and cross-shard loading with
asynchronous resets. They also check disabled progress, invalid worker settings,
and worker failures leaving no completed manifest.

A CPU fixture with four attempts, 1,000 recorded samples per attempt, two
environments, and 256-row shards produced 8,000 grid endpoints. Timings include
verification, preparation, normalization, and output hashing; multiprocessing
runs include spawn/startup costs:

| Version | Processes | Elapsed seconds |
|---|---:|---:|
| Previous implementation | Main process | 2.87 |
| Optimized implementation | Main process | 2.44 |
| Optimized implementation | 2 workers | 2.30 |
| Optimized implementation | 4 workers | 1.88 |

All four outputs had identical shard/index hashes, source identities, normalization,
counts, and series metadata. Peak individual process RSS was about 443 MiB,
including imported libraries; this is not total memory across workers. The small
fixture emphasizes startup costs and is not a million-timestep scaling benchmark.
No complete collection campaign was processed for this change. Measure the actual
collection and filesystem with `tools/profile_prepare.py` before choosing a worker
count; small datasets or bandwidth-limited disks may be faster in serial mode.

## Validation metric optimization and loss logging

The suite passed **75 tests in 20.52 seconds** after vectorizing wrench peak/lag
metrics, reusing batch data for group reports, and adding validation losses.
Tests cover lag and peak equivalence, constant/masked signals, MSE and Huber,
unequal label coverage, partial final batches, TensorBoard loss tags, and resume
compatibility with checkpoints lacking separate validation loader settings.

On a CPU synthetic dataset of 712 validation windows with varying wrench labels,
41 endpoints per window, an 8-channel model, and batch size 128, the complete
validation pass decreased from **0.552 s to 0.185 s** (about 3×). Both versions
were run after warmup without profiler instrumentation. Existing report values
matched within `1e-10` absolute/relative tolerance. A separate profile identified
repeated per-window lag/peak analysis as the dominant previous cost. No full
collection or GPU speedup was measured; real throughput also depends on the
network, storage, batch size, and loader workers.
