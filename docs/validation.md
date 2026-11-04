# Executed acceptance checks

These results validate pipeline behavior, not a production-trained surrogate. Only short acceptance training was run. Numerical details and provenance are in [validation_results.json](validation_results.json); the original imported files and PDF identity are in [imports.json](imports.json).

## Environment and source

The reviewed commit `300d3288b9bce83ccdc31b38bd7926f9484787e7` was available in a local checkout of the collection repository. `git archive` extracted its collection modules, docs and tests to `/tmp/liquid-source-300d3288`; that snapshot was made read-only. The source checkout and recordings were not edited. One file was also retrieved from GitHub at the explicit commit to confirm availability. The standalone package contains adapted imports, not dependencies on that checkout.

Acceptance used Python 3.10.19, PyTorch 2.8.0+cu126 and CPU computation. Tests using spawned workers require local IPC sockets; the sandbox blocked them, so those tests and the two-worker profile ran with authorized sandbox escalation. No simulation or collection was launched.

The supplied PDF was read from `hardpact_liquid_wrench_training_specification.pdf`. Its section 6 architecture is implemented. The user's discrete-objective override takes precedence over the document's instantaneous/JVP objectives and deployment suggestions.

## Regression and equivalence results

**34 tests passed in 9.92 seconds.** The suite includes 18 imported discrete regression tests and integration checks covering:

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

Preparation retained 7,206 selected grid endpoints, rejected 24 invalid grid endpoints/full intervals and 6 unselected grid records, and assembled 7,200 split intervals from native pieces. There were no unmerged partial pieces or episode-assumption rejected windows. The source remains unchanged; all derived artifacts are in ignored `artifacts/` directories.

The **default 249-sample/64-channel TCN** ran six CPU updates, batch size four, with two pretraining updates and a two-update physics ramp. All five heads remained supervised; each full batch reported 160 short and four long constraints. The short/long weights reached one. Validation selected best/latest checkpoints, and the JSON launcher generated config copies, a progress bar and TensorBoard events. No full training was launched.

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

These are small warm-cache acceptance profiles. Training throughput excludes startup precision checks, validation and checkpoint serialization; loader benchmarks include worker startup. Main-process RSS includes imported Torch libraries and excludes spawned-worker RSS. Each dataset instance caps its shard cache at eight mappings and each worker prefetches two batches. The initial early-exit profiling tool exposed a worker teardown error; it was corrected to bound the sampler, consume it completely and explicitly close persistent workers. The repeated profile completed successfully.

A wheel was built without resolving/installing collection dependencies, installed into a separate `/tmp` target, and imported from outside the repository. The launch script passed Bash syntax validation and the dry-run regression. Git ignore checks cover recordings, prepared data, weights, TensorBoard files, local configs and temporary artifacts.

## Reproduction and remaining scope

```bash
python -m pytest -q
python tools/compare_port.py --source /tmp/liquid-source-300d3288 \
  --attempt /path/to/representative/attempt-0000 --output artifacts/port-equivalence.json
liquid-tcn prepare --config configs/prepare.local.json --output data/prepared
GPU=cpu CONFIG=configs/smoke.local.json bash launch_train.sh
liquid-tcn evaluate --data data/prepared --checkpoint runs/smoke/best.pt \
  --split test --output runs/smoke/test.json
liquid-tcn export --checkpoint runs/smoke/best.pt --output exports/liquid.pt
python tools/check_export.py --data data/prepared --export exports/liquid.pt \
  --output artifacts/export_check.json
python tools/profile_loader.py --data data/prepared --workers 2 --output artifacts/loader.json
python tools/check_precision.py --data data/prepared --amp --output artifacts/precision.json
```

For the recorded smoke configuration, use the default model, six updates, two pretraining updates, two ramp updates, batch size four, validation/checkpoint interval two and two validation batches. Full evaluation uses all test windows. Local absolute source paths and exact source identities are preserved in the prepared manifest; compact recorded summaries are tracked in `validation_results.json`.

Not exercised: full model convergence, trained ablation comparisons, real Go2 smoke training, CUDA throughput/AMP/pinned-transfer profiling, large multi-robot campaigns, long-duration generalization, closed-loop stability or HardPACT deployment. Those are not implied by the regression results. Closed-loop integration is outside the requested implementation. The 200 Hz completed-interval convention is retained at export; it is not a validated next-step controller forecast.

## Collection-root discovery follow-up

After adding automatic discovery for live collection roots, the full regression suite passed **39 tests in 10.82 seconds**. Five new tests cover recursive candidate discovery, pruning saved-source trees, matching retry/robot groups, stable assignments as recordings are added, incomplete auxiliary streams, completion arriving during checksum verification, corrupt completed recordings, empty collections, conflicting input modes, and the root-based CLI. Automatic preparation produced exactly the same shard/index hashes, normalization, and window counts as manual source selection on the matched fixture. Discovery never starts collection or training and leaves unfinished recordings unchanged.
