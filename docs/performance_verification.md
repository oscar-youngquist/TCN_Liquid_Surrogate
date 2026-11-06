# Performance and verification record

Checked on 2026-11-06 using Python 3.11.18, PyTorch 2.15.0+cu132 and NumPy 2.4.6,
with CUDA hidden. These are small synthetic CPU checks, not Unity GPU capacity or
48-hour completion measurements. Raw results are in [performance_cpu.json](performance_cpu.json).

## Before/after comparison

The local baseline was commit `25f8382`. The comparison loads that commit's loader,
trainer, gradient combiner and scalar-reporting code. Both sides use the same
current model, discrete physics, normalization and synthetic inputs. This isolates
the performance edits and preserves the existing working-tree model architecture.

| Measurement | Before | After |
|---|---:|---:|
| Loader windows/s, fastest of four passes | 9,067 | 21,592 |
| 60-update training wall time | 0.575 s | 0.485 s |
| Loader time during training | 0.146 s | 0.102 s |
| Training compute section | 0.322 s | 0.330 s |
| Full validation | 0.034 s | 0.026 s |
| Peak process RSS | 750,172 KiB | 750,684 KiB |

The loader test uses 724 windows with the default 289-input geometry, batch size
64, 71-row shards and zero workers. It performs four passes over the same small
dataset, so the result mostly measures CPU copying/collation with warm OS caches.
The roughly 2.38× gain does not establish disk throughput on a remote filesystem.

The training test uses the small two-block fixture, batch size 8, one CPU thread,
10 pretraining updates and a 20-update physics ramp, then continues through update
60 with one-way PCGrad. The wall section includes loading, training, logging,
validation and checkpointing, but excludes initialization and precision checks.
It was about 16% shorter in this single short run. The compute section itself was
slightly slower, illustrating why subsecond timings should not be extrapolated
to production. Model weights, optimizer state, scheduler and sampler matched
bitwise/exactly at update 60. RSS is a sequential same-process high-water mark;
it is not evidence of independent process memory savings.

The optimized run additionally measured 0.00068 s in transfer dispatch, 0.0102 s
in logging and 0.0119 s in checkpointing. The old profile did not separate those
categories; absent values are not reported as zeros.

Reproduce from a checkout containing the baseline commit and test fixtures:

```bash
CUDA_VISIBLE_DEVICES='' python tools/compare_pipeline.py \
  --baseline 25f8382 --output /tmp/liquid_pipeline_comparison.json
```

This command creates and removes temporary synthetic data, runs short CPU tests,
and writes the requested JSON result. It does not launch the staged search.

## Regression coverage

The available suite passed 112 tests with two existing spawned-loader cases
excluded from the sandbox run. Those two cases passed separately outside the
sandbox, for 114 tests total. They cover shard-crossing multiworker loading and
exact checkpoint resume with two loader workers and PCGrad. In this environment,
those cases can stall on sandbox multiprocessing IPC.

The new staged tests include:

- Exact field equality for reordered, repeated and shard-crossing windows;
  invalidation of dataset verification receipts after a file changes.
- Exact model, optimizer, scheduler and validation parity with ordinary per-sample
  collation; the existing suite also checks sampler resume and nonfinite failures.
- Dependent candidate inheritance, eight distinct refinements at LR boundaries,
  equal-budget promotion, matched seeds and unchanged LR-related settings.
- CPU/RAM/GPU-cap arithmetic, scheduling waves, original-deadline preservation,
  bounded forced cleanup and per-device process limits with mock workers.
- A real tiny CPU six-stage lifecycle with reduced test-only budgets, twelve frozen
  final selections, all twelve held-out test evaluations and completed-run resume.
- Rejection of test evaluation before freezing or after selection-file mutation.

Existing benchmark tests cover warmup/measurement barriers, shared-interval
throughput arithmetic, startup memory retention, failure/timeout handling,
ordinary-training parity, final-model validation and worker shutdown.

Bash syntax checks, a quoted-path staged-launcher dry run, `git diff --check`, and
an installable wheel build passed. The wheel includes the new Python modules and
packaged baseline. No production sweep, scheduler submission, full training run,
or production GPU concurrency benchmark was launched.

GPU timing, twelve-process fit, long-run allocator growth, shared-filesystem I/O,
optional AMP and actual Unity allocation deadlines were not measured locally.
The launch-time preflight checks the requested allocation and stops on failure;
a CPU test is not a substitute for those measurements.
