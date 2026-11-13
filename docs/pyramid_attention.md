# Sparse temporal-pyramid attention

Set `model.architecture` to `tcn_pyramid_attention` to add attention between the
existing residual backbone and prediction heads. `tcn` remains the default.
[Example configuration](../configs/pyramid_attention.json) includes every attention
setting and uses MSE, PCGrad, 500 pretraining updates, a 1,500-update physics ramp,
physics weights of 1, and a 40,000-update maximum/scheduler horizon. Losses, physical conventions, normalization, checkpoint
cadence and selection policies are independent of the architecture choice. The example
uses the same training settings as the 40K full-model experiment and enables its
fixed-subset monitoring/shortlist policy for standalone training.

## Features and history

The current backbone has two causal convolutions per residual block. Its block
receptive fields are `[9,25,57,121,249]` samples. The default memory is:

| Block | Feature ages (ms) | Tokens |
|---|---|---:|
| 1 | 0, 100, 200, 300, 400, 500, 600, 800, 1000, 1200 | 10 |
| 2 | 0, 100, 200, 300, 400, 600, 800, 1000 | 8 |
| 3 | 0, 200, 400, 600, 800 | 5 |
| 4 | 0, 200, 400, 600 | 4 |
| 5 | 0 | 1 |

Ages must be nonnegative integer multiples of `1000 * sample_seconds`.
For other block counts, supply one age list per block; an empty list omits that
block from memory. The deepest current feature always supplies the query.

```text
block_rf[i] = 1 + 2 * (kernel_size - 1) * sum(dilations[:i+1])
required_history = max(deepest_rf, max(lag_steps + block_rf[block]))
input_length = required_history + long_intervals
```

The default needs **249 real samples**, spanning 1.24 seconds between endpoints.
The 1,200 ms first-block token needs 240 + 9 = 249 samples. A 289-sample training
pass still yields 41 contextual predictions for the 40 physics intervals.
**Existing default prepared data can be reused.** Changing feature width, heads,
or backend does not require new preprocessing.

If a custom schedule changes history length or the interval horizon, prepare a
separate dataset with the same `model` object in its preparation config. The
preprocessor derives history masks and compact window indices from those lengths;
the loader uses the same calculation. Loading mismatched window dimensions gives
an actionable error rather than truncating history or treating padding as data.
Production preparation retains the recorded 5 ms cadence; `sample_seconds` also
supports other cadences for model-only applications. Keep the original collection
groups and split assignments when preparing another window geometry.

## Attention calculation

For endpoint `t`, gather each selected block feature at `t - lag`. Gathered memory
has shape `[batch, prediction_times, 28, channels]`. The backbone is computed once
for the whole sequence, including the 41 shared physics endpoints. No independent
backbone pass is made per prediction time.

Query and memory features receive feature-wise LayerNorm. Learned block and age
embeddings are added to the normalized memory. The query projection is separate;
key and value projections share one fused linear operation. Each endpoint has
only its own historical memory:

```text
A = softmax(Q K^T / sqrt(head_dim)) V
h_t = z_t_deepest + alpha * output_projection(dropout(concat_heads(A)))
```

`alpha` is a learned scalar initialized to 0.1. Prediction heads remain the
existing `64 → 64 → 64 → 3` MLPs. The default attention dimension is the backbone
width, with four heads when divisible by four (otherwise one), no projection
bias, and zero dropout. LayerNorm never mixes times or environments.

[`pyramid.py`](../src/liquid_tcn/pyramid.py) adapts the explicit scaled dot product
and multi-head implementation by
[Sanjaya Subedi](https://sanjayasubedi.com.np/deeplearning/multihead-attention-from-scratch/).
The custom backend remains the default. `attention_backend="sdpa"` selects
functional PyTorch SDPA with identical projections and no causal triangle mask:
every gathered token is already at or before the query. Dropout acts on combined
output features before the output projection in both implementations, and is off
in evaluation. Diagnostic `return_attention=True` computes explicit per-head
weights without averaging; it uses the custom calculation even for an SDPA model.

## Loading and streaming

`build_model` is the common factory for training, resume, checkpoint selection,
evaluation, chronological assessment and export loading. Checkpoints serialize
the architecture, resolved age schedule and attention settings. Legacy checkpoints
without these fields reconstruct the original TCN with identical state-dict names
and shapes. Switching an old TCN checkpoint to an attention architecture is not a
resume operation; train the new architecture in a new output directory.

`CachedInference` retains convolution caches and one feature ring per selected
block, with `max(lag_steps)+1` slots. Each environment has independent reset and
warmup validity. `step` returns NaNs and `valid=False` until the complete history
has been observed. Reset on episode changes, input gaps or payload changes.
After warmup, predictions agree with full-context inference. These caches are
inference-only (`no_grad`); training gathers differentiable sequence features.

## Commands

Train using the existing default prepared geometry (up to 40,000 updates):

```bash
python -m liquid_tcn train --config configs/pyramid_attention.json \
  --data data/prepared_expanded --output runs/pyramid_full --device cuda:1
```

The example enables `monitoring.policy="shortlist"`: recovery checkpoints every
200 updates, one fixed representative 5% validation subset every 1,000 updates,
and patience 3 after update 2,000. The training configuration's `patience=40001`
disables the ordinary full-validation stopping counter; the shortlist policy
owns the three-check patience rule. Maximum updates and the cosine schedule
horizon are 40,000. Early stopping may finish sooner.

The subset is persisted at `runs/pyramid_full/monitor_subset.json`. Seed 1729
reproduces the experiment's subset on the same prepared dataset (31,763 identical
indices were verified against the local full-model run). Set `monitoring.subset`
to an existing subset JSON to copy its indices explicitly instead. Resume keeps
the local indices and checkpointed monitoring state.

The three best subset checkpoints plus the stopping checkpoint are compared on
full validation automatically after training finishes. Results live under
`runs/pyramid_full/selection/`; the selected checkpoint is also copied to
`runs/pyramid_full/best.pt`. No test data enters this selection. Resume with:

```bash
python -m liquid_tcn train --config configs/pyramid_attention.json \
  --data data/prepared_expanded --output runs/pyramid_full --device cuda:1 \
  --resume runs/pyramid_full/latest.pt
```

All optimizer, loss, precision, loader and logging settings match the saved local
40K full-model configuration, including two CPU threads and plain logging.
Configurations without `monitoring` retain the ordinary standalone behavior.

Evaluate the checkpoint selected on validation, then export it:

```bash
python -m liquid_tcn evaluate --data data/prepared_expanded \
  --checkpoint runs/pyramid_full/best.pt --split test \
  --device cuda:1 --batch-size 1024 --output runs/pyramid_full/test.json
python -m liquid_tcn export --checkpoint runs/pyramid_full/best.pt \
  --output runs/pyramid_full/export.pt
```

For the existing twelve-run experiment with fixed 5% monitoring and full-validation
shortlist selection, pass `--config configs/pyramid_attention.json` to
`scripts/launch_experiment_unity.sh`; all variants and seeds inherit the architecture.
The legacy sweep, staged sweep and concurrency benchmark accept
`--model-config configs/pyramid_attention.json`, applying only the model section.
Their trial grids, budgets, physics settings and selection rules remain unchanged.
Resume uses saved model settings; omit `--model-config` when resuming a sweep.
For completed runs with a monitoring shortlist, `scripts/evaluate_local_full.sh`
uses the same checkpoint reconstruction and assessment path.

## Verification and measured cost

Focused tests cover the explicit attention reference and SDPA gradients, multiple
query lengths, causality, finite gradients through all model parameters, history
and reset boundaries, selective cached resets, legacy state loading, and exact
small-run checkpoint resume followed by held-out evaluation/export loading.

The benchmark uses synthetic normalized inputs, FP32, synchronized CUDA boundaries,
three warmup iterations and ten measured iterations. Training timing includes
forward, squared-output scalar loss, backward and gradient clearing. It excludes
physics, PCGrad, optimizer, loading and logging, so it is **network cost**, not
end-to-end training throughput. Cached inference includes normalization and
validity checks after full history warmup. No `torch.compile` was used.

Measured on logical `cuda:1`, NVIDIA RTX 4090, PyTorch 2.15.0+cu132, two CPU threads:

| Model/backend | Parameters | Batch 1024 forward/backward (ms) | Windows/s | Peak allocated (MiB) | Cached 1 env median (ms) | Cached 64 env median (ms) |
|---|---:|---:|---:|---:|---:|---:|
| TCN | 252,943 | 36.72 | 27,890 | 1,982.0 | 0.610 | 0.724 |
| Pyramid/custom | 270,544 | 55.88 | 18,324 | 3,742.7 | 1.140 | 1.168 |
| Pyramid/SDPA | 270,544 | 71.33 | 14,357 | 6,097.4 | 1.009 | 1.116 |

Custom attention adds 17,601 parameters (+6.96%) and about 52% network training
time at batch 1024 in this short measurement. Its cached single-environment
latency is about 1.87× baseline. Results depend on batch size, kernels and hardware;
they are not a runtime acceptance criterion. The [raw measurements](pyramid_benchmark_cuda1.json)
include batches 16/64/1024, inference batches 1/16/64, latency percentiles and peak
allocated/reserved memory. Repeat on the deployment device with:

```bash
python tools/benchmark_pyramid.py --device cuda:1 --threads 2 \
  --train-batches 16 64 1024 --inference-batches 1 16 64 \
  --warmup 3 --iterations 10 --output artifacts/pyramid_benchmark.json
```

CPU is supported via `--device cpu`; CPU peak tensor memory is reported as
unavailable. No full training run or sweep was launched for these checks.

### Forward-only batch 4,096

A separate RTX 4090 (`cuda:1`) run used five warmups and twenty synchronized
measurements, FP32, `eval()` and `torch.inference_mode()`, without compilation.
Inputs were already on the GPU. No backward pass or optimizer ran.

| Model/backend | Full window median (ms) | Windows/s | Full window peak allocated (GiB) | Cached step median (ms) | Cached peak allocated (GiB) |
|---|---:|---:|---:|---:|---:|
| TCN | 53.838 | 76,067 | 1.263 | 1.808 | 0.328 |
| Pyramid/custom | 83.953 | 48,784 | 6.156 | 2.396 | 1.124 |
| Pyramid/SDPA | 97.940 | 41,822 | 5.005 | 2.796 | 1.096 |

Full-window timing processes 4,096 sequences of 289 samples and returns 41
contextual endpoints per sequence. Cached timing processes **one new sample for
each of 4,096 environments**, after 249 samples of history warmup. These are batch
latencies, not per-environment latencies. Cached timing includes normalization
and validity checks; full-window timing starts with normalized tensors.
The measurements exclude data loading and CPU-to-GPU transfer. Peak allocation
includes model, inputs, and active caches/workspace. Unused allocator blocks from
the full-window phase are cleared before measuring cached inference.

[Raw measurements](pyramid_benchmark_cuda1_batch4096.json) include mean, median,
p95, throughput and allocated/reserved memory. Reproduce with:

```bash
python tools/benchmark_pyramid.py --device cuda:1 --threads 2 \
  --skip-training --forward-batches 4096 --inference-batches 4096 \
  --warmup 5 --iterations 20 \
  --output artifacts/pyramid_batch4096.json
```
