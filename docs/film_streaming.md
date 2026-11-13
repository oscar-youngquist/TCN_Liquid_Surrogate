# FiLM conditioning and cached one-step assessment

`model.property_conditioning` accepts `"existing"` (default) or `"film"` for both
TCN architectures. Checkpoints without this field retain the original network
and state-dict layout. [Example FiLM config](../configs/pyramid_film.json) keeps the
40K full-model training settings, fixed 5% monitoring subset, and full-validation
shortlist selection. The existing pyramid config remains FiLM-disabled.

## Conditioning

The existing normalized 24-value liquid/payload vector contains density,
kinematic viscosity, surface-tension coefficient, actual mass and volume,
geometry, fill descriptors and mounting configuration. Robot motion remains in
the separate 15-channel time-varying input. No new statistics are fitted.

FiLM reuses the current property MLP and **supplements** its input concatenation:

```text
e = property_encoder(normalized_properties)
[delta_gamma, beta] = zero_initialized_linear(e)
z_after_block[i] = (1 + delta_gamma[i]) * z_block[i] + beta[i]
```

Each block has channel-wise scale and shift. The conditioned features feed both
the next block and sparse attention memory, with identical placement in sequence
and streaming inference. The final modulation projection starts at zero, so
turning FiLM on initially gives exactly the same outputs and common weights for
the same seed. Zero initialization does not consume RNG used by the backbone or
attention. Training can immediately update the modulation projection. Prediction
heads, normalization, masking and physics losses are unchanged.

The default FiLM projection adds 21,120 parameters: 274,063 total for the TCN,
291,664 for pyramid attention. Its width follows `embedding_dim` and `channels`;
no additional property-encoder hyperparameter is needed. Model construction in
training, sweeps, ablations, resume, evaluation and export uses the shared factory.
Starting FiLM training from scratch requires a new output directory; existing
checkpoints are loaded with their saved conditioning mode.

## Cached inference API and lifecycle

`CachedInference` and `load_export` retain their existing public interfaces.
The fast path registers properties once:

```python
from liquid_tcn.inference import load_export

cache = load_export("runs/pyramid_film/export.pt", environments=4096, device="cuda:1")
cache.set_properties(properties)  # [1,24] shared, or [4096,24] per environment
values, valid = cache.step(motion)  # [4096,15] physical inputs -> [4096,12]
cache.reset([0, 7])               # only these environments rebuild their history
```

For a bank of payloads, use `set_properties(bank, property_indices)` with one
integer bank index per environment. Repeated property rows are encoded once;
shared payloads retain one set of FiLM parameters rather than copying them for
every environment. Changed payloads automatically reset only affected histories.
The compatible `step(motion, properties)` call is still supported. Versioned
input tensors avoid repeated property comparisons. Inference tensors have no
mutation counter, so that form compares values; register them explicitly and
use `step(motion)` for a synchronization-light steady-state path.

Each convolution has an in-place input ring and precomputed chronological tap
indices. Only its kernel taps are gathered into a shared scratch buffer; no
history shifts, `torch.roll`, concatenation, or reconstructed input windows occur.
Pyramid features use bounded per-block rings and a reusable selected-token buffer.
Key/value projections are recomputed only for selected tokens: projected K/V are
not cached because relative-age embeddings must be applied at the current query.
Feature normalization and embedding order remain unchanged.

Caches are inference-only and do not enter autograd. Calls run under
`inference_mode()` and set the model to evaluation. Parameter mutation versions,
parameter replacement, loading new weights, and device/dtype changes invalidate
the caches and restart warmup. Reconstruct the model through `build_model` when
changing its architecture or conditioning mode. Call `invalidate()` explicitly
when changing normalization statistics or editing parameters created inside
`inference_mode`, which lack mutation counters. Do not modify weights through
`.data`, which bypasses PyTorch's mutation tracking.

Reset after gaps and episode boundaries. Until the full 249-sample default history
is available, values are NaN and `valid=False`. `return_all=True` returns all five
heads `[batch,5,3]` for assessment, including `S_B`; the default remains the
12-value force/moment/COM-position/COM-velocity interface.

## Test assessment in both execution modes

The existing selected-checkpoint assessment now runs the original full-sequence
assessment and repeats it with cached one-step predictions. It uses the **same
selected checkpoint**, collection splits, eligible intervals, frozen events and
training-only reference statistics. Streaming test results never enter checkpoint
selection or early stopping.

A chronological streaming pass maintains state within each episode and resets on
gaps/invalid intervals. Each physical endpoint is inferred once. Its bounded mmap
prediction files serve both the overlapping-window metrics and every offline
analysis: accumulated force/moment impulse error, transient response, difficult
conditions and tails, longer-horizon consistency with SPH floors, and approximate
mechanical work transfer. Physical-unit float32 predictions are re-normalized for
window supervision; this representation round trip can add tiny rounding error.
Chronological comparisons also report direct per-head prediction differences.

Outputs beneath an assessment directory:

- Existing `window_test.json`, `summary.json`, prediction caches and per-series
  results remain the full-sequence baseline.
- `streaming/` contains the corresponding one-step metrics, caches, and transient
  traces, with identical coverage and metric definitions.
- `streaming_comparison.json`, `.csv`, and `.md` report numeric differences.
- Twelve-run experiments additionally write `streaming_report/`, including seed
  means/SD, paired ablations, collection-group bootstrap intervals and the same
  horizon, tail and representative-event plots.

Completed full-sequence outputs are reused. Resume computes missing streaming
assessments without retraining or reselecting checkpoints. Checkpoint-selection
and training locks, atomic prediction files, and interruption handling remain in
place. Chronological prediction-cache timing is reported separately from the
window loading/metric pass; it is not a GPU kernel-latency benchmark. Sequential
assessment can take longer than batched full-sequence inference, especially when
processing individual episodes. No full test assessment was launched during
implementation; synthetic and small smoke checks exercise the paths.

The general `evaluate` command defaults to both modes for the test split and full
inference otherwise. `--inference-mode full`, `streaming`, or `both` controls it.
For partial evaluations (`--max-batches`), streaming replays only the requested
windows instead of building a complete chronological cache. Validation selection
continues to use the existing full-inference policy.

## Commands

Train a new FiLM model with the matched 40K settings:

```bash
python -m liquid_tcn train --config configs/pyramid_film.json \
  --data data/prepared_expanded --output runs/pyramid_film --device cuda:1
```

Finish selection if needed, then run both complete held-out assessments:

```bash
bash scripts/evaluate_local_full.sh --run runs/pyramid_film \
  --data data/prepared_expanded --device cuda:1 --threads 2
```

The same command works for `runs/pyramid_full` and the original TCN run. Existing
full assessment reports are reused; missing streaming outputs are added. For only
window-weighted test metrics, use:

```bash
python -m liquid_tcn evaluate --data data/prepared_expanded \
  --checkpoint runs/pyramid_film/best.pt --split test --inference-mode both \
  --device cuda:1 --batch-size 1024 --output runs/pyramid_film/test.json
```

To repeat all twelve-run experiment assessments and aggregation, without training:

```bash
bash scripts/launch_experiment_unity.sh --data /path/to/prepared \
  --output /path/to/existing_experiment --devices 0 --concurrency 12 --threads 2 \
  --resume --assess-only
```

## Equivalent-output batch-4,096 benchmark

```bash
python tools/benchmark_film_streaming.py --device cuda:1 --batch-size 4096 \
  --threads 2 --warmup 8 --repeats 5 --steps 20 \
  --output artifacts/film_streaming_benchmark.json
```

The benchmark compares 12-value **single-endpoint** outputs on identical weights,
GPU inputs, and valid histories. Full history uses 249 samples, unlike the earlier
289-sample/41-output training-window benchmark. Cached paths prefill 249 samples;
setup/prefill times are reported separately. Eight additional warmup steps precede
five repeated 20-step measurements using preallocated CUDA events, with explicit
synchronization outside the multi-step loop. No compilation is used. FiLM-enabled
weights use nonzero modulation to exercise the trained execution path.

FP32 comparisons explicitly disable TF32 only within the benchmark process;
training/evaluation precision settings are unchanged. The first run using PyTorch
defaults is retained in [this raw report](film_streaming_benchmark_cuda1_tf32_default.json):
both prior and optimized caches differed from full-history output by up to
3.4e-5. Under controlled FP32, all paths agree within 3.2e-8, and optimized versus
prior cache outputs match exactly in this synthetic check. These are measurements,
not runtime acceptance gates. The final [raw report](film_streaming_benchmark_cuda1.json)
records GPU/software versions, mean/median/p95, throughput, peak memory, persistent
cache storage, setup/prefill costs and relative speedups. OOMs are surfaced without
reducing the requested batch size.

### Measured batch-4096 results

RTX 4090, PyTorch 2.15.0+cu132, CUDA 13.2; mean milliseconds per batch of one-step outputs. Setup and warmup are excluded.

| Architecture | Conditioning | Full history ms | Prior cache ms | Optimized cache ms | Cache speedup | Optimized peak MiB |
|---|---|---:|---:|---:|---:|---:|
| tcn | existing | 43.454 | 1.758 | 1.085 | 1.62× | 361.7 |
| tcn | film | 49.170 | 1.807 | 1.107 | 1.63× | 361.8 |
| tcn_pyramid_attention | existing | 45.013 | 2.378 | 1.603 | 1.48× | 1240.8 |
| tcn_pyramid_attention | film | 50.687 | 2.434 | 1.630 | 1.49× | 1240.9 |

Peak memory includes live model, inputs and temporary allocations; persistent cache sizes and prefill costs are recorded separately in the raw JSON. These synthetic GPU timings are not full-dataset assessment throughput.

A separate two-window test smoke check using the trained local 40K checkpoint and ordinary runtime settings yielded normalized MSE 0.000784219250 (full) versus 0.000784452441 (streaming), a difference of +2.332e-7. This is a small compatibility check, not evidence of full-test accuracy equivalence. Full paired reports are needed to assess trained-model differences across conditions.
