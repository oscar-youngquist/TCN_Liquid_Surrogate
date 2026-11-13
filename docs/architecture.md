# Model architecture and tensor conventions

The TCN estimates liquid-on-robot reaction wrenches over completed 5 ms intervals
and liquid state at each interval endpoint. Inputs and predictions are sampled at
200 Hz. Training combines recorded labels with discrete Newton–Euler constraints.
Supervision is the primary objective for [one-way PCGrad](liquid_discrete_constraints.md#gradient-combination);
the weighted sum of physics losses is the secondary objective.
The [physics reference](liquid_discrete_constraints.md) defines those constraints;
[configuration](configuration.md) lists every configurable value.

## Coordinate frames and motion inputs

`W` is the inertial world frame. `B` is the robot base frame, with origin `O`.
`R` maps base vectors to world vectors. Orientations use wxyz quaternions.
All inputs use SI units unless a raw simulator coefficient is explicitly named.

Each motion sample has 15 values in this order:

| Indices | Quantity | Definition | Units |
|---|---|---|---|
| 0–2 | `g_B` | `R.T @ gravity_W` | m/s² |
| 3–5 | `u_O_B` | Base-origin inertial linear velocity expressed in B | m/s |
| 6–8 | `omega_B` | Base angular velocity expressed in B | rad/s |
| 9–11 | `held_a_B` | Recorded held linear acceleration, expressed in B | m/s² |
| 12–14 | `held_alpha_B` | Recorded held angular acceleration, expressed in B | rad/s² |

Held estimates are updated only after a complete feature interval. A reset clears
their validity. Preprocessing uses recorded values or causal coordinate
transformations of recorded constituents, without differentiating future motion.
[Recording format](liquid_surrogate_collection.md) describes the source fields.

## Payload inputs

The property vector is constant within an episode. Its exact order is defined by
[`PROPERTY_FIELDS`](../src/liquid_tcn/metadata.py):

| Indices | Values | Units |
|---|---|---|
| 0 | Density `rho` | kg/m³ |
| 1 | Genesis kinematic viscosity `mu` | m²/s |
| 2 | Genesis surface-tension coefficient `gamma` | Raw recorded coefficient |
| 3 | Actual solver liquid mass | kg |
| 4 | Actual rest-density volume | m³ |
| 5 | Nominal fill-box volume | m³ |
| 6–8 | Interior dimensions x, y, z | m |
| 9–11 | Fill dimensions x, y, z | m |
| 12–14 | Mount translation in B | m |
| 15–18 | Mount quaternion w, x, y, z | Dimensionless |
| 19–21 | Geometry scale factors x, y, z | Dimensionless |
| 22 | Fill offset | m |
| 23 | Particle size | m |

Actual mass and volume come from the recording, not from nominal box dimensions.
`mu` is kinematic viscosity; it is not interpreted as dynamic viscosity in Pa·s.
`gamma` is preserved without an assumed unit conversion.

## Network

The default network in [`model.py`](../src/liquid_tcn/model.py) consists of:

1. A property MLP: `24 → 32 → 32`, with SiLU after both linear layers.
2. Concatenation of the 32 property features with the 15 motion channels at each
   time, followed by a `1×1` convolution projecting 47 channels to 64.
3. Five residual blocks with dilations `[1, 2, 4, 8, 16]`. Each block has two
   causal convolutions with kernel size 5 and 64 channels. Its output is
   `x + SiLU(conv2(SiLU(conv1(x))))`.
4. Five independent `64 → 64 → 64 → 3` heads, each with SiLU between linear layers.

There is no dropout, batch normalization, or normalization across time.
Convolutions pad only on the left. The static property embedding is shared across
an episode's time samples; it contains no future motion.

For kernel size `k` and block dilations `d_i`, two convolutions per block give:

```text
receptive_field = 1 + 2 * (k - 1) * sum(d_i)
input_length = receptive_field + long_intervals
contextual_outputs = long_intervals + 1
```

Defaults give a receptive field of 249 samples and a 40-interval physics horizon.
A 289-sample pass therefore yields 41 fully contextual predictions. Outputs from
the first 248 positions depend on left padding and are discarded during training.

The optional [temporal-pyramid attention variant](pyramid_attention.md) adds sparse
attention before these heads. It derives context from block receptive fields plus
selected feature ages; the default schedule retains the same 249-sample history.

## Predictions and loss alignment

The model takes motion `[batch, time, 15]` and properties `[batch, 24]`. It returns
`[batch, contextual_time, 5, 3]` in normalized units, with head order:

| Head | Quantity | Physical units |
|---|---|---|
| 0 | Completed-interval liquid-on-robot force | N |
| 1 | Completed-interval moment about the fixed endpoint base origin | N m |
| 2 | Liquid COM position `r_B` relative to O | m |
| 3 | Rotating-coordinate velocity `v_B = dr_B/dt` | m/s |
| 4 | Internal angular momentum `S_B` about the liquid COM | kg m²/s |

All five heads receive supervised targets at all 41 contextual endpoints.
For physics, endpoint states `0:41` define 40 intervals, while wrenches `1:41`
describe those completed intervals. The wrench at endpoint 0 belongs to the
preceding interval and is excluded from this physics window.

Force and moment are expressed in the interval endpoint's base frame. Moment is
about the fixed world location of that endpoint's base origin. Predictions include
liquid weight; they are neither instantaneous forces nor forecasts for the next
interval. See [physics](liquid_discrete_constraints.md) for impulse transport,
reaction signs, normalization, and validity masks.

## Normalization and inference

Preprocessing fits per-coordinate means and population standard deviations using
only training data. Constant channels use unit scale. Motion, properties, and
supervised targets are normalized for network training. Physics and evaluation
convert predictions back to physical units.

`load_export` in [`inference.py`](../src/liquid_tcn/inference.py) restores the
architecture, weights, and normalization. `step` accepts one physical-unit motion
sample and property vector per environment and returns 12 physical-unit values:
`[force_xyz, moment_xyz, r_B_xyz, v_B_xyz]`. `S_B` is retained by the model for
training but omitted from this interface.

Each causal convolution caches its own past input activations. Environments have
independent caches and warmup counters. Call `reset` after episode resets, gaps,
invalid held inputs, or payload changes. Until an environment has a full receptive
field of real samples, its output is NaN with `valid=False`. The package provides
this inference interface; controller integration and timing are application concerns.

## FiLM and streaming assessment

See [FiLM conditioning and cached one-step assessment](film_streaming.md) for the optional property conditioning, cache lifecycle, measured batch-4096 inference costs, and repeated test assessments. Full-sequence and streaming results use the same validation-selected checkpoints and frozen event definitions. Resume reuses completed full-sequence results and computes missing streaming reports without retraining.
