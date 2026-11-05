# Discrete liquid-wrench physics objectives

The objective combines **supervision from smoothed-particle hydrodynamics (SPH)
recordings with short- and long-interval Newton–Euler consistency**. The model predicts completed **5 ms interval-average reaction
wrenches** and endpoint liquid states at 200 Hz. It need not reproduce individual
SPH-substep forces.

Implementation: [discrete losses](../src/liquid_tcn/ported/discrete/losses.py) and
[combined objective](../src/liquid_tcn/objectives.py). See [validation results](validation.md)
for numerical tests and [architecture](architecture.md) for tensor alignment.

## Endpoint and interval contract

Row `k` contains endpoint state at `t_k`; its force and moment heads describe
`(t_{k-1}, t_k]`. Both averages are expressed in endpoint base frame `B_k`.
The moment is about the **fixed world location of the endpoint base origin**,
not a continuously moving reference point. `R_k` maps base to world. Recorded
quaternions are `wxyz`; the module normalizes them before constructing `R`.

The five supervised heads are:

| Head | Target and meaning |
|---|---|
| `r_B` | Endpoint liquid CoM position relative to base origin, expressed in B |
| `v_B` | Endpoint rotating-frame coordinate derivative `dr_B/dt` |
| `S_B` | Endpoint internal liquid angular momentum about its CoM, expressed in B |
| `F_average_B` | Completed-interval average **liquid-on-robot** force |
| `tau_average_O_B` | Completed-interval average **liquid-on-robot** moment about the fixed endpoint base origin |

Use actual solver particle mass (summed over particles), recorded gravity, and
recorded base origin position/velocity/angular velocity. Mass must be positive
and constant and gravity spatially uniform and constant within an eligible
trajectory. The recorder supports a single liquid with uniform solver gravity
and no external fluid fields; preprocessing verifies mass/gravity consistency
in each eligible episode. The module also
checks these assumptions in each supplied window. Invalid assumptions mask the
window; they must not be interpreted as a low-loss success.

Completed-interval labels are not forecasts of the next interval. Training inputs
must obey the stated history convention. This validation does **not** establish
which completed prediction would be available at a deployment control instant.

## Equations

Let `x_O`, `u_O`, `omega_B`, and `R` be recorded robot states. For predicted heads:

```text
X_k = x_O_k + R_k r_k
U_k = u_O_k + R_k (v_k + omega_B_k × r_k)
P_k = m U_k
H_A_k = R_k S_k + (X_k - A) × P_k
```

`A` is a nearby fixed world point derived from recorded robot state. By default
it is the base origin at the start of the supplied window and remains fixed for
all 40 intervals. It must not depend on a predicted quantity.

```text
I_F_k     = h_k R_k F_average_B_k
I_tau_A_k = h_k R_k tau_average_O_B_k + (x_O_k - A) × I_F_k
G_A_k     = h_k ((X_(k-1) + X_k)/2 - A) × (m g_W)

e_r_k = r_k - r_(k-1) - h_k (v_(k-1) + v_k)/2
e_P_k = P_k - P_(k-1) - h_k m g_W + I_F_k
e_H_k = H_A_k - H_A_(k-1) - G_A_k + I_tau_A_k
```

Reaction signs are positive in the residual because liquid-on-robot impulses
are opposite to the impulses acting on the liquid. Momenta are computed from
the predicted heads, not supplied as ground-truth momentum inputs.

For 41 endpoints and 40 interval labels:

```text
e_r_long = r_end - r_start - sum(h_k (v_(k-1) + v_k)/2)
e_P_long = P_end - P_start - m g_W sum(h_k) + sum(I_F_k)
e_H_long = H_A_end - H_A_start - sum(G_A_k) + sum(I_tau_A_k)
```

**Sum interval impulses directly.** Do not apply trapezoidal weights to the
already averaged wrenches. Endpoint velocity and the stated gravity-moment
approximation use trapezoidal quadrature. No instantaneous acceleration,
position second derivative or substep-rate network output enters these losses.

With a common `A` and quadrature convention, summing short residuals gives the
long residual. If `A` changes by `d`, `e_H` changes by `-d × e_P`. Thus the joint
zero-residual solution is reference-invariant; an angular residual alone need
not be invariant when the linear momentum constraint is violated.

## Normalization and objective

Define `L = ||interior_dimensions||_2`, the diagonal of the nominal inner cavity:

```text
F0   = m |g|
tau0 = F0 L
u0   = sqrt(|g| L)
scales = (u0 T, F0 T, tau0 T) for (e_r, e_P, e_H)
```

`T` is the actual interval/window duration: 0.005 or 0.2 seconds with the default
configuration. Each channel loss is the mean squared Euclidean norm of its
normalized residual over valid samples. The physics loss sums the three channel
losses. The scales have units of position, linear momentum, and angular momentum,
respectively, so the normalized residuals are dimensionless.

Training supervises all five heads and adds weighted short/long consistency:

```text
L_total = L_supervised(r, v, S, F_interval_average, tau_interval_average)
          + lambda_short L_short + lambda_long L_long
```

For normalized prediction `y_hat`, normalized target `y`, and vector validity
mask `M[b,k,h]`, the supervised MSE is:

```text
L_supervised = sum_b,k,h,c M[b,k,h] * (y_hat[b,k,h,c] - y[b,k,h,c])^2
               / (3 * sum_b,k,h M[b,k,h])
```

Here `h` indexes the five heads and `c` indexes xyz. Each valid scalar has equal
weight. Huber supervision replaces each squared error with `e^2/2` for
`|e| <= delta` and `delta*(|e|-delta/2)` otherwise. The threshold is in normalized
units. Statistics are fitted only on the training split. Every head must have at
least one valid label in a training batch.

Predictions are converted back to physical units before physics evaluation.
Both endpoints and all five heads remain differentiable. Physics arithmetic runs
outside autocast, in float64 by default. Supervised pretraining sets both physics
weights to zero; a linear ramp then raises them to the configured values while
retaining supervision. See [training configuration](configuration.md) for defaults.

## Gradient combination

The default optimizer step uses conflict-only one-way PCGrad. Let
`g_s = grad(L_supervised)` and
`g_p = grad(lambda_short L_short + lambda_long L_long)`, where both lambdas include
the current training ramp. The primary objective includes supervision for force,
moment, position, velocity, and internal angular momentum. Physics is one
secondary objective, formed by summing its weighted terms before differentiation.

On the flattened parameters shared by both objectives:

```text
d = dot(g_s, g_p)
g_p_projected = g_p - d * g_s / (||g_s||² + eps)   if d < 0
                g_p                              otherwise
g_update = g_s + g_p_projected
```

The primary gradient is not projected. There is no gradient averaging, secondary
norm balancing, random task ordering, or pairwise projection between physics
components. Parameters used by only one objective receive that objective's
gradient unchanged. Parameters unused by both retain `grad=None`, preventing
optimizer momentum or weight decay from updating them.

The implementation uses the source helper's additive `eps=1e-12`. It keeps the
zero-primary case well defined, but leaves a small negative dot product after
projection when `eps` is significant relative to `||g_s||²`. The operation removes
the conflicting component up to this regularization and floating-point precision.
It is not a guarantee that supervised loss decreases after a finite AdamW step:
adaptive scaling, momentum, weight decay, and nonlinear curvature still apply.

The merged gradient is clipped globally before AdamW. Nonfinite objectives,
component gradients, or merged gradients raise an error. With both physics
weights zero, training uses only the supervised gradient. `gradient_strategy="sum"`
provides ordinary summed-loss gradients for comparison. The reported scalar
`loss` remains `L_supervised + L_secondary` under either strategy; PCGrad changes
the update direction, not the reported objective values or validation metric.

Implementation: [gradient combination](../src/liquid_tcn/gradients.py) and
[ported projection](../src/liquid_tcn/ported/pcgrad.py). The helper is extracted
from `project_aux_against_primary_pcgrad` at the revision listed in
[provenance](provenance.md). Tests compare it with a verbatim upstream fixture.

## API and masks

`discrete_constraints(...)` accepts tensors with batch dimension `B`:

| Argument | Shape |
|---|---|
| `r_B`, `v_B`, `S_B`, `x_O_W`, `u_O_W`, `omega_B`, `gravity_W` | `[B,N,3]` |
| `q_WB_wxyz` | `[B,N,4]` |
| `mass`, `episode_id`, `state_valid`, `history_valid` | `[B,N]` |
| `F_average_B`, `tau_average_O_B` | `[B,N-1,3]` |
| `h`, `interval_valid` | `[B,N-1]` |
| `interior_dimensions`, optional `reference_W` | `[B,3]` |

Short-only examples may use `N=2`; long windows use `N=41`. Every wrench maps to
the interval ending at the corresponding endpoint from index 1 onward. Supply
`omega_B = R^T omega_W`; the collector's `velocity_W` concatenates world linear
and angular velocity. There are no recorded-momentum arguments in this API.

`history_valid` must certify a full receptive field of causal 200 Hz samples
(**249 with the default architecture**) ending at that
endpoint, including valid held inputs, same episode identity, and valid
intervening transitions. It is required at the window start and at every
endpoint. The loader supplies this eligibility without spline reconstruction or
future held-input interpolation. The physics module cannot reconstruct an input
history from a two-endpoint tensor; callers must compute this mask from recorded history.

The result contains `short` and `long` dictionaries with dimensional residuals,
normalized residuals, validity masks, counts, per-channel losses, total losses
and `has_valid`. Invalid data/predictions are excluded from the loss; invalid raw
residuals are diagnostic only. Empty valid sets return zero loss and count zero;
**always check the count/has_valid**, rather than treating that zero as a pass.

Raw equations support varying positive durations. Defaults admit only full
5 ms short intervals and exactly 40 intervals totaling 200 ms for a long loss.
An explicitly irregular mathematical test can set the corresponding
`interval_seconds`, `window_seconds`, or `window_intervals` restriction to
`None`. Do not use that option to disguise partial production intervals.

The loader merges primary, exact-event and boundary streams by native tick and
record sequence. It retains old-episode terminal endpoints before same-time
resets. Another environment's reset may split an interval; only contiguous,
valid same-episode pieces can be combined, after converting their impulses to a
common world moment reference. Gaps, invalid transitions, unsupported liquid
states, unaligned starts/durations and incomplete histories/windows are rejected.

## Validation and reproduction

Run `python -m pytest -q` to check equilibrium, free fall, rotating frames,
reference shifts, short/long summation, masks, and gradients on synthetic data.
Use `liquid-tcn evaluate` for a trained model; it reports predicted and target
conservation residuals alongside supervised errors. [Validation results](validation.md)
include commands and the scope of recorded-data checks.

The numerical helpers in `liquid_tcn.ported.discrete.analyze` support source
recording diagnostics and are exercised by the tests. Its legacy campaign CLI
assumes the upstream experiment/source layout and is not a standalone package
command. Use `liquid-tcn evaluate` for checkpoint evaluation and
`tools/compare_port.py` for the optional [source comparison](provenance.md).

All offline arithmetic is float64, using nearby fixed references. This reduces
new arithmetic error but cannot recover precision already lost in stored
float32 poses or world-origin angular momentum. Native gravity is independently
integrated from native interval-start CoM positions, consistent with the recorded
kick/couple/drift solver order. Gravity recovered from precomputed balance fields
is only a diagnostic cross-check. Coupling impulses are never constructed from
momentum changes to force conservation.
