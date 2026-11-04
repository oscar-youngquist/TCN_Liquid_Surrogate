# Collection recording format

This package reads completed attempts produced by the HCR Genesis liquid
collector. It does not run the simulator or robot policies. The supported metadata
schema versions are 2 and 3. This document describes the recordings consumed by
[preprocessing](../src/liquid_tcn/preprocess.py); collector source and revision are
listed in [provenance](provenance.md).

## Preparing a collection

```bash
liquid-tcn prepare --root /path/to/collection --output data/prepared
```

Discovery recursively finds attempts, skips unfinished streams, and verifies
checksums on completed attempts. The result is a fixed snapshot. Run preparation
again with a new output directory to include recordings completed later.
[Configuration](configuration.md) describes splitting, held-out payloads, shard
sizes, and explicit source lists.

## Metadata used by the package

| File | Required information |
|---|---|
| `job.json` | `schema_version`, `selected.policy.robot`, `particle_size`; automatic grouping also uses `selected.seed` and `selected.round` |
| `resolved.json` | Positive `native_dt` dividing 5 ms exactly; `save_mode` distinguishes native debug recordings |
| `liquid.json` | `payload`, `geometry`, `mass_per_env`, and `rest_density_volume_per_env` |
| `result.json` | Completed stream status and checksummed chunk entries |
| `chunks.json` | Chunk index, when present; must agree with `result.json` |
| `provenance.json` | Optional source/environment provenance, retained by hash |

Payload properties include `rho`, `mu`, `gamma`, and `offset`. Geometry includes
`interior`, `fill_size`, `fill_box_volume`, `mount_translation_B`,
`mount_quaternion_wxyz`, and `scale`. The exact 24-value model input order and
units are defined in [architecture](architecture.md). Saved Python policy files
are never executed by the loader.

## Causal motion inputs

Each endpoint supplies `z = [g_B, u_O_B, omega_B, held_a_B, held_alpha_B]`,
five three-vectors. `R` maps base coordinates to world coordinates, so world
vectors are expressed in the base frame with `R.T`. If `z` is missing,
preprocessing reconstructs it from recorded gravity, base velocity, orientation,
and held acceleration. If held acceleration is missing but `z` exists, its last
six components recover the held values. Missing both is an error.

Held acceleration estimates use completed feature intervals and remain constant
until the next recorded update. They are unavailable immediately after reset.
The separate `base_acceleration_interval_average_W` label is not substituted for
a causal held input. See [architecture](architecture.md) for window alignment and
[physics](liquid_discrete_constraints.md) for the resulting losses.

## Stream layout and physical meaning


Each `round-N/collection/attempt-N` contains `job.json`, `status.json`,
`worker.log`, installed/source `provenance.json` plus source copies,
`resolved.json`, `training_config.py`, `robot.urdf`, `terrain.npz` (generated signed
height map and tile origins), `liquid.json`, numbered NPZ
chunks, a checksummed `chunks.json` index, and an atomic `result.json` only when
the worker finishes writing. An exception adds `failure.json`; a killed worker
may leave an unindexed `.tmp` file, which readers must ignore. NPZ needs no pickle.

With default 200 Hz saving, `events/` contains compact exact-time policy, command,
held-feature and reset events. `boundaries/` retains full pre-reset terminal and
post-reset states, and any partial interval flushed before an off-grid reset or
the end of a run. Each has its own bounded chunks and checksums; completion/resume
verification includes these auxiliary streams.

Chunk arrays have `[sample, environment, ...]` axes except scalar time/tick fields.
In 200 Hz mode the primary stream contains an initial state and one endpoint row
every 5 ms. States (including quaternions, CoM quantities and held inputs) are
endpoint values, never temporal averages. Force impulses and fixed-world-origin
angular impulses are summed over every contributing native interval on GPU before
host transfer. Angular impulse is shifted to the **fixed endpoint base origin**,
and both impulses are rotated into the endpoint base frame. This is not the
integral of moments about the continuously moving base origin.

Interval acceleration is the endpoint velocity difference divided by actual
duration. Every constituent substep must be valid; interval diagnostics retain
logical AND for containment/activation/finite state, OR for domain contact, maximum
wall penetration and minimum active count. A transient problem cannot disappear
merely because the endpoint recovered.

Native debug mode retains the previous per-substep state/event stream and its
coupling-phase reference pose. Both modes store `interval_duration_s` and
`interval_native_steps`. Event-only rows have zero duration and invalid interval
labels. Never infer integration duration from adjacent rows: repeated timestamps
and separate boundary records are intentional. Merge streams by `(native_step,
record_sequence)`; reset and terminal records preserve their exact ordering.

| Fields | Meaning |
| --- | --- |
| `time_s`, `native_step`, `target_grid_event` | Recorded time starts at zero after the player's initialization reset/step; 200 Hz grid membership |
| `record_sequence`, `interval_duration_s`, `interval_native_steps` | Ordering across streams, actual integration duration and contributing substep count |
| `episode_id`, `episode_native_step`, `reset`, `terminal`, `forced_reset` | Per-environment trajectory boundaries; unchanged environments retain IDs |
| `episode_selected` | False once that environment has completed its per-round quota; exclude these extra episodes from training |
| `policy_update`, `feature_update`, `command_update`, `commands` | Separate event streams; commands include heading where present |
| `x_O_W`, `q_WB_wxyz`, `velocity_W` | Base-link origin position, base-to-world rotation, world inertial `[linear, angular]` velocity |
| `gravity_W`, `held_acceleration_W`, `held_valid`, `z` | Gravity, held `[linear, angular]` estimates, availability, the 15 causal motion channels |
| `mass`, `com_W`, `com_vel_W`, `momentum_W` | Actual solver-particle aggregates |
| `r_B`, `v_B`, `S_B` | Liquid CoM relative position, **rotating-frame coordinate velocity**, inertial angular momentum about liquid CoM in B |
| `angular_momentum_world_origin_W` | Fixed-world-origin angular momentum, for impulse closure auditing |
| `coupling_*impulse*`, `coupling_origin_W`, `coupling_q_WB_wxyz` | Liquid-on-rigid impulses; fixed endpoint reference in 200 Hz mode, coupling-phase reference in native mode |
| `momentum_balance_*impulse*`, `*_closure_residual_impulse_W` | Native balances/residuals summed over the saved interval |
| `force_interval_average_B`, `moment_interval_average_O_B` | Impulse divided by recorded duration, expressed at its recorded reference pose; **not pointwise labels** |
| `base_acceleration_interval_average_W` | Endpoint velocity difference divided by recorded duration; metadata, not a TCN input |
| `finite`, `all_active`, `active_count`, `contained`, `domain_contact`, `attached` | State/transition validity evidence |
| `inside_nominal_cavity`, `max_wall_penetration_m`, `particle_min_B`, `particle_max_B` | Separate nominal inner-surface penetration diagnostics |
| `interval_finite`, `interval_contained`, `interval_all_active`, `interval_attached`, `interval_inside_nominal_cavity`, `interval_domain_contact`, `interval_max_wall_penetration_m`, `interval_min_active_count` | Conservative diagnostics across all constituent substeps, distinct from endpoint state |
| `state_valid`, `episode_supported`, `interval_valid` | Structural masks; spill/domain/inactive/nonfinite/attachment failures taint the episode until its reset |
| `instantaneous_wrench_valid`, `derivative_reconstruction_valid` | False until separately reconstructed and validated |

Initialization checks the actual sampled layout against the STL triangles (or
the box faces), including rounded corners and the bottom. This one-time mesh
audit may transfer the shared rest layout to CPU; hot-path reductions stay on
GPU. Initialization and every partial reset validate **actual** positions against
particle-radius clearance, activation and domain bounds. Resets restore only
selected liquid layouts, clear their momentum, reactivate particles and clear
their coupling-normal cache. During motion, soft SDF coupling can allow particle
centres inside the nominal wall material. This is recorded separately from
`contained`, which means centres remain inside the outer closed tank/lid envelope.
An outer-envelope crossing, inactive particle, domain contact, bad attachment or
nonfinite state invalidates the native transition. Nominal wall penetration is
not silently discarded or described as strict cavity containment; downstream
studies can reject it using the dedicated field.

The recorder observes native `Simulator.substep` and both sides of
`LegacyCoupler.couple`. All particle reduction runs on GPU; only compact aggregates
transfer to host by default. Reordered particle masses and active masks are used
at coupling. A particle's velocity difference measures the actual coupling
impulse, whose opposite is applied to the rigid solver at that particle's world
position. Angular impulse is retained both about the coupling-phase base origin
and the fixed world origin; shifting uses `I_tau_O = I_tau_A + (x_A-x_O) cross I_F`.
All rigid coupling is included. Valid contained liquid should interact only with
the container/robot; no per-geometry attribution is claimed. Additional fluid
entities, external fluid force fields and MPM/FEM-fluid coupling are rejected.

WCSPH pressure and DFSPH velocity corrections occur before coupling and are
included in complete-interval endpoint momentum changes. `particle.acc` omits
coupling (and DFSPH direct velocity corrections), so it is not used as a complete
acceleration. Force closure is `m*g*dt - delta(P) - I_F`. Angular closure uses a
fixed world origin, the gravity kick at interval-start particle positions, and
endpoint angular momentum. The installed kick/drift algorithm and its residual
are recorded rather than assuming exact internal angular-momentum conservation.
The legacy rigid solver integrates before coupling and receives the reaction for
its subsequent solve; liquid impulse timestamps are not shifted to hide this lag.

## Training eligibility

Primary, event, and boundary streams are merged by native tick and record
sequence. Duplicate copies must agree on shared fields. Terminal records are
processed before the corresponding reset, with separate episode identities for
each environment. Split intervals are assembled by summing impulses in a common
world reference and then expressing the result at the final endpoint.

Only selected, supported episodes with valid held inputs and contiguous 5 ms
intervals can yield windows. Each predicted endpoint needs a complete causal
history. Resets, gaps, invalid transitions, and incomplete histories are never
padded. Mass and gravity must remain consistent throughout an eligible episode.

Whole collection groups are assigned to train, validation, or test before
windowing. Related retries and robots stay together. Each row is stored once in
bounded `.npy` shards; compact indices select overlapping windows. Normalization
uses valid rows from training series with eligible windows. `manifest.json`
records schemas, source hashes, split assignments, statistics, and rejection
counts. Raw recordings are opened read-only.
