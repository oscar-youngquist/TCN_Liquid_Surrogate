# Liquid-surrogate collection

Run locally on `liquid_wrench_surrogate`. This implements collection for existing
coupled DACT Go1/Go2 policies; it does not train a TCN or deploy HardPACT. The
mathematical reference is `hardpact_liquid_wrench_training_specification.pdf`,
v2.0, November 3, 2026. Native impulses are **not instantaneous wrench labels**.
Every attempt explicitly records `pinn_ready: false` pending the reconstruction
and conservation audits below.

See [executed validation results](liquid_surrogate_validation.md) for the selected
contact settings, measured runtime/memory, live smoke checks and remaining limits.

## Top-level campaign launcher

Edit the five settings at the top of
[`launch_liquid_collection.sh`](../launch_liquid_collection.sh), then run it:

```bash
bash launch_liquid_collection.sh
```

```bash
PLATFORMS=(go1-dact go2-dact)
NUM_ENVS=2
EPISODES_PER_ENV=2
ROUNDS=1
EPISODE_SECONDS=5
```

Every selected platform runs every round. `ROUNDS` is the number of randomly
sampled payload configurations; the same round's payload is used across selected
platforms, retaining each robot's mounting height. For two platforms and ten
rounds there are twenty sequential fresh worker processes. Environments within a
worker share the payload and independently sample placements and commands.

The round finishes only when **every environment** has completed its quota.
Terminations count as completed episodes even when shorter than the time limit.
Faster-finishing environments continue stepping while the others finish; records
after their quota carry `episode_selected: false` and invalid interval labels.
Use that mask when selecting training data. Metadata reports both raw completed
counts and quota-capped `selected_episodes_per_env`. Episodes already in progress
when a worker is interrupted do not count as completed.

The episode limit is in **simulated seconds**, enforced at policy boundaries and
rounded down to a whole policy interval. The policy/controller timing is unchanged.
Maximum steps are derived from the quota and checkpoint clock. A conservative
wall-clock watchdog scales with the workload; it includes initialization and is
distinct from the per-episode simulated-time cap.

Go1 DACT and Go2 DACT use the existing manifest/checkpoint entries. `go2-hardpact`
is a reserved selector: choosing it fails before starting any worker because its
adapter/checkpoint is not implemented. It never silently substitutes DACT or
installs a QP dependency.

The launcher fixes `genesis_lr`, `cuda:0`, 200 Hz saving and the current box
contacts/resolution. Default output is a fresh UTC-stamped directory under
`runs/liquid_collection/`, with separate platform subdirectories. Optional CLI
arguments override the five settings and provide planning/resume controls:

```bash
bash launch_liquid_collection.sh --dry-run
bash launch_liquid_collection.sh --platforms go2-dact --num-envs 4 \
  --episodes-per-env 3 --rounds 5 --episode-seconds 10 \
  --seed 7 --output runs/liquid_collection/example
# Repeat the same settings and output, adding --resume, to continue that campaign.
```

`--dry-run` creates no output and launches no simulation. `--seed` defaults to 1;
change it for another reproducible sequence. `--max-seconds` optionally overrides
the per-worker wall-clock watchdog. Campaign settings and checkpoint identities
are immutable on resume. No large campaign is required to inspect the plan.

## Visual preview

```bash
# Two Go2 environments, new random payload each launch, translucent walls.
bash legged_gym/scripts/visualize_liquid_collection.sh

# Reproduce a sample with Go1 on its training terrain.
bash legged_gym/scripts/visualize_liquid_collection.sh \
  --robot go1 --seed 1 --num-envs 2 --terrain training --wall-opacity .25

# Inspect every environment, or save a headless rendered frame.
bash legged_gym/scripts/visualize_liquid_collection.sh --num-envs 4 --camera overview
bash legged_gym/scripts/visualize_liquid_collection.sh \
  --seed 1 --headless --steps 3 --frame /tmp/liquid-preview.png
```

The launcher uses `genesis_lr` and `cuda:0`. It samples one shared payload with
the collection planner's bounds, clearance rejection and fixed mounting bounds,
and prints its seed and resolved selection. `--seed` replays that selection;
`--manifest` selects a custom policy pool. No dataset recorder is enabled.
Defaults are two environments (maximum eight), 10 simulated seconds, a flat
inspection surface, blue SPH particles and 25% wall opacity. `--terrain training`
uses the selected policy's randomized training terrain. Controller commands are
sampled from the saved training scope in either case.
Walls use an explicit RGBA texture; particle instances are drawn from the actual
SPH positions to avoid this rasterizer's glass/particle occlusion. These visual
markers are disabled in dataset collection and do not change contact physics.

`--camera follow` tracks environment zero; `--follow-env` changes the index.
`--camera overview` frames the batch, and `--camera free` sets an initial close
view then leaves camera navigation to the viewer. Close the window or press
Ctrl-C to stop early. `--steps` bounds a quick check; `--dry-run` prints the sample
without starting Genesis. The existing preset-based Go1/Go2 players also now use
the translucent box shell with analytic-plane liquid contacts, 10 mm particles,
5 mm contact softness and four substeps.

## Launch and manifest

Use the existing patched environment, with GPU access:

```bash
PYTHON="$HOME/.conda/envs/genesis_lr/bin/python"

# Inspect reproducible choices without importing Genesis or allocating CUDA.
"$PYTHON" -m liquid_collection.plan liquid_collection/example_manifest.json \
  --output /tmp/liquid-plan --seed 1 --rounds 3 --num-envs 100 --dry-run

# Small negative-height, attachment, activation, finite-state, reset and output test.
"$PYTHON" -m liquid_collection.plan liquid_collection/example_manifest.json \
  --output /tmp/liquid-smoke --num-envs 2 --episodes 2 --max-steps 25 \
  --max-seconds 600 --smoke

# Runtime-budget benchmark: original mesh/12 mm/two-substep player, current
# box/10 mm/eight-substep player, then current collection; one process at a time.
"$PYTHON" -m liquid_collection.plan liquid_collection/example_manifest.json \
  --output /tmp/liquid-benchmark --num-envs 2 --episodes 2 --max-steps 50 \
  --max-seconds 600 --smoke --benchmark

# Example bounded collection, after checking the benchmark and masks.
"$PYTHON" -m liquid_collection.plan liquid_collection/example_manifest.json \
  --output /path/to/dataset --num-envs 4 --rounds 2 --episodes 8 \
  --max-steps 3000 --max-seconds 1800 --feature-dt .02
```

The coordinator launches `~/.conda/envs/genesis_lr/bin/python` in a fresh process
for every attempt on `cuda:0`. It sets writable Numba/Matplotlib cache paths under
`/tmp` when absent. No packages or QP solvers are installed. `--max-seconds` includes
scene construction/JIT compilation and enforces a process-group timeout.

Saving defaults to `--save-mode 200hz`: physics still uses every native substep,
but primary state records are written every 5 ms. `--save-mode native` explicitly
retains all substeps for reconstruction/debugging runs. Plans include the save
mode and schema version; use a new output directory for the new format rather
than trying to resume an older plan with different recording semantics.

The manifest has a `policies` array. Each entry requires a unique `id`, `robot`
(`go1` or `go2`), `checkpoint`, and saved Python `config`; `adapter` defaults to
`dact`. Paths resolve relative to the manifest. Add several entries for the same
robot to collect several policies. Each round uniformly selects one entry and
one shared continuous payload configuration. Round seeds are independent of the
number of planned rounds. Checkpoint/config SHA-256 identities are checked again
inside the worker. Saved Python configs are parsed, not executed.

For the lower-level coordinator, `--episodes-per-env` requires a quota in every
environment and takes precedence over `--episodes`. Without it, `--episodes` is a
total completed-episode budget per round; simultaneous batched
terminations can exceed it. Maximum steps/runtime can stop before the budget;
such attempts remain **incomplete**, with their data retained. Benchmark mode runs
the full step budget to give comparable timings. Smoke mode uses 200 ms episodes
and descending 16 cm stairs within the existing generator's support, putting
actual liquid below world Z=0; it is not
long enough to validate 249-tap histories. Forced smoke resets are not counted as
completed episodes. `--episode-seconds` overrides saved/smoke episode duration.

`--resume` verifies the same plan/input identities, checks completed chunk hashes,
skips completed rounds, and creates new attempt directories for failed,
interrupted or incomplete rounds. It restarts that round from its seed; it does
not pretend to restore opaque GPU solver state. Failed files are never replaced.
An advisory coordinator lock prevents concurrent writers. Each attempt snapshots
its own source, so fixes made between failed attempts remain auditable. Use a new
output directory to recollect already completed rounds after source changes.

`PolicyAdapter` explicitly owns initialization, batched inference/control, reset,
and cleanup. DACT uses the existing `play_pact_water.load_policy` and checkpoint
restoration helpers. Its history belongs to the environment. A future HardPACT
adapter must own/import its QP dependencies; DACT has no such dependency.

## Payload, terrain and clocks

Reference bounds are extracted from literal dictionaries in `liquid_payload_configs.py`:
density 730–1000, Genesis mu .001–.025, gamma .001–.010, offset .010–.0875 m,
X scale 1.4–1.8, Y 1.4–2.6, Z 1.2–2.2, and mounting X/Y 0–.05 m. By default,
`--property-expansion .10` expands rho/mu/gamma endpoints by 10%, while
`--geometry-expansion .02` expands scale/fill-offset endpoints by 2%. Both are
configurable (capped at 25% and 5% respectively); set both to zero for exactly
the original envelope. **Mounting X/Y bounds never expand**, and mounting Z
remains robot-specific. Sampling is independent and uniform within this envelope,
followed by rejection of unsafe
fills. These are combinations of existing bounds, not claims that every sampled
combination has been experimentally validated. Robot-specific mounting Z stays
at its existing task value.

Full fill dimensions are `(.17, .12, .185) * scale - offset`, retaining the legacy
center. Clearance is checked against four walls, bottom, and the lid underside
after its .007 m overlap. Each side must clear particle radius plus
`--safety-margin` (default .001 m). Nominal litre names are not measured volume.
Actual mass is read from `SPHSolver.particles_info.mass`; the installed solver
uses `0.8 * particle_size**3 * rho`, not cubic spacing alone. `liquid.json` records
actual count, mass, sum(m/rho), fill volume, dimensions and mounting transform.
Genesis mu enters the inspected acceleration kernel as kinematic viscosity
(m²/s), despite its name; it is not dynamic viscosity in Pa s. Gamma remains the
raw Genesis surface-tension coefficient. No unsupported physical-unit conversion
or real-fluid calibration is applied.

The collector generates the existing terrain once before constructing the SPH
domain. Bounds cover its actual signed heights and XY map plus a 3 m motion
margin. The default SPH lower Z of zero is overridden. Terrain origins and random
XY reset placement are retained; negative terrain is neither removed nor shifted
upward. Each environment independently samples a terrain column and level.
Out-of-bounds behavior terminates/resets the whole environment instead of the
legacy rigid-only teleport.

Saved terrain/command literals are restored. Missing inherited terrain fields
fall back to the current robot training class, and each fallback is recorded.
Current `Terrain.make_terrain` supplies the existing hardcoded terrain parameter
ranges and randomized difficulties [.5, .75, .9]; its source is snapshotted.
Curricula are disabled. Map rows/columns are reduced for small environment counts,
without changing tile dimensions or terrain parameter ranges. Commands use saved
training ranges and heading mode. Controller timing, scales, joint order and
observation contract come from the selected checkpoint config/weights. Joint
dynamics retain the existing player's saved-range midpoint convention.

`--substeps` and `--particle-size` expose explicit resolution/runtime experiments.
Collector and visual-player defaults are four substeps and .010 m particles with
inward-facing analytic-plane liquid contacts, .005 m contact softness and a
translucent box shell. These settings passed repeated ten-second containment trials;
see [containment experiments](liquid_containment_experiment.md). The benchmark's explicit
reference profile retains the pre-swap mesh/.012 m/two-substep settings.
Genesis's viscosity warning alone does not change
these settings. Increasing internal substeps
preserves policy and PD timing; coarsening particles is an explicit alternative,
not an automatic response. For a typical DACT checkpoint at the defaults:

| Clock | Interval |
| --- | --- |
| Policy inference | 10 ms |
| PD / `scene.step()` | 2 ms (decimation 5) |
| Internal rigid/SPH interval | 0.5 ms (substeps 4) |
| Surrogate target grid | 5 ms |
| Surrogate held-acceleration feature update | 20 ms, configurable independently |

The worker requires these grids to align to native steps; it fails instead of
silently resampling the controller. Feature estimates use velocity differences
over a **completed** feature interval. They are held in world coordinates and
rotated with each sample's current base quaternion. Updates are right-continuous.
Reset only clears affected environments' estimates/age and increments their
episode IDs. Policy, feature, command and reset events are distinct fields.

`--contact-geometry mesh` retains the existing STL tank/lid. The collector default
`boxes` replaces
these with six attached closed cuboids having the same nominal inner dimensions,
outer dimensions and lid overlap. Y walls overlap into X walls by 2 mm to close
SDF seams; their density is corrected to preserve the nonoverlapping shell's total
dry mass (small inertia changes remain explicit). Contact softness is half the
particle size, so contact response begins before a particle centre reaches a thin
wall. This avoids a single concave tank SDF and uses
2 mm SDF cells (maximum resolution 256). Installed Genesis still uses sampled
SDFs for boxes; this is not falsely described as analytical contact. The cuboids'
dry mass/inertia come from their actual geometry and recorded densities;
they replace the meshes, so no second dry container is added. Differences from
the rounded STL mass/inertia, part dimensions and actual SDF grids are recorded.
The same box contacts and translucent surfaces are used by the visual players.

## Output schema and physical meaning

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
and separate boundary records are intentional. Merge streams by `(time_s,
record_sequence)`; reset and terminal records preserve their exact ordering.

| Fields | Meaning |
| --- | --- |
| `time_s`, `native_step`, `target_grid_event` | Recorded time starts at zero after the player's initialization reset/step; 200 Hz grid membership |
| `record_sequence`, `interval_duration_s`, `interval_native_steps` | Ordering across streams, actual integration duration and contributing substep count |
| `episode_id`, `episode_native_step`, `reset`, `terminal`, `forced_reset` | Per-environment trajectory boundaries; unchanged environments retain IDs |
| `episode_selected` | False once that environment has completed its per-round quota; exclude these extra episodes from training |
| `policy_update`, `feature_update`, `command_update`, `commands` | Separate event streams; commands include heading where present |
| `x_O_W`, `q_WB_wxyz`, `velocity_W` | Base-link origin position, base-to-world rotation, world inertial `[linear, angular]` velocity |
| `gravity_W`, `held_acceleration_W`, `held_valid`, `z` | Gravity, held `[linear, angular]` estimates, availability, PDF's 15 causal inputs |
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

## Historical pointwise/PDF preprocessing diagnostics

The current training formulation is [discrete endpoint/interval-average
consistency](liquid_discrete_constraints.md), validated separately in the
[v1 report](liquid_discrete_validation_v1.md). The guidance below describes the
original continuous/JVP diagnostic objective and is retained for historical
interpretation; it is not a requirement to reconstruct instantaneous forces or
accelerations for the revised discrete losses. Historical validity flags remain
unchanged.

1. Validate chunk hashes including `events/` and `boundaries/`. Merge state and
   boundary rows by `(time_s, record_sequence)`, applying exact-time command and
   held-input updates from events. Split per-environment trajectories by episode ID and
   all unsupported transitions. Preserve terminal rows, then start a new sequence
   from the reset row. Repeated-time rows require explicit event semantics.
2. For each 5 ms grid time use the final right-continuous state/event row belonging
   to that episode. Require `held_valid`, valid constituent intervals and a complete
   249-sample history; do not pad across resets. Keep histories in each tap's own
   base frame. Do not derive input accelerations from the upcoming interval.
3. Default 200 Hz output does **not** preserve the dense native trajectory or
   native impulse distribution within each 5 ms bin. Native knots, high-frequency
   acceleration and impulse timing cannot be recovered exactly. Use explicitly
   recorded `--save-mode native` validation runs to assess any interpolation/JVP
   reconstruction; pointwise readiness remains false in both modes.
   Reconstruct a consistent differentiable base trajectory from the available pose
   and velocity endpoints: position derivatives must match reconstructed linear
   velocity; quaternion derivatives must match reconstructed angular velocity.
   SO(3) interpolation and translational Hermite/spline choices need independent
   derivative checks against finer data. The exported interval accelerations are
   characterized finite differences, not validated instantaneous accelerations.
   Never smooth held-estimate jumps into ramps or mix future feature updates into
   inputs. Any offline interpolation/lookahead assumption must be documented.
4. Measured coupling is impulsive. For pointwise supervised/JVP losses, define and
   validate a time-aligned continuous wrench reconstruction (including its sign,
   frame, reference point and numerical floor), or explicitly change the learning
   objective to interval averages. Do not use `I/dt` as an instantaneous endpoint
   label. This implementation deliberately does not mark either choice validated.
5. For local JVP sampling, test **all 249 shifted taps** against feature events,
   derivative jumps, reconstruction knots and a stated exclusion margin.
   `audit.interior_mask` implements that event-distance test; it additionally needs
   complete-history validity from step 1. Do not demand the whole history contain
   no feature events. Native knots at 0.25 ms still leave fractional interior samples,
   but use an exclusion margin smaller than half the knot spacing (e.g. 0.1 ms);
   the PDF's proposed 0.25 ms margin would leave no interior at this resolution.
6. For grid-aligned 200 ms integrated windows retain 41 target-grid endpoints and
   sum all 40 saved interval impulses (including split partial intervals from
   `boundaries/`). These preserve native world impulse sums, up to floating-point
   storage precision. Sum fixed-world-origin angular
   impulses, or shift each saved impulse with its own recorded reference pose;
   do not multiply an averaged force by an arbitrary endpoint lever arm. Convert
   to CoM angular balance using the matching trajectory. Audit native closure and
   200 Hz quadrature error on ground truth before applying physics losses.

```bash
"$PYTHON" -m liquid_collection.audit /path/to/round-00000/collection/attempt-0000
"$PYTHON" -m pytest -q tests/test_liquid_collection.py
```

## Persistence, environment and validation

Synchronous NPZ writes provide backpressure. Row count and byte limits bound the
buffer per stream (`--chunk-size 128`, `--max-buffer-mb 64`); stacking briefly duplicates a
chunk. A single oversized row fails clearly. No samples are silently dropped.
`--snapshots N` enables a separate bounded particle-debug stream every N native
steps; it deliberately incurs particle host transfers. Rendering, cameras and
markers are disabled. Successful/failed attempts retain compact data, logs and
their own provenance.

The installed `genesis_lr` reports Genesis 0.3.11 and Torch 2.8.0+cu126. Source
inspection confirmed the attached rigid-entity API, batched SPH setters and solver
particle mass fields; patched liquid-property prints remain intact. Each worker
copies relevant installed files and records SHA-256, distribution RECORD
comparison (when available), package versions, Python executable, repository head
and dirty status. It also records local collector/player/terrain sources and tank
meshes. The installation is never edited or replaced. To reproduce elsewhere,
preserve/clone the original environment, overlay the snapshotted Genesis sources
at their recorded relative paths, verify every hash, then rerun attachment,
activation, negative-height and partial-reset smoke tests. A plain pip installation
with the same version number is not evidence of equivalent local patches.

Benchmark results include initialization time, first-step time, subsequent mean
policy-step time, Torch allocated/reserved peaks, process RSS, and sampled
whole-process NVIDIA memory (includes Genesis allocations outside Torch, 0.5 s
sampling). The benchmark uses three sequential, matched profiles: original mesh
player (12 mm/two substeps), current player loop, and current collection. It writes
`benchmark.json` with recording overhead and `runtime_budget_passed`, comparing
collection against the original mesh player. A runtime-budget failure gives a
nonzero coordinator exit status. Compare these matched runs, not unrelated
terrain/payload presets. A
small-environment result does not establish throughput or memory at 100
environments; profile progressively before recommending that count.
