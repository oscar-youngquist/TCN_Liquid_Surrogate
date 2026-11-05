# Source provenance

The collection-format readers and discrete physics utilities derive from
[HCR_Genesis_PACT_Development](https://github.com/oscar-youngquist/HCR_Genesis_PACT_Development/tree/300d3288b9bce83ccdc31b38bd7926f9484787e7),
commit `300d3288b9bce83ccdc31b38bd7926f9484787e7`. [imports.json](imports.json)
lists source paths, destination paths, and SHA-256 hashes of the imported source
files. The upstream license is retained in [SOURCE_LICENSE](SOURCE_LICENSE).
Hashes identify the source revision, not the subsequently documented local files.

Package imports have been adapted to `liquid_tcn`. Comments, docstrings, diagnostic
report wording, and the documentation describe this standalone package. The data
and physics equations retain the source semantics. Production preprocessing adds
bounded stream merging, causal feature retention, shards, compact indices, and
split/statistics handling around those utilities. [Validation results](validation.md)
record numerical and data-equivalence checks.

The historical architecture document is identified by name, version, and hash in
`imports.json` for traceability. It is not distributed with this package and is
not required to understand or use it. The local [architecture guide](architecture.md),
[physics reference](liquid_discrete_constraints.md), [recording format](liquid_surrogate_collection.md),
and [configuration reference](configuration.md) define the implemented behavior.
The two recording/physics documents were adapted from the upstream documentation;
their hashes in `imports.json` refer to the upstream versions.

## Optional source-equivalence check

Normal installation, training, evaluation, and tests do not require the source
repository, Genesis, or robot policies. Only `tools/compare_port.py` requires a
source snapshot. To create one without modifying an existing checkout:

```bash
git clone --no-checkout https://github.com/oscar-youngquist/HCR_Genesis_PACT_Development.git /tmp/liquid-source-repo
mkdir /tmp/liquid-source-snapshot
git -C /tmp/liquid-source-repo archive 300d3288b9bce83ccdc31b38bd7926f9484787e7 \
  liquid_collection | tar -x -C /tmp/liquid-source-snapshot
chmod -R a-w /tmp/liquid-source-snapshot
python tools/compare_port.py --source /tmp/liquid-source-snapshot \
  --output artifacts/port-equivalence.json
```

Use fresh temporary paths. Add `--attempt /path/to/completed/attempt` to compare
merged rows, episodes, intervals, and window eligibility on a recording. Recorded
data are not bundled; the tool's synthetic physics comparison needs no recording.

## Conflict-only PCGrad

`src/liquid_tcn/ported/pcgrad.py` contains the verbatim
`project_aux_against_primary_pcgrad` function from
`rsl_rl/algorithms/encoder_pinn_grad_utils.py` at the same pinned commit above.
`imports.json` records the full source file hash and extracted symbol.
`tests/fixtures/pcgrad_reference.py` preserves an independent upstream fixture for
numerical equivalence checks. The always-orthogonal optimizer variants from the
source repository are not used.

`src/liquid_tcn/gradients.py` supplies the package integration: compute supervised
and summed physics gradients, project on shared parameters, preserve unused
parameter ownership, and return logging diagnostics. Projection has no external
optimizer, simulator, or policy dependency.
