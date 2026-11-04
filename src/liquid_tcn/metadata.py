"""Strict JSON metadata parsing; never imports or executes saved policy configs."""
import hashlib
import json
from pathlib import Path
import numpy as np

PROPERTY_FIELDS = (
    'rho_kg_m3', 'mu_kinematic_m2_s', 'gamma_genesis_raw', 'actual_mass_kg',
    'actual_rest_density_volume_m3', 'fill_box_volume_m3',
    'interior_x_m', 'interior_y_m', 'interior_z_m',
    'fill_x_m', 'fill_y_m', 'fill_z_m',
    'mount_x_m', 'mount_y_m', 'mount_z_m', 'mount_qw', 'mount_qx', 'mount_qy', 'mount_qz',
    'scale_x', 'scale_y', 'scale_z', 'fill_offset_m', 'particle_size_m')


def read_metadata(path):
    path = Path(path)
    result = {name: json.loads((path / (name + '.json')).read_text())
              for name in ('liquid', 'resolved', 'job')}
    if result['job'].get('schema_version') not in (2, 3):
        raise ValueError('Unsupported collection schema; expected version 2 or 3')
    dt = result['resolved']['native_dt']
    if dt <= 0 or not np.isclose(round(.005 / dt) * dt, .005, rtol=0, atol=1e-12):
        raise ValueError('Native clock must divide the 5 ms grid')
    result['payload_id'] = hashlib.sha256(json.dumps(result['liquid']['payload'], sort_keys=True).encode()).hexdigest()
    result['robot'] = result['job']['selected']['policy']['robot']
    return result


def properties(metadata, env):
    liquid, job = metadata['liquid'], metadata['job']
    p, g = liquid['payload'], liquid['geometry']
    values = [p['rho'], p['mu'], p['gamma'], liquid['mass_per_env'][env],
              liquid['rest_density_volume_per_env'][env], g['fill_box_volume'],
              *g['interior'], *g['fill_size'], *g['mount_translation_B'],
              *g['mount_quaternion_wxyz'], *g['scale'], p['offset'], job['particle_size']]
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(PROPERTY_FIELDS),) or not np.isfinite(values).all():
        raise ValueError('Invalid recorded payload properties')
    if min(values[0], values[3], values[4]) <= 0 or (values[6:12] <= 0).any():
        raise ValueError('Nonpositive mass, volume, density or dimensions')
    return values


def assign_splits(sources, seed=1, fractions=(.8, .1, .1), held_out_payloads=()):
    """Assign entire explicitly named collection groups before looking at rows.

    Group IDs must include related retries and all robots from a collection round.
    An explicit split is allowed but conflicting group assignments are rejected.
    """
    if len(fractions) != 3 or any(x < 0 for x in fractions) or not np.isclose(sum(fractions), 1):
        raise ValueError('Split fractions must be nonnegative and sum to one')
    groups = {}
    for source in sources:
        if not source.get('group'):
            raise ValueError('Every source needs an explicit collection group (including related retries)')
        groups.setdefault(source['group'], []).append(source)
    for group, members in groups.items():
        explicit = {s['split'] for s in members if 'split' in s}
        held = any(s['payload_id'] in held_out_payloads for s in members)
        if len(explicit) > 1 or (held and explicit and explicit != {'test'}):
            raise ValueError('Split leakage: conflicting group or held-out payload assignment')
        u = int(hashlib.sha256(f'{seed}:{group}'.encode()).hexdigest()[:16], 16) / 2**64
        split = 'test' if held else next(iter(explicit), 'train' if u < fractions[0] else
                                        'validation' if u < sum(fractions[:2]) else 'test')
        if split not in ('train', 'validation', 'test'):
            raise ValueError('Unknown split')
        for source in members:
            source['split'] = split
    # Same attempt may never be presented under two identities.
    paths = [str(Path(s['path']).resolve()) for s in sources]
    if len(set(paths)) != len(paths):
        raise ValueError('Duplicate source attempt')
    return sources
