"""Read-only, bounded chunk/grid preprocessing built on the reviewed data functions."""
from collections import Counter
from itertools import groupby
import heapq
import json
import os
from pathlib import Path
import time
import resource
import numpy as np
from .ported.audit import stream_rows
from .ported.storage import verified_complete, digest, atomic_json
from .data import environment_episodes, grid_intervals, state_ok
from .ported.discrete.analyze import rotation
from .metadata import read_metadata, properties, assign_splits, PROPERTY_FIELDS
from .discovery import IncompleteAttemptError, discover_sources
from .model import ModelConfig
from .normalization import Moments

SCHEMA_VERSION = 1
ROW_DTYPE = np.dtype([
    ('motion', '<f8', (15,)), ('held_acceleration_W', '<f8', (6,)),
    ('targets', '<f8', (5, 3)), ('target_valid', '?', (5,)),
    ('x_O_W', '<f8', (3,)), ('u_O_W', '<f8', (3,)), ('omega_B', '<f8', (3,)),
    ('q_WB_wxyz', '<f8', (4,)), ('mass', '<f8'), ('gravity_W', '<f8', (3,)),
    ('episode_id', '<i8'), ('tick', '<i8'), ('state_valid', '?'),
    ('interval_valid', '?'), ('history_valid', '?'), ('h', '<f8')])
INDEX_DTYPE = np.dtype([('series', '<i8'), ('end', '<i8')])


def verify_attempt(path):
    """Use ported completion verification, additionally reconcile both indices."""
    path = Path(path)
    if not verified_complete(path):
        raise IncompleteAttemptError(f'Incomplete attempt: {path}')
    identities = {}
    for sub in ('', 'events', 'boundaries'):
        root = path / sub
        if not root.exists():
            continue
        if not verified_complete(root):
            raise IncompleteAttemptError(f'Incomplete stream: {root}')
        result = json.loads((root / 'result.json').read_text())
        if (root / 'chunks.json').exists():
            if json.loads((root / 'chunks.json').read_text()) != result['chunks']:
                raise ValueError('Chunk index and completion manifest disagree')
            identities[str(Path(sub) / 'chunks.json')] = digest(root / 'chunks.json')
        identities[str(Path(sub) / 'result.json')] = digest(root / 'result.json')
        for c in result['chunks']:
            if Path(c['file']).name != c['file']:
                raise ValueError('Chunk path must be a local filename')
            identities[str(Path(sub) / c['file'])] = c['sha256']
    for name in ('liquid.json', 'resolved.json', 'job.json', 'provenance.json'):
        if (path / name).exists():
            identities[name] = digest(path / name)
    return identities


def iter_merged_rows(path):
    """Bounded equivalent of ported merged_rows: one decoded chunk per stream."""
    key = lambda r: (int(r['native_step']), int(r['record_sequence']))
    streams = [stream_rows(Path(path) / sub) for sub in ('', 'boundaries', 'events')]
    for identity, copies in groupby(heapq.merge(*streams, key=key), key=key):
        merged = {}
        for row in copies:
            for field, value in row.items():
                if field in merged and not np.array_equal(value, merged[field], equal_nan=True):
                    raise ValueError(f'Conflicting copies at {identity}: {field}')
                merged[field] = value
        yield merged


def grid_blocks(rows, stride, max_rows=100000):
    """One closed grid bin plus its exact start records. Finalize same-time events first."""
    block = []
    end = None
    for tick, group in groupby(rows, key=lambda r: int(r['native_step'])):
        group = list(group)
        if end is None:
            end = (tick // stride + 1) * stride
        if tick > end:
            yield block, end
            block = [r for r in block if int(r['native_step']) == end]
            end = ((tick - 1) // stride + 1) * stride
        block.extend(group)
        if len(block) > max_rows:
            raise ValueError('Too many records in one grid bin; refusing unbounded buffering')
    if block:
        yield block, end


def endpoint_row(state, interval, native_mode=False):
    out = np.zeros((), dtype=ROW_DTYPE)
    R = rotation(state)
    velocity = state['velocity_W']
    held = state.get('held_acceleration_W')
    z = state.get('z')
    if z is None:
        if held is None:
            raise ValueError('Missing both recorded z and held acceleration; no future reconstruction allowed')
        z = np.concatenate((R.T @ state['gravity_W'], R.T @ velocity[:3], R.T @ velocity[3:],
                            R.T @ held[:3], R.T @ held[3:]))
    if held is None:
        held = np.concatenate((R @ z[9:12], R @ z[12:15]))
    out['motion'], out['held_acceleration_W'] = z, held
    for name in ('x_O_W', 'q_WB_wxyz', 'mass', 'gravity_W', 'episode_id', 'tick'):
        out[name] = state[name]
    out['u_O_W'], out['omega_B'] = velocity[:3], R.T @ velocity[3:]
    out['targets'][2:] = np.stack([state[k] for k in ('r_B', 'v_B', 'S_B')])
    out['state_valid'] = state_ok(state) and all(np.isfinite(out[k]).all() for k in
        ('motion', 'held_acceleration_W', 'targets', 'x_O_W', 'u_O_W', 'omega_B', 'q_WB_wxyz', 'mass', 'gravity_W'))
    out['state_valid'] &= np.linalg.norm(state['q_WB_wxyz']) > 1e-6
    out['target_valid'][2:] = out['state_valid']
    if interval is not None:
        # Same common-reference split assembly as analyze.prepare_intervals.
        # Native debug labels use coupling-phase poses, so sum the recorded world
        # impulses directly instead of interpreting those labels as endpoint labels.
        anchor = state['x_O_W']
        IF, J = np.zeros(3), np.zeros(3)
        for part in interval['parts']:
            s = part['endpoint']
            if native_mode:
                force = part['coupling_force_impulse_W']
                moment = part['coupling_moment_impulse_world_origin_W'] - np.cross(anchor, force)
            else:
                rot = rotation(s)
                force = part['h'] * (rot @ part['force_interval_average_B'])
                moment = part['h'] * (rot @ part['moment_interval_average_O_B']) + np.cross(s['x_O_W'] - anchor, force)
            IF += force
            J += moment
        out['targets'][:2] = np.stack((R.T @ IF, R.T @ J)) / interval['h']
        out['h'] = interval['h']
        out['interval_valid'] = interval['valid'] and np.isfinite(out['targets'][:2]).all()
        out['target_valid'][:2] = out['interval_valid'] & out['state_valid']
    return out


class SeriesWriter:
    def __init__(self, root, identity, metadata, shard_rows, model):
        self.root, self.identity, self.metadata = root, identity, metadata
        self.shard_rows, self.model = shard_rows, model
        self.buffer = np.empty(shard_rows, dtype=ROW_DTYPE)
        self.used = self.total = self.shards = self.run = 0
        self.previous_tick = None
        self.mass = self.gravity = None
        self.assumptions_valid = True
        self.indices = open(root / f'series-{identity:06d}.idx', 'wb')
        self.window_count = 0

    def append(self, row, stride):
        if self.previous_tick is not None and int(row['tick']) <= self.previous_tick:
            raise ValueError('Duplicate or decreasing episode endpoint')
        if row['state_valid']:
            if self.mass is None:
                self.mass, self.gravity = float(row['mass']), row['gravity_W'].copy()
            self.assumptions_valid &= (row['mass'] > 0 and np.linalg.norm(row['gravity_W']) > 0 and
                np.isclose(row['mass'], self.mass, rtol=1e-7, atol=1e-9) and
                np.allclose(row['gravity_W'], self.gravity, rtol=1e-7, atol=1e-9) and
                np.isclose(row['mass'], self.metadata['properties'][3], rtol=1e-6, atol=1e-8))
        contiguous = self.previous_tick is not None and int(row['tick']) - self.previous_tick == stride
        self.run = self.run + 1 if contiguous and row['interval_valid'] and row['state_valid'] and self.run else int(row['state_valid'])
        row['history_valid'] = self.run >= self.model.receptive_field
        self.buffer[self.used] = row
        if self.run >= self.model.input_length:
            self.indices.write(np.array((self.identity, self.total), dtype=INDEX_DTYPE).tobytes())
            self.window_count += 1
        self.previous_tick = int(row['tick'])
        self.used += 1
        self.total += 1
        if self.used == self.shard_rows:
            self.flush()

    def flush(self):
        if self.used:
            np.save(self.root / f'series-{self.identity:06d}-{self.shards:06d}.npy', self.buffer[:self.used], allow_pickle=False)
            self.shards += 1
            self.used = 0

    def close(self):
        self.flush()
        self.indices.close()
        return dict(self.metadata, id=self.identity, rows=self.total, shards=self.shards,
                    windows=self.window_count if self.assumptions_valid else 0,
                    rejected_assumptions_windows=0 if self.assumptions_valid else self.window_count,
                    assumptions_valid=bool(self.assumptions_valid))


def prepare(config, output):
    from dataclasses import asdict
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model = ModelConfig(**config.get('model', {}))
    if model.property_dim != len(PROPERTY_FIELDS):
        raise ValueError('property_dim must match the 24 recorded property fields')
    shard_rows = int(config.get('shard_rows', 8192))
    if shard_rows < 1:
        raise ValueError('shard_rows must be positive')
    discovery = None
    if config.get('collection_root') is not None:
        if config.get('sources'):
            raise ValueError('Use collection_root or explicit sources, not both')
        sources, discovery = discover_sources(config['collection_root'], verify_attempt, exclude=(output,))
        atomic_json(output / 'discovery.json', discovery)
        if discovery['errors']:
            raise ValueError(f"Discovery found {len(discovery['errors'])} invalid completed attempts; see {output / 'discovery.json'}")
        if not sources:
            raise ValueError(f"No completed attempts available; see {output / 'discovery.json'}")
    else:
        if not config.get('sources'):
            raise ValueError('Provide collection_root or a nonempty sources list')
        sources = []
        for entry in config['sources']:
            source = dict(entry)
            meta = read_metadata(source['path'])
            source.update(payload_id=meta['payload_id'], robot=meta['robot'],
                          hashes=verify_attempt(source['path']), metadata=meta)
            sources.append(source)
    assign_splits(sources, config.get('seed', 1), config.get('fractions', (.8, .1, .1)),
                  config.get('held_out_payloads', ()))
    # Reject retries assigned to different user groups when the collector identity matches.
    retry_groups = {}
    for s in sources:
        selected = s['metadata']['job']['selected']
        key = json.dumps([selected.get('seed'), selected.get('round'), s['payload_id']], sort_keys=True)
        if key in retry_groups and retry_groups[key] != s['group']:
            raise ValueError('Related collector retries must share a group')
        retry_groups[key] = s['group']
    series, counts = [], Counter()
    for source_index, source in enumerate(sources):
        meta = source['metadata']
        dt = meta['resolved']['native_dt']
        stride = round(.005 / dt)
        active = {}
        assumptions = {}
        last_episode = {}
        native = meta['resolved'].get('save_mode') == 'native'
        for rows, end in grid_blocks(iter_merged_rows(source['path']), stride):
            if not rows:
                continue
            for env in range(len(rows[0]['episode_id'])):
                episodes = environment_episodes(rows, env, dt)
                for ep, episode in episodes.items():
                    if env in last_episode and ep < last_episode[env]:
                        # The retained boundary may include an old terminal; it
                        # is legitimate only if it contributes no new endpoint.
                        if any(t > end - stride for t in episode['nodes']):
                            raise ValueError('Decreasing per-environment episode identity')
                        continue
                    # Match the source episode-wide mass/gravity assumptions,
                    # including supported off-grid states before held inputs exist.
                    for state in episode['nodes'].values():
                        if not bool(state['episode_selected']) or not bool(state['episode_supported']):
                            continue
                        key = (env, ep)
                        mass, gravity = float(state['mass']), state['gravity_W']
                        if key not in assumptions:
                            assumptions[key] = [mass, gravity.copy(), True]
                        initial_mass, initial_gravity, _ = assumptions[key]
                        assumptions[key][2] &= bool(np.isfinite(mass) and mass > 0 and
                            np.isfinite(gravity).all() and np.linalg.norm(gravity) > 0 and
                            np.isclose(mass, initial_mass, rtol=1e-7, atol=1e-9) and
                            np.allclose(gravity, initial_gravity, rtol=1e-7, atol=1e-9))
                    # The retained start row is context, not another partial interval.
                    episode['pieces'] = [p for p in episode['pieces'] if p['end'] > end - stride]
                    grid, rejected = grid_intervals(episode, dt)
                    counts.update(rejected)
                    by_end = {p['end']: p for p in grid}
                    for tick in sorted(episode['nodes']):
                        if tick % stride:
                            continue
                        state = episode['nodes'][tick]
                        if env in active and active[env].metadata['episode'] == ep and active[env].previous_tick == tick:
                            continue
                        if not bool(state['episode_selected']):
                            counts['unselected_grid_endpoints'] += 1
                            continue
                        if env not in active or active[env].metadata['episode'] != ep:
                            if env in active:
                                series.append(active.pop(env).close())
                            # IDs are allocated separately from close order.
                            identity = counts['series_created']; counts['series_created'] += 1
                            details = dict(source=source_index, environment=env, episode=ep,
                                split=source['split'], group=source['group'], robot=source['robot'],
                                payload_id=source['payload_id'], properties=properties(meta, env).tolist(),
                                interior_dimensions=meta['liquid']['geometry']['interior'])
                            active[env] = SeriesWriter(output, identity, details, shard_rows, model)
                        row = endpoint_row(state, by_end.get(tick), native)
                        counts['grid_endpoints'] += 1
                        counts['invalid_grid_endpoints'] += int(not row['state_valid'])
                        active[env].append(row, stride)
                        last_episode[env] = ep
        series.extend(w.close() for w in active.values())
        for item in series:
            if item['source'] != source_index:
                continue
            check = assumptions.get((item['environment'], item['episode']))
            if check is None or not check[2]:
                item['assumptions_valid'] = False
                item['rejected_assumptions_windows'] += item['windows']
                item['windows'] = 0
    series.sort(key=lambda s: s['id'])
    stats = {'motion': Moments((15,)), 'targets': Moments((5, 3)), 'properties': Moments((24,))}
    split_counts = {}
    for split in ('train', 'validation', 'test'):
        total = 0
        with open(output / f'{split}.idx', 'wb') as dest:
            for s in series:
                if s['split'] != split or not s['assumptions_valid']:
                    continue
                index = output / f"series-{s['id']:06d}.idx"
                with open(index, 'rb') as src:
                    for block in iter(lambda: src.read(1024 * 1024), b''):
                        dest.write(block)
                total += s['windows']
                if split == 'train' and s['windows']:
                    stats['properties'].add(np.asarray(s['properties'])[None])
                    for shard in range(s['shards']):
                        rows = np.load(output / f"series-{s['id']:06d}-{shard:06d}.npy", mmap_mode='r')
                        stats['motion'].add(rows['motion'], rows['state_valid'][:, None])
                        stats['targets'].add(rows['targets'], rows['target_valid'][..., None])
        split_counts[split] = total
    if not split_counts['train']:
        raise ValueError('No eligible training windows; prepared output is incomplete')
    counts['eligible_windows'] = sum(split_counts.values())
    counts['assumption_rejected_windows'] = sum(s['rejected_assumptions_windows'] for s in series)
    counts['grid_endpoints_without_complete_window'] = sum(s['rows'] - s['windows'] for s in series)
    normalization = {k: v.result() for k, v in stats.items()}
    artifacts = {p.name: digest(p) for p in output.iterdir() if p.suffix in ('.npy', '.idx')}
    manifest = dict(schema_version=SCHEMA_VERSION, imported_commit='300d3288b9bce83ccdc31b38bd7926f9484787e7',
        model=asdict(model), config=config, discovery=discovery, shard_rows=shard_rows, series=series,
        sources=[{k: v for k, v in s.items() if k != 'metadata'} for s in sources],
        property_fields=PROPERTY_FIELDS, normalization=normalization, counts=dict(counts),
        windows=split_counts, artifacts=artifacts,
        wrench_convention='completed 5 ms interval; endpoint base frame; fixed endpoint base origin; liquid-on-robot',
        profile=dict(seconds=time.perf_counter()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    atomic_json(output / 'manifest.json', manifest)  # Completion marker written last.
    return manifest
