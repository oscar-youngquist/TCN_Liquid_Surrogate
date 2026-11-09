"""Streaming audit; no training-readiness claims are inferred from finite data."""
import argparse
import json
import heapq
from pathlib import Path
from collections import deque
import numpy as np
from .storage import digest


def interior_mask(candidates, boundaries, start, end, history=249, grid_dt=.005, margin=.0001):
    """Require every history tap to lie inside the episode and away from events.

    Tap j is candidates - j*grid_dt. Its distance to the nearest boundary
    must exceed margin. This diagnostic is not the training window mask.

    Boundaries must include feature updates and reconstruction knots. Entire
    history validity (including spills) must be checked separately by the caller.
    """
    times = np.asarray(candidates)[:, None]-np.arange(history)[None, :]*grid_dt
    events = np.sort(np.r_[start, boundaries, end])
    i = np.searchsorted(events, times)
    left = events[np.maximum(i-1, 0)]
    right = events[np.minimum(i, len(events)-1)]
    return ((times > start) & (times < end) &
            (np.minimum(abs(times-left), abs(right-times)) > margin)).all(-1)


def stream_rows(path, verify=False):
    """Stream one chunk at a time; checksum auditing is explicitly opt-in."""
    path = Path(path)
    index = path/'chunks.json'
    if index.exists():
        chunks = json.loads(index.read_text())
    elif (path/'result.json').exists():
        chunks = json.loads((path/'result.json').read_text())['chunks']
    else:
        return
    previous = (-1., -1)
    sequence = 0
    for chunk in chunks:
        if verify and digest(path/chunk['file']) != chunk['sha256']:
            raise ValueError('Chunk checksum mismatch')
        with np.load(path/chunk['file'], allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        for i in range(len(arrays['time_s'])):
            row = {key: value[i] for key, value in arrays.items()}
            row.setdefault('record_sequence', sequence)
            sequence += 1
            order = (float(row['time_s']), int(row['record_sequence']))
            if order < previous:
                raise ValueError('Nonmonotonic sample times/order')
            previous = order
            yield row


def audit(path):
    path = Path(path)
    counts = dict(rows=0, native_intervals=0, valid_intervals=0, resets=0, terminals=0,
                  policy_events=0, feature_events=0, max_force_closure_impulse=0., max_moment_closure_impulse=0.,
                  valid_249_sample_histories=0, valid_history_plus_200ms_windows=0,
                  max_200ms_force_closure_impulse=0., max_200ms_moment_closure_impulse=0.)
    previous_time = -1.
    previous_episode = None
    history = deque(maxlen=289)
    pending = None
    cumulative = None

    def consume_grid(row):
        history.append(row)
        for length, key in ((249, 'valid_249_sample_histories'), (289, 'valid_history_plus_200ms_windows')):
            if len(history) < length:
                continue
            segment = list(history)[-length:]
            if not np.allclose(np.diff([r[0] for r in segment]), .005, rtol=0, atol=1e-8):
                continue
            valid = np.stack([r[2] for r in segment]).all(0)
            valid &= (np.stack([r[1] for r in segment]) == segment[0][1]).all(0)
            counts[key] += int(valid.sum())
            if length == 289 and valid.any():
                residual = segment[-1][3]-segment[-41][3]
                for label, part in (('force', slice(0, 3)), ('moment', slice(3, 6))):
                    key = f'max_200ms_{label}_closure_impulse'
                    counts[key] = max(counts[key], float(np.linalg.norm(residual[valid, part], axis=-1).max()))

    separate_events = (path/'events').exists()
    rows = heapq.merge(stream_rows(path, verify=True), stream_rows(path/'boundaries', verify=True),
                      key=lambda r: (float(r['time_s']), int(r['record_sequence'])))
    for d in rows:
        timestamp, ids = float(d['time_s']), d['episode_id']
        if previous_episode is not None and (ids < previous_episode).any():
            raise ValueError('Episode IDs decreased')
        previous_episode = ids
        if cumulative is None:
            cumulative = np.zeros((len(ids), 6), dtype=np.float64)
        if pending is not None and timestamp > pending[0]:
            consume_grid(pending)
            pending = None
        residual = np.concatenate((d['force_closure_residual_impulse_W'],
                                   d['moment_closure_residual_impulse_W']), -1)
        cumulative += np.where(d['interval_valid'][:, None], residual, 0.)
        if d['target_grid_event']:
            valid = d['state_valid'] & d['held_valid']
            if 'episode_supported' in d:
                valid &= d['episode_supported']
            if 'episode_selected' in d:
                valid &= d['episode_selected']
            pending = (timestamp, ids.copy(), valid.copy(), cumulative.copy())
        counts['rows'] += 1
        counts['native_intervals'] += int(d['interval_native_steps'][0]) if 'interval_native_steps' in d else int(timestamp > max(0., previous_time))
        previous_time = timestamp
        counts['valid_intervals'] += int(d['interval_valid'].sum())
        if not separate_events:
            for key, field in (('resets', 'reset'), ('terminals', 'terminal'),
                               ('policy_events', 'policy_update'), ('feature_events', 'feature_update')):
                counts[key] += int(d[field].sum())
        for label, field in (('force', 'force_closure_residual_impulse_W'), ('moment', 'moment_closure_residual_impulse_W')):
            residual = np.linalg.norm(d[field][d['interval_valid']], axis=-1)
            counts[f'max_{label}_closure_impulse'] = max(counts[f'max_{label}_closure_impulse'],
                                                       float(residual.max(initial=0)))
    if separate_events:
        for event in stream_rows(path/'events', verify=True):
            for key, field in (('resets', 'reset'), ('terminals', 'terminal'),
                               ('policy_events', 'policy_update'), ('feature_events', 'feature_update')):
                counts[key] += int(event[field].sum())
    if pending is not None:
        consume_grid(pending)
    counts['pinn_ready'] = False
    return counts


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt', type=Path)
    print(json.dumps(audit(p.parse_args().attempt), indent=2))
