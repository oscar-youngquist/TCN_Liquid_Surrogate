"""Diagnostic episode extraction with recorded causal motion retained.

Episode extraction, interval assembly, and window eligibility live in ported.discrete.data.
This adapter only retains z/held inputs and applies same-tick, right-continuous
held-input events; it never interpolates from future samples.
"""
import numpy as np
from .ported.discrete.data import environment_episodes as reference_episodes
from .ported.discrete.data import grid_intervals, eligible_windows, state_ok
from .ported.discrete.analyze import rotation


def environment_episodes(rows, env, native_dt):
    # Callers of this diagnostic adapter provide a finite iterable; production
    # passes at most one 5 ms bin plus the start boundary.
    rows = list(rows)
    episodes = reference_episodes(rows, env, native_dt)
    for row in rows:
        ep, tick = int(row['episode_id'][env]), int(row['native_step'])
        state = episodes[ep]['nodes'].get(tick)
        if state is None:
            continue
        for key in ('z', 'held_acceleration_W'):
            if key in row:
                state[key] = np.asarray(row[key][env], dtype=np.float64).copy()
        if 'x_O_W' not in row and 'held_acceleration_W' in row and 'z' in state:
            R = rotation(state)
            held = state['held_acceleration_W']
            state['z'][9:12] = R.T @ held[:3]
            state['z'][12:15] = R.T @ held[3:]
    return episodes
