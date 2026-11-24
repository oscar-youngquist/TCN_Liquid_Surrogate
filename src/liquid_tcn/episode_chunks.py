"""Chronological episode streams over prepared endpoint shards."""
import json
from pathlib import Path
import random

import numpy as np
import torch


def _recorded_resets(source):
    root = Path(source['path'])
    if not root.is_dir():
        raise FileNotFoundError(f'Recorded reset metadata is unavailable: {root}')
    boundaries = sorted((root / 'boundaries').glob('chunk-*.npz'))
    primary = sorted(root.glob('chunk-*.npz'))
    records = {}
    for path in [*(primary[:1] if boundaries else primary), *boundaries]:
        with np.load(path) as chunk:
            indices, environments = np.nonzero(chunk['reset'] & chunk['episode_selected'])
            if len(indices):
                ticks, episodes = chunk['native_step'], chunk['episode_id']
                motion = chunk['z']
                for i, env in zip(indices, environments):
                    records[(env, int(episodes[i, env]))] = (int(ticks[i]), motion[i, env].copy())
    return records


def episode_start(reader, series):
    if not hasattr(reader, 'reset_records'):
        reader.reset_records = {}
    source = series['source']
    if source not in reader.reset_records:
        reader.reset_records[source] = _recorded_resets(reader.manifest['sources'][source])
    reset = reader.reset_records[source].get((series['environment'], series['episode']))
    if reset is None or not np.isfinite(reset[1]).all():
        return None
    first = reader.read(series, 0, 1)[0]
    return reset[1] if int(first['tick']) == reset[0] else None


def episode_end(reader, series):
    """Retain the contiguous prefix from reset; never bridge missing inputs."""
    sid = series['id']
    if sid not in reader.clocks:
        source = reader.manifest['sources'][series['source']]
        resolved = Path(source['path']) / 'resolved.json'
        if resolved.is_file():
            native = json.loads(resolved.read_text()).get('native_dt')
            if native is not None:
                reader.clocks[sid] = native
    stride = reader.native_dt(series)
    if not series.get('assumptions_valid', True) or not stride:
        return 0
    stride = round(.005 / stride)
    previous = None
    for start in range(0, series['rows'], reader.block_rows):
        rows = reader.read(series, start, min(start + reader.block_rows, series['rows']))
        okay = np.isfinite(rows['motion']).all(axis=-1)
        ticks = rows['tick']
        okay[1:] &= np.diff(ticks) == stride
        if previous is not None:
            okay[0] &= ticks[0] - previous == stride
        missing = np.flatnonzero(~okay)
        if len(missing):
            return start + int(missing[0])
        previous = ticks[-1]
    return series['rows']


def chunk_motion(reader, series, start, stop, history, reset_motion):
    rows = reader.read(series, max(0, start - history), stop)
    motion = rows['motion'].astype(np.float32)
    padding = max(0, history - start)
    return np.concatenate((np.broadcast_to(reset_motion, (padding, 15)), motion)).astype(np.float32)


class EpisodeChunks:
    """One ordered chunk per live stream, with durable optimizer-consumed state."""

    def __init__(self, data, model, batch_size, seed, steps=400, split='train', series_ids=None):
        from .trajectories import Trajectories
        self.reader = Trajectories(data, split)
        self.dataset = self.reader.dataset
        self.manifest = self.reader.manifest
        if not model.episode_training:
            self.dataset.check_model(model)
        self.model = model
        self.history = model.receptive_field - 1
        self.horizon = model.consistency_intervals if model.episode_training else model.long_intervals
        self.batch_size, self.seed, self.steps = batch_size, seed, steps
        self.episodes = {}
        for series in self.reader.series:
            if series_ids is not None and series['id'] not in series_ids:
                continue
            reset = episode_start(self.reader, series)
            if reset is None:
                continue
            end = episode_end(self.reader, series)
            if end:
                self.episodes[series['id']] = (series, reset, end)
        if not self.episodes:
            raise ValueError(f'No {split} episodes with recorded reset samples and contiguous inputs')
        self.epoch, self.position = 0, 0
        self.limit = None
        self.states, self.tails = {}, {}
        self._schedule()

    def _epoch_schedule(self, epoch):
        order = list(self.episodes)
        random.Random(self.seed + epoch * 1000003).shuffle(order)
        schedule = []
        for start in range(0, len(order), self.batch_size):
            group = order[start:start + self.batch_size]
            for pos in range(0, max(self.episodes[sid][2] for sid in group), self.steps):
                schedule.append([(sid, pos) for sid in group if self.episodes[sid][2] > pos])
        return schedule

    def _schedule(self):
        self.schedule = self._epoch_schedule(self.epoch)
        self.length = len(self.schedule)

    def updates_for_epochs(self, epochs):
        return sum(len(self._epoch_schedule(epoch)) for epoch in range(epochs))

    def __len__(self):
        return self.length

    def next_epoch(self):
        self.position = 0
        self.epoch += 1
        self._schedule()

    def advance(self, _count):
        self.position += 1

    def state_dict(self):
        return dict(epoch=self.epoch, position=self.position, seed=self.seed,
                    batch_size=self.batch_size, steps=self.steps, states=self.states, tails=self.tails)

    def load_state_dict(self, saved):
        if (saved['batch_size'], saved['steps']) != (self.batch_size, self.steps):
            raise ValueError('Resume requires the original episode stream and chunk sizes')
        self.epoch, self.position, self.seed = saved['epoch'], saved['position'], saved['seed']
        self.states, self.tails = saved['states'], saved['tails']
        self._schedule()

    def __iter__(self):
        stop = self.length if self.limit is None else min(self.length, self.position + self.limit)
        for entries in self.schedule[self.position:stop]:
            yield self.batch(entries)

    def batch(self, entries):
        values = []
        lengths = [min(self.steps, self.episodes[sid][2] - pos) for sid, pos in entries]
        width = max(lengths)
        for (sid, pos), length in zip(entries, lengths):
            series, reset, _ = self.episodes[sid]
            stop = pos + length
            start = max(0, pos - self.horizon)
            rows = self.reader.read(series, start, stop)
            prefix = self.horizon - (pos - start)
            backfill = reset.copy()
            if self.model.reset_backfill == "stationary":
                backfill[3:] = 0
            motion = chunk_motion(self.reader, series, pos, stop, self.history, backfill)
            motion = np.pad(motion, ((0, width - length), (0, 0)), mode='edge')
            fields = {}
            for key in rows.dtype.names:
                if key in ('motion', 'held_acceleration_W'):
                    continue
                field = rows[key].copy()
                widths = ((prefix, width - length),) + ((0, 0),) * (field.ndim - 1)
                fields[key] = np.pad(field, widths, mode='edge')
            real = np.arange(self.horizon + width)
            present = (real >= prefix) & (real < self.horizon + length)
            fields['history_valid'] = present
            fields['state_valid'] &= present
            fields['interval_valid'] &= present
            fields['score_endpoints'] = (real >= self.horizon) & present
            fields.update(motion=motion, properties=np.asarray(series['properties'], dtype=np.float32),
                          interior_dimensions=np.asarray(series['interior_dimensions']),
                          series=sid, end=stop-1, start=pos, length=length, reset_motion=reset.astype(np.float32))
            values.append(fields)
        return {k: torch.from_numpy(np.stack([v[k] for v in values])) for k in values[0]}

    def predict(self, model, norm, batch):
        properties = norm.normalize('properties', batch['properties'])
        entries = list(zip(batch['series'].tolist(), batch['start'].tolist(), batch['length'].tolist()))
        states = ()
        if model.config.temporal_memory == 'mamba_stateful':
            initial = model.temporal.initial_state(properties, norm.normalize('motion', batch['reset_motion']))
            states = tuple(torch.stack([default[i] if pos == 0 else self.states[sid][j].to(default)
                                        for i, (sid, pos, _) in enumerate(entries)])
                           for j, default in enumerate(initial))
        prediction, states = model(norm.normalize('motion', batch['motion']), properties,
                                   memory_state=states, return_state=True, memory_lengths=batch['length'])
        history = []
        for i, (sid, pos, length) in enumerate(entries):
            tail = self.tails.get(sid) if pos else None
            prefix = (prediction.new_zeros(self.horizon, 5, 3) if tail is None else tail.to(prediction))
            history.append(prefix)
            if pos + length == self.episodes[sid][2]:
                self.states.pop(sid, None)
                self.tails.pop(sid, None)
            else:
                self.states[sid] = tuple(state[i].detach().clone() for state in states)
                self.tails[sid] = torch.cat((prefix, prediction[i, :length]))[-self.horizon:].detach().clone()
        return torch.cat((torch.stack(history), prediction), dim=1)
