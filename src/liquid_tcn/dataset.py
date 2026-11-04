"""Worker-local read-only mmap access and resumable bounded-memory sampling."""
from collections import OrderedDict
import json
import math
import os
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from .preprocess import INDEX_DTYPE, SCHEMA_VERSION
from .ported.storage import digest


class WindowDataset(Dataset):
    def __init__(self, path, split='train', cache_size=8, verify=False):
        self.path = Path(path)
        self.manifest = json.loads((self.path / 'manifest.json').read_text())
        if self.manifest['schema_version'] != SCHEMA_VERSION:
            raise ValueError('Unsupported prepared schema')
        self.split, self.cache_size = split, max(1, cache_size)
        self.length = self.manifest['windows'][split]
        self.index = None
        self.cache = OrderedDict()
        self.pid = None
        self.series = {s['id']: s for s in self.manifest['series']}
        c = self.manifest['model']
        self.rf = 1 + 2 * (c['kernel_size'] - 1) * sum(c['dilations'])
        self.input_length = self.rf + c['long_intervals']
        if (self.path / f'{split}.idx').stat().st_size != self.length * INDEX_DTYPE.itemsize:
            raise ValueError('Truncated window index')
        if verify:
            for name, expected in self.manifest['artifacts'].items():
                if digest(self.path / name) != expected:
                    raise ValueError(f'Prepared artifact checksum mismatch: {name}')

    def __len__(self):
        return self.length

    def __getstate__(self):
        state = self.__dict__.copy()
        # Spawned workers reopen their own mappings instead of pickling live
        # mmap/cache state. _local also guards against inherited state after fork.
        state.update(index=None, cache=OrderedDict(), pid=None)
        return state

    def _local(self):
        if self.pid != os.getpid():
            self.cache = OrderedDict()
            self.index = np.memmap(self.path / f'{self.split}.idx', dtype=INDEX_DTYPE, mode='r') if self.length else None
            self.pid = os.getpid()

    def _shard(self, series, shard):
        key = (series, shard)
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            # Read-only .npy mappings avoid repeated NPZ decompression. LRU eviction
            # bounds open shard mappings per worker; it does not materialize a dataset.
            self.cache[key] = np.load(self.path / f'series-{series:06d}-{shard:06d}.npy', mmap_mode='r', allow_pickle=False)
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        return self.cache[key]

    def __getitem__(self, index):
        if not 0 <= index < self.length:
            raise IndexError(index)
        self._local()
        entry = self.index[index]
        # A compact index stores just series ID and inclusive final row. Overlap
        # is represented by shared shard rows, not duplicated full training windows.
        sid, end = int(entry['series']), int(entry['end'])
        begin = end - self.input_length + 1
        shard_rows = self.manifest['shard_rows']
        pieces = []
        pos = begin
        while pos <= end:
            shard, offset = divmod(pos, shard_rows)
            take = min(end - pos + 1, shard_rows - offset)
            pieces.append(np.array(self._shard(sid, shard)[offset:offset + take], copy=True))
            pos += take
        rows = np.concatenate(pieces)
        # Copy only this window (possibly across shard boundaries). Keep all
        # 289 motion inputs but only the final 41 labels/recorded physics states:
        # these align exactly with the model's RF-1 contextual-output trim.
        endpoints = rows[self.rf - 1:]
        details = self.series[sid]
        # Preserve metadata at float64; network inputs alone are converted to FP32.
        out = {k: torch.from_numpy(np.array(endpoints[k], copy=True)) for k in endpoints.dtype.names
               if k not in ('motion', 'held_acceleration_W')}
        out['motion'] = torch.from_numpy(rows['motion'].astype(np.float32))
        out['properties'] = torch.tensor(details['properties'], dtype=torch.float32)
        out['interior_dimensions'] = torch.tensor(details['interior_dimensions'], dtype=torch.float64)
        out['series'] = sid
        out['end'] = end
        return out


class EpochSampler(Sampler):
    """Seeded affine permutation, O(1) state, no dataset-sized randperm.

    Acknowledgement is explicit: DataLoader prefetch must never advance saved
    training progress. Every index appears once per epoch; ordering is less mixed
    than a full random permutation, a deliberate memory/reproducibility tradeoff.
    """
    def __init__(self, length, seed=1, epoch=0, position=0):
        self.length, self.seed, self.epoch, self.position = length, seed, epoch, position
        self.limit = None

    def __iter__(self):
        n = self.length
        if not n:
            return
        rng = random.Random(self.seed + self.epoch * 1000003)
        a = rng.randrange(1, n + 1)
        # An affine map modulo n is a permutation exactly when gcd(a,n)=1.
        # This visits every window once with constant-memory reproducible state.
        while math.gcd(a, n) != 1:
            a = a % n + 1
        b = rng.randrange(n)
        stop = n if self.limit is None else min(n, self.position + self.limit)
        for pos in range(self.position, stop):
            yield (a * pos + b) % n

    def __len__(self):
        return self.length - self.position

    def advance(self, count):
        self.position += count
        if self.position > self.length:
            raise ValueError('Sampler advanced beyond epoch')

    def next_epoch(self):
        self.epoch += 1
        self.position = 0

    def state_dict(self):
        return dict(length=self.length, seed=self.seed, epoch=self.epoch, position=self.position)

    def load_state_dict(self, state):
        if state['length'] != self.length or state['seed'] != self.seed:
            raise ValueError('Sampler identity mismatch')
        self.epoch, self.position = state['epoch'], state['position']


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def make_loader(dataset, batch_size=16, num_workers=0, prefetch_factor=2,
                pin_memory=True, seed=1, sampler=None):
    options = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory and torch.cuda.is_available(),
                   worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(seed), sampler=sampler)
    # PyTorch forbids worker prefetch/persistence options with zero workers.
    # A dedicated generator seeds workers without consuming model RNG state.
    if num_workers:
        options.update(prefetch_factor=prefetch_factor, persistent_workers=True, multiprocessing_context='spawn')
    return DataLoader(dataset, **options)


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def close_loader(loader):
    """Shut persistent workers down while Python/Torch IPC is still alive."""
    iterator = getattr(loader, '_iterator', None)
    if iterator is not None:
        iterator._shutdown_workers()
        loader._iterator = None
