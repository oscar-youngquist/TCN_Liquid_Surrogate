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
from torch.utils.data._utils.collate import default_collate
from .preprocess import INDEX_DTYPE, SCHEMA_VERSION
from .ported.storage import digest


class WindowDataset(Dataset):
    def __init__(self, path, split='train', cache_size=8, verify=False, verification_receipt=None):
        self.path = Path(path)
        self.manifest = json.loads((self.path / 'manifest.json').read_text())
        if self.manifest['schema_version'] != SCHEMA_VERSION:
            raise ValueError('Unsupported prepared schema')
        self.split, self.cache_size = split, max(1, cache_size)
        self.length = self.manifest['windows'][split]
        self.index = None
        self.cache = OrderedDict()
        self.pid = None
        self.metadata_cache = OrderedDict()
        self.series = {s['id']: s for s in self.manifest['series']}
        c = self.manifest['model']
        self.rf = 1 + 2 * (c['kernel_size'] - 1) * sum(c['dilations'])
        self.input_length = self.rf + c['long_intervals']
        if (self.path / f'{split}.idx').stat().st_size != self.length * INDEX_DTYPE.itemsize:
            raise ValueError('Truncated window index')
        if verification_receipt is not None:
            from .integrity import check_receipt
            check_receipt(self.path, verification_receipt)
        elif verify:
            for name, expected in self.manifest['artifacts'].items():
                if digest(self.path / name) != expected:
                    raise ValueError(f'Prepared artifact checksum mismatch: {name}')

    def __len__(self):
        return self.length

    def __getstate__(self):
        state = self.__dict__.copy()
        # Spawned workers reopen their own mappings instead of pickling live
        # mmap/cache state. _local also guards against inherited state after fork.
        state.update(index=None, cache=OrderedDict(), metadata_cache=OrderedDict(), pid=None)
        return state

    def _local(self):
        if self.pid != os.getpid():
            self.cache = OrderedDict()
            self.metadata_cache = OrderedDict()
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

    def _metadata(self, sid):
        if sid not in self.metadata_cache:
            details = self.series[sid]
            self.metadata_cache[sid] = (np.asarray(details['properties'], dtype=np.float32),
                                        np.asarray(details['interior_dimensions'], dtype=np.float64))
            if len(self.metadata_cache) > self.cache_size * 16:
                self.metadata_cache.popitem(last=False)
        self.metadata_cache.move_to_end(sid)
        return self.metadata_cache[sid]

    def __getitem__(self, index):
        batch = self.__getitems__([index]).batch
        return {k: int(v[0]) if k in ('series', 'end') else v[0] for k, v in batch.items()}

    def __getitems__(self, indices):
        """Gather a batch by shard, then restore sampler order without window copies.

        Only motion needs the full causal history. Labels and recorded physics
        constituents need the contextual endpoints. Allocate each field once per
        batch; no structured-window concatenation or per-sample tensor conversion.
        """
        self._local()
        if not indices or any(i < 0 or i >= self.length for i in indices):
            raise IndexError(indices)
        entries = self.index[indices]
        count, endpoints = len(indices), self.input_length - self.rf + 1
        sid = int(entries[0]['series'])
        dtype = self._shard(sid, int(entries[0]['end']) // self.manifest['shard_rows']).dtype
        arrays = {}
        for key in dtype.names:
            if key in ('motion', 'held_acceleration_W'):
                continue
            field = dtype.fields[key][0]
            arrays[key] = np.empty((count, endpoints) + field.shape, dtype=field.base)
        arrays['motion'] = np.empty((count, self.input_length, 15), dtype=np.float32)
        arrays['properties'] = np.stack([self._metadata(int(e['series']))[0] for e in entries])
        arrays['interior_dimensions'] = np.stack([self._metadata(int(e['series']))[1] for e in entries])
        arrays['series'] = entries['series'].astype(np.int64)
        arrays['end'] = entries['end'].astype(np.int64)
        reads = {}
        for i, entry in enumerate(entries):
            sid, end = int(entry['series']), int(entry['end'])
            begin = end - self.input_length + 1
            pos = begin
            while pos <= end:
                shard, offset = divmod(pos, self.manifest['shard_rows'])
                take = min(end-pos+1, self.manifest['shard_rows']-offset)
                reads.setdefault((sid, shard), []).append((i, offset, take, pos-begin))
                pos += take
        for (sid, shard), segments in reads.items():
            rows = self._shard(sid, shard)
            for i, offset, take, destination in segments:
                arrays['motion'][i, destination:destination+take] = rows['motion'][offset:offset+take]
                skip = max(0, self.rf-1-destination)
                if skip < take:
                    dest = destination+skip-(self.rf-1)
                    for key in dtype.names:
                        if key not in ('motion', 'held_acceleration_W'):
                            arrays[key][i, dest:dest+take-skip] = rows[key][offset+skip:offset+take]
        return FetchedBatch({k: torch.from_numpy(v) for k, v in arrays.items()})


class FetchedBatch(list):
    """Default DataLoader collation still works; our loader avoids its extra stack."""
    def __init__(self, batch):
        self.batch = batch
        # Our collator consumes .batch directly. Construct sample views only
        # for callers using PyTorch's default collator or direct list iteration.
        super().__init__()

    def __len__(self):
        return len(self.batch['motion'])

    def __getitem__(self, index):
        return {k: v[index] for k, v in self.batch.items()}

    def __iter__(self):
        return (self[i] for i in range(len(self)))


def collate_windows(samples):
    return samples.batch if isinstance(samples, FetchedBatch) else default_collate(samples)


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
    options = dict(collate_fn=collate_windows, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory and torch.cuda.is_available(),
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
