"""Bounded synchronous chunks: disk writes provide backpressure, never drop data."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class ChunkWriter:
    def __init__(self, path, chunk_size=128, max_bytes=64*1024*1024):
        if chunk_size < 1 or max_bytes < 1:
            raise ValueError('Buffer bounds must be positive')
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        if list(self.path.glob('chunk-*')):
            raise ValueError('Attempt directory already contains chunks')
        self.chunk_size, self.max_bytes = chunk_size, max_bytes
        self.rows, self.chunks, self.bytes = [], [], 0
        self.failed = False
        self.schema = None

    def append(self, row):
        if self.failed:
            raise RuntimeError('Writer failed; attempt cannot be completed')
        row = {k: np.array(v, copy=True) for k, v in row.items()}
        schema = {k: (v.shape, str(v.dtype)) for k, v in row.items()}
        if self.schema is not None and schema != self.schema:
            self.failed = True
            raise ValueError('Chunk schema changed')
        self.schema = schema
        size = sum(v.nbytes for v in row.values())
        if size > self.max_bytes:
            self.failed = True
            raise ValueError('Single sample exceeds writer byte limit')
        if self.bytes+size > self.max_bytes:
            self.flush()
        self.rows.append(row)
        self.bytes += size
        if len(self.rows) >= self.chunk_size:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        name = f'chunk-{len(self.chunks):06d}.npz'
        tmp = self.path/(name+'.tmp')
        try:
            with tmp.open('wb') as f:
                np.savez(f, **{k: np.stack([r[k] for r in self.rows]) for k in self.rows[0]})
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path/name)
            self.chunks.append(dict(file=name, samples=len(self.rows), sha256=digest(self.path/name)))
            atomic_json(self.path/'chunks.json', self.chunks)
            self.rows.clear()
            self.bytes = 0
        except BaseException:
            self.failed = True
            raise

    def finish(self, metadata):
        if self.failed:
            raise RuntimeError('Writer failed; refusing completion')
        self.flush()
        atomic_json(self.path/'result.json', dict(metadata, chunks=self.chunks))


def verified_complete(path):
    path = Path(path)
    result_path = path/'result.json'
    if not result_path.exists():
        return False
    result = json.loads(result_path.read_text())
    if result.get('status') != 'complete':
        return False
    for chunk in result['chunks']:
        if not (path/chunk['file']).is_file() or digest(path/chunk['file']) != chunk['sha256']:
            raise ValueError(f'Corrupt completed attempt: {path}')
    for stream in result.get('auxiliary_streams', []):
        if not verified_complete(path/stream):
            return False
    return True
