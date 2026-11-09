"""Atomic trusted-local checkpoints, including RNG and acknowledged sampler position."""
import os
from pathlib import Path
import random
import numpy as np
import torch


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state, source_device=None, destination_device=None):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        # Allocations may expose fewer GPUs or move this worker to another logical
        # index. Restore available states, then map the worker's own CUDA stream.
        for index, value in enumerate(state['cuda'][:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(value.cpu(),index)
        if (source_device is not None and destination_device is not None
                and str(source_device).startswith('cuda') and str(destination_device).startswith('cuda')):
            source = torch.device(source_device).index or 0
            if source >= len(state['cuda']):
                raise ValueError('Checkpoint lacks the source worker CUDA RNG state')
            torch.cuda.set_rng_state(state['cuda'][source].cpu(),torch.device(destination_device))


def atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'wb') as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def load_checkpoint(path, device='cpu'):
    # Training checkpoints include Python/NumPy RNG state. Only load trusted files.
    return torch.load(path, map_location=device, weights_only=False)
