"""Streaming training-only population statistics, with explicit physical-unit floor."""
import numpy as np
import torch


class Moments:
    def __init__(self, shape):
        self.count = np.zeros(shape, dtype=np.int64)
        self.mean = np.zeros(shape, dtype=np.float64)
        self.m2 = np.zeros(shape, dtype=np.float64)

    def add(self, values, mask=None):
        values = np.asarray(values, dtype=np.float64)
        mask = np.isfinite(values) if mask is None else np.broadcast_to(mask, values.shape) & np.isfinite(values)
        n = mask.sum(0)
        mean = np.where(mask, values, 0).sum(0) / np.maximum(n, 1)
        m2 = np.where(mask, (values - mean)**2, 0).sum(0)
        total = self.count + n
        delta = mean - self.mean
        self.mean += delta * n / np.maximum(total, 1)
        self.m2 += m2 + delta**2 * self.count * n / np.maximum(total, 1)
        self.count = total

    def result(self):
        if (self.count == 0).any():
            raise ValueError('No training observations for a normalization channel')
        std = np.sqrt(self.m2 / self.count)
        # A constant training channel must not amplify unseen variation by 1e8.
        std = np.where(std < 1e-8, 1., std)
        return dict(count=self.count.tolist(), mean=self.mean.tolist(), std=std.tolist())


class Normalization:
    def __init__(self, stats):
        self.stats = stats

    def _params(self, name, value):
        s = self.stats[name]
        return (torch.as_tensor(s['mean'], device=value.device, dtype=value.dtype),
                torch.as_tensor(s['std'], device=value.device, dtype=value.dtype))

    def normalize(self, name, value):
        mean, std = self._params(name, value)
        return (value - mean) / std

    def denormalize(self, name, value):
        mean, std = self._params(name, value)
        return value * std + mean
