"""Streaming physical-unit, peak, temporal-lag and conservation reports."""
import math
import numpy as np
import torch
from .dataset import to_device
from .model import HEADS
from .objectives import objective


class Metrics:
    def __init__(self):
        self.count = np.zeros(5, dtype=np.int64)
        self.sse = np.zeros((5, 3)); self.sae = self.sse.copy(); self.bias = self.sse.copy()
        self.maximum = np.zeros(5)
        self.normalized_sum = 0.; self.normalized_count = 0
        self.peak_sum = np.zeros(2); self.peak_max = np.zeros(2); self.peak_count = 0
        self.lag_sum = np.zeros(2); self.lag_count = np.zeros(2, dtype=np.int64)
        self.physics = {}
        self.windows = 0

    def add(self, prediction, batch, result, select=None):
        p = prediction.detach().cpu().numpy()
        y = batch['targets'].detach().cpu().numpy()
        valid = batch['target_valid'].detach().cpu().numpy()
        if select is not None:
            p, y, valid = p[select], y[select], valid[select]
        error = p - y
        self.windows += len(p)
        self.count += valid.sum((0, 1))
        masked = np.where(valid[..., None], error, 0)
        self.sse += (masked**2).sum((0, 1)); self.sae += abs(masked).sum((0, 1)); self.bias += masked.sum((0, 1))
        self.maximum = np.maximum(self.maximum, np.linalg.norm(masked, axis=-1).max((0, 1)))
        for i in range(len(p)):
            for head in (0, 1):
                norm_y = np.linalg.norm(y[i, :, head], axis=-1)
                peak = np.argmax(np.where(valid[i, :, head], norm_y, -np.inf))
                pe = np.linalg.norm(error[i, peak, head])
                self.peak_sum[head] += pe
                self.peak_max[head] = max(self.peak_max[head], pe)
                # Lag of vector signal via demeaned cross-correlation; positive
                # lag means prediction follows target. Never join episodes/windows.
                a, b = p[i, :, head], y[i, :, head]
                if valid[i, :, head].all() and len(a) >= 10:
                    a, b = a - a.mean(0), b - b.mean(0)
                    if np.linalg.norm(a) > 1e-10 and np.linalg.norm(b) > 1e-10:
                        lags = range(-min(10, len(a)//4), min(10, len(a)//4)+1)
                        correlations = []
                        for lag in lags:
                            aa = a[max(lag, 0):len(a)+min(lag, 0)]
                            bb = b[max(-lag, 0):len(b)+min(-lag, 0)]
                            correlations.append((aa*bb).sum() / max(np.linalg.norm(aa)*np.linalg.norm(bb), 1e-30))
                        self.lag_sum[head] += list(lags)[int(np.argmax(correlations))] * .005
                        self.lag_count[head] += 1
            self.peak_count += 1
        for horizon in ('short', 'long'):
            r = result['physics'][horizon]
            v = r['valid'].detach().cpu().numpy()
            raw = r['residual'].detach().cpu().numpy()
            normalized = r['normalized'].detach().cpu().numpy()
            if select is not None:
                v, raw, normalized = v[select], raw[select], normalized[select]
            acc = self.physics.setdefault(horizon, dict(count=0, raw=np.zeros(3), normalized=np.zeros(3), maximum=np.zeros(3)))
            acc['count'] += int(v.sum())
            acc['raw'] += (raw[v]**2).sum(axis=(0, 2))
            acc['normalized'] += (normalized[v]**2).sum(axis=(0, 2))
            if v.any():
                acc['maximum'] = np.maximum(acc['maximum'], np.linalg.norm(raw[v], axis=-1).max(0))

    def report(self):
        if not self.windows:
            return dict(status='inconclusive', windows=0)
        return dict(status='measured', windows=self.windows,
            heads={head: dict(count=int(self.count[i]),
                rmse_axis=np.sqrt(self.sse[i]/max(self.count[i], 1)).tolist(),
                mae_axis=(self.sae[i]/max(self.count[i], 1)).tolist(),
                bias_axis=(self.bias[i]/max(self.count[i], 1)).tolist(), maximum_vector_error=float(self.maximum[i]))
                for i, head in enumerate(HEADS)},
            peak_wrench={head: dict(mean_error_at_target_peak=float(self.peak_sum[i]/self.peak_count),
                maximum_error_at_target_peak=float(self.peak_max[i]), windows=self.peak_count,
                mean_lag_seconds=float(self.lag_sum[i]/self.lag_count[i]) if self.lag_count[i] else None,
                lag_valid_windows=int(self.lag_count[i])) for i, head in enumerate(HEADS[:2])},
            conservation={h: dict(count=a['count'],
                rms_vector=np.sqrt(a['raw']/max(a['count'], 1)).tolist(),
                normalized_rms_vector=np.sqrt(a['normalized']/max(a['count'], 1)).tolist(),
                maximum_vector=a['maximum'].tolist()) for h, a in self.physics.items()})


@torch.no_grad()
def evaluate(model, loader, normalization, device='cpu', max_batches=None):
    model.eval()
    overall, ground_truth, groups = Metrics(), Metrics(), {}
    sums = np.zeros(5); counts = np.zeros(5, dtype=np.int64)
    dataset = loader.dataset
    for number, batch in enumerate(loader):
        if max_batches is not None and number >= max_batches:
            break
        batch = to_device(batch, device)
        prediction = model(normalization.normalize('motion', batch['motion']),
                           normalization.normalize('properties', batch['properties']))
        result = objective(prediction, batch, normalization)
        sums += result['supervised_sums'].cpu().numpy()
        counts += result['supervised_counts'].cpu().numpy()
        overall.add(result['physical'], batch, result)
        truth = objective(normalization.normalize('targets', batch['targets']), batch, normalization)
        ground_truth.add(batch['targets'], batch, truth)
        ids = batch['series'].cpu().tolist()
        for group in {f"{dataset.series[s]['robot']}/{dataset.series[s]['payload_id']}" for s in ids}:
            selected = np.array([f"{dataset.series[s]['robot']}/{dataset.series[s]['payload_id']}" == group for s in ids])
            groups.setdefault(group, Metrics()).add(result['physical'], batch, result, selected)
    report = overall.report()
    report.update(metric_name='normalized_supervised_mse_all_heads',
        selection_metric=float(sums.sum()/(3*counts.sum())) if counts.sum() else None,
        normalized_mse_per_head=(sums / np.maximum(3*counts, 1)).tolist() if counts.sum() else None,
        ground_truth_conservation=ground_truth.report().get('conservation'),
        per_robot_payload={k: v.report() for k, v in groups.items()},
        coverage=dict(eligible_windows=len(dataset), evaluated_windows=overall.windows,
            fraction=overall.windows/len(dataset) if len(dataset) else 0,
            preparation_counts=dataset.manifest['counts']),
        weighting='window-weighted; overlapping endpoints counted per window',
        units=dict(force='N', moment='N m', r_B='m', v_B='m/s', S_B='kg m^2/s'),
        lag_definition='demeaned vector cross-correlation within each window, +/- 50 ms; positive means prediction delayed')
    return report
