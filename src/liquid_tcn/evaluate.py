"""Streaming physical-unit, peak, temporal-lag and conservation reports."""
import math
from itertools import islice
import numpy as np
import torch
from .dataset import to_device
from .model import HEADS
from .objectives import objective, loss_components


def wrench_metrics(prediction, target, valid):
    """Vectorize windows and wrench heads; loop only over the 21 lag offsets."""
    p, y, mask = prediction[:, :, :2], target[:, :, :2], valid[:, :, :2]
    indices = np.argmax(np.where(mask, np.linalg.norm(y, axis=-1), -np.inf), axis=1)
    errors = np.linalg.norm(p - y, axis=-1)
    peaks = np.take_along_axis(errors, indices[:, None, :], axis=1)[:, 0]
    peak_valid = mask.any(axis=1)
    peaks = np.where(peak_valid, peaks, 0.)
    length = p.shape[1]
    a, b = p - p.mean(axis=1, keepdims=True), y - y.mean(axis=1, keepdims=True)
    lag_valid = (mask.all(axis=1) & (np.linalg.norm(a, axis=(1, 3)) > 1e-10)
                 & (np.linalg.norm(b, axis=(1, 3)) > 1e-10) & (length >= 10))
    if not lag_valid.any():
        return dict(peaks=peaks, peak_valid=peak_valid,
                    lag_seconds=np.zeros_like(peaks), lag_valid=lag_valid)
    lags = np.arange(-min(10, length//4), min(10, length//4)+1)
    correlations = []
    for lag in lags:
        aa = a[:, max(lag, 0):length+min(lag, 0)]
        bb = b[:, max(-lag, 0):length+min(-lag, 0)]
        denominator = np.maximum(np.linalg.norm(aa, axis=(1, 3))*np.linalg.norm(bb, axis=(1, 3)), 1e-30)
        correlations.append((aa*bb).sum(axis=(1, 3)) / denominator)
    lag_seconds = lags[np.argmax(np.stack(correlations), axis=0)] * .005
    return dict(peaks=peaks, peak_valid=peak_valid, lag_seconds=lag_seconds, lag_valid=lag_valid)


def numpy_physics(result):
    return {h: {k: result['physics'][h][k].detach().cpu().numpy()
                for k in ('valid', 'residual', 'normalized')} for h in ('short', 'long')}


def batch_metrics(prediction, batch, result):
    # Transfer once per batch and reuse the result for overall and group reports.
    p = prediction.detach().cpu().numpy()
    y = batch['targets'].detach().cpu().numpy()
    valid = batch['target_valid'].detach().cpu().numpy()
    return dict(p=p, y=y, valid=valid, physics=numpy_physics(result),
                wrench=wrench_metrics(p, y, valid))


class Metrics:
    def __init__(self):
        self.count = np.zeros(5, dtype=np.int64)
        self.sse = np.zeros((5, 3)); self.sae = self.sse.copy(); self.bias = self.sse.copy()
        self.maximum = np.zeros(5)
        self.normalized_sum = 0.; self.normalized_count = 0
        self.peak_sum = np.zeros(2); self.peak_max = np.zeros(2); self.peak_count = np.zeros(2, dtype=np.int64)
        self.lag_sum = np.zeros(2); self.lag_count = np.zeros(2, dtype=np.int64)
        self.physics = {}
        self.windows = 0

    def add(self, prediction, batch, result, select=None, prepared=None):
        data = prepared if prepared is not None else batch_metrics(prediction, batch, result)
        if select is None:
            select = slice(None)
        p, y, valid = (data[k][select] for k in ('p','y','valid'))
        error = p - y
        self.windows += len(p)
        self.count += valid.sum((0, 1))
        masked = np.where(valid[..., None], error, 0)
        self.sse += (masked**2).sum((0, 1)); self.sae += abs(masked).sum((0, 1)); self.bias += masked.sum((0, 1))
        self.maximum = np.maximum(self.maximum, np.linalg.norm(masked, axis=-1).max((0, 1)))
        wrench = {k: value[select] for k, value in data['wrench'].items()}
        self.peak_sum += wrench['peaks'].sum(0)
        self.peak_max = np.maximum(self.peak_max, wrench['peaks'].max(0))
        self.peak_count += wrench['peak_valid'].sum(0)
        self.lag_sum += np.where(wrench['lag_valid'], wrench['lag_seconds'], 0.).sum(0)
        self.lag_count += wrench['lag_valid'].sum(0)
        self.add_conservation(data['physics'], select)

    def add_conservation(self, physics, select=slice(None)):
        for horizon in ('short', 'long'):
            r = physics[horizon]
            v, raw, normalized = (r[k][select] for k in ('valid', 'residual', 'normalized'))
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
            peak_wrench={head: dict(mean_error_at_target_peak=float(self.peak_sum[i]/self.peak_count[i]) if self.peak_count[i] else None,
                maximum_error_at_target_peak=float(self.peak_max[i]), windows=int(self.peak_count[i]),
                mean_lag_seconds=float(self.lag_sum[i]/self.lag_count[i]) if self.lag_count[i] else None,
                lag_valid_windows=int(self.lag_count[i])) for i, head in enumerate(HEADS[:2])},
            conservation=self.conservation_report())

    def conservation_report(self):
        return {h: dict(count=a['count'],
                rms_vector=np.sqrt(a['raw']/max(a['count'], 1)).tolist(),
                normalized_rms_vector=np.sqrt(a['normalized']/max(a['count'], 1)).tolist(),
                maximum_vector=a['maximum'].tolist()) for h, a in self.physics.items()}



@torch.no_grad()
def evaluate(model, loader, normalization, device='cpu', max_batches=None, *,
             supervised='mse', huber_delta=1., lambda_short=0., lambda_long=0., physics_dtype='float64'):
    model.eval()
    overall, ground_truth, groups = Metrics(), Metrics(), {}
    sums = np.zeros(5); mse_sums = np.zeros(5); counts = np.zeros(5, dtype=np.int64)
    dataset = loader.dataset
    batches = loader if max_batches is None else islice(loader, max_batches)
    for batch in batches:
        cpu_batch = batch
        batch = to_device(batch, device)
        prediction = model(normalization.normalize('motion', batch['motion']),
                           normalization.normalize('properties', batch['properties']))
        result = objective(prediction, batch, normalization, supervised=supervised,
                           huber_delta=huber_delta, physics_dtype=physics_dtype)
        sums += result['supervised_sums'].cpu().numpy()
        counts += result['supervised_counts'].cpu().numpy()
        if supervised == 'mse':
            mse_sums += result['supervised_sums'].cpu().numpy()
        else:
            target = normalization.normalize('targets', batch['targets'].to(prediction.dtype))
            valid = batch['target_valid'].bool() & torch.isfinite(target).all(-1)
            safe = torch.where(valid[..., None], target, torch.zeros_like(target))
            error = torch.where(valid[..., None], (prediction-safe).square(), torch.zeros_like(prediction))
            mse_sums += error.sum((0, 1, 3)).cpu().numpy()
        prepared = batch_metrics(result['physical'], cpu_batch, result)
        overall.add(result['physical'], batch, result, prepared=prepared)
        truth = objective(normalization.normalize('targets', batch['targets']), batch, normalization)
        # Ground truth is used only for conservation floors, not zero-error
        # prediction metrics or target-versus-itself cross-correlations.
        ground_truth.add_conservation(numpy_physics(truth))
        ids = cpu_batch['series'].tolist()
        labels = np.array([f"{dataset.series[s]['robot']}/{dataset.series[s]['payload_id']}" for s in ids])
        for group in np.unique(labels):
            selected = labels == group
            groups.setdefault(group, Metrics()).add(result['physical'], batch, result, selected, prepared=prepared)
    report = overall.report()
    report.update(metric_name='normalized_supervised_mse_all_heads',
        selection_metric=float(mse_sums.sum()/(3*counts.sum())) if counts.sum() else None,
        normalized_mse_per_head=(mse_sums / np.maximum(3*counts, 1)).tolist() if counts.sum() else None,
        ground_truth_conservation=ground_truth.conservation_report() or None,
        per_robot_payload={k: v.report() for k, v in groups.items()},
        coverage=dict(eligible_windows=len(dataset), evaluated_windows=overall.windows,
            fraction=overall.windows/len(dataset) if len(dataset) else 0,
            preparation_counts=dataset.manifest['counts']),
        weighting='window-weighted; overlapping endpoints counted per window',
        units=dict(force='N', moment='N m', r_B='m', v_B='m/s', S_B='kg m^2/s'),
        lag_definition='demeaned vector cross-correlation within each window, +/- 50 ms; positive means prediction delayed')
    # Reduce numerators and valid counts over the entire validation pass before
    # forming losses. Averaging per-batch means would bias partial batches.
    losses = None
    if counts.sum() and (counts > 0).all():
        physics = {h: dict(loss_by_channel=torch.as_tensor(a['normalized']/a['count']))
                   for h, a in overall.physics.items() if a['count'] > 0}
        if len(physics) == 2:
            aggregate = dict(supervised_sums=torch.as_tensor(sums),
                             supervised_counts=torch.as_tensor(counts), physics=physics)
            components = loss_components(aggregate, lambda_short, lambda_long)
            supervision = float(sums.sum()/(3*counts.sum()))
            secondary = components['short_weighted'] + components['long_weighted']
            losses = dict(supervised=supervision, short=float(physics['short']['loss_by_channel'].sum()),
                long=float(physics['long']['loss_by_channel'].sum()), secondary=secondary,
                loss=supervision+secondary, components=components, supervised_kind=supervised,
                lambda_short=lambda_short, lambda_long=lambda_long,
                supervised_counts=counts.tolist(),
                physics_counts={h: a['count'] for h,a in overall.physics.items()})
    report['losses'] = losses
    return report
