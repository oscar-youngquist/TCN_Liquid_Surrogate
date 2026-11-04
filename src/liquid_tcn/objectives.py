"""Five supervised heads plus the unchanged reviewed discrete physics."""
import torch
from torch.nn import functional as F
from .ported.discrete.losses import discrete_constraints


def objective(prediction, batch, normalization, lambda_short=0., lambda_long=0.,
              supervised='mse', huber_delta=1., physics_dtype='float64'):
    if not torch.isfinite(prediction).all():
        raise FloatingPointError('Nonfinite network predictions')
    target = normalization.normalize('targets', batch['targets'].to(prediction.dtype))
    valid = batch['target_valid'].bool() & torch.isfinite(target).all(-1)
    safe_target = torch.where(valid[..., None], target, torch.zeros_like(target))
    if supervised == 'mse':
        error = (prediction - safe_target).square()
    elif supervised == 'huber':
        error = F.huber_loss(prediction, safe_target, reduction='none', delta=huber_delta)
    else:
        raise ValueError('supervised must be mse or huber')
    counts = valid.sum((0, 1))
    if (counts == 0).any():
        raise ValueError('Empty supervision for one or more heads')
    sums = torch.where(valid[..., None], error, torch.zeros_like(error)).sum((0, 1, 3))
    # Each valid scalar has equal weight; missing labels never dilute a head.
    supervised_loss = sums.sum() / (3 * counts.sum())
    if physics_dtype not in ('float32', 'float64'):
        raise ValueError('Physics arithmetic must be float32 or float64')
    dtype = getattr(torch, physics_dtype)
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        physical = normalization.denormalize('targets', prediction.to(dtype))
        args = {k: batch[k].to(dtype) for k in
                ('x_O_W', 'u_O_W', 'omega_B', 'q_WB_wxyz', 'mass', 'gravity_W', 'interior_dimensions')}
        args.update(r_B=physical[:, :, 2], v_B=physical[:, :, 3], S_B=physical[:, :, 4],
            F_average_B=physical[:, 1:, 0], tau_average_O_B=physical[:, 1:, 1],
            h=batch['h'][:, 1:].to(dtype), episode_id=batch['episode_id'],
            state_valid=batch['state_valid'], history_valid=batch['history_valid'],
            interval_valid=batch['interval_valid'][:, 1:],
            window_intervals=prediction.shape[1]-1, window_seconds=.005*(prediction.shape[1]-1))
        physics = discrete_constraints(**args)
        for horizon in ('short', 'long'):
            if not physics[horizon]['has_valid']:
                raise ValueError(f'Empty valid {horizon} constraints; refusing a zero-loss success')
            if not torch.isfinite(physics[horizon]['loss']):
                raise FloatingPointError(f'Nonfinite {horizon} physics')
        total = supervised_loss + lambda_short * physics['short']['loss'] + lambda_long * physics['long']['loss']
    return dict(loss=total, supervised=supervised_loss, supervised_sums=sums, supervised_counts=counts,
                physics=physics, physical=physical)
