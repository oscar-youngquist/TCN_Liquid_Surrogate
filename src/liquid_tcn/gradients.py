"""Combine supervised and physics gradients without changing the primary gradient."""
import math
import torch
from .ported.pcgrad import project_aux_against_primary_pcgrad


def backward_objectives(primary, secondary, parameters, strategy='pcgrad', eps=1e-12):
    """Set parameter gradients, returning diagnostics before clipping/AdamW.

    The secondary scalar is the sum of all ramped, weighted physics terms.
    Conflict is measured once across shared parameters, not per layer or loss.
    Parameters unused by both objectives retain grad=None (no optimizer decay).
    Passing secondary=None implements supervised-only training.
    """
    if strategy not in ('pcgrad', 'sum'):
        raise ValueError('gradient_strategy must be pcgrad or sum')
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError('pcgrad_eps must be finite and positive')
    parameters = [p for p in parameters if p.requires_grad]
    if not parameters:
        raise ValueError('No trainable parameters')
    for loss in (primary, secondary):
        if loss is not None and not torch.isfinite(loss).all():
            raise FloatingPointError('Nonfinite objective before gradient projection')

    def gradients(loss, retain_graph=False):
        if loss is None or not loss.requires_grad:
            return [None] * len(parameters)
        return torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=retain_graph)

    a = gradients(primary, retain_graph=secondary is not None)
    b = gradients(secondary)
    finite = [torch.isfinite(g).all() for g in (*a, *b) if g is not None]
    if finite and not torch.stack(finite).all():
        raise FloatingPointError('Nonfinite objective gradient before projection')
    shared = [i for i, (x, y) in enumerate(zip(a, b)) if x is not None and y is not None]
    diagnostics = dict(projected=0., dot_before=0., dot_after=0., cosine_before=0.,
                       primary_norm=0., secondary_norm=0.)
    projected = None
    if shared:
        primary_flat = torch.cat([a[i].reshape(-1) for i in shared])
        secondary_flat = torch.cat([b[i].reshape(-1) for i in shared])
        # Preserve upstream arithmetic, including its additive epsilon. This can
        # leave a small negative dot product near zero; it is not exact orthogonality.
        projected, changed = (project_aux_against_primary_pcgrad(primary_flat, secondary_flat, eps)
                              if strategy == 'pcgrad' else (secondary_flat, False))
        dot = torch.dot(primary_flat, secondary_flat)
        norm_a, norm_b = primary_flat.norm(), secondary_flat.norm()
        diagnostics = dict(projected=float(changed), dot_before=float(dot),
            dot_after=float(torch.dot(primary_flat, projected)),
            cosine_before=float(dot / (norm_a * norm_b).clamp_min(eps)),
            primary_norm=float(norm_a), secondary_norm=float(norm_b))
        if not all(math.isfinite(value) for value in diagnostics.values()):
            raise FloatingPointError('Nonfinite gradient projection diagnostics')
    offset = 0
    finite = []
    for p, x, y in zip(parameters, a, b):
        if x is not None and y is not None:
            size = p.numel()
            gradient = x + projected[offset:offset + size].view_as(p)
            offset += size
        else:
            gradient = x if y is None else y
        if gradient is not None:
            finite.append(torch.isfinite(gradient).all())
        p.grad = None if gradient is None else gradient.detach().clone()
    if finite and not torch.stack(finite).all():
        raise FloatingPointError('Nonfinite merged gradient')
    return diagnostics
