"""Conflict-only one-way gradient projection; source identity is in docs/imports.json."""
from typing import Tuple
import torch


def project_aux_against_primary_pcgrad(
    primary_grad: torch.Tensor,
    aux_grad: torch.Tensor,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, bool]:
    """
    Project an auxiliary gradient using standard PCGrad logic.

    If dot(aux, primary) < 0, remove from aux the component that conflicts with
    primary. If dot(aux, primary) >= 0, keep aux unchanged.

    Args:
        primary_grad:
            Primary objective gradient, e.g. VAE encoder gradient.

        aux_grad:
            Auxiliary objective gradient, e.g. PINN encoder gradient.

    Returns:
        projected_aux_grad:
            Auxiliary gradient after optional projection.

        was_projected:
            True if a conflicting component was removed.
    """
    dot = torch.dot(aux_grad, primary_grad)

    if dot < 0.0:
        projected = aux_grad - dot * primary_grad / (primary_grad.norm() ** 2 + eps)
        return projected, True

    return aux_grad, False
