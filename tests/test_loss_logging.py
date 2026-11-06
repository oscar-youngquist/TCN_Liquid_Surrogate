import pytest
import torch
from liquid_tcn.objectives import loss_components
from liquid_tcn.model import HEADS


def test_loss_component_means_and_contributions_with_unequal_coverage():
    # Three coordinates per valid vector. The synthetic sums can represent MSE
    # or Huber errors; logging must use the actual objective's unreduced sums.
    counts = torch.tensor([1, 2, 3, 4, 5])
    means = torch.tensor([1., 2., 3., 4., 5.], dtype=torch.float64)
    sums = (3 * counts * means).requires_grad_()
    result = dict(supervised_sums=sums, supervised_counts=counts,
                  physics={h:dict(loss_by_channel=torch.tensor([1.,2.,3.], requires_grad=True))
                           for h in ('short','long')})
    values = loss_components(result, .25, 0.)
    for i, head in enumerate(HEADS):
        assert values[f'supervised/{head}'] == means[i].item()
        assert values[f'supervised_contribution/{head}'] == pytest.approx(sums[i].item()/(3*counts.sum().item()))
    assert sum(values[f'supervised_contribution/{h}'] for h in HEADS) == pytest.approx((sums.sum()/(3*counts.sum())).item())
    assert values['short_weighted/P'] == .5
    assert values['short_weighted'] == 1.5
    assert values['long/H'] == 3. and values['long_weighted/H'] == 0.
    assert all(isinstance(value, float) for value in values.values())
    assert sums.grad is None
