"""Reference equivalence and optimizer integration for one-way PCGrad."""
from pathlib import Path
import runpy
import pytest
import torch
from liquid_tcn.gradients import backward_objectives
from liquid_tcn.ported.pcgrad import project_aux_against_primary_pcgrad

reference = runpy.run_path(str(Path(__file__).parent / 'fixtures/pcgrad_reference.py'))['project_aux_against_primary_pcgrad']


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('case', ['conflict', 'aligned', 'orthogonal', 'zero_primary', 'tiny_primary'])
def test_projection_matches_imported_reference(dtype, case):
    a = torch.tensor([1., 2., -3.], dtype=dtype)
    b = dict(conflict=-a, aligned=a, orthogonal=torch.tensor([2., -1., 0.], dtype=dtype),
             zero_primary=-a, tiny_primary=-a)[case]
    if case == 'zero_primary': a.zero_()
    if case == 'tiny_primary': a *= 1e-9
    before = a.clone()
    expected, changed = reference(a, b)
    actual, projected = project_aux_against_primary_pcgrad(a, b)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(a, before, rtol=0, atol=0)
    assert projected == changed


def test_secondary_terms_are_summed_before_one_global_projection():
    p = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    # Individual projection would discard -2*x and retain 3*x, yielding 4.
    # The weighted secondary sum has gradient (+1, 2), so no projection occurs.
    primary = p[0]
    secondary = 2 * (-p[0] + p[1]) + 3 * p[0]
    info = backward_objectives(primary, secondary, [p])
    torch.testing.assert_close(p.grad, torch.tensor([2., 2.], dtype=p.dtype))
    assert info['projected'] == 0


def test_global_shared_projection_and_unused_parameter_ownership():
    a = torch.nn.Parameter(torch.tensor([0.], dtype=torch.float64))
    b = torch.nn.Parameter(torch.tensor([0.], dtype=torch.float64))
    primary_only = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    aux_only = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    unused = torch.nn.Parameter(torch.tensor([1.], dtype=torch.float64))
    primary = a.sum() + 2*b.sum() + 100*primary_only.sum()
    secondary = -3*a.sum() + b.sum() + 4*aux_only.sum()
    info = backward_objectives(primary, secondary, [a,b,primary_only,aux_only,unused])
    g = torch.tensor([1.,2.], dtype=torch.float64)
    aux = torch.tensor([-3.,1.], dtype=torch.float64)
    projected, _ = reference(g, aux)
    torch.testing.assert_close(torch.cat([a.grad,b.grad]), g+projected)
    assert info['projected'] == 1 and info['dot_before'] == -1
    assert abs(info['dot_after']) < 1e-10
    assert primary_only.grad == 100 and aux_only.grad == 4
    assert unused.grad is None
    optimizer = torch.optim.AdamW([unused], lr=.1, weight_decay=.1)
    optimizer.step()
    assert unused.item() == 1


@pytest.mark.parametrize('strategy', ['pcgrad', 'sum'])
def test_supervised_only_is_exact_and_sum_matches_backward(strategy):
    p = torch.nn.Parameter(torch.tensor([1.,2.]))
    backward_objectives(p.square().sum(), None, [p], strategy)
    torch.testing.assert_close(p.grad, 2*p)
    if strategy == 'sum':
        expected = torch.autograd.grad(p.square().sum() - 3*p.sum(), p)[0]
        backward_objectives(p.square().sum(), -3*p.sum(), [p], strategy)
        torch.testing.assert_close(p.grad, expected)


def test_nonfinite_gradients_cannot_be_hidden_by_projection():
    p = torch.nn.Parameter(torch.tensor(0.))
    with pytest.raises(FloatingPointError, match='Nonfinite objective gradient'):
        backward_objectives(p, p.sqrt(), [p])


@pytest.mark.parametrize('eps', [0., -1., float('nan'), float('inf')])
def test_invalid_epsilon(eps):
    p = torch.nn.Parameter(torch.tensor(1.))
    with pytest.raises(ValueError, match='pcgrad_eps'):
        backward_objectives(p, None, [p], eps=eps)
