from dataclasses import asdict

import torch

from helpers import make_attempt
from test_mamba_memory import config
from test_pyramid import identity_norm
from liquid_tcn.episode_chunks import EpisodeChunks
from liquid_tcn.model import CachedInference, build_model
from liquid_tcn.objectives import objective
from liquid_tcn.preprocess import prepare


def test_reset_state_gradients_detachment_and_stepwise_parity(tmp_path):
    cfg = config('mamba_stateful')
    source = make_attempt(tmp_path/'raw', samples=110, shift=.25, initial_reset=True)
    data = tmp_path/'prepared'
    prepare(dict(model=asdict(cfg), sources=[dict(path=str(source), group='train', split='train')]), data)
    chunks = EpisodeChunks(data, cfg, batch_size=2, seed=1, steps=50)
    norm = identity_norm()
    model = build_model(cfg)
    first, second, third = iter(chunks)
    assert torch.all(first['reset_motion'][:, 3] != 0)
    assert not first['score_endpoints'][:, :40].any()
    assert first['score_endpoints'][:, 40:].all()
    prediction = chunks.predict(model, norm, first)
    result = objective(prediction, first, norm, 1., 1.)
    assert int(result['physics']['short']['count']) == 2 * 49
    assert int(result['physics']['long']['count']) == 2 * 10
    result['loss'].backward()
    for p in (model.temporal.initializer[-1].weight, model.temporal.in_proj.weight):
        assert p.grad is not None and p.grad.abs().sum() > 0
    assert all(state.grad_fn is None for states in chunks.states.values() for state in states)
    assert all(tail.grad_fn is None for tail in chunks.tails.values())

    # Checkpointed continuation reuses the same detached state and boundary predictions.
    chunks.advance(len(first['series']))
    saved = chunks.state_dict()
    resumed = EpisodeChunks(data, cfg, batch_size=2, seed=1, steps=50)
    resumed.load_state_dict(dict(saved, states=saved['states'].copy(), tails=saved['tails'].copy()))
    assert resumed.position == 1
    torch.testing.assert_close(next(iter(resumed))['start'], second['start'])
    model.eval()
    with torch.no_grad():
        carried = chunks.predict(model, norm, second)
        torch.testing.assert_close(carried, resumed.predict(model, norm, second))
        result = objective(carried, second, norm, 1., 1.)
        assert int(result['physics']['short']['count']) == 2 * 50
        assert int(result['physics']['long']['count']) == 2 * 50
        result = objective(chunks.predict(model, norm, third), third, norm, 1., 1.)
        assert int(result['physics']['long']['count']) == 2 * 10

        # A learned nonzero state makes accidental advancement during prefill visible.
        model.temporal.initializer[-1].bias.fill_(.05)
        props, reset = first['properties'], first['reset_motion']
        rf = cfg.receptive_field
        real = first['motion'][:, rf-1:]
        cache = CachedInference(model, norm, 2)
        cache.set_properties(props)
        cache.initialize_reset(reset)

        def chronological(x, p, r):
            history = r[:, None].expand(-1, rf-1, -1)
            state = model.temporal.initial_state(p, r)
            return model(torch.cat((history, x), dim=1), p, memory_state=state)

        both = chronological(real, props, reset)
        next_reset = reset[:1] + .2
        changed = chronological(real[:1, 19:], props[:1], next_reset)
        for step in range(real.shape[1]):
            pred, valid = cache.step(real[:, step], reset=[0] if step == 19 else None,
                                     reset_motion=next_reset if step == 19 else None, return_all=True)
            assert valid.all()
            torch.testing.assert_close(pred[1], both[1, step], atol=2e-5, rtol=2e-4)
            expected = both[0, step] if step < 19 else changed[0, step-19]
            torch.testing.assert_close(pred[0], expected, atol=2e-5, rtol=2e-4)
