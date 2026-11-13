from dataclasses import asdict
import pytest
import torch
from liquid_tcn.model import ModelConfig, build_model, CachedInference, ReferenceCachedInference
from liquid_tcn.checkpoint import atomic_save, load_checkpoint
from test_pyramid import identity_norm


def config(architecture='tcn_pyramid_attention', conditioning='film'):
    return ModelConfig(architecture=architecture, property_conditioning=conditioning,
        channels=8, embedding_dim=4, kernel_size=3, dilations=(1,2), long_intervals=4,
        attention_ages_ms=((0,20,40),(0,)))


@pytest.mark.parametrize('architecture',['tcn','tcn_pyramid_attention'])
def test_film_identity_backward_and_checkpoint(tmp_path, architecture):
    torch.manual_seed(7); existing=build_model(config(architecture,'existing'))
    torch.manual_seed(7); film=build_model(config(architecture))
    x=torch.randn(2,25,15); p=torch.randn(2,24)
    torch.testing.assert_close(film(x,p),existing(x,p),atol=0,rtol=0)
    film(x,p).square().mean().backward()
    assert film.film.weight.grad.abs().sum()>0
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in film.parameters())
    path=tmp_path/'model.pt'
    atomic_save(path,dict(model_config=asdict(film.config),model=film.state_dict()))
    saved=load_checkpoint(path);restored=build_model(saved['model_config'])
    restored.load_state_dict(saved['model'])
    torch.testing.assert_close(restored(x,p),film(x,p))
    legacy=asdict(existing.config);legacy.pop('property_conditioning')
    original=build_model(legacy);original.load_state_dict(existing.state_dict(),strict=True)
    torch.testing.assert_close(original(x,p),existing(x,p),atol=0,rtol=0)


@pytest.mark.parametrize('architecture',['tcn','tcn_pyramid_attention'])
def test_streaming_wrap_reset_payload_change_and_lifecycle(architecture):
    torch.manual_seed(5);model=build_model(config(architecture)).eval()
    with torch.no_grad():
        model.film.weight.normal_(0,.05);model.film.bias.normal_(0,.02)
    motion=torch.randn(3,65,15);properties=torch.randn(1,24)
    cached=CachedInference(model,identity_norm(),3)
    calls=[]
    handle=model.embedding.register_forward_hook(lambda *args:calls.append(1))
    cached.set_properties(properties)
    assert cached.modulation.shape[0]==1
    before=len(calls)
    # Versioned shared payload is encoded only once, not once per environment/step.
    for i in range(3):cached.step(motion[:,i],properties)
    assert len(calls)==before
    cached.reset(); starts=[0,0,0]; per_env=properties.expand(3,-1).clone()
    reference=ReferenceCachedInference(model,identity_norm(),3)
    for i in range(65):
        if i==25:
            per_env[1,0]+=1
            cached.set_properties(per_env);starts[1]=i
            reference.reset([1])
        if i==40:
            cached.reset([0]);reference.reset([0]);starts[0]=i
        y,valid=cached.step(motion[:,i],return_all=True)
        prior,prior_valid=reference.step(motion[:,i],per_env,return_all=True)
        torch.testing.assert_close(valid,prior_valid)
        torch.testing.assert_close(y,prior,atol=1e-6,rtol=1e-5,equal_nan=True)
        for env in range(3):
            if i-starts[env]+1>=model.config.receptive_field:
                expected=model(motion[env:env+1,starts[env]:i+1],per_env[env:env+1])[0,-1]
                torch.testing.assert_close(y[env],expected,atol=1e-6,rtol=1e-5)
            else:assert not valid[env] and torch.isnan(y[env]).all()
    # Normal load_state_dict changes parameter versions and invalidates history.
    model.load_state_dict(model.state_dict())
    assert not cached.step(motion[:,0])[1].any()
    model.double()
    assert not cached.step(motion[:,0].double())[1].any()
    assert cached.inputs.dtype==torch.float64
    handle.remove()


def test_inplace_properties_and_payload_bank():
    model=build_model(config()).eval();cache=CachedInference(model,identity_norm(),3)
    bank=torch.randn(2,24);mapping=torch.tensor([0,1,0]);x=torch.randn(3,15)
    cache.set_properties(bank,mapping)
    for _ in range(model.config.receptive_field):cache.step(x)
    values=bank[mapping]
    cache.step(x,values)
    values[0,2]+=1
    _,valid=cache.step(x,values)
    assert valid.tolist()==[False,True,True]
    with torch.inference_mode():
        unversioned=values.clone()
        cache.step(x,unversioned)
        unversioned[2,2]+=1
        _,valid=cache.step(x,unversioned)
        assert not valid[2]
