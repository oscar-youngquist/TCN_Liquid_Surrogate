"""Small causal/streaming and real trainer checks for the optional architecture."""
from dataclasses import asdict
import math
import numpy as np
import pytest
import torch
from liquid_tcn.model import ModelConfig, LiquidTCN, build_model, CachedInference
from liquid_tcn.pyramid import SparseCrossAttention
from liquid_tcn.normalization import Normalization
from liquid_tcn.dataset import WindowDataset, make_loader
from liquid_tcn.checkpoint import load_checkpoint
from liquid_tcn.train import train
from liquid_tcn.evaluate import evaluate
from liquid_tcn.inference import load_export
from helpers import preparation, make_attempt


def config(**kw):
    return ModelConfig(**dict(dict(architecture='tcn_pyramid_attention', channels=8,
        embedding_dim=4,kernel_size=2,dilations=(1,2),long_intervals=4,
        attention_ages_ms=((0,20,40),(0,10))), **kw))


def identity_norm():
    return Normalization({k:dict(mean=np.zeros(s).tolist(),std=np.ones(s).tolist())
        for k,s in [('motion',(15,)),('properties',(24,)),('targets',(5,3))]})


def test_context_and_legacy_loading():
    c=ModelConfig(architecture='tcn_pyramid_attention')
    assert c.block_receptive_fields==(9,25,57,121,249)
    assert c.receptive_field==249 and c.input_length==289
    assert sum(map(len,c.attention_lags))==28
    assert config().receptive_field==11  # first block RF=3, oldest lag=8
    assert config(sample_seconds=.01).receptive_field==8
    old=dict(channels=8,embedding_dim=4,kernel_size=2,dilations=[1,2],long_intervals=4)
    legacy=LiquidTCN(ModelConfig(**old))
    restored=build_model(old)
    restored.load_state_dict(legacy.state_dict(),strict=True)
    assert type(restored) is LiquidTCN
    x=torch.randn(2,11,15);p=torch.randn(2,24)
    torch.testing.assert_close(restored(x,p),legacy(x,p),atol=0,rtol=0)
    with pytest.raises(ValueError,match='integer multiples'):
        config(attention_ages_ms=((0,1),(0,)))
    with pytest.raises(ValueError,match='Supply attention_ages_ms'):
        ModelConfig(architecture='tcn_pyramid_attention',dilations=(1,2))


@pytest.mark.parametrize('queries',[1,3])
def test_custom_attention_reference_and_sdpa(queries):
    torch.manual_seed(13)
    module=SparseCrossAttention(8,12,4,bias=True).double()
    q=torch.randn(2,5,queries,8,dtype=torch.float64,requires_grad=True)
    mem=torch.randn(2,5,7,8,dtype=torch.float64,requires_grad=True)
    output,weights=module(q,mem,True)
    # Intentionally straightforward per-head reference with the same weights.
    qp=module.q(q);kp,vp=module.kv(mem).chunk(2,-1)
    results=[];references=[]
    for h in range(4):
        sl=slice(h*3,(h+1)*3)
        w=torch.softmax(qp[...,sl] @ kp[...,sl].transpose(-1,-2)/math.sqrt(3),-1)
        references.append(w);results.append(w @ vp[...,sl])
    expected=module.projection(torch.cat(results,-1))
    torch.testing.assert_close(output,expected)
    torch.testing.assert_close(weights,torch.stack(references,-3))
    gradients=torch.autograd.grad(output.square().sum(),(q,mem),retain_graph=True)
    module.backend='sdpa'
    other=module(q,mem)
    torch.testing.assert_close(output,other)
    for a,b in zip(gradients,torch.autograd.grad(other.square().sum(),(q,mem))):
        torch.testing.assert_close(a,b)
    module.dropout.p=.5;module.eval()
    torch.testing.assert_close(module(q,mem),other)


@pytest.mark.parametrize('backend',['custom','sdpa'])
def test_causality_gradients_cache_and_selective_reset(backend):
    torch.manual_seed(5)
    c=config(attention_backend=backend)
    model=build_model(c)
    x=torch.randn(2,36,15);p=torch.randn(2,24)
    y,weights=model(x,p,return_attention=True)
    assert y.shape==(2,26,5,3) and weights.shape==(2,26,4,1,5)
    y.square().mean().backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
    changed=x.clone();changed[:,20:]+=100
    torch.testing.assert_close(model(changed,p)[:,:10],model(x,p)[:,:10],atol=0,rtol=0)
    cache=CachedInference(model,identity_norm(),2)
    sizes=[tuple(r.shape) for r in cache.histories]
    for i in range(36):
        actual,valid=cache.step(x[:,i],p,reset=[0] if i==17 else None)
        for env in (0,1):
            begin=17 if env==0 and i>=17 else 0
            eligible=i-begin+1>=c.receptive_field
            assert bool(valid[env])==eligible
            if eligible:
                expected=model(x[env:env+1,begin:i+1],p[env:env+1])[:,-1,:4].flatten(1)[0]
                torch.testing.assert_close(actual[env],expected,atol=1e-6,rtol=1e-5)
            else:
                assert torch.isnan(actual[env]).all()
    assert [tuple(r.shape) for r in cache.histories]==sizes
    assert torch.isnan(model(x,p,contextual_only=False)[:,:c.receptive_field-1]).all()


def test_prepared_history_reset_and_alignment(tmp_path):
    from liquid_tcn.preprocess import prepare
    attempt=make_attempt(tmp_path/'attempt',samples=50,reset_at=25)
    c=config()
    manifest=prepare(dict(model=asdict(c),shard_rows=7,
        sources=[dict(path=str(attempt),group='a',split='train')]),tmp_path/'data')
    ds=WindowDataset(tmp_path/'data');ds.check_model(c);ds._local()
    for entry in ds.index:
        sid,end=int(entry['series']),int(entry['end'])
        begin=end-c.input_length+1
        # Read history across shard boundaries and verify no reset is included.
        ticks=[];episodes=[]
        for index in range(begin,end+1):
            row=ds._shard(sid,index//7)[index%7]
            ticks.append(row['tick']);episodes.append(row['episode_id'])
        assert len(set(episodes))==1 and np.all(np.diff(ticks)==5)
    batch=next(iter(make_loader(ds,batch_size=2)))
    assert batch['motion'].shape[1]==c.input_length
    assert batch['targets'].shape[1]==c.long_intervals+1
    assert build_model(c)(batch['motion'],batch['properties']).shape==batch['targets'].shape
    with pytest.raises(ValueError,match='Prepare a separate dataset'):
        ds.check_model(config(attention_ages_ms=((0,100),(0,))))


def test_attention_train_resume_evaluate_and_export(tmp_path):
    path,_=preparation(tmp_path,model=asdict(config()))
    cfg=dict(updates=3,pretrain_updates=1,ramp_updates=1,batch_size=2,threads=2,
             num_workers=0,validation_every=3,checkpoint_every=1,validation_batches=1,patience=10,progress=False)
    train(path,tmp_path/'full',cfg)
    train(path,tmp_path/'resumed',cfg,max_updates=1)
    train(path,tmp_path/'resumed',resume=tmp_path/'resumed/latest.pt')
    full=load_checkpoint(tmp_path/'full/latest.pt');saved=load_checkpoint(tmp_path/'resumed/latest.pt')
    assert full['update']==saved['update']==3 and full['sampler']==saved['sampler']
    for key in full['model']:
        torch.testing.assert_close(full['model'][key],saved['model'][key],rtol=0,atol=0)
    model=build_model(saved['model_config']);model.load_state_dict(saved['model'])
    report=evaluate(model,make_loader(WindowDataset(path,'test'),batch_size=2),
                    Normalization(saved['normalization']),max_batches=1)
    assert math.isfinite(report['selection_metric'])
    # Export interface accepts the same model/normalization payload as a checkpoint.
    cached=load_export(tmp_path/'resumed/latest.pt',environments=2)
    assert cached.model.config.architecture=='tcn_pyramid_attention'


def test_model_config_reaches_sweeps_ablations_and_benchmark(tmp_path):
    from liquid_tcn.sweep import generate_trials, read_model_config
    from liquid_tcn.staged_plan import candidates
    from liquid_tcn.experiment import configurations
    from liquid_tcn.benchmark import resolve_configs
    from pathlib import Path
    selected_path=Path(__file__).resolve().parents[1]/'configs/pyramid_attention.json'
    model=read_model_config(selected_path)
    trials=generate_trials(tmp_path/'data',tmp_path/'sweep',['cpu'],2,model)
    assert len(trials)==16 and all(v['model']==model for v in trials.values())
    assert all(cfg['model']==model for _,cfg in candidates('1',trials['00'],{}))
    ablations=configurations(tmp_path/'data',tmp_path/'experiment',['cpu'],2,selected=trials['00'])
    assert len(ablations)==12 and all(v['model']==model for v in ablations.values())
    benchmark=resolve_configs(tmp_path/'data',tmp_path/'benchmark','cpu',2,model)
    assert all(v['model']==model for phase in benchmark.values() for v in phase.values())
