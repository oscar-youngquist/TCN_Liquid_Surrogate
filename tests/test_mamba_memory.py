"""Official Mamba CUDA checks are optional; indexing/physics checks run on CPU."""
from dataclasses import asdict
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import pytest
import torch
from helpers import make_attempt
from test_pyramid import identity_norm
from liquid_tcn.model import ModelConfig, build_model, CachedInference
from liquid_tcn.preprocess import prepare
from liquid_tcn.sequences import SequenceDataset
from liquid_tcn.dataset import make_loader, to_device
from liquid_tcn.objectives import objective
from liquid_tcn.normalization import Normalization


def config(memory='mamba'):
    return ModelConfig(architecture='tcn_pyramid_attention',property_conditioning='film',
        channels=8,embedding_dim=4,kernel_size=2,dilations=[1,2],
        attention_ages_ms=[[0,20,40],[0]],temporal_memory=memory)


def prepared(tmp_path):
    sources=[]
    for i,split in enumerate(('train','validation','test')):
        source=make_attempt(tmp_path/split,samples=180,reset_at=90,shift=i)
        sources.append(dict(path=str(source),group=split,split=split))
    path=tmp_path/'data'
    manifest=prepare(dict(model=asdict(config('none')),sources=sources,shard_rows=17),path)
    return path,manifest


def test_clip_boundaries_and_fixed_long_horizon(tmp_path):
    path,manifest=prepared(tmp_path)
    ds=SequenceDataset(path,config(),burn_in=3,scored_intervals=60)
    assert len(ds)>0
    batch=next(iter(make_loader(ds,batch_size=2,num_workers=0)))
    assert batch['motion'].shape[1]==config().receptive_field+3+60
    assert batch['targets'].shape[1]==61
    assert (batch['episode_id']==batch['episode_id'][:,:1]).all()
    assert (batch['tick'][:,1:]-batch['tick'][:,:-1]==5).all()
    norm=Normalization(manifest['normalization'])
    p=norm.normalize('targets',batch['targets']).float().requires_grad_()
    result=objective(p,batch,norm,1.,1.)
    assert int(result['physics']['short']['count'])==2*60
    assert int(result['physics']['long']['count'])==2*21
    values=[]
    for start in range(21):
        sl={k:(v[:,start:start+41] if k not in ('motion','properties','interior_dimensions','series','end') else v) for k,v in batch.items()}
        values.append(objective(p[:,start:start+41],sl,norm)['physics']['long']['loss'])
    torch.testing.assert_close(result['physics']['long']['loss'],torch.stack(values).mean())
    for horizon in ('short','long'):
        g=torch.autograd.grad(result['physics'][horizon]['loss'],p,retain_graph=True)[0]
        assert torch.isfinite(g).all()
    # Prepared windows and their history contract remain unchanged.
    assert manifest['model']['long_intervals']==40
    from liquid_tcn.staged_validation import segment_monitoring_subset
    subset=segment_monitoring_subset(path,tmp_path/'subset.json')
    assert subset['sampling_unit']=='segment' and subset['indices']
    assert segment_monitoring_subset(path,tmp_path/'subset.json')==subset


def official_cuda():
    if not torch.cuda.is_available():pytest.skip('Official Mamba requires CUDA')
    pytest.importorskip('mamba_ssm')
    device='cuda:1' if torch.cuda.device_count()>1 else 'cuda:0'
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    return device


def test_official_causality_cache_resets_and_physics_gradients(tmp_path):
    device=official_cuda();torch.manual_seed(31)
    model=build_model(config()).to(device).eval()
    x=torch.randn(2,45,15,device=device);props=torch.randn(2,24,device=device)
    y=model(x,props)
    changed=x.clone();changed[:,25:]+=20
    torch.testing.assert_close(model(changed,props)[:,:15],y[:,:15])
    cache=CachedInference(model,identity_norm(),2)
    cache.set_properties(props)
    pointers=[s.data_ptr() for s in cache.recurrent_states]
    for i in range(45):
        actual,valid=cache.step(x[:,i],reset=[0] if i==19 else None,return_all=True)
        for env in (0,1):
            start=19 if env==0 and i>=19 else 0
            if i-start+1>=model.config.receptive_field:
                expected=model(x[env:env+1,start:i+1],props[env:env+1])[0,-1]
                torch.testing.assert_close(actual[env],expected,atol=2e-5,rtol=2e-4)
    assert pointers==[s.data_ptr() for s in cache.recurrent_states]
    untouched=[s[1].clone() for s in cache.recurrent_states]
    changed_props=props.clone();changed_props[0,0]+=1
    cache.set_properties(changed_props)
    for s,other in zip(cache.recurrent_states,untouched):
        assert not s[0].any()
        torch.testing.assert_close(s[1],other,atol=0,rtol=0)
    cache.invalidate()
    assert all(not s.any() for s in cache.recurrent_states)
    path,manifest=prepared(tmp_path)
    ds=SequenceDataset(path,config(),3,60)
    batch=to_device(next(iter(make_loader(ds,batch_size=2))),device)
    norm=Normalization(manifest['normalization'])
    model.train()
    prediction=model(norm.normalize('motion',batch['motion']),norm.normalize('properties',batch['properties']))
    result=objective(prediction,batch,norm,1.,1.)
    for horizon in ('short','long'):
        g=torch.autograd.grad(result['physics'][horizon]['loss'],model.temporal.block.in_proj.weight,retain_graph=True)[0]
        assert torch.isfinite(g).all() and g.abs().sum()>0
    # The first burn-in input still participates in the differentiable scan.
    assert prediction.shape[1]==64


def test_official_train_resume_and_chronological_evaluation(tmp_path,monkeypatch):
    device=official_cuda()
    from liquid_tcn.train import train
    from liquid_tcn.checkpoint import load_checkpoint
    from liquid_tcn.staged_validation import standalone_policy
    import liquid_tcn.staged_validation as policy
    monkeypatch.setattr(policy,'MONITOR_EVERY',1)
    path,manifest=prepared(tmp_path);out=tmp_path/'run'
    settings=dict(policy='shortlist',seed=1729,subset=None)
    cfg=dict(updates=2,pretrain_updates=0,ramp_updates=1,batch_size=2,threads=2,
        sequence_burn_in=3,sequence_scored_intervals=60,device=device,progress=False,
        validation_every=1000,checkpoint_every=200,validation_batch_size=8,weight_decay=.001)
    p=standalone_policy(path,out,settings,2)
    train(path,out,cfg,max_updates=1,expected_model=asdict(config()),staged_policy=p)
    first=load_checkpoint(out/'latest.pt')
    assert first['update']==1
    p=standalone_policy(path,out,settings,2)
    train(path,out,cfg,resume=out/'latest.pt',staged_policy=p)
    saved=load_checkpoint(out/'latest.pt');assert saved['update']==2
    assert saved['sampler']['position']==first['sampler']['position']+2
    assert [g['weight_decay'] for g in saved['optimizer']['param_groups']]==[.001,0.]
    assert json.loads((out/'monitor_subset.json').read_text())['sampling_unit']=='segment'
    assert (out/'selection/selection.json').exists()
    from liquid_tcn.assessment import assess_checkpoint
    assess_checkpoint(path,out/'best.pt',out/'assessment',out/'shared',saved['config'],device,block_rows=31,window_only=True)
    comp=json.loads((out/'assessment/streaming_comparison.json').read_text())
    assert abs(comp['differences']['window_test.json']['/selection_metric']['difference'])<1e-4
