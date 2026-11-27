from dataclasses import asdict
import json
import numpy as np
import pytest
import torch
from helpers import make_attempt
from test_film_streaming import config
from test_pyramid import identity_norm
from liquid_tcn.preprocess import prepare
from liquid_tcn.dataset import WindowDataset, make_loader, close_loader
from liquid_tcn.reset_backfill import BackfilledWindows, MixedWindowSampler, build_startup_index, validation_diagnostics, episode_evaluation
from liquid_tcn.model import build_model, CachedInference
from liquid_tcn.normalization import Normalization
from liquid_tcn.objectives import objective


@pytest.fixture
def prepared(tmp_path):
    model=asdict(config());model['long_intervals']=40
    sources=[]
    for i,split in enumerate(('train','validation','test')):
        path=make_attempt(tmp_path/split,samples=120,shift=float(i),reset_at=0)
        sources.append(dict(path=str(path),group=split,split=split))
    data=tmp_path/'prepared'
    prepare(dict(model=model,shard_rows=7,sources=sources,progress=False),data)
    info=build_startup_index(data)
    return data,info


def test_alignment_masks_and_legacy(prepared):
    data,info=prepared;old=WindowDataset(data);mixed=BackfilledWindows(data)
    assert len(info['episodes']['train'])==1  # env1's arbitrary start is not a reset.
    for i in (0,len(old)//2,len(old)-1):
        a,b=old[i],mixed[i]
        for key in a:torch.testing.assert_close(a[key] if torch.is_tensor(a[key]) else torch.tensor(a[key]), b[key] if torch.is_tensor(b[key]) else torch.tensor(b[key]),equal_nan=True)
    for age in (0,1,mixed.rf-2):
        batch=mixed.__getitems__([mixed.ordinary_length+age]).batch
        from liquid_tcn.trajectories import Trajectories
        reader=Trajectories(data,'train');series=reader.dataset.series[int(batch['series'][0])]
        rows=reader.read(series,0,age+41)
        expected=np.concatenate((np.repeat(rows['motion'][:1],mixed.rf-1-age,axis=0),rows['motion']))
        np.testing.assert_array_equal(batch['motion'][0],expected.astype(np.float32))
        for key in ('targets','target_valid','state_valid','interval_valid','tick','x_O_W','u_O_W'):
            np.testing.assert_array_equal(batch[key][0],rows[key][age:age+41])
        assert batch['input_context_valid'].all()
        if age==0:
            assert not batch['target_valid'][0,0,:2].any()
            assert not batch['history_valid'][0,0]
            model=build_model(config_from(data));norm=Normalization(mixed.manifest['normalization'])
            result=objective(model(norm.normalize('motion',batch['motion']),norm.normalize('properties',batch['properties'])),batch,norm,1.,1.)
            assert result['physics']['short']['count']>0
            assert result['physics']['long']['count']==1
            batch['interval_valid'][0,5]=False
            bad=objective(model(norm.normalize('motion',batch['motion']),norm.normalize('properties',batch['properties'])),batch,norm,1.,1.)
            assert bad['physics']['long']['count']==0
            assert bad['physics']['short']['count']==result['physics']['short']['count']-1
            batch['target_valid'][:,:,:2]=False;batch['interval_valid'][:]=False
            missing=objective(model(norm.normalize('motion',batch['motion']),norm.normalize('properties',batch['properties'])),batch,norm,1.,1.)
            assert torch.isfinite(missing['loss']) and missing['supervised_counts'][:2].sum()==0
            assert missing['physics']['short']['count']==0


def config_from(data):return json.loads((data/'manifest.json').read_text())['model']


def test_gap_does_not_create_reset(prepared):
    data,info=prepared
    sid=info['episodes']['train'][0]['series']
    path=data/f'series-{sid:06d}-000007.npy'
    rows=np.load(path);rows['tick'][1]+=1;np.save(path,rows)  # Gap at row 50, across shards.
    other=build_startup_index(data,data/'gap_index')
    assert other['windows']['train']==10  # Only endpoints 0..49 can precede the gap.
    assert len(other['episodes']['train'])==1


def test_consumed_resume_with_prefetch(prepared):
    data,_=prepared;ds=BackfilledWindows(data);sampler=MixedWindowSampler(ds,.2,10,7)
    complete=list(sampler);assert len([i for i in complete if i<ds.ordinary_length])==ds.ordinary_length
    assert len(set(i for i in complete if i<ds.ordinary_length))==ds.ordinary_length
    loader=make_loader(ds,batch_size=10,num_workers=2,sampler=sampler)
    try:
        it=iter(loader)
        for _ in range(2):
            batch=next(it);sampler.advance(len(batch['motion']));sampler.observe(batch)
        state=sampler.state_dict();assert state['position']==20
        resumed=MixedWindowSampler(ds,.2,10,99);resumed.load_state_dict(state)
        assert list(resumed)==complete[20:]
        assert resumed.state_dict()==state
        assert sampler.startup_seen==4
        list(it)  # Drain worker IPC without acknowledging prefetched samples.
        assert sampler.state_dict()==state
    finally:close_loader(loader)


def test_backfilled_cached_async_resets():
    torch.set_num_threads(2);torch.manual_seed(9)
    model=build_model(config()).eval();norm=identity_norm()
    # Non-identity input normalization proves raw motion is repeated first.
    norm.stats['motion']['mean']=[.7]*15;norm.stats['motion']['std']=[1.9]*15
    with torch.no_grad():model.film.weight.normal_(0,.03)
    motion=torch.randn(3,36,15);props=torch.randn(3,24);cache=CachedInference(model,norm,3)
    cache.set_properties(props);cache.initialize_reset(motion[:,0]);starts=[0,0,0]
    for t in range(36):
        if t==17:
            before=[v[[0,2]].clone() for v in cache.caches+cache.histories if v is not None]
            cursor=cache.cursor;cache.initialize_reset(motion[1:2,t],[1]);starts[1]=t
            assert cache.cursor==cursor
            for a,b in zip(before,[v[[0,2]] for v in cache.caches+cache.histories if v is not None]):torch.testing.assert_close(a,b,atol=0,rtol=0)
        y,valid=cache.step(motion[:,t],return_all=True);assert valid.all()
        for env in range(3):
            start=starts[env];inputs=torch.cat((motion[env:env+1,start:start+1].expand(-1,model.config.receptive_field-1,-1),motion[env:env+1,start:t+1]),1)
            expected=norm.denormalize('targets',model(norm.normalize('motion',inputs),norm.normalize('properties',props[env:env+1])))[0,-1]
            torch.testing.assert_close(y[env],expected,rtol=2e-5,atol=2e-6)
    legacy=CachedInference(model,norm,3);legacy.set_properties(props)
    assert not legacy.step(motion[:,0])[1].any()


def test_mixed_training_smoke_and_reports(prepared,tmp_path,monkeypatch):
    from liquid_tcn.train import train
    from liquid_tcn.checkpoint import load_checkpoint
    data,_=prepared;out=tmp_path/'run'
    settings=dict(startup_fraction=.2,batch_size=10,updates=3,num_workers=0,threads=2,
                  pretrain_updates=0,ramp_updates=1,validation_every=2,checkpoint_every=1,validation_batches=1,
                  startup_validation_ages=3,device='cpu')
    report=train(data,out,settings,max_updates=2,logging_mode='plain')
    assert report['final_update']==2
    saved=load_checkpoint(out/'latest.pt');assert saved['sampler']['startup_seen']==4
    metrics=[json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines()]
    assert all(np.isfinite([m['loss'],m['gradient_norm'],m['short'],m['long']]).all() for m in metrics)
    assert all(m['gradient_norm']>0 for m in metrics)
    train(data,out,settings,resume=out/'latest.pt',logging_mode='plain')
    saved=load_checkpoint(out/'latest.pt');assert saved['sampler']['startup_seen']==6
    uninterrupted=tmp_path/'uninterrupted'
    train(data,uninterrupted,settings,logging_mode='plain')
    reference=load_checkpoint(uninterrupted/'latest.pt')
    for key,value in saved['model'].items():torch.testing.assert_close(value,reference['model'][key],rtol=0,atol=0)
    model=build_model(saved['model_config']);model.load_state_dict(saved['model']);norm=Normalization(saved['normalization'])
    diagnostics=validation_diagnostics(model,norm,data,settings,'cpu')
    assert diagnostics['periods']['0_200ms']['heads']['r_B']['count']>0
    episode=episode_evaluation(model,norm,data,settings,'cpu')
    assert sum(v['heads']['r_B']['count'] for v in episode['periods'].values())==120
    import liquid_tcn.staged_validation as staged
    monkeypatch.setattr(staged,'MONITOR_EVERY',1)
    stageout=tmp_path/'shortlist'
    policy=staged.standalone_policy(data,stageout,dict(policy='shortlist'),1)
    train(data,stageout,dict(settings,updates=1),staged_policy=policy,logging_mode='plain')
    assert (stageout/'reset-validation-1.json').exists()
    selected=json.loads((stageout/'selection/best_validation.json').read_text())
    assert selected['reset_backfill']['periods']['0_200ms']['heads']['r_B']['count']>0


def test_configuration_preserves_recipe():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    old=json.loads((root/'configs/pyramid_film.json').read_text())
    new=json.loads((root/'configs/pyramid_film_reset_backfill.json').read_text())
    assert new['model']==old['model'] and new['monitoring']==old['monitoring']
    assert new['training']['startup_fraction']==.2
    for key,value in old['training'].items():assert new['training'][key]==value
