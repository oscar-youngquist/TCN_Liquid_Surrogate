import copy
import json
import numpy as np
import pytest
import torch
from liquid_tcn.model import LiquidTCN, ModelConfig, CachedInference
from liquid_tcn.normalization import Normalization
from liquid_tcn.preprocess import iter_merged_rows, prepare, endpoint_row, verify_attempt
from liquid_tcn.dataset import WindowDataset, make_loader, EpochSampler
from liquid_tcn.metadata import assign_splits
from liquid_tcn.ported.discrete.data import merged_rows, environment_episodes, grid_intervals, eligible_windows
from liquid_tcn.objectives import objective
from liquid_tcn.train import train
from liquid_tcn.checkpoint import load_checkpoint
from helpers import make_attempt, preparation


def identity_norm():
    return Normalization({k:dict(mean=np.zeros(s).tolist(),std=np.ones(s).tolist())
        for k,s in [('motion',(15,)),('properties',(24,)),('targets',(5,3))]})


def test_default_lengths_causality_and_output_alignment():
    torch.manual_seed(7)
    m=LiquidTCN(); c=m.config
    assert c.receptive_field==249 and c.input_length==289
    x=torch.randn(2,289,15); p=torch.randn(2,24)
    y=m(x,p); assert y.shape==(2,41,5,3)
    changed=x.clone();changed[:,270:]+=100
    torch.testing.assert_close(m(changed,p)[:,:22],y[:,:22],atol=0,rtol=0)
    # The first fully contextual output is exactly the final output on 249 inputs.
    torch.testing.assert_close(m(x[:,:249],p)[:,-1],y[:,0])


def test_cached_batch_and_independent_resets():
    torch.manual_seed(2)
    m=LiquidTCN(ModelConfig(channels=8, embedding_dim=4,kernel_size=3,dilations=(1,2)))
    x=torch.randn(2,55,15);p=torch.randn(2,24)
    cache=CachedInference(m,identity_norm(),2)
    for i in range(55):
        reset=[0] if i==25 else None
        y,valid=cache.step(x[:,i],p,reset)
        if i>=m.config.receptive_field-1:
            expected=m(x[1:2,:i+1],p[1:2])[:,-1,:4].flatten(1)
            torch.testing.assert_close(y[1:2],expected,atol=1e-6,rtol=1e-5)
        begin=25 if i>=25 else 0
        if i-begin>=m.config.receptive_field-1:
            expected=m(x[:1,begin:i+1],p[:1])[:,-1,:4].flatten(1)
            torch.testing.assert_close(y[:1],expected,atol=1e-6,rtol=1e-5)
        else:
            assert not valid[0] and torch.isnan(y[0]).all()


def test_prepare_reference_eligibility_and_cross_shard_loader(tmp_path):
    p=make_attempt(tmp_path/'attempt',samples=330, reset_at=310)
    sources=[dict(path=str(p),group='one',split='train')]
    m=prepare(dict(sources=sources,shard_rows=71),tmp_path/'data')
    rows=merged_rows(p)
    streamed=list(iter_merged_rows(p))
    assert len(rows)==len(streamed)
    for a,b in zip(rows,streamed):
        for k in a:np.testing.assert_equal(a[k],b[k])
    expected=0
    for env in range(2):
        for ep in environment_episodes(rows,env,.001).values():
            grid,_=grid_intervals(ep,.001)
            expected+=len(eligible_windows(grid,ep['nodes'],5)[1])
    assert m['windows']['train']==expected
    ds=WindowDataset(tmp_path/'data',verify=True)
    for i in [0,len(ds)-1]:
        b=ds[i]
        assert b['motion'].shape==(289,15) and b['targets'].shape==(41,5,3)
        assert b['history_valid'].all() and (b['episode_id']==b['episode_id'][0]).all()
        assert torch.all(b['tick'][1:]-b['tick'][:-1]==5)
        assert b['targets'][0,0,2]==pytest.approx(-19.62)
    assert len(ds.cache)<=ds.cache_size
    a=next(iter(make_loader(ds,batch_size=2,num_workers=0)))
    b=next(iter(make_loader(ds,batch_size=2,num_workers=2)))
    for key in a:torch.testing.assert_close(a[key],b[key])


def test_normalization_leakage_and_split_rejection(tmp_path):
    path,m=preparation(tmp_path)
    train_stats=m['normalization']['motion']['mean']
    assert 0<train_stats[3]<1
    assert all(s['group'] in {'a','b','c'} for s in m['series'])
    with pytest.raises(ValueError,match='leakage'):
        assign_splits([dict(path='a',group='same',split='train',payload_id='p'),dict(path='b',group='same',split='test',payload_id='p')])
    sources=[dict(path='a',group='a',payload_id='held'),dict(path='b',group='b',payload_id='held')]
    assert all(s['split']=='test' for s in assign_splits(sources,held_out_payloads=['held']))


def test_objective_all_heads_both_endpoints_and_empty(tmp_path):
    path,m=preparation(tmp_path)
    b=next(iter(make_loader(WindowDataset(path),batch_size=2)))
    norm=Normalization(m['normalization'])
    y=torch.randn(2,5,5,3,requires_grad=True)
    result=objective(y,b,norm,1.,1.)
    result['loss'].backward()
    assert torch.isfinite(y.grad).all() and (y.grad.abs().sum((0,1,3))>0).all()
    assert y.grad[:,0,2:].norm()>0 and y.grad[:,-1,2:].norm()>0
    assert result['physics']['short']['count']==8 and result['physics']['long']['count']==2
    b['history_valid'][:]=False
    with pytest.raises(ValueError,match='Empty valid'):objective(y,b,norm)
    y.data[0,0,0,0]=float('nan')
    with pytest.raises(FloatingPointError):objective(y,b,norm)


def test_checksum_and_completion(tmp_path):
    p=make_attempt(tmp_path/'attempt',samples=5)
    verify_attempt(p)
    chunk=p/'chunk-000000.npz'
    with open(chunk,'ab') as f:f.write(b'corruption')
    with pytest.raises(ValueError,match='Corrupt'):verify_attempt(p)


def test_sampler_acknowledgement():
    s=EpochSampler(101,seed=5)
    order=list(s)
    assert len(set(order))==101 and s.position==0
    s.advance(17)
    t=EpochSampler(101,seed=5);t.load_state_dict(s.state_dict())
    assert list(t)==order[17:]


@pytest.mark.parametrize('workers,strategy', [(0, 'pcgrad'), (2, 'pcgrad'), (0, 'sum')])
def test_checkpoint_exact_resume_and_evaluation(tmp_path, workers, strategy):
    path,_=preparation(tmp_path)
    cfg=dict(updates=4,pretrain_updates=1,ramp_updates=2,batch_size=2,threads=2,num_workers=workers,
             validation_every=2,checkpoint_every=1,validation_batches=1,patience=10,gradient_strategy=strategy)
    train(path,tmp_path/'full',cfg)
    train(path,tmp_path/'resume',cfg,max_updates=2)
    train(path,tmp_path/'resume',cfg,resume=tmp_path/'resume'/'latest.pt')
    a=load_checkpoint(tmp_path/'full'/'latest.pt');b=load_checkpoint(tmp_path/'resume'/'latest.pt')
    assert a['update']==b['update']==4 and a['sampler']==b['sampler']
    for name in a['model']:torch.testing.assert_close(a['model'][name],b['model'][name],atol=0,rtol=0)
    assert a['validation']['selection_metric']==b['validation']['selection_metric']
    assert (tmp_path/'resume'/'best.pt').exists()
    assert (tmp_path/'resume'/'config.json').exists()
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events=EventAccumulator(str(tmp_path/'resume'/'tensorboard')).Reload()
    assert [e.step for e in events.Scalars('train/loss')]==[1,2,3,4]
    assert events.Scalars('validation/selection_metric')[-1].step==4
    assert [e.step for e in events.Scalars('train/pcgrad/projected')]==[1,2,3,4]
    records=[json.loads(line) for line in (tmp_path/'full'/'metrics.jsonl').read_text().splitlines()]
    assert records[0]['secondary']==0 and records[0]['projection']['projected']==0
    assert records[-1]['secondary'] > 0
    assert a['config']['gradient_strategy']==strategy


def test_tiny_batch_overfit(tmp_path):
    path,manifest=preparation(tmp_path)
    torch.manual_seed(1)
    model=LiquidTCN(ModelConfig(**manifest['model']))
    norm=Normalization(manifest['normalization'])
    batch=next(iter(make_loader(WindowDataset(path),batch_size=2)))
    opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=0)
    losses=[]
    for _ in range(80):
        opt.zero_grad()
        y=model(norm.normalize('motion',batch['motion']),norm.normalize('properties',batch['properties']))
        o=objective(y,batch,norm)
        losses.append(float(o['loss'].detach()));o['loss'].backward();opt.step()
    assert losses[-1]<losses[0]*.03


def test_recorded_motion_fallback_and_split_impulse_assembly():
    from test_ported_discrete import row
    from liquid_tcn.ported.discrete.losses import quaternion_matrix
    from liquid_tcn.ported.discrete.analyze import prepare_intervals, CRITERIA
    rows=[row(0,0),row(2,1,n=2),row(2,2,terminal=(False,True)),
          row(2,3,ep=(0,1),reset=(False,True)),row(5,4,ep=(0,1),n=3)]
    for r in rows:
        r['held_acceleration_W']=np.ones((2,6))*.3
        r['velocity_W'][:]=np.arange(6)*.1
    for i,theta in [(1,.3),(4,.8)]:
        r=rows[i];r['q_WB_wxyz'][0]=[np.cos(theta/2),0,0,np.sin(theta/2)]
        r['x_O_W'][0]=[i*.1,-i*.2,0]
        R=quaternion_matrix(torch.tensor(r['q_WB_wxyz'][0])).numpy()
        f=np.array([.1,.2,-.3])*i;j=np.array([.02,-.03,.05])*i;h=float(r['interval_duration_s'][0])
        r['force_interval_average_B'][0]=R.T@f/h
        r['moment_interval_average_O_B'][0]=R.T@(j-np.cross(r['x_O_W'][0],f))/h
        r['coupling_force_impulse_W'][0]=f;r['coupling_moment_impulse_world_origin_W'][0]=j
    for i in (2,3):
        for k in ('q_WB_wxyz','x_O_W'):rows[i][k][0]=rows[1][k][0]
    ep=environment_episodes(rows,0,.001)[0]
    grid,_=grid_intervals(ep,.001)
    s=ep['nodes'][5];s['held_acceleration_W']=rows[-1]['held_acceleration_W'][0]
    output=endpoint_row(s,grid[0])
    original,_,_=prepare_intervals(ep,ep,.001,CRITERIA)
    np.testing.assert_allclose(output['targets'][0],original[0]['F_B'],rtol=1e-14,atol=1e-14)
    np.testing.assert_allclose(output['targets'][1],original[0]['tau_B'],rtol=1e-14,atol=1e-14)
    native=endpoint_row(s,grid[0],native_mode=True)
    np.testing.assert_allclose(output['targets'],native['targets'],rtol=1e-14,atol=1e-14)
    R=quaternion_matrix(torch.tensor(s['q_WB_wxyz'])).numpy()
    np.testing.assert_allclose(output['motion'][9:12],R.T@s['held_acceleration_W'][:3])
    s['z']=output['motion'].copy()
    np.testing.assert_allclose(endpoint_row(s,grid[0])['motion'],output['motion'])


def test_preparation_masks_gaps_unsupported_and_changed_mass(tmp_path):
    from liquid_tcn.preprocess import SeriesWriter, ROW_DTYPE
    c=ModelConfig(channels=4,kernel_size=2,dilations=(1,),long_intervals=2)
    meta=dict(episode=0,properties=[1000.,.01,.001,2.])
    w=SeriesWriter(tmp_path,0,meta,3,c)
    for i in range(20):
        r=np.zeros((),dtype=ROW_DTYPE)
        r['tick']=i*5+(5 if i>=10 else 0)
        r['state_valid']=i!=5;r['interval_valid']=True;r['mass']=2.;r['gravity_W'][2]=-9.81
        if i==19:r['mass']=3.
        w.append(r,5)
    result=w.close()
    assert not result['assumptions_valid'] and result['windows']==0
    assert result['rejected_assumptions_windows']>0
    idx=np.memmap(tmp_path/'series-000000.idx',dtype=[('series','<i8'),('end','<i8')],mode='r')
    ends=set(idx['end'].tolist())
    assert not ends.intersection({5,6,7,8,9,10,11,12,13})


def test_export_roundtrip(tmp_path):
    from liquid_tcn.checkpoint import atomic_save
    from liquid_tcn.inference import load_export
    c=ModelConfig(channels=4,kernel_size=2,dilations=(1,))
    m=LiquidTCN(c)
    from dataclasses import asdict
    atomic_save(tmp_path/'export.pt',dict(model=m.state_dict(),model_config=asdict(c),normalization=identity_norm().stats))
    frozen=load_export(tmp_path/'export.pt',environments=2)
    x=torch.randn(2,10,15);p=torch.randn(2,24)
    for i in range(10):y,valid=frozen.step(x[:,i],p)
    expected=m(x,p)[:,-1,:4].flatten(1)
    torch.testing.assert_close(y,expected)
    assert valid.all() and not any(t.requires_grad for t in frozen.model.parameters())


def test_diagnostic_preserves_right_continuous_held_events():
    from test_ported_discrete import row
    from liquid_tcn.data import environment_episodes as enriched
    r=row(5,1,n=5);r['z']=np.zeros((2,15));r['held_acceleration_W']=np.zeros((2,6))
    event={k:r[k] for k in ('native_step','time_s','episode_id','held_valid','episode_selected','reset','terminal')}
    event['record_sequence']=2;event['held_acceleration_W']=np.ones((2,6))
    state=enriched([r,event],0,.001)[0]['nodes'][5]
    np.testing.assert_array_equal(state['held_acceleration_W'],np.ones(6))
    np.testing.assert_array_equal(state['z'][9:],np.ones(6))


def test_json_launcher_dry_run(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/'configs/combined.json').read_text())
    config['data']=str(tmp_path/'not prepared')
    config['output']=str(tmp_path/'not created')
    path=tmp_path/'selected config.json';path.write_text(json.dumps(config))
    result=subprocess.run(['bash',str(root/'launch_train.sh'),'--dry-run'],capture_output=True,text=True,check=True,
        env=dict(os.environ,PYTHON=sys.executable,GPU='cpu',CONFIG=str(path),PYTHONPATH=str(root/'src')))
    resolved=json.loads(result.stdout)
    assert resolved['training']['device']=='cpu' and resolved['data']==config['data']
    assert not (tmp_path/'not created').exists()


def test_epoch_budget_is_derived_from_training_windows(tmp_path):
    path,manifest=preparation(tmp_path)
    config=dict(epochs=1,updates=999,batch_size=40,pretrain_updates=1,ramp_updates=1,
        validation_every=2,validation_batches=1,threads=2,progress=False)
    result=train(path,tmp_path/'epoch-run',config)
    expected=int(np.ceil(manifest['windows']['train']/40))
    assert result['final_update']==expected
    saved=load_checkpoint(tmp_path/'epoch-run'/'latest.pt')
    assert saved['effective_updates']==expected and saved['sampler']['position']==manifest['windows']['train']


def test_pre_pcgrad_checkpoint_requires_sum_strategy(tmp_path):
    path, _ = preparation(tmp_path)
    cfg = dict(updates=2, pretrain_updates=1, ramp_updates=1, batch_size=2,
               threads=2, validation_every=2, checkpoint_every=1,
               validation_batches=1, gradient_strategy='sum')
    output = tmp_path / 'legacy'
    train(path, output, cfg, max_updates=1)
    checkpoint = output / 'latest.pt'
    state = load_checkpoint(checkpoint)
    del state['config']['gradient_strategy']
    del state['config']['pcgrad_eps']
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match='identical data provenance and configuration'):
        train(path, output, dict(cfg, gradient_strategy='pcgrad'), resume=checkpoint)
    train(path, output, cfg, resume=checkpoint)
    restored = load_checkpoint(checkpoint)
    assert restored['update'] == 2
    assert restored['config']['gradient_strategy'] == 'sum'
