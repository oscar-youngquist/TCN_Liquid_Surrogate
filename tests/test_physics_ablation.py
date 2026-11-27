from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from test_reset_backfill import prepared
from liquid_tcn.physics_ablation import resolve, VARIANTS
from liquid_tcn.staged_validation import monitoring_subset, StageValidation
from liquid_tcn.reset_backfill import BackfilledWindows, MixedWindowSampler
from liquid_tcn.model import build_model, ModelConfig
from liquid_tcn.normalization import Normalization
from liquid_tcn.ablation_evaluation import prepare_protocol, infer_batch, score_episode, segments, HORIZONS
from liquid_tcn.trajectories import Trajectories
from liquid_tcn.train import DEFAULTS, weights
from liquid_tcn.ported.storage import atomic_json


def setup_settings(data,tmp_path):
    ds=BackfilledWindows(data)
    baseline=tmp_path/'baseline';baseline.mkdir()
    training=dict(DEFAULTS,startup_fraction=.2,startup_index=str(data/'reset_backfill'),batch_size=10,
                  updates=40000,pretrain_updates=500,ramp_updates=1500,pretrain_lr=.001,physics_lr=.001,
                  gradient_strategy='pcgrad',validation_batch_size=10,threads=2,device='cpu')
    atomic_json(baseline/'config.json',dict(model=ds.manifest['model'],training=training,normalization=ds.manifest['normalization']))
    return dict(data=str(data),baseline_run=str(baseline),startup_index=str(data/'reset_backfill'),
                evaluation_batch_size=2,evaluation_chunk_size=17,bootstrap_repetitions=20)


def test_matrix_weights_order_monitor_and_patience(prepared,tmp_path,monkeypatch):
    data,_=prepared;settings=setup_settings(data,tmp_path)
    m=resolve(settings,tmp_path/'experiment','cpu',2)
    assert len(m['trials'])==12
    ds=BackfilledWindows(data);initial={};ordering={}
    for trial,c in m['trials'].items():
        t=c['training'];assert weights(t,2000)==VARIANTS[c['variant']]
        assert t['updates']==40000 and t['startup_fraction']==.2 and t['batch_size']==10
        torch.manual_seed(c['seed']);model=build_model(c['model'])
        state=torch.cat([p.detach().flatten() for p in model.parameters()])
        order=list(MixedWindowSampler(ds,.2,10,c['seed']))[:30]
        if c['seed'] in initial:
            torch.testing.assert_close(state,initial[c['seed']],atol=0,rtol=0);assert order==ordering[c['seed']]
        initial[c['seed']]=state;ordering[c['seed']]=order
    assert ordering[1]!=ordering[2]
    path=tmp_path/'subset.json';subset=monitoring_subset(data,path,1729,.2)
    assert subset['windows']==max(1,int(np.floor(subset['total_windows']*.2+.5)))
    assert len(set(subset['indices']))==subset['windows']
    assert monitoring_subset(data,path,999,.2)==subset
    import liquid_tcn.staged_validation as sv
    policy=StageValidation('6',40000,path,patience_checks=6)
    output=tmp_path/'policy';output.mkdir()
    log=SimpleNamespace(emit=lambda *a,**k:None);writer=SimpleNamespace(add_scalar=lambda *a,**k:None)
    training=dict(m['baseline']['training'],startup_fraction=0,min_delta=.1)
    scores=iter([1.,1.,.95,.95,.95,.95,.95,.95])
    def monitoring(*args,**kwargs):
        return dict(selection_metric=next(scores),physics={},physics_objective=0.,total_objective=1.,lambda_short=1.,lambda_long=1.)
    monkeypatch.setattr(sv,'monitoring_mse',monitoring)
    from threading import Event
    validation=Trajectories(data,'validation').dataset
    policy.attach(model,Normalization(ds.manifest['normalization']),validation,None,training,output,None,0,
                  lambda name:(output/name).write_bytes(b'checkpoint'),Event(),log,writer)
    for update in range(1000,8000,1000):
        policy.after_update(update);assert not policy.patience_stop
    assert policy.state['stale']==5 and policy.state['eligibility_update']==2000
    policy.after_update(8000);assert policy.patience_stop and policy.state['effective_patience']==6
    policy.close()


def test_cached_chunk_continuity_masks_and_sliding_physics(prepared,tmp_path):
    data,_=prepared;reader=Trajectories(data,'test');config=reader.manifest['model']
    norm=Normalization(reader.manifest['normalization']);model=build_model(config).eval()
    protocol_root=tmp_path/'protocol'
    protocol=prepare_protocol(data,data/'reset_backfill',protocol_root,config,norm.stats)
    entries=protocol['episodes'];entry=entries[0];series=reader.dataset.series[entry['series']]
    # A recorded gap invalidates input context and physics; it is not a new reset.
    shard=data/f"series-{entry['series']:06d}-000007.npy"
    rows=np.load(shard);rows['tick'][1]+=1;np.save(shard,rows)
    protocol_root=tmp_path/'gap_protocol'
    reader=Trajectories(data,'test')
    protocol=prepare_protocol(data,data/'reset_backfill',protocol_root,config,norm.stats)
    entry=protocol['episodes'][0];entries=[entry]
    rows=reader.read(series,0,entry['rows'])
    other_rows=rows[:90].copy();other_rows['tick']=other_rows['tick'][0]+np.arange(90)*entry['stride']
    other_rows['motion'][:,3]+=1.25
    other_id=entry['series']+10000
    other_series=dict(series,id=other_id,rows=90)
    other_entry=dict(entry,series=other_id,rows=90,reset=dict(entry['reset'],motion=other_rows['motion'][0].tolist()))
    fake_batch=SimpleNamespace(dataset=SimpleNamespace(series={entry['series']:series,other_id:other_series}),
        read=lambda s,a,b:(rows if s['id']==entry['series'] else other_rows)[a:b].copy())
    results=[]
    for chunk in (7,23):
        out=tmp_path/f'predictions_{chunk}';out.mkdir()
        infer_batch(fake_batch,[entry,other_entry],model,norm,protocol,out,'cpu',chunk)
        results.append(np.load(out/f"prediction-{entry['series']:06d}.npy"))
        other_prediction=np.load(out/f'prediction-{other_id:06d}.npy')
        rf=ModelConfig(**config).receptive_field
        padded=np.concatenate((np.repeat(other_rows['motion'][:1],rf-1,axis=0),other_rows['motion']))
        with torch.no_grad():
            other_expected=norm.denormalize('targets',model(norm.normalize('motion',torch.tensor(padded,dtype=torch.float32)[None]),
                norm.normalize('properties',torch.tensor(series['properties'],dtype=torch.float32)[None]))).numpy()[0]
        np.testing.assert_allclose(other_prediction,other_expected,rtol=2e-5,atol=2e-6)
    np.testing.assert_allclose(results[0],results[1],rtol=0,atol=0,equal_nan=True)
    rows=reader.read(series,0,entry['rows']);rf=ModelConfig(**config).receptive_field
    raw=np.concatenate((np.repeat(rows['motion'][:1],rf-1,axis=0),rows['motion']))
    with torch.no_grad():
        expected=norm.denormalize('targets',model(norm.normalize('motion',torch.tensor(raw,dtype=torch.float32)[None]),
            norm.normalize('properties',torch.tensor(series['properties'],dtype=torch.float32)[None]))).numpy()[0]
    with np.load(protocol_root/entry['mask']) as masks:context=masks['context']
    np.testing.assert_allclose(results[0][context],expected[context],rtol=2e-5,atol=2e-6)
    assert not context[50:50+rf-1].any()
    bad=results[0].copy();bad[0]=np.nan
    with pytest.raises(FloatingPointError,match='missing/nonfinite'):score_episode(reader,entry,bad,protocol,protocol_root,17)
    # Long, fixed-clock physical fixture exercises every sliding horizon and gap.
    count=2111;physical=np.repeat(rows[1:2],count,axis=0)
    physical['tick']=np.arange(count)*5;physical['episode_id']=1;physical['history_valid']=True
    physical['h']=.005;physical['state_valid']=True;physical['interval_valid']=True;physical['target_valid']=True
    physical['target_valid'][0,:2]=False
    valid=physical['target_valid'].copy();edges=np.ones(count,bool);edges[0]=False;edges[500]=False
    entry=dict(entry,rows=count,profile_cohort=True,valid_head_counts=valid.sum(0).tolist(),valid_intervals=int(edges.sum()),excluded_input_context=0,excluded_intervals=1)
    np.savez_compressed(protocol_root/entry['mask'],ticks=physical['tick'],times=np.arange(count)*.005,context=np.ones(count,bool),target_valid=valid,edges=edges)
    fake=SimpleNamespace(dataset=SimpleNamespace(series={entry['series']:series}),read=lambda s,a,b:physical[a:b].copy())
    report=score_episode(fake,entry,physical['targets'],protocol,protocol_root,137)
    for h in HORIZONS:
        n=sum(max(0,b-a-h) for a,b in segments(edges))
        key=f'impulse/reset_inclusive/{h*5}ms/force/vector'
        assert report['metrics'][key]['n']==n and report['metrics'][key]['square']==0
        for channel in ('r','P','H'):
            a=report['metrics'][f'consistency/reset_inclusive/{h*5}ms/predicted/{channel}/vector']
            b=report['metrics'][f'consistency/reset_inclusive/{h*5}ms/sph/{channel}/vector']
            assert a==b
    assert report['metrics']['reconstruction/overall/r_B/vector']['n']==count
    assert report['metrics']['reconstruction/profile_9_10s/r_B/vector']['n']==201


def test_paired_bootstrap():
    from liquid_tcn.ablation_report import bootstrap_pair
    reports={};key='error'
    for seed in (1,2,3):
        for variant,factor in [('full',1.),('supervised',4.)]:
            reports[f'{variant}-s{seed}']=dict(groups={g:{key:dict(n=10,square=factor*10,reduction='rms')} for g in ('a','b')},
                metrics={key:dict(reduction='rms')})
    result=bootstrap_pair(reports,'supervised',[1,2,3],key,20)
    assert result['mean_difference']==-1 and result['ci95']==[-1,-1]
    reports['supervised-s2']['groups']['a'][key]['n']=9
    with pytest.raises(ValueError,match='scoring counts differ'):bootstrap_pair(reports,'supervised',[1,2,3],key,20)


def test_tiny_orchestration_selection_evaluation_resume(prepared,tmp_path):
    data,_=prepared;settings=setup_settings(data,tmp_path)
    cfg=tmp_path/'settings.json';atomic_json(cfg,settings);output=tmp_path/'experiment'
    command=[sys.executable,'-m','liquid_tcn.physics_ablation','--config',str(cfg),'--output',str(output),'--device','cpu','--threads','2','--concurrency','2','--smoke']
    first=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    assert first.returncode==0,first.stdout+'\n'+'\n'.join(p.read_text() for p in output.glob('trials/*/console/*.log'))
    state=json.loads((output/'state.json').read_text());assert state['status']=='complete' and len(state['trials'])==4
    before={p:p.stat().st_mtime_ns for p in output.glob('trials/*/evaluation/summary.json')}
    selected={p:p.stat().st_mtime_ns for p in output.glob('trials/*/run/selection/best.pt')}
    for trial in state['trials']:
        report=json.loads((output/'trials'/trial/'complete.json').read_text())
        assert report['selection']['selection_complete'] and report['selection']['effective_patience']==6
        assert report['coverage']['missing_predictions']==0
    # Simulate interruption after evaluation finished but before completion was acknowledged.
    (output/'trials/supervised-s1/complete.json').unlink()
    resumed=subprocess.run(command+['--resume'],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    assert resumed.returncode==0,resumed.stdout
    assert all(p.stat().st_mtime_ns==value for p,value in {**before,**selected}.items())
    state=json.loads((output/'state.json').read_text())
    assert len(state['trials']['supervised-s1']['attempts'])==2
    assert all(len(v['attempts'])==1 for k,v in state['trials'].items() if k!='supervised-s1')
    assert len(list((output/'report').glob('*.png')))==3 and len(list((output/'report').glob('*.pdf')))==3
