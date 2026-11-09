import copy
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pytest
import torch
from liquid_tcn.dataset import WindowDataset, make_loader, EpochSampler
from liquid_tcn.integrity import verify_once, check_receipt
from liquid_tcn.staged_plan import candidates, identity, BUDGETS, rank
from liquid_tcn.staged_sweep import verify_test_gate, initialize, summarize_seeds
from liquid_tcn.sweep import generate_trials, Coordinator
from liquid_tcn.sweep_resources import cpu_limit, capacity, original_deadline, forecast
from liquid_tcn.ported.storage import atomic_json
from helpers import preparation


def baseline(tmp_path):
    cfg=generate_trials(tmp_path/'data',tmp_path/'out',['cpu'],1)['00']
    cfg['training']['patience']=20001
    return cfg


def test_dependent_plan_and_dedup(tmp_path):
    base=baseline(tmp_path);history={}
    for stage,count in [('1',5),('2',3),('3',3),('4',4),('5',8)]:
        values=[c for _,c in candidates(stage,base,history)]
        assert len(values)==count
        assert len({identity(c) for c in values})==count
        if stage!='1':
            assert identity(history[str(int(stage)-1)][0]) in {identity(c) for c in values}
        assert all(c['training']['updates']==20000 for c in values)
        history[stage]=values[::-1]  # Force non-baseline inheritance.
    finals=candidates('6',base,history)
    assert len(finals)==12
    for label,c in finals:
        expected=copy.deepcopy(history['5'][0]['training'])
        for k in ('seed','lambda_short','lambda_long'):
            expected[k]=c['training'][k]
        assert c['training']==expected
        assert c['training']['seed'] in [1,2,3]
    assert {c['training']['seed'] for label,c in finals if label.startswith('short_')}=={1,2,3}


def test_refinement_at_both_lr_boundaries(tmp_path):
    base=baseline(tmp_path)
    for lr in [.0001,.001]:
        history={}
        for s in ['1','2','3','4']:
            history[s]=[c for _,c in candidates(s,base,history)]
        history['4'][0]['training']['physics_lr']=lr
        values=candidates('5',base,history)
        assert len({identity(c) for _,c in values})==8
        assert all(.0001<=c['training']['physics_lr']<=.001 for _,c in values)


def test_promotion_requires_equal_complete_budgets():
    state={'trials':{t:{'stages':{'1':dict(status='complete',result=dict(update=3000,best_update=3000,normalized_mse=1.,selection_complete=True,scope='stage_endpoint'))}} for t in ['b','a']}}
    assert rank(['b','a'],state,'1')==['a','b']
    state['trials']['a']['stages']['1']['result']['update']=2999
    with pytest.raises(ValueError,match='Incomplete'):rank(['a','b'],state,'1')


def test_batch_loader_equivalence_and_receipt(tmp_path):
    data,_=preparation(tmp_path)
    ds=WindowDataset(data)
    indices=[len(ds)-1,0,13,0,4]
    batch=next(iter(make_loader(ds,batch_size=len(indices),sampler=indices,pin_memory=False)))
    for n,i in enumerate(indices):
        ds._local();entry=ds.index[i];sid,end=int(entry['series']),int(entry['end'])
        rows=np.concatenate([np.asarray(ds._shard(sid,j)) for j in range(end//ds.manifest['shard_rows']+1)])
        rows=rows[end-ds.input_length+1:end+1]
        for k in rows.dtype.names:
            if k=='held_acceleration_W':continue
            expected=rows[k].astype('float32') if k=='motion' else rows[k][ds.rf-1:]
            assert np.array_equal(batch[k][n].numpy(),expected,equal_nan=True),k
    verify_once(data,tmp_path/'receipt.json')
    check_receipt(data,tmp_path/'receipt.json')
    artifact=data/next(iter(ds.manifest['artifacts']))
    artifact.touch()
    with pytest.raises(ValueError,match='changed'):check_receipt(data,tmp_path/'receipt.json')


def test_deadline_and_capacity():
    assert original_deadline(10,48,{}) is None
    assert original_deadline(10,48,{'SLURM_JOB_END_TIME':'100'}) is None
    assert cpu_limit(12,2,dict(cpus=8))==3
    with pytest.raises(ValueError):cpu_limit(12,8,dict(cpus=4))
    worker=dict(peak_rss_kib=1024,memory={'startup':dict(peak_reserved_bytes=100,free_visible_device_bytes=900)})
    assert capacity(12,2,dict(cpus=40,available_ram_bytes=100*1024**2),{'cuda:0':worker})=={'cuda:0':6}
    w=dict(training=dict(seconds=10,updates=10),validation=dict(seconds=20),startup_including_initial_checkpoint_seconds=2,checkpoint_seconds=1)
    f=forecast([('cpu',1000,1)]*3,{'cpu':2},{'workers':[w]*2},100,100)
    assert f['training_seconds']==2*(3+1000+20+5)*1.25
    assert f['shutdown_reporting_seconds']==0


def test_no_test_before_freeze(tmp_path):
    atomic_json(tmp_path/'state.json',{})
    with pytest.raises(ValueError,match='frozen'):verify_test_gate(tmp_path,'a')


def test_seed_statistics_are_paired():
    rows=[]
    for variant,offset in [('full',1),('short',2),('long',3),('supervised',4)]:
        for seed in [1,2,3]:
            rows.append(dict(selection=dict(label=f'{variant}_seed{seed}',seed=seed),report=dict(heads={'force':{'rmse_axis':[seed+offset]*3}})))
    r=summarize_seeds(rows)
    assert r['across_seeds']['full']['/heads/force/rmse_axis/0']['sample_std']==1
    assert r['full_minus_ablation']['short']['/heads/force/rmse_axis/0']['mean']==-1
    assert r['full_minus_ablation']['short']['/heads/force/rmse_axis/0']['sample_std']==0


def test_dry_run_is_non_mutating(tmp_path):
    completed=subprocess.run([sys.executable,'-m','liquid_tcn.staged_sweep','--data',str(tmp_path/'missing'),
        '--output',str(tmp_path/'out'),'--devices','0,1','--dry-run'],capture_output=True,text=True)
    assert completed.returncode==0,completed.stderr
    plan=json.loads(completed.stdout)
    assert len(plan['stage1'])==5
    assert plan['max_concurrency']==12
    assert not (tmp_path/'out').exists()


def test_coordinator_ignores_historical_deadline(tmp_path):
    folder=tmp_path/'trials/a'
    (folder/'console').mkdir(parents=True)
    atomic_json(folder/'resolved.json',dict(training=dict(device='cpu',seed=1)))
    state=dict(trials={'a':dict(attempts=[],stages={})})
    c=Coordinator(tmp_path,{'threads':1,'deadline':0},state,concurrency=1,deadline=0,
        command_factory=lambda *_:[sys.executable,'-c','pass'],inspect=lambda *_:{})
    assert c.run('1',['a'])
    assert state['trials']['a']['stages']['1']['status']=='complete'


def test_tiny_cpu_staged_lifecycle_and_resume(tmp_path, monkeypatch):
    """Real train/checkpoint/evaluate pipeline; shrink only test stage budgets.

    Dispatch inline to keep this a small CPU integration test. Separate process
    lifecycle/barrier/cleanup behavior is covered by sweep and benchmark tests.
    """
    import liquid_tcn.staged_sweep as sweep
    from liquid_tcn.train import DEFAULTS
    data, prepared=preparation(tmp_path)
    cfg=baseline(tmp_path)
    cfg['model']=prepared['model']
    cfg['data']=str(data)
    cfg['training'].update(batch_size=2,validation_every=1,checkpoint_every=1,device='cpu',threads=1,
                           pretrain_updates=0,ramp_updates=1,validation_batch_size=16)
    monkeypatch.setattr(sweep,'generate_trials',lambda *args:{'00':copy.deepcopy(cfg)})
    import liquid_tcn.staged_validation as validation_policy
    monkeypatch.setattr(validation_policy,'MONITOR_EVERY',1)
    monkeypatch.setattr(validation_policy,'RECOVERY_EVERY',2)
    cfg['training']['min_delta']=1e9
    original_candidates=sweep.candidates
    def tiny_candidates(stage,base,history):
        values=original_candidates(stage,base,history)
        if stage=='3':
            for (_,config),(pretrain,ramp) in zip(values,[(0,1),(1,1),(1,2)]):
                config['training'].update(pretrain_updates=pretrain,ramp_updates=ramp)
        return values
    monkeypatch.setattr(sweep,'candidates',tiny_candidates)
    for n in range(1,7):monkeypatch.setitem(BUDGETS,str(n),n if n<6 else 12)
    resources=dict(cpus=8,available_ram_bytes=2**30)
    monkeypatch.setattr(sweep,'available_resources',lambda:resources)
    worker=dict(peak_rss_kib=1024,memory={},training=dict(seconds=.001,updates=50),
        validation=dict(seconds=.001),startup_including_initial_checkpoint_seconds=.001,checkpoint_seconds=.001)
    monkeypatch.setattr(sweep,'preflight',lambda *args,**kwargs:dict(workers=[worker]))
    monkeypatch.setattr(sweep,'profile_loader_settings',lambda *args,**kwargs:{})
    class InlineCoordinator:
        def __init__(self,root,manifest,state,*args,**kwargs):
            self.root,self.manifest,self.state=root,manifest,state
        def run(self,stage,ids):
            for trial in ids:
                row=self.state['trials'][trial]
                if row['stages'].get(stage,{}).get('status')=='complete':continue
                checkpoint=self.root/'trials'/trial/'run/latest.pt'
                assert sweep.run_worker(self.root,trial,stage,checkpoint.exists())==0
                result=sweep.inspect(self.root,trial,stage,self.manifest)
                row['stages'][stage]=dict(status='complete',result=result)
                atomic_json(self.root/'state.json',self.state)
            return True
        def stop(self):pass
    monkeypatch.setattr(sweep,'Coordinator',InlineCoordinator)
    args=['--data',str(data),'--output',str(tmp_path/'sweep'),'--devices','cpu','--threads','1','--concurrency','2']
    assert sweep.main(args)==0
    state=json.loads((tmp_path/'sweep/state.json').read_text())
    frozen=json.loads((tmp_path/'sweep/final_selections.json').read_text())
    assert len(frozen['selections'])==12
    assert state['status']=='complete'
    assert any(e['stopping_reason']=='patience' for e in frozen['selections'])
    for e in frozen['selections']:
        selected=state['trials'][e['trial']]['stages']['6']['result']
        assert len(selected['evaluations'])<=4
        assert selected['selected_update']<=selected['update']
    assert sweep.main(args+['--resume'])==0
    after=json.loads((tmp_path/'sweep/staged_sweep.json').read_text())
    original_code=after['code']
    assert sweep.main(args+['--resume'])==0
    assert json.loads((tmp_path/'sweep/staged_sweep.json').read_text())['code']==original_code
    result=json.loads((tmp_path/'sweep/report.json').read_text())
    assert result['completed_test_runs']==12
    assert result['seed_summary']['across_seeds']['full']['/heads/force/rmse_axis/0']['n']==3
    # Mutating a frozen identity prevents even a previously completed test replay.
    frozen['selections'][0]['best_update']+=1
    atomic_json(tmp_path/'sweep/final_selections.json',frozen)
    with pytest.raises(ValueError,match='frozen'):sweep.verify_test_gate(tmp_path/'sweep',frozen['selections'][0]['trial'])


def test_training_matches_reference_collation(tmp_path, monkeypatch):
    import liquid_tcn.train as training
    from liquid_tcn.checkpoint import load_checkpoint
    from torch.utils.data._utils.collate import default_collate
    data,_=preparation(tmp_path)
    config=dict(batch_size=3,updates=3,pretrain_updates=0,ramp_updates=1,threads=1,
                device='cpu',progress=False,gradient_strategy='pcgrad',validation_every=3)
    training.train(data,tmp_path/'fast',config)
    original=training.make_loader
    def reference(*args,**kwargs):
        loader=original(*args,**kwargs)
        loader.collate_fn=default_collate  # Restack individual sample views as before.
        return loader
    monkeypatch.setattr(training,'make_loader',reference)
    training.train(data,tmp_path/'reference',config)
    a,b=[load_checkpoint(tmp_path/p/'latest.pt') for p in ['fast','reference']]
    for key in a['model']:assert torch.equal(a['model'][key],b['model'][key])
    assert a['sampler']==b['sampler']
    assert a['scheduler']==b['scheduler']
    assert a['validation']==b['validation']
    for param,values in a['optimizer']['state'].items():
        for k,v in values.items():assert torch.equal(v,b['optimizer']['state'][param][k])


def test_external_stop_cleanup_is_bounded(tmp_path):
    folder=tmp_path/'trials/a'
    (folder/'console').mkdir(parents=True)
    atomic_json(folder/'resolved.json',dict(training=dict(device='cpu',seed=1)))
    state=dict(trials={'a':dict(attempts=[],stages={})})
    def worker(*_):
        return [sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)']
    c=Coordinator(tmp_path,{'threads':1},state,1,worker,deadline=time.time()+.3,shutdown_timeout=.1)
    from threading import Timer
    timer=Timer(.3,c.stop);timer.start()
    start=time.monotonic()
    assert not c.run('1',['a'])
    timer.join()
    assert time.monotonic()-start<3
    assert state['trials']['a']['stages']['1']['status']=='interrupted'
    assert state['trials']['a']['attempts'][0]['exit_code'] is not None


def test_device_slots_limit_independent_mock_workers(tmp_path):
    state=dict(trials={})
    ids=[str(i) for i in range(6)]
    for i in ids:
        folder=tmp_path/'trials'/i
        (folder/'console').mkdir(parents=True)
        device='cuda:0' if int(i)<4 else 'cuda:1'
        atomic_json(folder/'resolved.json',dict(training=dict(device=device,seed=1)))
        state['trials'][i]=dict(attempts=[],stages={})
    def worker(root,trial,*_):
        path=str(root/'trials'/trial/'interval.json')
        script='import json,time; start=time.monotonic(); time.sleep(.3); open('+repr(path)+',"w").write(json.dumps([start,time.monotonic()]))'
        return [sys.executable,'-c',script]
    def inspect(*_):return {}
    c=Coordinator(tmp_path,{'threads':1},state,12,worker,inspect,max_concurrency=12,device_limits={'cuda:0':2,'cuda:1':1})
    assert c.run('1',ids)
    for device,limit in [('cuda:0',2),('cuda:1',1)]:
        intervals=[json.loads((tmp_path/'trials'/i/'interval.json').read_text()) for i in ids
                   if state['trials'][i]['attempts'][0]['device']==device]
        events=sorted([(a,1) for a,b in intervals]+[(b,-1) for a,b in intervals])
        live=0
        for _,delta in events:
            live+=delta
            assert live<=limit



def test_fixed_subset_mse_and_policy_resume(tmp_path,monkeypatch):
    from liquid_tcn.staged_validation import monitoring_subset,monitoring_mse,StageValidation
    import liquid_tcn.staged_validation as policy_module
    from liquid_tcn.evaluate import evaluate
    from liquid_tcn.normalization import Normalization
    from liquid_tcn.model import LiquidTCN,ModelConfig
    from liquid_tcn.train import train
    from liquid_tcn.checkpoint import load_checkpoint
    data,manifest=preparation(tmp_path)
    path=tmp_path/'subset.json'
    subset=monitoring_subset(data,path)
    assert monitoring_subset(data,path)==subset
    assert subset['windows']==max(1,int(len(WindowDataset(data,'validation'))*.05+.5))
    assert len(set(subset['indices']))==subset['windows']
    ds=WindowDataset(data,'validation')
    loader=make_loader(ds,batch_size=3,sampler=subset['indices'],pin_memory=False)
    model=LiquidTCN(ModelConfig(**manifest['model']))
    norm=Normalization(manifest['normalization'])
    monitor=monitoring_mse(model,loader,norm,'cpu')
    full=evaluate(model,loader,norm,'cpu')
    assert monitor['selection_metric']==pytest.approx(full['selection_metric'],rel=1e-7)
    for horizon in ('short','long'):
        assert monitor['physics'][horizon]['loss']==pytest.approx(full['losses'][horizon],rel=1e-7)
        assert monitor['physics'][horizon]['valid_count']==full['losses']['physics_counts'][horizon]
    monkeypatch.setattr(policy_module,'MONITOR_EVERY',1)
    monkeypatch.setattr(policy_module,'RECOVERY_EVERY',2)
    config=dict(updates=12,pretrain_updates=0,ramp_updates=1,min_delta=1e9,patience=20001,
                threads=1,device='cpu',batch_size=2,progress=False)
    run=tmp_path/'trial/run'
    first=StageValidation('6',12,path,time.time()+100)
    train(data,run,config,max_updates=2,staged_policy=first)
    saved=load_checkpoint(run/'latest.pt')
    assert saved['stage_monitoring']['stale']==1
    assert not (run.parent/'6/selection.json').exists()
    second=StageValidation('6',12,path,time.time()+100)
    train(data,run,config,resume=run/'latest.pt',staged_policy=second)
    result=json.loads((run.parent/'6/selection.json').read_text())
    assert result['stopping_reason']=='patience' and result['update']==4
    assert result['monitoring']['entry_update']==0
    assert len(result['monitoring']['checks'])==4
    assert len(result['evaluations'])<=4
    lowest=sorted(result['monitoring']['checks'],key=lambda r:(r['selection_metric'],r['update']))[:3]
    assert [r['update'] for r in result['monitoring']['shortlist']]==[r['update'] for r in lowest]
    hashes={p.name:p.read_bytes() for p in (run.parent/'6').glob('validation-*.json')}
    # Resuming selection reuses full-validation reports, even if evaluation is unavailable.
    monkeypatch.setattr(policy_module,'evaluate',lambda *a,**k:pytest.fail('re-evaluated cached checkpoint'))
    third=StageValidation('6',12,path,time.time()+100)
    train(data,run,config,resume=run/'latest.pt',staged_policy=third)
    assert hashes=={p.name:p.read_bytes() for p in (run.parent/'6').glob('validation-*.json')}
