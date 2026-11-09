"""Focused finite-precision warning and allocation/code-change recovery checks."""
import copy
import json
from pathlib import Path
import random
import numpy as np
import pytest
import torch
from liquid_tcn.checkpoint import rng_state, load_checkpoint
from liquid_tcn.dataset import WindowDataset,make_loader
from liquid_tcn.model import LiquidTCN,ModelConfig
from liquid_tcn.normalization import Normalization
from liquid_tcn import train as training
from helpers import preparation


def test_precision_warning_persists_and_preserves_state(tmp_path,monkeypatch):
    data,manifest=preparation(tmp_path)
    model=LiquidTCN(ModelConfig(**manifest['model']))
    optimizer=torch.optim.AdamW(model.parameters())
    batch=next(iter(make_loader(WindowDataset(data),batch_size=2)))
    norm=Normalization(manifest['normalization'])
    cfg=dict(training.DEFAULTS,device='cpu',threads=1)
    real=training.objective
    def discrepancy(pred,*args,**kwargs):
        result=real(pred,*args,**kwargs)
        if pred.dtype==torch.float32:
            for key in ('loss','supervised','secondary'):
                result[key]=result[key]*1.02
        # Deliberately consume all RNG streams to test restoration.
        random.random();np.random.rand();torch.rand(1)
        return result
    monkeypatch.setattr(training,'objective',discrepancy)
    before={k:v.clone() for k,v in model.state_dict().items()}
    opt_before=copy.deepcopy(optimizer.state_dict());rng=rng_state()
    path=tmp_path/'precision.json'
    class Log:
        def emit(self,event,**fields):
            assert path.exists() and json.loads(path.read_text())['float32']['status']=='warning'
            assert event=='precision_warning'
    report=training.precision_check(model,batch,norm,cfg,diagnostic_path=path,
        context=dict(checkpoint='latest.pt',update=3000),log=Log())
    assert report['float32']['relative_gradient_error']>cfg['precision_rtol']
    assert report['context']['update']==3000 and report['status']=='warning'
    assert all(torch.equal(before[k],v) for k,v in model.state_dict().items())
    assert optimizer.state_dict()==opt_before
    after=rng_state()
    assert rng['python']==after['python'] and np.array_equal(rng['numpy'][1],after['numpy'][1])
    assert torch.equal(rng['torch'],after['torch'])
    for value in model.parameters():assert value.grad is None
    # AMP remains an acceptance gate, even though finite FP32 differences only warn.
    with pytest.raises(ValueError,match='amp failed'):
        training.precision_check(model,batch,norm,dict(cfg,amp=True,amp_rtol=0.),diagnostic_path=path)
    assert json.loads(path.read_text())['failure']['precision_mode']=='amp'
    def nonfinite(pred,*args,**kwargs):
        result=real(pred,*args,**kwargs);result['loss']=result['loss']*float('nan');return result
    monkeypatch.setattr(training,'objective',nonfinite)
    with pytest.raises(FloatingPointError):training.precision_check(model,batch,norm,cfg,diagnostic_path=path)
    assert json.loads(path.read_text())['status']=='failed'
    monkeypatch.setattr(training,'objective',real)
    backward=training.backward_objectives
    def bad_gradient(*args,**kwargs):
        params=list(args[2]);backward(args[0],args[1],params,*args[3:],**kwargs)
        params[0].grad.fill_(float('inf'))
    monkeypatch.setattr(training,'backward_objectives',bad_gradient)
    with pytest.raises(FloatingPointError,match='gradient'):
        training.precision_check(model,batch,norm,cfg,diagnostic_path=path)
    assert json.loads(path.read_text())['failure']['error_type']=='FloatingPointError'


def test_expired_allocation_resume_keeps_completed_trials(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import sys
    import time
    import liquid_tcn.staged_sweep as sweep
    import liquid_tcn.sweep_resume as recovery
    import liquid_tcn.staged_validation as policy
    from liquid_tcn.ported.storage import atomic_json,digest
    from liquid_tcn.integrity import verify_once
    from liquid_tcn.staged_plan import BUDGETS
    data,prepared=preparation(tmp_path)
    root=tmp_path/'sweep';root.mkdir()
    args=SimpleNamespace(data=data,resume=False,allow_code_change=False,threads=1,concurrency=2,budget_hours=48)
    manifest,state=sweep.initialize(root,args,['cpu'],time.time())
    monkeypatch.setitem(BUDGETS,'1',2);monkeypatch.setitem(BUDGETS,'2',4)
    monkeypatch.setattr(policy,'MONITOR_EVERY',1)
    baseline=sweep.generate_trials(data,root,['cpu'],1)['00']
    baseline['model']=prepared['model']
    baseline['training'].update(batch_size=2,pretrain_updates=0,ramp_updates=1,patience=20001)
    for trial,lr in [('t000',.001),('t005',.0003),('t006',.0001)]:
        cfg=copy.deepcopy(baseline);cfg['training']['physics_lr']=lr
        cfg['output']=str(root/'trials'/trial/'run')
        (root/'trials'/trial/'console').mkdir(parents=True)
        atomic_json(root/'trials'/trial/'resolved.json',cfg)
        manifest['configs'][trial]=digest(root/'trials'/trial/'resolved.json')
        state['trials'][trial]=dict(attempts=[],stages={})
    verify_once(data,root/'verification.json');policy.monitoring_subset(data,root/'monitor_subset.json')
    manifest.update(monitor_subset_sha256=digest(root/'monitor_subset.json'),deadline=0,budget_hours=48)
    def persist():
        atomic_json(root/'staged_sweep.json',manifest);atomic_json(root/'state.json',state)
    persist()
    for trial,stage in [('t000','1'),('t005','2'),('t006','2')]:
        assert sweep.run_worker(root,trial,stage)==0
        state['trials'][trial]['stages'][stage]=dict(status='complete',result=sweep.inspect(root,trial,stage,manifest))
        persist()
    old_checkpoint=load_checkpoint(root/'trials/t000/run/latest.pt')
    old_hash=digest(root/'trials/t000/run/latest.pt')
    old_monitor=copy.deepcopy(old_checkpoint['stage_monitoring'])
    old_code=copy.deepcopy(manifest['code'])
    completed=copy.deepcopy({t:state['trials'][t] for t in ['t005','t006']})
    monkeypatch.setattr(training,'precision_check',lambda *a,**kw:pytest.fail('automatic precision probe'))
    state['trials']['t000']['stages']['2']=dict(status='failed',error='old float32 gate')
    state['rankings']['1']=['t000']
    persist()
    assert digest(root/'trials/t000/run/latest.pt')==old_hash
    manifest['code'] = {'source_sha256':'historical-code'}
    manifest['runtime'] = {'python':'old-runtime'}
    manifest['devices'] = ['cuda:99']
    manifest['configs'] = {t:'obsolete-hash' for t in manifest['configs']}
    persist()
    import shutil
    moved = tmp_path/'relocated-sweep'
    shutil.copytree(root,moved)
    root = moved
    args.resume=True;args.threads=2;args.concurrency=1;args.budget_hours=.001
    restored,restored_state=sweep.initialize(root,args,['cpu'],time.time())
    assert restored['deadline']==0 and restored['devices']==['cuda:99']
    completed=copy.deepcopy({t:restored_state['trials'][t] for t in ['t005','t006']})
    assert all(r['stages']['2']['result']['update']==4 for r in completed.values())
    assert restored_state['rankings']==state['rankings']
    assert load_checkpoint(root/'trials/t000/run/latest.pt')['stage_monitoring']==old_monitor
    recovery.assign_devices(root,restored,{'cpu':1},2)
    atomic_json(root/'staged_sweep.json',restored)
    launched=[]
    def command(r,trial,stage,resume):
        launched.append((trial,resume))
        script=("from pathlib import Path; from liquid_tcn.staged_sweep import run_worker; "
                "from liquid_tcn.staged_plan import BUDGETS; import liquid_tcn.staged_validation as p; "
                "BUDGETS['2']=4; p.MONITOR_EVERY=1; "
                f"raise SystemExit(run_worker(Path({str(root)!r}),{trial!r},'2',True))")
        return [sys.executable,'-c',script]
    coordinator=sweep.Coordinator(root,restored,restored_state,1,command,sweep.inspect,
        max_concurrency=12,device_limits={'cpu':1},deadline=0)
    assert coordinator.run('2',['t000','t005','t006'])
    assert launched==[('t000',True)]
    assert {t:restored_state['trials'][t] for t in completed}==completed
    assert load_checkpoint(root/'trials/t000/1/latest.pt')['stage_monitoring']==old_monitor
    new=load_checkpoint(root/'trials/t000/run/latest.pt')
    assert new['update']==4 and new['sampler']['position']==old_checkpoint['sampler']['position']+4
    assert new['stage_monitoring']['entry_update']==2
    assert new['config']==old_checkpoint['config']  # Thread allocation is runtime-only.
    assert len(sweep.rank(['t000','t005','t006'],restored_state,'2'))==3
    # Allocation remapping does not rewrite immutable trial configs.
    recovery.assign_devices(root,restored,{'cuda:0':1},2)
    assert set(restored['execution']['assignments'].values())=={'cuda:0'}
    assert restored['configs']['t000']=='obsolete-hash'


def test_standalone_resume_without_verification(tmp_path,monkeypatch):
    from liquid_tcn.ported.storage import atomic_json
    from liquid_tcn.checkpoint import atomic_save
    data,manifest=preparation(tmp_path)
    manifest['artifacts']={k:'obsolete' for k in manifest['artifacts']}
    atomic_json(data/'manifest.json',manifest)
    monkeypatch.setattr(training,'precision_check',lambda *a,**kw:pytest.fail('automatic probe'))
    cfg=dict(updates=4,batch_size=2,threads=1,device='cpu',validation_every=4,
             checkpoint_every=1,validation_batches=1,pretrain_updates=1)
    output=tmp_path/'run'
    training.train(data,output,cfg,max_updates=2)
    path=output/'latest.pt';saved=load_checkpoint(path)
    saved['data_manifest_sha256']='obsolete'
    saved['config']['device']='cuda:99'
    atomic_save(path,saved)
    training.train(data,output,dict(device='cpu',weight_decay=.01),resume=path,max_updates=3)
    result=load_checkpoint(path)
    assert result['update']==3 and result['sampler']['position']==saved['sampler']['position']+2
    assert result['config']['batch_size']==2
    assert result['optimizer']['param_groups'][0]['weight_decay']==.01
    assert result['scheduler']['last_epoch']==saved['scheduler']['last_epoch']+1
    assert all(v['step']==3 for v in result['optimizer']['state'].values())
