"""Selected full model and three matched-seed ablations, followed by held-out assessment."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time
from threading import Event
from .sweep import Coordinator, generate_trials, validate_devices, read_json, event
from .ported.storage import atomic_json
from .staged_validation import StageValidation, monitoring_subset

VARIANTS={'full':(1.,1.),'supervised':(0.,0.),'short':(1.,0.),'long':(0.,1.)}


def configurations(data, root, devices, threads, updates=20000, selected=None):
    baseline=copy.deepcopy(selected or generate_trials(data,root,devices,threads)['00'])
    baseline['training'].update(supervised='mse',physics_lr=.001,pretrain_updates=500,ramp_updates=1500,
        updates=updates,epochs=None,patience=updates+1,checkpoint_every=200,validation_every=1000,
        validation_batches=None,validation_batch_size=1024,validation_num_workers=0,progress=False,threads=threads)
    configs={}
    for variant,(short,long) in VARIANTS.items():
        for seed in (1,2,3):
            trial=f'{variant}-s{seed}'
            cfg=copy.deepcopy(baseline)
            cfg.update(data=str(Path(data).resolve()),output=str(Path(root).resolve()/'trials'/trial/'run'),resume=None,max_updates=None)
            cfg['training'].update(seed=seed,device=devices[len(configs)%len(devices)],lambda_short=short,lambda_long=long)
            configs[trial]=cfg
    return configs


def initialize(root, args, devices):
    manifest_path=root/'experiment.json'
    if args.resume or args.assess_only:
        manifest=read_json(manifest_path);state=read_json(root/'state.json')
        from .sweep_resume import relocate
        relocate(root,args.data,manifest,state)
        # Execution resources are invocation options, not candidate identities.
    else:
        if any(p.name!='.coordinator.lock' for p in root.iterdir()):
            raise ValueError('Use an empty output directory or --resume')
        selected=read_json(args.config) if args.config else None
        configs=configurations(args.data,root,devices,args.threads,args.max_updates or 20000,selected)
        manifest=dict(data=str(args.data.resolve()),threads=args.threads,configs=list(configs),started=time.time(),
                      max_updates=args.max_updates or 20000,assessment_block_rows=args.block_rows)
        state=dict(trials={})
        for trial,cfg in configs.items():
            folder=root/'trials'/trial;(folder/'console').mkdir(parents=True)
            atomic_json(folder/'resolved.json',cfg)
            state['trials'][trial]=dict(attempts=[],stages={})
    if args.max_updates is not None:
        manifest['max_updates']=args.max_updates
        for trial in manifest['configs']:
            cfg=read_json(root/'trials'/trial/'resolved.json')
            cfg['training'].update(updates=args.max_updates,patience=args.max_updates+1)
            atomic_json(root/'trials'/trial/'resolved.json',cfg)
    manifest['execution']=dict(threads=args.threads,assignments={t:devices[i%len(devices)] for i,t in enumerate(manifest['configs'])})
    manifest['assessment_block_rows']=args.block_rows
    manifest.setdefault('sessions',[]).append(dict(started=time.time(),devices=devices,threads=args.threads,concurrency=args.concurrency))
    atomic_json(manifest_path,manifest);atomic_json(root/'state.json',state)
    monitoring_subset(args.data,root/'monitor_subset.json')
    return manifest,state


def command(root,trial,stage,resume):
    result=[sys.executable,'-u','-m','liquid_tcn.experiment','--output',str(root),'--worker',trial,'--worker-stage',stage]
    if resume:result.append('--resume')
    return result


def inspect(root,trial,stage,manifest):
    if stage=='train':
        value=read_json(root/'trials'/trial/'6/selection.json')
        if not value['selection_complete']:raise ValueError('Selection unfinished')
        return value
    value=read_json(root/'trials'/trial/'assessment/summary.json')
    if value['status']!='complete':raise ValueError('Assessment unfinished')
    return dict(status='complete',report=str(root/'trials'/trial/'assessment/summary.json'))


def run_worker(root,trial,stage,resume=False):
    cfg=read_json(root/'trials'/trial/'resolved.json');manifest=read_json(root/'experiment.json')
    execution=dict(device=manifest['execution']['assignments'][trial],threads=manifest['execution']['threads'])
    if stage=='train':
        from .train import train
        policy=StageValidation('6',cfg['training']['updates'],root/'monitor_subset.json')
        result=train(cfg['data'],cfg['output'],cfg['training'],
            resume=str(Path(cfg['output'])/'latest.pt') if resume else None,
            expected_model=cfg['model'],logging_mode='plain',trial_id=trial,staged_policy=policy,execution=execution)
        return int(result['interrupted'] or not (Path(cfg['output']).parent/'6/selection.json').exists())
    # All twelve validation choices are persisted before this stage is dispatched.
    selections=read_json(root/'selected.json')
    selected=selections[trial]
    checkpoint=root/'trials'/trial/'6/best.pt'
    destination=root/'trials'/trial/'assessment';destination.mkdir(parents=True,exist_ok=True)
    stopped=Event();previous={s:signal.signal(s,lambda *_:stopped.set()) for s in (signal.SIGTERM,signal.SIGINT)}
    import torch
    torch.set_num_threads(execution['threads'])
    with open(Path(cfg['output'])/'.training.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            from .batch_logging import BatchLogger
            log=BatchLogger(trial,0);log.emit('assessment_start',checkpoint=str(checkpoint))
            heartbeat=[time.monotonic()]
            def cancelled():
                if time.monotonic()-heartbeat[0]>=60:
                    log.emit('assessment_heartbeat');heartbeat[0]=time.monotonic()
                return stopped.is_set()
            # Preserve existing window-weighted assessment separately.
            if not (destination/'window_test.json').exists():
                from .checkpoint import load_checkpoint
                from .model import LiquidTCN,ModelConfig
                from .normalization import Normalization
                from .dataset import WindowDataset,make_loader,close_loader
                from .evaluate import evaluate
                from .benchmark import CancellableLoader
                class Cancellation:
                    def cancelled(self):return stopped.is_set()
                saved=load_checkpoint(checkpoint)
                model=LiquidTCN(ModelConfig(**saved['model_config'])).to(execution['device']);model.load_state_dict(saved['model'])
                loader=make_loader(WindowDataset(cfg['data'],'test'),batch_size=1024,num_workers=0)
                try:
                    report=evaluate(model,CancellableLoader(loader,Cancellation()),Normalization(saved['normalization']),execution['device'],
                        supervised='mse',lambda_short=cfg['training']['lambda_short'],lambda_long=cfg['training']['lambda_long'],physics_dtype=cfg['training']['physics_dtype'])
                    if stopped.is_set():raise InterruptedError('Test assessment interrupted')
                    report['weighting']='overlapping eligible windows'
                    atomic_json(destination/'window_test.json',report)
                finally:close_loader(loader)
                del model
            from .offline import assess
            assess(cfg['data'],checkpoint,destination,root/'shared_assessment',execution['device'],manifest['assessment_block_rows'],cancelled)
            if stopped.is_set():raise InterruptedError('Assessment interrupted')
            log.emit('assessment_complete',selected_update=selected['selected_update'])
            return 0
        finally:
            for s,h in previous.items():signal.signal(s,h)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--config',type=Path,help='Optional selected configuration; omitted uses packaged sweep baseline plus selected settings')
    p.add_argument('--devices',default='0');p.add_argument('--concurrency',type=int,default=12);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--max-updates',type=int);p.add_argument('--block-rows',type=int,default=2048)
    p.add_argument('--resume',action='store_true');p.add_argument('--assess-only',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--worker',help=argparse.SUPPRESS);p.add_argument('--worker-stage',help=argparse.SUPPRESS)
    args=p.parse_args(argv);root=args.output.resolve()
    if args.worker:return run_worker(root,args.worker,args.worker_stage,args.resume)
    if args.data is None:p.error('--data is required')
    if min(args.concurrency,args.threads,args.block_rows)<1 or (args.max_updates is not None and args.max_updates<1):p.error('Counts must be positive')
    devices=validate_devices(args.devices,False)
    if args.dry_run:
        print(json.dumps(configurations(args.data,root,devices,args.threads,args.max_updates or 20000,read_json(args.config) if args.config else None),indent=2));return 0
    root.mkdir(parents=True,exist_ok=True)
    with open(root/'.coordinator.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest,state=initialize(root,args,devices)
        current=[None];stopped=Event()
        def stop(*_):
            stopped.set()
            if current[0]:current[0].stop()
        previous={s:signal.signal(s,stop) for s in (signal.SIGTERM,signal.SIGINT)}
        try:
            ids=list(manifest['configs'])
            if not args.assess_only:
                current[0]=Coordinator(root,manifest,state,args.concurrency,command,inspect)
                if not current[0].run('train',ids):raise InterruptedError('Training incomplete; resume from saved progress')
            if not all(state['trials'][t]['stages'].get('train',{}).get('status')=='complete' for t in ids):
                raise ValueError('Assessment requires all twelve validation selections')
            if not (root/'selected.json').exists():
                atomic_json(root/'selected.json',{t:read_json(root/'trials'/t/'6/selection.json') for t in ids})
            if stopped.is_set():raise InterruptedError('Experiment interrupted')
            from .offline import shared_events
            event('assessment_reference_start')
            heartbeat=[time.monotonic()]
            def cancelled():
                if time.monotonic()-heartbeat[0]>=60:
                    event('assessment_reference_heartbeat');heartbeat[0]=time.monotonic()
                return stopped.is_set()
            shared_events(args.data,root/'shared_assessment',cancelled)
            event('assessment_reference_complete')
            current[0]=Coordinator(root,manifest,state,args.concurrency,command,inspect)
            if stopped.is_set() or not current[0].run('assess',ids):raise InterruptedError('Assessment incomplete; resume to continue')
            from .experiment_report import write_report
            write_report(root,manifest,state)
            if stopped.is_set():raise InterruptedError('Experiment interrupted during reporting')
            state['status']='complete';return 0
        except BaseException as exc:
            state.update(status='incomplete',error=str(exc));event('experiment_incomplete',error=str(exc));return 1
        finally:
            atomic_json(root/'state.json',state)
            for s,h in previous.items():signal.signal(s,h)


if __name__=='__main__':raise SystemExit(main())
