"""Fresh matched-seed physics ablations; training and evaluation share worker slots."""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import signal
import sys
import time
from .ported.storage import atomic_json
from .sweep import Coordinator, read_json, event
from .staged_validation import monitoring_subset, StageValidation

VARIANTS={'supervised':(0.,0.),'short':(1.,0.),'long':(0.,1.),'full':(1.,1.)}
REPO=Path(__file__).resolve().parents[2]


def absolute(path):
    path=Path(path)
    return str(path.resolve() if path.is_absolute() else (REPO/path).resolve())


def resolve(settings,root,device,threads,smoke=False):
    baseline=read_json(Path(absolute(settings['baseline_run']))/'config.json')
    config=dict(model=baseline['model'],training=baseline['training'],data=absolute(settings['data']))
    config['training']=dict(config['training'],device=device,threads=threads,progress=False,
                            startup_index=absolute(settings['startup_index']))
    # Experiment changes are explicit; all other successful run_1 settings persist.
    config['training']['startup_fraction']=.20
    config['training']['updates']=40000
    seeds=(1,2,3)
    policy=dict(fraction=.20,seed=settings.get('monitor_seed',1729),patience_checks=6,monitor_every=1000,recovery_every=200)
    if smoke:
        seeds=(1,)
        config['training'].update(updates=2,batch_size=5,validation_batch_size=5,pretrain_updates=0,ramp_updates=1,
            startup_validation_ages=2,num_workers=0,validation_num_workers=0)
        policy.update(monitor_every=1,recovery_every=1)
    trials={}
    for seed in seeds:
        for variant,(short,long) in VARIANTS.items():
            trial=f'{variant}-s{seed}';c=copy.deepcopy(config)
            c['training'].update(seed=seed,lambda_short=short,lambda_long=long)
            c.update(output=str(root/'trials'/trial/'run'),variant=variant,seed=seed,kind='ablation')
            trials[trial]=c
    references=[absolute(p) for p in settings.get('historical_checkpoints',[])]
    for i,path in enumerate(references):
        trial=f'historical-{i+1}';c=copy.deepcopy(config)
        c.update(output=str(root/'trials'/trial/'run'),kind='historical',variant=trial,seed=None,checkpoint=path)
        trials[trial]=c
    return dict(data=config['data'],baseline_run=absolute(settings['baseline_run']),baseline=baseline,
        settings=settings,policy=policy,trials=trials,historical_checkpoints=references,
        seeds=list(seeds),smoke=smoke,threads=threads,device=device,started=time.time())


def worker(root,trial):
    manifest=read_json(root/'ablation.json');c=read_json(root/'trials'/trial/'resolved.json')
    folder=root/'trials'/trial;run=Path(c['output']);run.mkdir(parents=True,exist_ok=True)
    device=manifest['execution']['assignments'][trial];threads=manifest['execution']['threads']
    if c['kind']=='ablation':
        selection=run/'selection/selection.json'
        if not selection.exists():
            from .train import train
            policy=StageValidation('6',c['training']['updates'],root/'monitor_subset.json',selection_root=run/'selection',
                **{k:manifest['policy'][k] for k in ('patience_checks','monitor_every','recovery_every')})
            checkpoint=run/'latest.pt'
            result=train(c['data'],run,c['training'],resume=str(checkpoint) if checkpoint.exists() else None,
                expected_model=c['model'],launch_config=c,logging_mode='plain',trial_id=trial,staged_policy=policy,
                execution=dict(device=device,threads=threads))
            if result['interrupted']:return 143
        selected=read_json(selection)
        checkpoint=selected['best_checkpoint']
        event('selected_for_test',trial=trial,selected_update=selected['selected_update'],stopping_reason=selected['stopping_reason'],
              monitor_windows=selected['monitor_windows'],effective_patience=selected['effective_patience'])
    else:
        checkpoint=c['checkpoint'];selected=dict(kind='historical',checkpoint=checkpoint,selection='externally supplied; no test selection')
    # The training lock also covers evaluation so retries cannot share run outputs.
    with open(run/'.training.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from .ablation_evaluation import evaluate_checkpoint
        report=evaluate_checkpoint(c['data'],checkpoint,root/'evaluation_protocol',folder/'evaluation',device,
            manifest['settings'].get('evaluation_batch_size',64),manifest['settings'].get('evaluation_chunk_size',400),threads)
        atomic_json(folder/'complete.json',dict(trial=trial,variant=c['variant'],seed=c['seed'],kind=c['kind'],selection=selected,
            evaluation=str(folder/'evaluation/summary.json'),coverage=report['coverage']))
    return 0


def command(root,trial,stage,resume):
    return [sys.executable,'-u','-m','liquid_tcn.physics_ablation','--output',str(root),'--worker',trial]


def inspect(root,trial,stage,manifest):
    return read_json(root/'trials'/trial/'complete.json')


def initialize(root,args):
    if args.resume:
        manifest=read_json(root/'ablation.json');state=read_json(root/'state.json')
        for trial,row in state['trials'].items():
            for attempt in row['attempts']:
                if attempt['status']=='running':attempt['status']='interrupted'
            done=root/'trials'/trial/'complete.json'
            if done.exists():row['stages']['ablation']=dict(status='complete',result=read_json(done))
            else:row['stages']['ablation']=dict(status='pending')
    else:
        if any(p.name!='.coordinator.lock' for p in root.iterdir()):
            raise ValueError('Choose a fresh output directory or use --resume')
        settings=read_json(args.config)
        if args.historical_checkpoint:settings['historical_checkpoints']=args.historical_checkpoint
        manifest=resolve(settings,root,args.device,args.threads,args.smoke)
        state=dict(trials={},status='prepared')
        for trial,c in manifest['trials'].items():
            folder=root/'trials'/trial;(folder/'console').mkdir(parents=True)
            atomic_json(folder/'resolved.json',c)
            state['trials'][trial]=dict(attempts=[],stages={})
    manifest['execution']=dict(threads=args.threads,assignments={t:args.device for t in state['trials']})
    manifest['concurrency']=args.concurrency
    atomic_json(root/'ablation.json',manifest);atomic_json(root/'state.json',state)
    policy=manifest['policy']
    subset=monitoring_subset(manifest['data'],root/'monitor_subset.json',policy['seed'],policy['fraction'])
    event('experiment_policy',runs=len(state['trials']),smoke=manifest['smoke'],device=args.device,concurrency=args.concurrency,
        monitor_windows=subset['windows'],eligible_mature_windows=subset['total_windows'],monitor_fraction=subset['fraction'],
        effective_patience=policy['patience_checks'],monitor_every=policy['monitor_every'],checkpoint_every=policy['recovery_every'])
    from .ablation_evaluation import prepare_protocol
    first=next(c for c in manifest['trials'].values() if c['kind']=='ablation')
    prepare_protocol(manifest['data'],first['training']['startup_index'],root/'evaluation_protocol',first['model'],
                     manifest['baseline']['normalization'],manifest['historical_checkpoints'])
    return manifest,state


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=REPO/'configs/reset_backfill_physics_ablation.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--concurrency',type=int,default=2)
    parser.add_argument('--threads',type=int,default=2);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--dry-run',action='store_true');parser.add_argument('--smoke',action='store_true',help='Four tiny two-update seed-1 runs; use a tiny prepared fixture')
    parser.add_argument('--historical-checkpoint',action='append')
    parser.add_argument('--worker',help=argparse.SUPPRESS)
    args=parser.parse_args(argv);root=args.output.resolve()
    if args.worker:return worker(root,args.worker)
    if min(args.threads,args.concurrency)<1:parser.error('Positive concurrency and CPU threads required')
    if args.dry_run:
        settings=read_json(args.config)
        if args.historical_checkpoint:settings['historical_checkpoints']=args.historical_checkpoint
        print(json.dumps(resolve(settings,root,args.device,args.threads,args.smoke),indent=2));return 0
    root.mkdir(parents=True,exist_ok=True)
    with open(root/'.coordinator.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest,state=initialize(root,args)
        coordinator=Coordinator(root,manifest,state,args.concurrency,command_factory=command,inspect=inspect,
                                heartbeat_seconds=30,shutdown_timeout=None)
        previous={s:signal.signal(s,coordinator.stop) for s in (signal.SIGTERM,signal.SIGINT)}
        try:
            success=coordinator.run('ablation',list(state['trials']))
            state['status']='complete' if success else 'interrupted' if coordinator.stopping else 'failed'
            atomic_json(root/'state.json',state)
            from .ablation_report import write_report
            write_report(root)
            event('experiment_'+state['status'],output=str(root),completed=sum((root/'trials'/t/'complete.json').exists() for t in state['trials']))
            return 0 if success else 143 if coordinator.stopping else 1
        finally:
            for s,handler in previous.items():signal.signal(s,handler)


if __name__=='__main__':raise SystemExit(main())
