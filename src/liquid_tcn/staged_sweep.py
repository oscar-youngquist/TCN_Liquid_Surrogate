"""Resumable dependent search and matched-seed ablations in an existing allocation."""
import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import statistics
import sys
import time

from .ported.storage import atomic_json, digest
from .sweep import (BASELINE, BASELINE_COMMIT, Coordinator, code_provenance, event,
                    generate_trials, inspect_result, read_json, runtime_versions, validate_devices)
from .staged_plan import BUDGETS, SEEDS, STAGES, candidates, identity, plan, rank
from .sweep_resources import available_resources, capacity, cpu_limit, forecast, profile_loader_settings


def frozen_selections(root, state):
    """All twelve eligible checkpoints are selected before any test process starts."""
    entries = state['plans']['6']['entries']
    if len(entries) != 12 or len({e['label'] for e in entries}) != 12:
        raise ValueError('Final confirmation requires twelve distinct variant/seed selections')
    rank([e['trial'] for e in entries], state, '6')
    rows = []
    for entry in entries:
        result = state['trials'][entry['trial']]['stages']['6']['result']
        config_path = root/'trials'/entry['trial']/'resolved.json'
        cfg = read_json(config_path)
        rows.append(dict(**entry, seed=cfg['training']['seed'], checkpoint=result['best_checkpoint'],
                         checkpoint_sha256=result['best_sha256'], config_sha256=digest(config_path),
                         validation_metric=result['normalized_mse'], best_update=result['best_update'],
                         stopping_reason=result['stopping_reason'],stopping_update=result['update'],subset_coverage=result['subset_coverage']))
    return dict(selections=rows, rule='best full-validation normalized_supervised_mse_all_heads',
                dataset_sha256=state['data_manifest_sha256'])


def verify_test_gate(root, trial):
    state = read_json(root/'state.json')
    path = root/'final_selections.json'
    # Preserve validation-before-test sequencing without authenticating metadata.
    if not path.exists():
        raise ValueError('Test evaluation requires frozen final selections')
    frozen_selections(root, state)
    frozen = read_json(path)
    return next(e for e in frozen['selections'] if e['trial'] == trial)


def run_worker(root, trial, stage, resume=False):
    """Run a candidate using current execution resources and saved progress."""
    config = read_json(root/'trials'/trial/'resolved.json')
    manifest = read_json(root/'staged_sweep.json')
    execution = dict(threads=manifest.get('execution',{}).get('threads',config['training']['threads']),
        device=manifest.get('execution',{}).get('assignments',{}).get(trial,config['training']['device']))
    event('worker_context', stage=stage, trial=trial, seed=config['training']['seed'],
          device=execution['device'],elapsed_seconds=time.time()-manifest['started'])
    if stage != 'test':
        from .train import train
        from .staged_validation import StageValidation
        policy = StageValidation(stage,BUDGETS[stage],root/'monitor_subset.json')
        result = train(config['data'], config['output'], config['training'],
            resume=str(Path(config['output'])/'latest.pt') if resume else None,
            max_updates=BUDGETS[stage], expected_model=config['model'], launch_config=config,
            logging_mode='plain', trial_id=f'stage{stage}/{trial}/seed{config["training"]["seed"]}/{execution["device"]}',
            staged_policy=policy,execution=execution)
        return int(result['interrupted'] or not (Path(config['output']).parent/stage/'selection.json').exists())
    with open(Path(config['output'])/'.training.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return evaluate_selected(root, trial, config, manifest)


def evaluate_selected(root, trial, config, manifest):
    """Repeat the selected test-window assessment with cached one-step outputs."""
    selected=verify_test_gate(root,trial)
    folder=root/'trials'/trial
    cached=folder/'test.json'
    destination=folder/'test_assessment'
    destination.mkdir(exist_ok=True)
    if cached.exists() and not (destination/'window_test.json').exists():
        atomic_json(destination/'window_test.json',read_json(cached)['report'])
    from threading import Event
    from .assessment import assess_checkpoint
    from .batch_logging import BatchLogger
    import torch
    stopped=Event()
    previous={s:signal.signal(s,lambda *_:stopped.set()) for s in (signal.SIGTERM,signal.SIGINT)}
    device=manifest.get('execution',{}).get('assignments',{}).get(trial,config['training']['device'])
    torch.set_num_threads(manifest.get('execution',{}).get('threads',config['training']['threads']))
    start=time.monotonic()
    try:
        result=assess_checkpoint(config['data'],selected['checkpoint'],destination,destination/'shared',
            config['training'],device,cancelled=stopped.is_set,log=BatchLogger(trial,0),window_only=True)
        if stopped.is_set():raise InterruptedError('Test evaluation interrupted')
        if not cached.exists():
            atomic_json(cached,dict(selection=selected,split='test',seconds=time.monotonic()-start,
                frozen_sha256=digest(root/'final_selections.json'),report=result))
        atomic_json(folder/'test_streaming.json',dict(selection=selected,split='test',
            report=read_json(destination/'streaming/window_test.json')))
    finally:
        for s,h in previous.items():signal.signal(s,h)
    return 0


def command(root, trial, stage, resume):
    result = [sys.executable, '-u', '-m', 'liquid_tcn.staged_sweep', '--worker', trial,
              '--worker-stage', stage, '--output', str(root)]
    if resume and stage != 'test':
        result.append('--resume')
    return result


def inspect(root, trial, stage, manifest):
    if stage == 'test':
        verify_test_gate(root, trial)
        return read_json(root/'trials'/trial/'test.json')
    result = read_json(root/'trials'/trial/stage/'selection.json')
    rank([trial], {'trials':{trial:{'stages':{stage:dict(status='complete',result=result)}}}},stage)
    return result


def initialize(root, args, devices, started):
    path = root/'staged_sweep.json'
    provenance = dict(data=str(args.data.resolve()), data_manifest_sha256=digest(args.data/'manifest.json'),
                      devices=devices, threads=args.threads, code=code_provenance(), runtime=runtime_versions())
    if args.resume:
        from .sweep_resume import resume_session
        return resume_session(root,args,devices,started,inspect)
    if any(p.name != '.coordinator.lock' for p in root.iterdir()):
        raise ValueError('Use a new empty output directory, or --resume')
    manifest = dict(**provenance, started=started,
        sessions=[dict(id=f'session-{time.time_ns()}',started=started,code=provenance['code'],runtime=provenance['runtime'],
                       devices=devices,threads=args.threads,requested_concurrency=args.concurrency,
                       slurm_job_id=os.environ.get('SLURM_JOB_ID'),status='initializing')], baseline_commit=BASELINE_COMMIT, baseline_sha256=digest(BASELINE),
        metric='normalized_supervised_mse_all_heads', configs={}, plan=plan(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    state = dict(trials={}, plans={}, rankings={}, registry={}, adjustments=[], status='initializing',
                 data_manifest_sha256=manifest['data_manifest_sha256'])
    atomic_json(path, manifest)
    atomic_json(root/'state.json', state)
    (root/'baseline.json').write_bytes(BASELINE.read_bytes())
    (root/'data_manifest.json').write_bytes((args.data/'manifest.json').read_bytes())
    return manifest, state


def preflight(root, baseline, slots, threads, manifest, current, stopped, name):
    from .benchmark import PhaseRunner
    folder = root/name
    # A failed/interrupted measurement is preserved; retries get a new folder.
    if folder.exists():
        folder = root/f'{name}_{time.time_ns()}'
    folder.mkdir()
    specs = {}
    for device, count in slots.items():
        for i in range(count):
            wid = f'{device.replace(":","_")}_{i}'
            worker_dir = folder/wid
            worker_dir.mkdir()
            config = copy.deepcopy(baseline)
            config['training'].update(device=device,threads=threads)
            config['output'] = str(worker_dir/'run')
            atomic_json(worker_dir/'resolved.json', config)
            specs[wid] = dict(worker_id=wid, worker_dir=str(worker_dir), barrier_dir=str(folder/'barriers'),
                config_sha256=digest(worker_dir/'resolved.json'), data_manifest_sha256=manifest['data_manifest_sha256'],
                warmup_updates=200, measured_updates=50, barrier_timeout=None,
                verification_receipt=str(root/'verification.json'))
    current[0] = PhaseRunner(folder, specs, threads, timeout=None)
    if stopped[0]:
        raise InterruptedError('Stopped before preflight')
    return current[0].run()


def register(root, config, state, manifest, slots):
    key = identity(config)
    if key in state['registry']:
        return state['registry'][key]
    trial = f't{len(state["registry"]):03d}'
    folder = root/'trials'/trial
    (folder/'console').mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(config)
    cfg.update(output=str(folder/'run'), resume=None, max_updates=None)
    # Weighted round robin by allowed slots, fixed thereafter for exact resume.
    assigned = [read_json(root/'trials'/t/'resolved.json')['training']['device'] for t in state['trials']]
    cfg['training']['device'] = min(slots, key=lambda d:(assigned.count(d)/slots[d], d))
    atomic_json(folder/'resolved.json', cfg)
    manifest['configs'][trial] = digest(folder/'resolved.json')
    if 'execution' in manifest:
        manifest['execution']['assignments'][trial]=cfg['training']['device']
    state['registry'][key] = trial
    state['trials'][trial] = dict(attempts=[], stages={})
    return trial


def metric_values(value, prefix=''):
    if isinstance(value, dict):
        for k,v in value.items():
            if k not in ('count','windows','lag_valid_windows','ground_truth_conservation'):
                yield from metric_values(v, prefix+'/'+k)
    elif isinstance(value, list):
        for i,v in enumerate(value):
            yield from metric_values(v,prefix+'/'+str(i))
    elif isinstance(value,(float,int)) and not isinstance(value,bool) and math.isfinite(value):
        yield prefix, value


def summarize_seeds(results):
    """Sample SD is across matched seeds, never across overlapping windows."""
    rows = {}
    for row in results:
        selection = row['selection']
        variant = selection['label'].split('_seed')[0]
        report = row['report']
        selected = {k:report[k] for k in ('selection_metric','normalized_mse_per_head','heads','conservation','peak_wrench','per_robot_payload','per_maneuver') if k in report}
        rows.setdefault(variant,{})[selection['seed']] = dict(metric_values(selected))
    means, paired = {}, {}
    def stats(values):
        return dict(n=len(values), mean=statistics.mean(values), sample_std=statistics.stdev(values) if len(values)>1 else None)
    for variant, seeds in rows.items():
        keys = set.intersection(*(set(v) for v in seeds.values()))
        means[variant] = {k:stats([v[k] for v in seeds.values()]) for k in sorted(keys)}
    for variant in ('supervised','short','long'):
        if variant in rows and 'full' in rows:
            seeds = sorted(set(rows['full']) & set(rows[variant]))
            keys = set.intersection(*(set(rows['full'][s]) & set(rows[variant][s]) for s in seeds))
            paired[variant] = {k:dict(**stats([rows['full'][s][k]-rows[variant][s][k] for s in seeds]),
                differences={str(s):rows['full'][s][k]-rows[variant][s][k] for s in seeds}) for k in sorted(keys)}
    return dict(across_seeds=means, full_minus_ablation=paired, definition='sample SD ddof=1 across seeds; paired differences match seed; not window variability')


def report(root, manifest, state):
    tests = [r['stages']['test']['result'] for r in state['trials'].values() if r['stages'].get('test',{}).get('status')=='complete']
    value = dict(status=state['status'], elapsed_seconds=time.time()-manifest['started'],
        provenance=manifest, plans=state['plans'],
        rankings=state['rankings'], adjustments=state['adjustments'], resources=state.get('resources'),
        forecast=state.get('forecast'), loader_profile=state.get('loader_profile'), trials=state['trials'], seed_summary=summarize_seeds(tests) if len(tests)==12 else None,
        completed_test_runs=len(tests), measurement_definitions=dict(
            head_units=dict(force='N',moment='N m',r_B='m',v_B='m/s',S_B='kg m^2/s'),
            conservation_channels=['kinematic displacement [m]','linear momentum [N s]','angular momentum [N m s]'],
            normalization='Residual divided by duration times reference [sqrt(g L), m g, m g L]; L=Euclidean norm of interior dimensions (container diagonal). Fixed recorded reference conventions.',
            peaks='Vector error at target magnitude peak within each contextual window; mean and maximum across windows.',
            lag='Demeaned vector cross-correlation over valid nonconstant windows; offsets up to +/-50 ms, positive means prediction lags.',
            sampling='Overlapping windows; endpoint counts are window-weighted.',
            maneuver='Unavailable unless present in prepared series metadata; no inferred maneuver labels.'))
    atomic_json(root/'report.json', value)
    lines = ['# Staged sweep', '', f'Status: **{state["status"]}**. Test runs complete: {len(tests)}/12.', '',
             '| Stage | Budget | Candidates | Winner |', '|---|---:|---:|---|']
    for stage,p in state['plans'].items():
        lines.append(f'| {stage} | {BUDGETS[stage]} | {len(p["entries"])} | {(state["rankings"].get(stage) or ["incomplete"])[0]} |')
    lines += ['', 'Per-run physical errors, conservation residuals, peak/lag metrics, grouping coverage, checkpoint identities, config hashes and provenance are in [report.json](report.json).',
              'Seed summaries use sample standard deviation (ddof=1). Full-minus-ablation differences are paired by seed. Test results do not enter rankings.']
    if '6' in state['plans']:
        lines += ['', '| Final run | Stop reason/update | Selected update | Full validation MSE | Subset coverage |',
                  '|---|---|---:|---:|---:|']
        for entry in state['plans']['6']['entries']:
            r=state['trials'][entry['trial']]['stages'].get('6',{}).get('result')
            if r:
                lines.append(f"| {entry['label']} | {r['stopping_reason']} / {r['update']} | {r['best_update']} | {r['normalized_mse']:.6g} | {r['subset_coverage']:.3%} |")
    if value['seed_summary']:
        lines += ['', '| Variant | Force x RMSE mean ± seed SD (N) | Moment x RMSE mean ± seed SD (N m) |', '|---|---:|---:|']
        for variant,metrics in value['seed_summary']['across_seeds'].items():
            cells=[]
            for head in ('force','moment'):
                m=metrics[f'/heads/{head}/rmse_axis/0'];cells.append(f'{m["mean"]:.5g} ± {m["sample_std"]:.5g}')
            lines.append(f'| {variant} | {cells[0]} | {cells[1]} |')
    (root/'report.md').write_text('\n'.join(lines)+'\n')


def main(argv=None):
    started = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--devices', default='0')
    parser.add_argument('--concurrency', type=int, default=12)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--budget-hours', type=float, default=None, help='Deprecated and ignored; Slurm controls wall time')
    parser.add_argument('--allow-code-change',action='store_true',help='Ignored legacy argument')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--model-config', type=Path, help='Model JSON override for a new sweep')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    parser.add_argument('--worker-stage', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = args.output.resolve()
    if args.worker:
        return run_worker(root, args.worker, args.worker_stage, args.resume)
    if args.budget_hours is not None:
        print('Warning: --budget-hours is deprecated and ignored; Slurm controls wall time.',file=sys.stderr,flush=True)
    if args.data is None:
        parser.error('--data is required')
    if args.concurrency < 1 or args.threads < 1 :
        parser.error('Concurrency 1..12 and positive threads required')
    devices = validate_devices(args.devices, not args.dry_run)
    baseline = generate_trials(args.data, root, devices, args.threads)['00']
    baseline['training'].update(patience=20001)
    from .sweep import read_model_config
    if args.model_config:
        baseline['model'] = read_model_config(args.model_config)
    elif args.resume and (root/'staged_sweep.json').exists():
        baseline['model'] = read_json(root/'staged_sweep.json').get('model', baseline['model'])
    if args.dry_run:
        print(json.dumps(dict(plan=plan(), stage1=candidates('1', baseline,{}), max_concurrency=args.concurrency,
            devices=devices, threads=args.threads,wall_time_control='Slurm/signals only', preflight='disabled; requested execution resources are used'),indent=2))
        return 0
    if (root == args.data.resolve() or args.data.resolve() in root.parents
            or any((p/'sweep.json').exists() or (p/'benchmark.json').exists() for p in (root,*root.parents))
            or any((p/'staged_sweep.json').exists() for p in root.parents)):
        raise ValueError('Sweep outputs must be separate from prepared data')
    root.mkdir(parents=True, exist_ok=True)
    current, stopped = [None], [False]
    def stop(*_):
        stopped[0] = True
        if current[0] is not None:
            current[0].stop()
    with open(root/'.coordinator.lock','a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest,state = initialize(root,args,devices,started)
        manifest['model'] = baseline['model']
        previous = {s:signal.signal(s,stop) for s in (signal.SIGTERM,signal.SIGINT)}
        def persist():
            atomic_json(root/'staged_sweep.json',manifest)
            atomic_json(root/'state.json',state)
        try:
            # A surviving worker from a killed coordinator still owns its lock.
            # Never launch a new preflight alongside that old allocation's work.
            for path in root.rglob('.training.lock'):
                with open(path, 'a') as worker_lock:
                    try:
                        fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        raise ValueError(f'Existing worker still owns {path}; wait for it to exit')
            from .staged_validation import monitoring_subset
            monitoring_subset(args.data,root/'monitor_subset.json')
            slots = {d: args.concurrency // len(devices) + (i < args.concurrency % len(devices))
                     for i,d in enumerate(devices)}
            slots = {d:n for d,n in slots.items() if n}
            from .sweep_resume import assign_devices
            assign_devices(root,manifest,slots,args.threads)
            state['resources'] = dict(requested=args.concurrency,slots=slots,actual=args.concurrency)
            manifest['sessions'][-1]['resources'] = copy.deepcopy(state['resources'])
            persist()
            history={}
            for stage, _, _ in STAGES:
                if stopped[0]:
                    raise InterruptedError('Sweep interrupted')
                if stage in state['rankings']:
                    history[stage]=[read_json(root/'trials'/t/'resolved.json') for t in state['rankings'][stage]]
                    continue
                if stage not in state['plans']:
                    generated=candidates(stage,baseline,history)
                    entries=[dict(label=label,trial=register(root,cfg,state,manifest,slots)) for label,cfg in generated]
                    state['plans'][stage]=dict(entries=entries,parent_rankings=copy.deepcopy(state['rankings']),rule=plan()['refinement_rule'] if stage=='5' else 'dependent incumbent plus prescribed alternatives')
                    persist()
                entries=state['plans'][stage]['entries']
                current[0]=Coordinator(root,manifest,state,sum(slots.values()),command,inspect,max_concurrency=12,
                    device_limits=slots)
                if stopped[0] or not current[0].run(stage,[e['trial'] for e in entries]):
                    raise RuntimeError(f'Stage {stage} incomplete; no promotion')
                state['rankings'][stage]=rank([e['trial'] for e in entries],state,stage)
                history[stage]=[read_json(root/'trials'/t/'resolved.json') for t in state['rankings'][stage]]
                persist()
                report(root,manifest,state)
            frozen=frozen_selections(root,state)
            if not (root/'final_selections.json').exists():
                atomic_json(root/'final_selections.json',frozen)
            persist()
            current[0]=Coordinator(root,manifest,state,sum(slots.values()),command,inspect,max_concurrency=12,
                device_limits=slots)
            for entry in frozen['selections']:
                stage=state['trials'][entry['trial']]['stages'].get('test',{})
                if stage.get('status')=='complete' and not (root/'trials'/entry['trial']/'test_streaming.json').exists():
                    stage['status']='pending_streaming'
            if stopped[0] or not current[0].run('test',[e['trial'] for e in frozen['selections']]):
                raise RuntimeError('Final test evaluation incomplete')
            atomic_json(root/'test_streaming_summary.json',summarize_seeds([
                read_json(root/'trials'/e['trial']/'test_streaming.json') for e in frozen['selections']]))
            state['status']='complete'
            return 0
        except BaseException as error:
            state.update(status='incomplete',error_type=type(error).__name__,error=str(error))
            event('staged_sweep_incomplete',error=str(error))
            return 1
        finally:
            manifest['sessions'][-1].update(finished=time.time(),status=state['status'])
            persist()
            report(root,manifest,state)
            for s,h in previous.items():
                signal.signal(s,h)


if __name__=='__main__':
    raise SystemExit(main())
