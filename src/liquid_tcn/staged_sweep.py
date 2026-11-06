"""Deadline-bound dependent search and matched-seed ablations in an existing allocation."""
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
from .sweep_resources import available_resources, capacity, cpu_limit, forecast, original_deadline, profile_loader_settings
from .integrity import verify_once, check_receipt


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
        if digest(result['best_checkpoint']) != result['best_sha256']:
            raise ValueError('Selected checkpoint changed')
        rows.append(dict(**entry, seed=cfg['training']['seed'], checkpoint=result['best_checkpoint'],
                         checkpoint_sha256=result['best_sha256'], config_sha256=digest(config_path),
                         validation_metric=result['normalized_mse'], best_update=result['best_update'],
                         stopping_reason=result['stopping_reason'],stopping_update=result['update'],subset_coverage=result['subset_coverage']))
    return dict(selections=rows, rule='best full-validation normalized_supervised_mse_all_heads',
                dataset_sha256=state['data_manifest_sha256'])


def verify_test_gate(root, trial):
    state = read_json(root/'state.json')
    path = root/'final_selections.json'
    if not state.get('frozen_sha256') or digest(path) != state['frozen_sha256']:
        raise ValueError('Test evaluation requires frozen final selections')
    frozen = read_json(path)
    if frozen != frozen_selections(root, state):
        raise ValueError('Final selections no longer match completed confirmation results')
    return next(e for e in frozen['selections'] if e['trial'] == trial)


def run_worker(root, trial, stage, resume=False):
    """Runtime receipts/log IDs stay outside checkpoint configuration equality."""
    config = read_json(root/'trials'/trial/'resolved.json')
    manifest = read_json(root/'staged_sweep.json')
    check_receipt(config['data'], root/'verification.json')
    event('worker_context', stage=stage, trial=trial, seed=config['training']['seed'],
          device=config['training']['device'], remaining_seconds=manifest['deadline']-time.time())
    if stage != 'test':
        from .train import train
        from .staged_validation import StageValidation
        policy = StageValidation(stage,BUDGETS[stage],root/'monitor_subset.json',manifest['deadline']-1800)
        result = train(config['data'], config['output'], config['training'],
            resume=str(Path(config['output'])/'latest.pt') if resume else None,
            max_updates=BUDGETS[stage], expected_model=config['model'], launch_config=config,
            logging_mode='plain', trial_id=f'stage{stage}/{trial}/seed{config["training"]["seed"]}/{config["training"]["device"]}',
            verification_receipt=root/'verification.json',staged_policy=policy)
        return int(result['interrupted'] or not (Path(config['output']).parent/stage/'selection.json').exists())
    with open(Path(config['output'])/'.training.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return evaluate_selected(root, trial, config, manifest)


def evaluate_selected(root, trial, config, manifest):
    selected = verify_test_gate(root, trial)
    cached=root/'trials'/trial/'test.json'
    if cached.exists():
        result=read_json(cached)
        if result['selection']!=selected or result['frozen_sha256']!=digest(root/'final_selections.json'):
            raise ValueError('Cached test selection changed')
        return 0
    event('validation_start',scope='test',trial=trial,update=selected['best_update'])
    from .checkpoint import load_checkpoint
    from .dataset import WindowDataset, make_loader, close_loader
    from .evaluate import evaluate
    from .model import LiquidTCN, ModelConfig
    from .normalization import Normalization
    from .benchmark import CancellableLoader, synchronize
    from threading import Event
    import torch
    stopped = Event()
    class Cancellation:
        def cancelled(self):
            return stopped.is_set() or time.time() >= manifest['deadline']-1800
    previous = {s:signal.signal(s, lambda *_:stopped.set()) for s in (signal.SIGTERM, signal.SIGINT)}
    device = config['training']['device']
    torch.set_num_threads(config['training']['threads'])
    saved = load_checkpoint(selected['checkpoint'])
    if saved['config'] != config['training'] or saved['data_manifest_sha256'] != manifest['data_manifest_sha256']:
        raise ValueError('Test checkpoint provenance mismatch')
    model = LiquidTCN(ModelConfig(**saved['model_config'])).to(device)
    model.load_state_dict(saved['model'])
    dataset = WindowDataset(config['data'], 'test')
    loader = make_loader(dataset, batch_size=1024, num_workers=0, seed=config['training']['seed'])
    start = time.monotonic()
    try:
        report = evaluate(model, CancellableLoader(loader, Cancellation()), Normalization(saved['normalization']), device,
            supervised=config['training']['supervised'], huber_delta=config['training']['huber_delta'],
            lambda_short=config['training']['lambda_short'], lambda_long=config['training']['lambda_long'],
            physics_dtype=config['training']['physics_dtype'])
        synchronize(device)
        if report['status'] != 'measured' or report['coverage']['fraction'] != 1:
            raise ValueError('Incomplete test coverage')
        atomic_json(root/'trials'/trial/'test.json', dict(selection=selected, split='test',
            seconds=time.monotonic()-start, frozen_sha256=digest(root/'final_selections.json'), report=report))
    finally:
        close_loader(loader)
        for s,h in previous.items():
            signal.signal(s,h)
    return 0


def command(root, trial, stage, resume):
    result = [sys.executable, '-u', '-m', 'liquid_tcn.staged_sweep', '--worker', trial,
              '--worker-stage', stage, '--output', str(root)]
    if resume and stage != 'test':
        result.append('--resume')
    return result


def inspect(root, trial, stage, manifest):
    config_path = root/'trials'/trial/'resolved.json'
    if digest(config_path) != manifest['configs'][trial]:
        raise ValueError('Resolved configuration changed')
    if stage != 'test':
        from .checkpoint import load_checkpoint
        cfg = read_json(config_path)
        result = read_json(root/'trials'/trial/stage/'selection.json')
        rank([trial], {'trials':{trial:{'stages':{stage:dict(status='complete',result=result)}}}},stage)
        validation=read_json(result['per_payload_report'])
        if validation['coverage']['fraction']!=1 or validation['selection_metric']!=result['normalized_mse']:
            raise ValueError('Selected score requires matching full validation')
        if result['subset_sha256'] != digest(root/'monitor_subset.json'):
            raise ValueError('Selection monitoring subset changed')
        for kind in ('best','latest'):
            if digest(result[kind+'_checkpoint'])!=result[kind+'_sha256']:
                raise ValueError('Selected checkpoint changed')
            saved = load_checkpoint(result[kind+'_checkpoint'])
            expected = result['best_update'] if kind=='best' else result['update']
            if kind=='latest' and saved.get('stage_monitoring')!=result['monitoring']:
                raise ValueError('Monitoring resume state mismatch')
            if saved['config']!=cfg['training'] or saved['update']!=expected or saved['data_manifest_sha256']!=manifest['data_manifest_sha256']:
                raise ValueError('Selection checkpoint provenance mismatch')
        return result
    selected = verify_test_gate(root, trial)
    value = read_json(root/'trials'/trial/'test.json')
    if value['selection'] != selected or value['frozen_sha256'] != digest(root/'final_selections.json'):
        raise ValueError('Test report selection mismatch')
    return value


def initialize(root, args, devices, started):
    path = root/'staged_sweep.json'
    provenance = dict(data=str(args.data.resolve()), data_manifest_sha256=digest(args.data/'manifest.json'),
                      devices=devices, threads=args.threads, code=code_provenance(), runtime=runtime_versions())
    if args.resume:
        manifest, state = read_json(path), read_json(root/'state.json')
        if any(manifest[k] != v for k,v in provenance.items()):
            raise ValueError('Resume requires identical data, code, runtime, devices and threads')
        if args.budget_hours != manifest['budget_hours']:
            raise ValueError('Resume cannot change the original wall-clock budget')
        for trial, expected in manifest['configs'].items():
            if digest(root/'trials'/trial/'resolved.json') != expected:
                raise ValueError('Resolved configuration changed')
        for row in state['trials'].values():
            for stage, result in row['stages'].items():
                if stage != 'test' and result['status'] == 'complete':
                    for kind in ('best','latest'):
                        r = result['result']
                        if digest(r[kind+'_checkpoint']) != r[kind+'_sha256']:
                            raise ValueError('Archived checkpoint changed')
        # A new allocation can shorten, never extend, the original deadline.
        manifest['deadline'] = min(manifest['deadline'], original_deadline(started, args.budget_hours))
        atomic_json(path, manifest)
        return manifest, state
    if any(p.name != '.coordinator.lock' for p in root.iterdir()):
        raise ValueError('Use a new empty output directory, or --resume')
    manifest = dict(**provenance, started=started, deadline=original_deadline(started,args.budget_hours),
        budget_hours=args.budget_hours, baseline_commit=BASELINE_COMMIT, baseline_sha256=digest(BASELINE),
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
            config['training']['device'] = device
            config['output'] = str(worker_dir/'run')
            atomic_json(worker_dir/'resolved.json', config)
            specs[wid] = dict(worker_id=wid, worker_dir=str(worker_dir), barrier_dir=str(folder/'barriers'),
                config_sha256=digest(worker_dir/'resolved.json'), data_manifest_sha256=manifest['data_manifest_sha256'],
                warmup_updates=200, measured_updates=50, barrier_timeout=max(1,min(3600,manifest['deadline']-time.time()-1800)),
                verification_receipt=str(root/'verification.json'))
    current[0] = PhaseRunner(folder, specs, threads, max(1,min(3600,manifest['deadline']-time.time()-1800)), absolute_deadline=manifest['deadline']-1800)
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
        remaining_seconds=max(0,manifest['deadline']-time.time()), provenance=manifest, plans=state['plans'],
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
    parser.add_argument('--budget-hours', type=float, default=48)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--worker', help=argparse.SUPPRESS)
    parser.add_argument('--worker-stage', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = args.output.resolve()
    if args.worker:
        return run_worker(root, args.worker, args.worker_stage, args.resume)
    if args.data is None:
        parser.error('--data is required')
    if not 1 <= args.concurrency <= 12 or args.threads < 1 or not math.isfinite(args.budget_hours) or args.budget_hours <= .5:
        parser.error('Concurrency 1..12, positive threads, budget > 0.5 hours required')
    devices = validate_devices(args.devices, not args.dry_run)
    baseline = generate_trials(args.data, root, devices, args.threads)['00']
    baseline['training'].update(patience=20001)
    if args.dry_run:
        print(json.dumps(dict(plan=plan(), stage1=candidates('1', baseline,{}), max_concurrency=args.concurrency,
            devices=devices, threads=args.threads, budget_hours=args.budget_hours, preflight='one probe per device, then actual resource-limited concurrent cohort'),indent=2))
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
        previous = {s:signal.signal(s,stop) for s in (signal.SIGTERM,signal.SIGINT)}
        def persist():
            atomic_json(root/'staged_sweep.json',manifest)
            atomic_json(root/'state.json',state)
        try:
            if time.time() >= manifest['deadline']-1800:
                raise TimeoutError('Original budget exhausted; deadline is not reset on resume')
            # A surviving worker from a killed coordinator still owns its lock.
            # Never launch a new preflight alongside that old allocation's work.
            for path in root.rglob('.training.lock'):
                with open(path, 'a') as worker_lock:
                    try:
                        fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        raise ValueError(f'Existing worker still owns {path}; wait for it to exit')
            verify_once(args.data,root/'verification.json')
            from .staged_validation import monitoring_subset
            monitoring_subset(args.data,root/'monitor_subset.json')
            subset_hash=digest(root/'monitor_subset.json')
            if manifest.get('monitor_subset_sha256',subset_hash)!=subset_hash:
                raise ValueError('Fixed monitoring subset changed')
            manifest['monitor_subset_sha256']=subset_hash
            persist()
            resources = available_resources()
            cpu_limit(args.concurrency,args.threads,resources)
            if 'measurement' not in state:
                probes = {}
                for device in devices:
                    measured=preflight(root,baseline,{device:1},args.threads,manifest,current,stopped,'probe_'+device.replace(':','_'))
                    probes[device]=measured['workers'][0]
                slots=capacity(args.concurrency,args.threads,resources,probes)
                state['resources']=dict(available=resources,requested=args.concurrency,slots=slots,actual=sum(slots.values()),probes=probes)
                if sum(slots.values()) != args.concurrency:
                    state['adjustments'].append(dict(reason='CPU/RAM/measured GPU limits',requested=args.concurrency,actual=sum(slots.values())))
                event('resource_plan',requested=args.concurrency,actual=sum(slots.values()),slots=slots,available=resources)
                state['loader_profile']=profile_loader_settings(args.data,baseline,slots,resources,manifest['deadline']-1800)
                state['measurement']=preflight(root,baseline,slots,args.threads,manifest,current,stopped,'concurrent_preflight')
                persist()
            else:
                slots=state['resources']['slots']
                new_slots=capacity(args.concurrency,args.threads,resources,state['resources']['probes'])
                if slots != new_slots:
                    raise ValueError('Allocation capacity changed; existing placement/preflight cannot be reused safely')
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
                # Reserve remaining mandatory search, all finals and test. Use worst
                # fresh-start costs for unknown future configurations, not optimistic reuse.
                jobs=[]
                for entry in entries:
                    cfg=read_json(root/'trials'/entry['trial']/'resolved.json')
                    completed = max([r.get('result',{}).get('update',0) for r in state['trials'][entry['trial']]['stages'].values() if r.get('status')=='complete'] or [0])
                    remaining=max(0,BUDGETS[stage]-completed)
                    jobs.append((cfg['training']['device'],remaining,math.ceil(remaining/1000),4 if stage=='6' else 1))
                placement=[d for d,n in slots.items() for _ in range(n)]
                for future,_,number in STAGES:
                    if int(future)>int(stage) and future!='5':
                        jobs += [(placement[i%len(placement)],BUDGETS[future],BUDGETS[future]//1000,4 if future=='6' else 1) for i in range(number)]
                data_manifest=read_json(root/'data_manifest.json')
                estimate=forecast(jobs,slots,state['measurement'],data_manifest['windows']['test'],data_manifest['windows']['validation'])
                # Optional joint refinement is the first work removed. All retained
                # candidates still have a common 10k budget; finals always have 20k.
                if stage=='5':
                    while len(entries)>1 and estimate['total_seconds']>manifest['deadline']-time.time():
                        dropped=entries.pop()
                        state['adjustments'].append(dict(stage=stage,removed=dropped,reason='Reserve finals/test/shutdown within deadline'))
                        jobs.pop(len(entries))
                        estimate=forecast(jobs,slots,state['measurement'],data_manifest['windows']['test'],data_manifest['windows']['validation'])
                state['forecast']=estimate
                persist()
                event('forecast',stage=stage,remaining_seconds=manifest['deadline']-time.time(),**estimate)
                if estimate['total_seconds']>manifest['deadline']-time.time():
                    raise TimeoutError('Measured forecast cannot fit mandatory equal-budget work plus finals/test/reporting reserve')
                # Stop this stage early enough to preserve future reserved work.
                future_jobs=jobs[len(entries):]
                reserved=forecast(future_jobs,slots,state['measurement'],data_manifest['windows']['test'],data_manifest['windows']['validation'])['total_seconds']
                current[0]=Coordinator(root,manifest,state,sum(slots.values()),command,inspect,max_concurrency=12,
                    device_limits=slots,deadline=manifest['deadline']-reserved)
                if stopped[0] or not current[0].run(stage,[e['trial'] for e in entries]):
                    raise RuntimeError(f'Stage {stage} incomplete; no promotion')
                state['rankings'][stage]=rank([e['trial'] for e in entries],state,stage)
                history[stage]=[read_json(root/'trials'/t/'resolved.json') for t in state['rankings'][stage]]
                persist()
                report(root,manifest,state)
            frozen=frozen_selections(root,state)
            if state.get('frozen_sha256'):
                if digest(root/'final_selections.json') != state['frozen_sha256'] or read_json(root/'final_selections.json') != frozen:
                    raise ValueError('Frozen checkpoint selection changed')
            else:
                atomic_json(root/'final_selections.json',frozen)
                state['frozen_sha256']=digest(root/'final_selections.json')
                persist()
            current[0]=Coordinator(root,manifest,state,sum(slots.values()),command,inspect,max_concurrency=12,
                device_limits=slots,deadline=manifest['deadline']-1800)
            if stopped[0] or not current[0].run('test',[e['trial'] for e in frozen['selections']]):
                raise RuntimeError('Final test evaluation incomplete')
            state['status']='complete'
            return 0
        except BaseException as error:
            state.update(status='incomplete',error_type=type(error).__name__,error=str(error))
            event('staged_sweep_incomplete',error=str(error))
            return 1
        finally:
            persist()
            report(root,manifest,state)
            for s,h in previous.items():
                signal.signal(s,h)


if __name__=='__main__':
    raise SystemExit(main())
