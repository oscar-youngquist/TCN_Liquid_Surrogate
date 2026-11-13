"""Bounded, resumable two-stage experiments; never submits scheduler jobs."""
import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import time

from .ported.storage import atomic_json, digest

BASELINE_COMMIT = '5c7792d0840ddfc039f48877d077cacd1da2038e'
BASELINE_SHA256 = 'ade364a795b42c258a90e4927ff605e088e08b46bedfa05e6a322b2003675155'
BASELINE = Path(__file__).parent / 'baselines/combined_5c7792d.json'
BUDGETS = {'screen': 5000, 'continue': 20000}
OVERRIDES = [
    {},
    *[dict(supervised='huber', huber_delta=d) for d in (.25, .5, 1., 2.)],
    dict(physics_lr=3e-4), dict(physics_lr=1e-4),
    dict(lambda_short=.3, lambda_long=1), dict(lambda_short=1, lambda_long=3),
    dict(lambda_short=.3, lambda_long=3),
    dict(pretrain_updates=500, ramp_updates=1500),
    dict(lambda_short=0, lambda_long=0), dict(lambda_short=1, lambda_long=0),
    dict(lambda_short=0, lambda_long=1),
    dict(supervised='huber', huber_delta=.5, lambda_short=.5, lambda_long=.5),
    dict(supervised='huber', huber_delta=1., lambda_short=.5, lambda_long=.5),
]


def read_json(path):
    return json.loads(Path(path).read_text())


def event(kind, **fields):
    print(json.dumps(dict(timestamp=datetime.now(timezone.utc).isoformat(), event=kind, **fields),
                     allow_nan=False), flush=True)


def code_provenance():
    """Hash installed Python sources as well as recording the checkout commit.

    A dirty tree is allowed for development, but its actual source is fingerprinted.
    Resuming with different code is rejected instead of claiming reproducibility.
    """
    package = Path(__file__).parent
    files = {str(p.relative_to(package)): digest(p) for p in sorted(package.rglob('*.py'))}
    fingerprint = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=package,
                                         stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return dict(commit=commit, source_sha256=fingerprint, source_files=files)


def worker_environment(threads):
    """Bound each independent worker's math-library pools without remapping GPUs."""
    env = os.environ.copy()
    env.pop('LIQUID_RUN_DEADLINE', None)
    env.update(OMP_MAX_ACTIVE_LEVELS='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS=str(threads),
               MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
               NUMEXPR_NUM_THREADS=str(threads))
    return env


def runtime_versions():
    import numpy
    import torch
    return dict(python=sys.version, torch=torch.__version__, numpy=numpy.__version__, cuda=torch.version.cuda)


def read_model_config(path):
    """Accept either a model-only JSON or the model section of a run config."""
    value = read_json(path)
    from .model import ModelConfig
    from dataclasses import asdict
    return asdict(ModelConfig(**value.get('model', value)))


def generate_trials(data, output, devices, threads, model_config=None):
    from .train import DEFAULTS
    baseline = read_json(BASELINE)
    if model_config is not None:
        baseline['model'] = model_config
    trials = {}
    for number, overrides in enumerate(OVERRIDES):
        trial_id = f'{number:02d}'
        config = copy.deepcopy(baseline)
        config.update(data=str(Path(data).resolve()), output=str(Path(output).resolve()/'trials'/trial_id/'run'),
                      resume=None, max_updates=None)
        config['training'] = dict(DEFAULTS, **config['training'])
        config['training'].update(overrides)
        config['training'].update(validation_every=1000, patience=5, checkpoint_every=500,
                                  validation_batches=None, num_workers=0, validation_num_workers=0,
                                  threads=threads, device=devices[number % len(devices)], progress=False)
        trials[trial_id] = config
    return trials


def validate_devices(value, check_available=True):
    devices = value.split(',')
    if devices == ['cpu']:
        return devices
    if not devices or any(not d.isdigit() for d in devices) or len(set(devices)) != len(devices):
        raise ValueError('--devices must be comma-separated logical GPU indices, or cpu')
    return ['cuda:' + str(int(d)) for d in devices]


def initialize(root, data, devices, threads, resume, model_config=None):
    """Create a run or reopen saved progress using current paths and resources."""
    manifest_path = root/'sweep.json'
    if manifest_path.exists():
        if not resume:
            raise ValueError('Sweep already exists: use --resume')
        manifest = read_json(manifest_path)
        state = read_json(root/'state.json')
        from .sweep_resume import relocate
        relocate(root, data, manifest, state)
        manifest['execution'] = dict(threads=threads,assignments={t:devices[i%len(devices)] for i,t in enumerate(state['trials'])})
        for trial,device in manifest['execution']['assignments'].items():
            path = root/'trials'/trial/'resolved.json'
            cfg = read_json(path)
            cfg['training'].update(device=device,threads=threads)
            atomic_json(path,cfg)
        atomic_json(manifest_path,manifest)
        atomic_json(root/'state.json',state)
        return manifest, state
    if resume:
        raise ValueError('Cannot resume: sweep.json does not exist')
    if any(p.name != '.coordinator.lock' for p in root.iterdir()):
        raise ValueError('Refusing to overwrite an unrelated or partially initialized directory')
    configs = generate_trials(data, root, devices, threads, model_config)
    # The prepared manifest includes schema, split, normalization, source hashes,
    # and every artifact checksum. Artifact hashes are historical metadata.
    (root/'baseline.json').write_bytes(BASELINE.read_bytes())
    (root/'data_manifest.json').write_bytes((data/'manifest.json').read_bytes())
    immutable = ['baseline.json', 'data_manifest.json']
    state = dict(trials={})
    for trial_id, config in configs.items():
        folder = root/'trials'/trial_id
        (folder/'console').mkdir(parents=True)
        path = folder/'resolved.json'
        atomic_json(path, config)
        immutable.append(str(path.relative_to(root)))
        state['trials'][trial_id] = dict(attempts=[], stages={})
    manifest = dict(format_version=1, baseline_commit=BASELINE_COMMIT,
        baseline_file='configs/combined.json', baseline_sha256=digest(BASELINE),
        data=str(data.resolve()), data_manifest_sha256=digest(data/'manifest.json'),
        devices=devices, threads=threads, code=code_provenance(),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        runtime=runtime_versions(), budgets=BUDGETS, metric='normalized_supervised_mse_all_heads',
        trial_ids=list(configs), immutable_files={p: digest(root/p) for p in immutable})
    atomic_json(root/'state.json', state)
    atomic_json(manifest_path, manifest)
    return manifest, state


def inspect_result(root, trial_id, stage, manifest, *, budget=None, allow_early_stop=True):
    """Legacy best-validation selection and stage completion checks.

    The staged sweep uses its explicit endpoint/shortlist selection record instead;
    a legacy historical-best score must never promote a staged search candidate.
    """
    from .checkpoint import load_checkpoint
    from .model import ModelConfig
    config = read_json(root/'trials'/trial_id/'resolved.json')
    run = Path(config['output'])
    latest = load_checkpoint(run/'latest.pt')
    best = load_checkpoint(run/'best.pt')
    required = BUDGETS[stage] if budget is None else budget
    if latest['update'] != required and not (allow_early_stop and
            stage == 'continue' and latest['stale'] >= config['training']['patience']
            and BUDGETS['screen'] <= latest['update'] <= BUDGETS[stage]):
        raise ValueError(f"Incomplete {stage}: checkpoint update {latest['update']}, required {required}")
    report = best['validation']
    score = report['selection_metric']
    if (score is None or not math.isfinite(score) or report['coverage']['fraction'] != 1
            or report['status'] != 'measured' or best['update'] > latest['update']):
        raise ValueError('Best checkpoint lacks a consistent finite full-validation MSE')
    archive = root/'trials'/trial_id/stage
    archive.mkdir(exist_ok=True)
    for name in ('best.pt', 'latest.pt'):
        shutil.copyfile(run/name, archive/(name + '.tmp'))
        os.replace(archive/(name + '.tmp'), archive/name)
    report_path = root/'trials'/trial_id/f'{stage}_best_validation.json'
    atomic_json(report_path, report)
    return dict(update=latest['update'], best_update=best['update'], normalized_mse=score,
        heads=report['heads'], peak_wrench=report['peak_wrench'], conservation=report['conservation'],
        coverage=report['coverage'], per_payload_report=str(report_path),
        best_checkpoint=str(archive/'best.pt'), latest_checkpoint=str(archive/'latest.pt'),
        resume_checkpoint=str(run/'latest.pt'),
        best_sha256=digest(run/'best.pt'), latest_sha256=digest(run/'latest.pt'),
        profile=read_json(run/'profile.json') if (run/'profile.json').exists() else None)


def select_trials(state):
    trials = state['trials']
    incomplete = [t for t, row in trials.items() if row['stages'].get('screen', {}).get('status') != 'complete']
    if incomplete:
        raise ValueError('Screening failed/incomplete; no promotion: ' + ', '.join(incomplete))
    ranked = sorted((t for t in trials if t != '00'),
                    key=lambda t: (trials[t]['stages']['screen']['result']['normalized_mse'], t))
    return ['00'] + ranked[:3]


def worker_command(root, trial_id, stage, resume):
    command = [sys.executable, '-u', '-m', 'liquid_tcn', 'train', '--config',
               str(root/'trials'/trial_id/'resolved.json'), '--max-updates', str(BUDGETS[stage]),
               '--logging-mode', 'plain', '--trial-id', trial_id]
    if resume:
        command += ['--resume', str(root/'trials'/trial_id/'run/latest.pt')]
    return command


class Coordinator:
    """One owner serializes state/events; training runs in independent processes."""
    def __init__(self, root, manifest, state, concurrency=4, command_factory=worker_command,
                 inspect=inspect_result, heartbeat_seconds=60, max_concurrency=4, deadline=None, device_limits=None, shutdown_timeout=300):
        if concurrency < 1:
            raise ValueError('concurrency must be positive')
        self.root, self.manifest, self.state = root, manifest, state
        self.concurrency, self.command_factory, self.inspect = concurrency, command_factory, inspect
        self.heartbeat_seconds = heartbeat_seconds
        self.stopping = False
        self.device_limits, self.shutdown_timeout = device_limits, shutdown_timeout  # deadline is ignored.
        self.stop_time = None

    def stop(self, number=None, frame=None):
        self.stopping = True

    def trial_device(self, trial_id):
        return self.manifest.get('execution',{}).get('assignments',{}).get(
            trial_id, read_json(self.root/'trials'/trial_id/'resolved.json')['training']['device'])

    def persist(self):
        atomic_json(self.root/'state.json', self.state)

    def run(self, stage, ids):
        pending = [t for t in ids if self.state['trials'][t]['stages'].get(stage, {}).get('status') != 'complete']
        active = {}
        heartbeat = time.monotonic()
        try:
            while pending or active:
                while pending and len(active) < self.concurrency and not self.stopping:
                    position = 0
                    if self.device_limits:
                        occupied = {}
                        for _, _, _, a in active.values():
                            occupied[a['device']] = occupied.get(a['device'], 0)+1
                        eligible = [i for i, t in enumerate(pending) if occupied.get(
                            self.trial_device(t), 0) < self.device_limits.get(
                            self.trial_device(t), 0)]
                        if not eligible:
                            if not active:
                                raise ValueError('No resource slot for pending trial')
                            break
                        position = eligible[0]
                    trial_id = pending.pop(position)
                    row = self.state['trials'][trial_id]
                    checkpoint = self.root/'trials'/trial_id/'run/latest.pt'
                    if stage == 'continue' and not checkpoint.exists():
                        raise ValueError(f'Trial {trial_id}: continuation requires latest.pt')
                    run = checkpoint.parent
                    if (run/'.training.lock').exists():
                        with open(run/'.training.lock', 'a') as worker_lock:
                            try:
                                fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            except BlockingIOError:
                                raise ValueError(f'Trial {trial_id} still has an active worker; wait for its allocation to end')
                    console = self.root/'trials'/trial_id/'console'/f'{len(row["attempts"])+1:04d}_{stage}.log'
                    attempt = dict(stage=stage, status='running', hostname=socket.gethostname(), started=datetime.now(timezone.utc).isoformat(),
                                   console=str(console), exit_code=None,
                                   resume_checkpoint_sha256=digest(checkpoint) if checkpoint.exists() else None)
                    config_path = self.root/'trials'/trial_id/'resolved.json'
                    if config_path.exists():
                        c = read_json(config_path)['training']
                        attempt.update(device=self.trial_device(trial_id), seed=c['seed'])
                    row['attempts'].append(attempt)
                    row['stages'][stage] = dict(status='running')
                    self.persist()
                    handle = open(console, 'x')
                    env = worker_environment(self.manifest.get('execution',{}).get('threads',self.manifest['threads']))
                    try:
                        process = subprocess.Popen(self.command_factory(self.root, trial_id, stage, checkpoint.exists()),
                                                   stdout=handle, stderr=subprocess.STDOUT, env=env)
                    except OSError as error:
                        handle.close()
                        attempt.update(status='failed', error=str(error), exit_code=None)
                        row['stages'][stage] = dict(status='failed', error=str(error))
                        self.persist()
                        event('trial_failed', trial_id=trial_id, stage=stage, error=str(error))
                        continue
                    attempt['pid'] = process.pid
                    self.persist()
                    active[trial_id] = (process, handle, time.monotonic(), attempt)
                    event('trial_started', trial_id=trial_id, stage=stage, pid=process.pid, console=str(console))
                if self.stopping:
                    if self.stop_time is None:
                        self.stop_time = time.monotonic()
                    for process, _, _, attempt in active.values():
                        if time.monotonic()-self.stop_time > self.shutdown_timeout and process.poll() is None:
                            process.kill()
                        if not attempt.get('stop_requested'):
                            process.send_signal(signal.SIGTERM)
                            attempt['stop_requested'] = True
                    pending.clear()
                for trial_id, (process, handle, started, attempt) in list(active.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    handle.close()
                    attempt.update(exit_code=code, wall_seconds=time.monotonic()-started,
                                   finished=datetime.now(timezone.utc).isoformat())
                    result = None
                    try:
                        if attempt.get('stop_requested'):
                            raise InterruptedError('Worker termination was requested')
                        if code != 0:
                            # Keep the traceback/OOM details in the separate console log.
                            with open(attempt['console'], 'rb') as stream:
                                stream.seek(0, 2)
                                stream.seek(max(0, stream.tell()-65536))
                                tail = stream.read().decode(errors='replace').lower()
                            attempt['failure_kind'] = 'out_of_memory' if ('out of memory' in tail or 'outofmemoryerror' in tail) else 'worker_exit'
                            raise ValueError(f'{attempt["failure_kind"]}: worker exit {code}; inspect {attempt["console"]}')
                        result = self.inspect(self.root, trial_id, stage, self.manifest)
                        attempt['resources'] = result.get('profile')
                        status = 'complete'
                    except Exception as error:
                        status = 'interrupted' if attempt.get('stop_requested') else 'failed'
                        attempt['error'] = str(error)
                    attempt['status'] = status
                    self.state['trials'][trial_id]['stages'][stage] = dict(status=status, result=result,
                        error=attempt.get('error'), exit_code=code)
                    self.persist()
                    event('trial_' + status, trial_id=trial_id, stage=stage, exit_code=code,
                          error=attempt.get('error'))
                    del active[trial_id]
                if time.monotonic()-heartbeat >= self.heartbeat_seconds:
                    statuses = [row['stages'].get(stage, {}).get('status') for row in self.state['trials'].values()]
                    event('heartbeat', stage=stage, running=list(active), pending=len(pending),
                          completed=statuses.count('complete'), failed=statuses.count('failed'),
                          elapsed_seconds=None if 'started' not in self.manifest else time.time()-self.manifest['started'],
                          workers={t: {k:a.get(k) for k in ('device','seed','started')} for t, (_,_,_,a) in active.items()})
                    heartbeat = time.monotonic()
                if active:
                    time.sleep(.2)
        finally:
            # Also drain children after unexpected coordinator errors; no orphaned
            # training processes should outlive the lock protecting their run dirs.
            for process, _, _, _ in active.values():
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)
            finish_by = time.monotonic()+self.shutdown_timeout
            for trial_id, (process, handle, started, attempt) in active.items():
                try:
                    code = process.wait(timeout=max(0,finish_by-time.monotonic()))
                except subprocess.TimeoutExpired:
                    process.kill()
                    code = process.wait()
                handle.close()
                attempt.update(exit_code=code, status='interrupted', wall_seconds=time.monotonic()-started,
                               error='Coordinator aborted; worker drained before releasing lock')
                self.state['trials'][trial_id]['stages'][stage] = dict(status='interrupted', exit_code=code)
            self.persist()
        return not self.stopping and all(
            self.state['trials'][t]['stages'].get(stage, {}).get('status') == 'complete' for t in ids)


def write_report(root, manifest, state):
    rows = []
    for trial_id, row in state['trials'].items():
        rows.append(dict(trial_id=trial_id, stages=row['stages'], attempts=row['attempts'],
            total_wall_seconds=sum(a.get('wall_seconds', 0) for a in row['attempts']),
            configuration=str(root/'trials'/trial_id/'resolved.json'),
            best_checkpoint=str(root/'trials'/trial_id/'run/best.pt'),
            latest_checkpoint=str(root/'trials'/trial_id/'run/latest.pt'),
            tensorboard=str(root/'trials'/trial_id/'run/tensorboard')))
    report = dict(metric=manifest['metric'], selection_split='validation', trials=rows)
    atomic_json(root/'report.json', report)
    lines = ['# Sweep results', '', 'Selection: full-validation normalized supervised MSE. Test data is not used.', '',
             '| Trial | Screen | Continue | Best normalized MSE | Reports |', '|---|---|---|---:|---|']
    for row in rows:
        stages = row['stages']
        screen, continuation = (stages.get(s, {}) for s in ('screen', 'continue'))
        result = continuation.get('result') or screen.get('result') or {}
        path = result.get('per_payload_report')
        link = f'[physical errors / payloads]({path})' if path else '—'
        lines.append(f"| {row['trial_id']} | {screen.get('status', 'pending')} | {continuation.get('status', 'pending')} | {result.get('normalized_mse', '—')} | {link} |")
    (root/'report.md').write_text('\n'.join(lines) + '\n')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--stage', choices=('screen', 'continue', 'auto', 'report'), default='screen')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--concurrency', type=int, default=4)
    parser.add_argument('--devices', default='0', help='Visible logical GPU indices, e.g. 0 or 0,1; cpu for testing')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--model-config', type=Path, help='Model JSON override for a new sweep')
    args = parser.parse_args(argv)
    if args.concurrency < 1 or args.threads < 1:
        parser.error('concurrency must be positive and threads must be positive')
    root = args.output.resolve()
    try:
        devices = validate_devices(args.devices, not args.dry_run and args.stage != 'report')
        if args.dry_run:
            configs = generate_trials(args.data, root, devices, args.threads, read_model_config(args.model_config) if args.model_config else None)
            print(json.dumps(dict(stage=args.stage, concurrency=args.concurrency, budgets=BUDGETS,
                                  baseline_commit=BASELINE_COMMIT, trials=configs), indent=2))
            return 0
        root.mkdir(parents=True, exist_ok=True)
        with open(root/'.coordinator.lock', 'a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError('Another coordinator owns this sweep directory')
            if args.stage == 'report':
                manifest, state = read_json(root/'sweep.json'), read_json(root/'state.json')
                write_report(root, manifest, state)
                event('report', path=str(root/'report.json'))
                return 0
            manifest, state = initialize(root, args.data.resolve(), devices, args.threads, args.resume, read_model_config(args.model_config) if args.model_config else None)
            # The worker lock prevents a surviving worker from sharing its output
            # with a retry, even if the previous coordinator was killed.
            for row in state['trials'].values():
                for attempt in row['attempts']:
                    if attempt['status'] == 'running':
                        attempt.update(status='interrupted', error='Coordinator exited before recording worker completion')
                        row['stages'][attempt['stage']]['status'] = 'interrupted'
            coordinator = Coordinator(root, manifest, state, args.concurrency)
            previous = {s: signal.signal(s, coordinator.stop) for s in (signal.SIGTERM, signal.SIGINT)}
            try:
                success = True
                if args.stage in ('screen', 'auto'):
                    success = coordinator.run('screen', manifest['trial_ids'])
                if success and not coordinator.stopping and args.stage in ('continue', 'auto'):
                    selected = select_trials(state)
                    selection_path = root/'selection.json'
                    selection = dict(trial_ids=selected, metric=manifest['metric'],
                        screening={t: state['trials'][t]['stages']['screen']['result'] for t in manifest['trial_ids']})
                    if selection_path.exists():
                        if read_json(selection_path) != selection:
                            raise ValueError('Recorded promotion differs from screening results')
                    else:
                        atomic_json(selection_path, selection)
                    event('selection', trial_ids=selected)
                    success = coordinator.run('continue', selected)
                return 0 if success else (143 if coordinator.stopping else 1)
            finally:
                write_report(root, manifest, state)
                for s, handler in previous.items():
                    signal.signal(s, handler)
    except Exception as error:
        event('failure', error_type=type(error).__name__, error=str(error))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
