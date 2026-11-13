"""Single versus four independent workers, using the sweep's baseline trainer."""
import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import resource
import socket
import subprocess
import sys
from threading import Event
import time

from .batch_logging import BatchLogger
from .ported.storage import atomic_json, digest
from .sweep import (BASELINE, BASELINE_COMMIT, code_provenance, generate_trials,
                    read_json, runtime_versions, validate_devices, worker_environment)

BARRIERS = ('warmup_done', 'measured_done', 'validation_ready', 'validation_done')


def synchronize(device):
    if str(device).startswith('cuda'):
        import torch
        torch.cuda.synchronize(device)


def memory_snapshot(device):
    if not str(device).startswith('cuda'):
        return dict(status='unavailable', reason='CPU run', peak_allocated_bytes=None,
                    peak_reserved_bytes=None, total_visible_device_bytes=None)
    import torch
    result = dict(status='measured', peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                  total_visible_device_bytes=torch.cuda.get_device_properties(device).total_memory)
    try:
        free, total = torch.cuda.mem_get_info(device)
        result.update(free_visible_device_bytes=free, total_visible_device_bytes=total)
    except RuntimeError as error:
        result['free_memory_unavailable'] = str(error)
    return result


def reset_peaks(device):
    if str(device).startswith('cuda'):
        import torch
        torch.cuda.reset_peak_memory_stats(device)


def device_identity(device):
    result = dict(logical_device=str(device), hostname=socket.gethostname(),
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  physical_device_telemetry=dict(status='unavailable',
                      reason='NVML/physical-device and MIG attribution are not collected; CUDA capacity describes the visible device or instance'))
    if str(device).startswith('cuda'):
        import torch
        props = torch.cuda.get_device_properties(device)
        result.update(name=props.name, uuid=str(props.uuid) if hasattr(props, 'uuid') else None,
                      compute_capability=[props.major, props.minor], total_visible_device_bytes=props.total_memory)
    else:
        result.update(name='CPU', total_visible_device_bytes=None)
    return result


class BarrierClient:
    """File rendezvous controlled by the parent; aborts are shared by all workers."""
    def __init__(self, folder, worker_id, timeout, stop_requested, log):
        self.folder, self.worker_id = Path(folder), worker_id
        self.timeout, self.stop_requested, self.log = timeout, stop_requested, log
        self.waits = {}
        self.training_stop_requested = None

    def cancelled(self):
        return (self.stop_requested.is_set()
                or (self.training_stop_requested is not None and self.training_stop_requested.is_set())
                or (self.folder/'abort.json').exists())

    def wait(self, name):
        started = time.monotonic()
        atomic_json(self.folder/f'{self.worker_id}.{name}.ready.json', dict(monotonic=started))
        self.log.emit('barrier_wait', barrier=name)
        while True:
            if self.cancelled():
                raise InterruptedError(f'Benchmark aborted at {name}')
            if (self.folder/f'{name}.release.json').exists():
                break
            time.sleep(.01)
        self.waits[name] = time.monotonic()-started
        self.log.emit('barrier_release', barrier=name, wait_seconds=self.waits[name])


class MeasurementHooks:
    """Runtime-only observer; no model/config/RNG mutation or second training loop.

    The first measured timestamp precedes the next DataLoader fetch. Each end
    timestamp follows ordinary TensorBoard/JSONL/console logging and GPU sync.
    The final TensorBoard flush belongs to the last latency. Maintenance is
    deferred until the measured_done barrier so fast workers cannot serialize a
    checkpoint while their peers are still being timed.
    """
    defer_maintenance = True

    def __init__(self, spec, barrier):
        self.spec, self.barrier = spec, barrier
        self.warmup = spec['warmup_updates']
        self.stop = self.warmup + spec['measured_updates']
        self.latencies, self.compute, self.checkpoints = [], [], []
        self.memory = {}
        self.error = None

    def bind(self, device, stop_requested, log):
        self.device, self.stop_requested, self.log = device, stop_requested, log
        self.barrier.training_stop_requested = stop_requested
        if self.barrier.cancelled():
            stop_requested.set()

    def checkpoint(self, update, name, seconds):
        self.checkpoints.append(dict(update=update, name=name, seconds=seconds))

    def ready(self, update):
        if update != 0:
            raise ValueError('Benchmark workers must start fresh')
        synchronize(self.device)
        self.memory['startup'] = memory_snapshot(self.device)
        reset_peaks(self.device)
        self.warmup_start = time.monotonic()
        self.log.emit('benchmark_warmup_start', warmup_updates=self.warmup)

    def wait_safely(self, name):
        # A timed-out/interrupted training barrier must return to the trainer,
        # which writes latest.pt at this completed update before exiting.
        try:
            if self.stop_requested.is_set():
                raise InterruptedError('Training stop requested')
            self.barrier.wait(name)
            return True
        except (InterruptedError, TimeoutError) as error:
            self.error = str(error)
            self.stop_requested.set()
            atomic_json(self.barrier.folder/f'{self.barrier.worker_id}.failure.json', dict(error=self.error, error_type=type(error).__name__))
            self.log.emit('benchmark_abort', error=self.error)
            return False

    def update_end(self, update, compute_seconds, writer):
        if update == self.warmup or update == self.stop:
            writer.flush()
            synchronize(self.device)
        now = time.monotonic()
        if update == self.warmup:
            self.warmup_end = now
            self.memory['warmup'] = memory_snapshot(self.device)
            if not self.wait_safely('warmup_done'):
                return
            synchronize(self.device)
            reset_peaks(self.device)
            self.start = self.previous = time.monotonic()
        elif update > self.warmup:
            self.latencies.append(now-self.previous)
            self.compute.append(compute_seconds)
            self.previous = now
            if update == self.stop:
                self.end = now
                self.memory['measured'] = memory_snapshot(self.device)
                if self.wait_safely('measured_done'):
                    reset_peaks(self.device)

    def costs(self, timing):
        keys = ('loader_seconds', 'transfer_seconds', 'training_seconds', 'logging_seconds')
        current = {k: timing[k] for k in keys}
        if not hasattr(self, 'cost_baseline'):
            self.cost_baseline = current
        self.section_costs = {k: current[k]-self.cost_baseline[k] for k in keys}
        # The callback precedes update_end, so warmup totals are captured exactly.
        if timing['updates'] == self.warmup:
            self.cost_baseline = current

    def report(self):
        import numpy as np
        if self.error or len(self.latencies) != self.spec['measured_updates']:
            raise ValueError(self.error or 'Incomplete measured updates')
        elapsed = self.end-self.start
        return dict(start_monotonic=self.start, end_monotonic=self.end,
            section_costs=getattr(self, 'section_costs', {}), updates=len(self.latencies), seconds=elapsed, updates_per_second=len(self.latencies)/elapsed,
            latency_seconds=self.latencies, median_latency_seconds=float(np.median(self.latencies)),
            p95_latency_seconds=float(np.percentile(self.latencies, 95)),
            compute_section_seconds=sum(self.compute),
            compute_section_updates_per_second=len(self.compute)/sum(self.compute),
            compute_section_definition='Host timed forward, backward, optimizer, scheduler, GPU synchronize; excludes loader and per-update metric logging, not pure GPU kernel time')


class CancellableLoader:
    """Let the existing evaluator observe cancellation between batches."""
    def __init__(self, loader, barrier):
        self.loader, self.dataset, self.barrier = loader, loader.dataset, barrier

    def __iter__(self):
        for batch in self.loader:
            if self.barrier.cancelled():
                raise InterruptedError('Validation cancelled')
            yield batch


def evaluate_latest(config, spec, barrier, log, memory):
    from .checkpoint import load_checkpoint
    from .dataset import WindowDataset, make_loader, close_loader
    from .evaluate import evaluate
    from .model import build_model, ModelConfig
    from .normalization import Normalization
    from .train import weights
    started = time.monotonic()
    checkpoint = Path(config['output'])/'latest.pt'
    saved = load_checkpoint(checkpoint)
    expected_update = spec['warmup_updates'] + spec['measured_updates']
    if saved['update'] != expected_update:
        raise ValueError('Benchmark training did not complete its measured updates')
    device = config['training']['device']
    model = build_model(ModelConfig(**saved['model_config'])).to(device)
    model.load_state_dict(saved['model'])
    dataset = WindowDataset(config['data'], 'validation', config['training']['cache_size'])
    loader = make_loader(dataset, batch_size=1024, num_workers=0,
                         pin_memory=config['training']['pin_memory'], seed=config['training']['seed'])
    norm = Normalization(saved['normalization'])
    short, long = weights(saved['config'], saved['update']-1)
    synchronize(device)
    memory['validation_setup'] = memory_snapshot(device)
    setup_seconds = time.monotonic()-started
    try:
        barrier.wait('validation_ready')
        log.emit('benchmark_validation_start', checkpoint=str(checkpoint), split='validation')
        synchronize(device)
        reset_peaks(device)
        start = time.monotonic()
        report = evaluate(model, CancellableLoader(loader, barrier), norm, device, None,
                          supervised=saved['config']['supervised'], huber_delta=saved['config']['huber_delta'],
                          lambda_short=short, lambda_long=long, physics_dtype=saved['config']['physics_dtype'])
        synchronize(device)
        end = time.monotonic()
        memory['validation'] = memory_snapshot(device)
        if (report['status'] != 'measured' or report['coverage']['fraction'] != 1
                or report['selection_metric'] is None or not math.isfinite(report['selection_metric'])):
            raise ValueError('Validation must have full coverage and a finite metric')
        barrier.wait('validation_done')
        report.update(split='validation', checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint))
        path = Path(spec['worker_dir'])/'validation.json'
        atomic_json(path, report)
        log.emit('benchmark_validation_end', seconds=end-start, validation_metric=report['selection_metric'])
        return dict(start_monotonic=start, end_monotonic=end, seconds=end-start,
                    windows=report['coverage']['evaluated_windows'], setup_seconds=setup_seconds,
                    windows_per_second=report['coverage']['evaluated_windows']/(end-start), report=str(path),
                    checkpoint=str(checkpoint), checkpoint_sha256=report['checkpoint_sha256'])
    finally:
        close_loader(loader)


def worker(spec_path):
    import torch
    from .train import train
    spec = read_json(spec_path)
    folder = Path(spec['worker_dir'])
    log = BatchLogger(spec['worker_id'], spec['warmup_updates']+spec['measured_updates'])
    stopped = Event()
    previous = {s: signal.signal(s, lambda *_: stopped.set()) for s in (signal.SIGINT, signal.SIGTERM)}
    barrier = BarrierClient(spec['barrier_dir'], spec['worker_id'], spec['barrier_timeout'], stopped, log)
    try:
        torch.set_num_interop_threads(1)
        config = read_json(folder/'resolved.json')
        if stopped.is_set():
            raise InterruptedError('Benchmark interrupted before training')
        hooks = MeasurementHooks(spec, barrier)
        result = train(config['data'], config['output'], config['training'], max_updates=log.budget,
                       expected_model=config['model'], launch_config=config, logging_mode='plain',
                       trial_id=spec['worker_id'], runtime_hooks=hooks)
        if result['interrupted'] or stopped.is_set() or hooks.error:
            raise InterruptedError(hooks.error or 'Benchmark training interrupted')
        training = hooks.report()
        with open(Path(config['output'])/'.training.lock', 'a') as evaluation_lock:
            fcntl.flock(evaluation_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            validation = evaluate_latest(config, spec, barrier, log, hooks.memory)
        if barrier.cancelled():
            raise InterruptedError('Benchmark interrupted after validation')
        result = dict(status='complete', worker_id=spec['worker_id'], training=training, validation=validation,
            memory=hooks.memory, device=device_identity(config['training']['device']),
            startup_seconds=hooks.warmup_start-spec['launch_monotonic']-sum(c['seconds'] for c in hooks.checkpoints if c['update'] == 0),
            startup_including_initial_checkpoint_seconds=hooks.warmup_start-spec['launch_monotonic'],
            warmup_seconds=hooks.warmup_end-hooks.warmup_start,
            checkpoint_seconds=sum(c['seconds'] for c in hooks.checkpoints), checkpoints=hooks.checkpoints,
            barrier_wait_seconds=barrier.waits, total_seconds=time.monotonic()-spec['launch_monotonic'],
            configuration=str(folder/'resolved.json'), config_sha256=spec['config_sha256'],
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            cpu_threads=config['training']['threads'], runtime=runtime_versions(), profile=result)
        atomic_json(folder/'result.json', result)
        log.emit('benchmark_worker_complete', measured_updates=training['updates'])
        return 0
    except BaseException as error:
        message = dict(status='failed', error_type=type(error).__name__, error=str(error),
                       failure_kind='out_of_memory' if 'outofmemory' in type(error).__name__.lower() or 'out of memory' in str(error).lower() else 'worker_failure')
        atomic_json(folder/'failure.json', message)
        atomic_json(Path(spec['barrier_dir'])/f'{spec["worker_id"]}.failure.json', message)
        log.emit('failure', **message)
        return 1
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)


def aggregate(workers, section, count):
    """Shared interval includes start skew and tail imbalance; never sum rates."""
    entries = [w[section] for w in workers]
    start = min(w['start_monotonic'] for w in entries)
    end = max(w['end_monotonic'] for w in entries)
    total = sum(w[count] for w in entries)
    if end <= start or total <= 0:
        raise ValueError('Empty or invalid benchmark interval')
    return dict(start_monotonic=start, end_monotonic=end, wall_seconds=end-start,
                count=total, per_second=total/(end-start),
                start_skew_seconds=max(w['start_monotonic'] for w in entries)-start)


def comparison(phases):
    single, four = phases['single']['workers'], phases['four']['workers']
    result = {}
    for name, workers in (('single', single), ('four', four)):
        result[name] = dict(training=aggregate(workers, 'training', 'updates'),
                            validation=aggregate(workers, 'validation', 'windows'))
    result['aggregate_speedup'] = result['four']['training']['per_second']/result['single']['training']['per_second']
    result['per_trial_slowdown'] = {w['worker_id']: w['training']['seconds']/single[0]['training']['seconds'] for w in four}
    result['validation_aggregate_speedup'] = result['four']['validation']['per_second']/result['single']['validation']['per_second']
    return result


def default_command(spec_path):
    return [sys.executable, '-u', '-m', 'liquid_tcn.benchmark', '--worker-spec', str(spec_path)]


class PhaseRunner:
    """Launch one fixed-size cohort and drain it before the next phase starts."""
    def __init__(self, folder, specs, threads, timeout, shutdown_timeout=300,
                 command_factory=default_command, heartbeat_seconds=60, absolute_deadline=None):
        self.folder, self.specs = Path(folder), specs
        self.threads, self.timeout, self.shutdown_timeout = threads, timeout, shutdown_timeout
        self.command_factory, self.heartbeat_seconds = command_factory, heartbeat_seconds
        self.absolute_deadline = absolute_deadline
        self.stopped = False
        self.log = BatchLogger(self.folder.name, 0)
        self.state = dict(status='running', workers={})

    def stop(self, *_):
        self.stopped = True

    def persist(self):
        atomic_json(self.folder/'state.json', self.state)

    def run(self):
        active = {}
        phase_start = time.monotonic()
        next_barrier = 0
        deadline = None if self.timeout is None else phase_start+self.timeout
        heartbeat = phase_start
        barrier_dir = self.folder/'barriers'
        barrier_dir.mkdir(exist_ok=True)
        try:
            for worker_id, spec in self.specs.items():
                if self.stopped:
                    raise InterruptedError('Benchmark interrupted during dispatch')
                worker_dir = Path(spec['worker_dir'])
                spec['launch_monotonic'] = time.monotonic()
                spec_path = worker_dir/'worker_spec.json'
                atomic_json(spec_path, spec)
                console_path = worker_dir/'console.log'
                handle = open(console_path, 'x')
                try:
                    process = subprocess.Popen(self.command_factory(spec_path), stdout=handle,
                                               stderr=subprocess.STDOUT, env=worker_environment(self.threads))
                except BaseException:
                    handle.close()
                    raise
                active[worker_id] = (process, handle)
                self.state['workers'][worker_id] = dict(pid=process.pid, status='running', console=str(console_path), exit_code=None)
                self.persist()
                self.log.emit('worker_started', worker_id=worker_id, pid=process.pid, console=str(console_path))
            while active:
                if self.stopped:
                    raise InterruptedError('Benchmark interrupted')
                failures = list(barrier_dir.glob('*.failure.json'))
                if failures:
                    raise RuntimeError(f'Worker failure: {read_json(failures[0])}')
                for worker_id, (process, handle) in list(active.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    handle.close()
                    self.state['workers'][worker_id].update(exit_code=code, status='complete' if code == 0 else 'failed')
                    self.persist()
                    del active[worker_id]
                    if code != 0 or next_barrier != len(BARRIERS):
                        raise RuntimeError(f'Worker {worker_id} exited {code} before successful phase completion; see console.log')
                    self.log.emit('worker_complete', worker_id=worker_id, exit_code=code)
                if next_barrier < len(BARRIERS):
                    name = BARRIERS[next_barrier]
                    if all((barrier_dir/f'{w}.{name}.ready.json').exists() for w in self.specs):
                        atomic_json(barrier_dir/f'{name}.release.json', dict(monotonic=time.monotonic()))
                        self.log.emit('cohort_release', barrier=name, workers=list(self.specs))
                        next_barrier += 1
                        deadline = None if self.timeout is None else time.monotonic()+self.timeout
                if time.monotonic()-heartbeat >= self.heartbeat_seconds:
                    self.log.emit('heartbeat', running=list(active), completed=sum(w['status'] == 'complete' for w in self.state['workers'].values()),
                                  failed=sum(w['status'] == 'failed' for w in self.state['workers'].values()))
                    heartbeat = time.monotonic()
                if active:
                    time.sleep(.02)
            results = [read_json(Path(s['worker_dir'])/'result.json') for s in self.specs.values()]
            for result, spec in zip(results, self.specs.values()):
                if result['status'] != 'complete' or result['training']['updates'] != spec['measured_updates']:
                    raise ValueError('Worker reported incomplete measurements')
            self.state.update(status='complete', workers_results=results, elapsed_seconds=time.monotonic()-phase_start)
            self.persist()
            return dict(workers=results, elapsed_seconds=self.state['elapsed_seconds'])
        except BaseException as error:
            self.state.update(status='failed', error_type=type(error).__name__, error=str(error))
            atomic_json(barrier_dir/'abort.json', dict(error=str(error)))
            self.log.emit('failure', error_type=type(error).__name__, error=str(error))
            raise
        finally:
            # Signal everyone before waiting for anyone. Graceful training exits
            # write latest.pt; a wedged worker is killed only after a bounded grace.
            for process, _ in active.values():
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)
            finish_by = time.monotonic()+self.shutdown_timeout
            heartbeat = time.monotonic()
            while any(p.poll() is None for p, _ in active.values()) and time.monotonic() < finish_by:
                if time.monotonic()-heartbeat >= self.heartbeat_seconds:
                    self.log.emit('heartbeat', shutting_down=True, running=[w for w, (p, _) in active.items() if p.poll() is None])
                    heartbeat = time.monotonic()
                time.sleep(.05)
            for worker_id, (process, handle) in active.items():
                forced = process.poll() is None
                if forced:
                    process.kill()
                code = process.wait()
                handle.close()
                self.state['workers'][worker_id].update(status='forced_kill' if forced else 'stopped', exit_code=code)
            self.persist()


def resolve_configs(data, output, device, threads, model_config=None):
    baseline = generate_trials(data, output, [device], threads, model_config)['00']
    configs = {}
    for phase, count in (('single', 1), ('four', 4)):
        configs[phase] = {}
        for i in range(count):
            worker_id = f'{phase}_{i:02d}'
            config = copy.deepcopy(baseline)
            config['output'] = str(Path(output)/phase/worker_id/'run')
            configs[phase][worker_id] = config
    return configs


def table(summary):
    rows = ['| Processes | Aggregate updates/s | Speedup | Per-trial slowdown | Validation windows/s | Max worker peak allocated/reserved GiB |',
            '|---:|---:|---:|---:|---:|---:|']
    for phase, count in (('single', 1), ('four', 4)):
        c = summary['comparison'][phase]
        workers = summary['phases'][phase]['workers']
        peaks = [w['memory']['measured'] for w in workers]
        allocated = [p['peak_allocated_bytes'] for p in peaks if p['peak_allocated_bytes'] is not None]
        reserved = [p['peak_reserved_bytes'] for p in peaks if p['peak_reserved_bytes'] is not None]
        memory = f'{max(allocated)/2**30:.2f}/{max(reserved)/2**30:.2f}' if allocated and reserved else 'unavailable'
        speedup = summary['comparison']['aggregate_speedup'] if count == 4 else 1
        slowdowns = list(summary['comparison']['per_trial_slowdown'].values()) if count == 4 else [1]
        slowdown = f'{min(slowdowns):.2f}–{max(slowdowns):.2f}x'
        rows.append(f'| {count} | {c["training"]["per_second"]:.3f} | {speedup:.2f}x | {slowdown} | {c["validation"]["per_second"]:.2f} | {memory} |')
    return '\n'.join(rows)+'\n'


def main(argv=None):
    invocation_start = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--devices', default='0')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--warmup-updates', type=int, default=200)
    parser.add_argument('--measured-updates', type=int, default=300)
    parser.add_argument('--barrier-timeout', type=float, default=3600)
    parser.add_argument('--shutdown-timeout', type=float, default=300)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--model-config', type=Path, help='Model JSON override')
    parser.add_argument('--worker-spec', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_spec:
        return worker(args.worker_spec)
    if args.data is None or args.output is None:
        parser.error('--data and --output are required')
    log = BatchLogger('benchmark', args.warmup_updates+args.measured_updates)
    try:
        if (min(args.threads, args.warmup_updates, args.measured_updates) < 1
                or args.warmup_updates+args.measured_updates > 20000
                or not all(math.isfinite(t) and t > 0 for t in (args.barrier_timeout, args.shutdown_timeout))):
            raise ValueError('Positive counts/timeouts required, with warmup + measured <= 20000')
        devices = validate_devices(args.devices, not args.dry_run)
        if len(devices) != 1:
            raise ValueError('Benchmark requires exactly one logical device for both phases')
        root, data = args.output.resolve(), args.data.resolve()
        from .sweep import read_model_config
        configs = resolve_configs(data, root, devices[0], args.threads, read_model_config(args.model_config) if args.model_config else None)
        if args.dry_run:
            print(json.dumps(dict(configs=configs, warmup_updates=args.warmup_updates,
                                  measured_updates=args.measured_updates, max_updates=args.warmup_updates+args.measured_updates,
                                  validation=dict(split='validation', batch_size=1024, num_workers=0),
                                  required_worker_cpu_threads=4*args.threads), indent=2))
            return 0
        if root == data or data in root.parents or any((p/'sweep.json').exists() or (p/'staged_sweep.json').exists() for p in (root, *root.parents)):
            raise ValueError('Benchmark output must be separate from prepared data and production sweeps')
        root.mkdir(parents=True, exist_ok=True)
        with open(root/'.coordinator.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if any(p.name != '.coordinator.lock' for p in root.iterdir()):
                raise ValueError('Benchmark requires a new empty output directory; runs are never overwritten or resumed')
            start = invocation_start
            (root/'baseline.json').write_bytes(BASELINE.read_bytes())
            (root/'data_manifest.json').write_bytes((data/'manifest.json').read_bytes())
            provenance = dict(format_version=1, baseline_commit=BASELINE_COMMIT, baseline_sha256=digest(BASELINE),
                code=code_provenance(), data_manifest_sha256=digest(data/'manifest.json'), runtime=runtime_versions(),
                device=dict(logical_device=devices[0], cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')), cpu=dict(threads_per_worker=args.threads, four_worker_threads=4*args.threads,
                    slurm_cpus_per_task=os.environ.get('SLURM_CPUS_PER_TASK'),
                    affinity_cpus=len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count()),
                warmup_updates=args.warmup_updates, measured_updates=args.measured_updates,
                barrier_timeout=args.barrier_timeout, shutdown_timeout=args.shutdown_timeout,
                maintenance='scheduled checkpoint/validation deferred; initial and final checkpoints retained')
            atomic_json(root/'benchmark.json', provenance)
            summary = dict(status='running', provenance=provenance, phases={})
            current = [None]
            stopped = Event()
            def stop(*_):
                stopped.set()
                if current[0] is not None:
                    current[0].stop()
            previous = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
            try:
                for phase in ('single', 'four'):
                    if stopped.is_set():
                        raise InterruptedError('Benchmark interrupted between phases')
                    specs = {}
                    folder = root/phase
                    folder.mkdir()
                    for worker_id, config in configs[phase].items():
                        worker_dir = folder/worker_id
                        worker_dir.mkdir()
                        atomic_json(worker_dir/'resolved.json', config)
                        specs[worker_id] = dict(worker_id=worker_id, worker_dir=str(worker_dir), barrier_dir=str(folder/'barriers'),
                            config_sha256=digest(worker_dir/'resolved.json'), data_manifest_sha256=provenance['data_manifest_sha256'],
                            warmup_updates=args.warmup_updates, measured_updates=args.measured_updates, barrier_timeout=args.barrier_timeout,
                            verification_receipt=str(root/'verification.json'))
                    current[0] = PhaseRunner(folder, specs, args.threads, args.barrier_timeout, args.shutdown_timeout)
                    if stopped.is_set():
                        raise InterruptedError('Benchmark interrupted before phase launch')
                    log.emit('phase_start', phase=phase, processes=len(specs))
                    summary['phases'][phase] = current[0].run()
                    # Query identity in the workers, avoiding a parent CUDA context.
                    provenance['device'] = summary['phases'][phase]['workers'][0]['device']
                    atomic_json(root/'benchmark.json', provenance)
                    atomic_json(root/'summary.json', summary)
                if stopped.is_set():
                    raise InterruptedError('Benchmark interrupted before completion')
                summary.update(status='complete', comparison=comparison(summary['phases']))
                rendered = table(summary)
                (root/'comparison.md').write_text(rendered)
                print(rendered, flush=True)
                return 0
            except BaseException as error:
                summary.update(status='failed', error_type=type(error).__name__, error=str(error),
                               failed_phase=current[0].folder.name if current[0] else None,
                               phase_state=str(current[0].folder/'state.json') if current[0] else None)
                raise
            finally:
                summary['total_elapsed_seconds'] = time.monotonic()-start
                atomic_json(root/'summary.json', summary)
                for s, handler in previous.items():
                    signal.signal(s, handler)
    except BaseException as error:
        log.emit('failure', error_type=type(error).__name__, error=str(error))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
