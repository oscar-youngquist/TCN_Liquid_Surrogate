"""No production benchmark: mock cohorts and tiny CPU trainer fixtures only."""
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Event, Timer
import time

import pytest
import torch

from liquid_tcn import benchmark as b
from liquid_tcn.batch_logging import BatchLogger
from liquid_tcn.checkpoint import load_checkpoint
from liquid_tcn.ported.storage import atomic_json, digest
from liquid_tcn.sweep import generate_trials
from liquid_tcn.train import train
from helpers import preparation
from test_sweep import assert_nested_equal


def specs_for(folder, count=4, warmup=1, measured=2):
    folder.mkdir()
    specs = {}
    for i in range(count):
        worker_id = f'w{i}'
        worker_dir = folder/worker_id
        worker_dir.mkdir()
        specs[worker_id] = dict(worker_id=worker_id, worker_dir=str(worker_dir), barrier_dir=str(folder/'barriers'),
                               barrier_timeout=4, warmup_updates=warmup, measured_updates=measured)
    return specs


def mock_command(tmp_path, behavior='normal'):
    script = tmp_path/f'mock_{behavior}.py'
    script.write_text('''import json, os, pathlib, signal, sys, time
from threading import Event
from liquid_tcn.benchmark import BarrierClient, BARRIERS
from liquid_tcn.batch_logging import BatchLogger
from liquid_tcn.ported.storage import atomic_json
spec = json.loads(pathlib.Path(sys.argv[1]).read_text())
mode = sys.argv[2]
w = spec['worker_id']; folder = pathlib.Path(spec['worker_dir'])
stop = Event()
signal.signal(signal.SIGTERM, lambda *_: stop.set())
client = BarrierClient(spec['barrier_dir'], w, spec['barrier_timeout'], stop, BatchLogger(w, 0))
try:
    if mode == 'fail' and w == 'w0':
        raise RuntimeError('CUDA out of memory (mock)')
    if mode == 'hang' and w == 'w0':
        while not stop.is_set(): time.sleep(.01)
        raise InterruptedError('stopped missing worker')
    time.sleep(int(w[1:])*.03)
    client.wait('warmup_done')
    start = time.monotonic(); time.sleep(.03 + int(w[1:])*.01); end = time.monotonic()
    client.wait('measured_done')
    time.sleep(.02)  # mock checkpoint/setup outside the measured interval
    client.wait('validation_ready')
    vstart = time.monotonic(); time.sleep(.02); vend = time.monotonic()
    client.wait('validation_done')
    atomic_json(folder/'result.json', dict(status='complete', worker_id=w,
        training=dict(start_monotonic=start,end_monotonic=end,updates=spec['measured_updates']),
        validation=dict(start_monotonic=vstart,end_monotonic=vend,windows=10)))
except BaseException as error:
    atomic_json(folder/'stopped.json', dict(error=str(error)))
    if mode == 'fail' and w == 'w0':
        atomic_json(pathlib.Path(spec['barrier_dir'])/(w+'.failure.json'), dict(error=str(error)))
    sys.exit(1)
''')
    return lambda path: [sys.executable, str(script), str(path), behavior]


def test_config_uses_trial_zero_and_preserves_defaults(tmp_path, monkeypatch, capsys):
    configs = b.resolve_configs('/data', tmp_path, 'cuda:0', 2)
    baseline = generate_trials('/data', tmp_path, ['cuda:0'], 2)['00']
    assert [len(configs[p]) for p in ('single', 'four')] == [1, 4]
    for workers in configs.values():
        for config in workers.values():
            assert config['training'] == baseline['training']
            assert config['model'] == baseline['model']
            assert config['training']['updates'] == 20000
            assert config['training']['batch_size'] == 1024
            assert config['resume'] is None
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '7')
    root = tmp_path/'absent'
    assert b.main(['--data', '/missing', '--output', str(root), '--dry-run']) == 0
    dry = json.loads(capsys.readouterr().out)
    assert dry['max_updates'] == 500 and dry['validation']['split'] == 'validation'
    assert not root.exists() and os.environ['CUDA_VISIBLE_DEVICES'] == '7'
    assert b.main(['--data', '/missing', '--output', str(root), '--devices', '0,1', '--dry-run']) == 1
    assert b.main(['--data', '/missing', '--output', str(root), '--measured-updates', '20000', '--dry-run']) == 1


def test_timing_arithmetic_uses_shared_wall_interval():
    def worker(name, start, end, vstart, vend):
        return dict(worker_id=name, training=dict(start_monotonic=start, end_monotonic=end, seconds=end-start, updates=300),
                    validation=dict(start_monotonic=vstart, end_monotonic=vend, windows=1000))
    one = worker('single', 0, 10, 20, 25)
    four = [worker(str(i), 100+i, 120+i, 140+i, 150+i) for i in range(4)]
    result = b.comparison(dict(single=dict(workers=[one]), four=dict(workers=four)))
    assert result['single']['training']['per_second'] == 30
    assert result['four']['training']['per_second'] == pytest.approx(1200/23)
    assert result['aggregate_speedup'] == pytest.approx((1200/23)/30)
    assert result['four']['validation']['per_second'] == pytest.approx(4000/13)
    assert all(x == 2 for x in result['per_trial_slowdown'].values())
    assert result['four']['training']['per_second'] != sum(300/20 for _ in four)


def test_mock_barriers_and_concurrent_validation(tmp_path, capsys):
    folder = tmp_path/'phase'
    specs = specs_for(folder)
    runner = b.PhaseRunner(folder, specs, 1, 5, command_factory=mock_command(tmp_path), heartbeat_seconds=.05)
    result = runner.run()
    workers = result['workers']
    warmup_ready = max(json.loads((folder/'barriers'/f'{w}.warmup_done.ready.json').read_text())['monotonic'] for w in specs)
    assert min(w['training']['start_monotonic'] for w in workers) >= warmup_ready
    assert min(w['validation']['start_monotonic'] for w in workers) > max(w['training']['end_monotonic'] for w in workers)
    assert len(workers) == 4
    assert all(w['exit_code'] == 0 for w in runner.state['workers'].values())
    out = capsys.readouterr().out
    assert 'heartbeat' in out and '\r' not in out and '\x1b' not in out


@pytest.mark.parametrize('behavior', ['fail'])
def test_cohort_failure_timeout_stops_remaining_workers(tmp_path, behavior):
    folder = tmp_path/'phase'; specs = specs_for(folder)
    runner = b.PhaseRunner(folder, specs, 1, .8, shutdown_timeout=2,
                           command_factory=mock_command(tmp_path, behavior))
    with pytest.raises((RuntimeError, TimeoutError)):
        runner.run()
    assert runner.state['status'] == 'failed'
    assert (folder/'barriers/abort.json').exists()
    assert all(w['exit_code'] is not None for w in runner.state['workers'].values())
    assert not (folder/'barriers/validation_ready.release.json').exists()


def test_coordinator_interruption_drains_workers(tmp_path):
    folder = tmp_path/'phase'; specs = specs_for(folder)
    runner = b.PhaseRunner(folder, specs, 1, 10, shutdown_timeout=2,
                           command_factory=mock_command(tmp_path, 'hang'))
    timer = Timer(.7, runner.stop)
    timer.start()
    try:
        with pytest.raises(InterruptedError):
            runner.run()
    finally:
        timer.join()
    assert all(w['exit_code'] is not None for w in runner.state['workers'].values())


class ImmediateBarrier:
    def __init__(self, folder):
        self.folder = folder
        self.worker_id = 'cpu'
        self.calls, self.waits = [], {}

    def wait(self, name):
        self.calls.append((name, time.monotonic()))

    def cancelled(self):
        return False


def test_cpu_hooks_keep_ordinary_training_exact_and_validate_latest(tmp_path, capsys):
    data, manifest = preparation(tmp_path)
    baseline = generate_trials(data, tmp_path, ['cpu'], 1)['00']
    baseline['model'] = manifest['model']
    baseline['training'].update(batch_size=2, pretrain_updates=1, ramp_updates=2)
    cfg = baseline['training']
    capsys.readouterr()
    train(data, tmp_path/'ordinary', cfg, max_updates=4, logging_mode='plain')
    folder = tmp_path/'measured'; folder.mkdir()
    spec = dict(warmup_updates=2, measured_updates=2, launch_monotonic=time.monotonic(),
                worker_dir=str(folder), data_manifest_sha256=digest(data/'manifest.json'))
    barrier = ImmediateBarrier(folder)
    hooks = b.MeasurementHooks(spec, barrier)
    train(data, folder/'run', cfg, max_updates=4, logging_mode='plain', runtime_hooks=hooks)
    a = load_checkpoint(tmp_path/'ordinary/latest.pt')
    c = load_checkpoint(folder/'run/latest.pt')
    for field in ('model', 'optimizer', 'scheduler', 'sampler', 'config', 'rng'):
        assert_nested_equal(a[field], c[field])
    assert c['effective_updates'] == 20000
    assert not (folder/'run/best.pt').exists()
    report = hooks.report()
    assert report['updates'] == 2 and len(report['latency_seconds']) == 2
    assert sum(report['latency_seconds']) == pytest.approx(report['seconds'])
    assert report['seconds'] > 0 and report['p95_latency_seconds'] >= report['median_latency_seconds']
    assert [c['update'] for c in hooks.checkpoints] == [0, 4]
    assert [c[0] for c in barrier.calls] == ['warmup_done', 'measured_done']
    baseline['output'] = str(folder/'run')
    val = b.evaluate_latest(baseline, spec, barrier, BatchLogger('cpu', 4), hooks.memory)
    evaluated = json.loads(Path(val['report']).read_text())
    assert evaluated['coverage']['fraction'] == 1
    assert evaluated['coverage']['evaluated_windows'] == manifest['windows']['validation']
    assert evaluated['split'] == 'validation' and evaluated['checkpoint'].endswith('latest.pt')
    assert [c[0] for c in barrier.calls][-2:] == ['validation_ready', 'validation_done']
    assert hooks.memory['startup']['status'] == 'unavailable'
    output = capsys.readouterr()
    assert '\r' not in output.out+output.err and '\x1b' not in output.out+output.err


def test_hook_timeout_preserves_safe_checkpoint(tmp_path):
    data, manifest = preparation(tmp_path)
    cfg = generate_trials(data, tmp_path, ['cpu'], 1)['00']['training']
    cfg['batch_size'] = 2
    class BrokenBarrier(ImmediateBarrier):
        def wait(self, name):
            raise TimeoutError('missing peer')
    spec = dict(warmup_updates=1, measured_updates=2)
    hooks = b.MeasurementHooks(spec, BrokenBarrier(tmp_path))
    result = train(data, tmp_path/'run', cfg, max_updates=3, logging_mode='plain', runtime_hooks=hooks)
    assert result['interrupted']
    assert load_checkpoint(tmp_path/'run/latest.pt')['update'] == 1
    with pytest.raises(ValueError, match='missing peer'):
        hooks.report()


def test_barrier_ignores_legacy_timeout_and_is_abort_aware(tmp_path):
    client = b.BarrierClient(tmp_path, 'a', .03, Event(), BatchLogger('a', 0))
    release = Timer(.06, lambda: atomic_json(tmp_path/'warmup_done.release.json', {}))
    release.start()
    client.wait('warmup_done')
    release.join()
    client.training_stop_requested = Event()
    client.training_stop_requested.set()
    with pytest.raises(InterruptedError):
        client.wait('measured_done')
    client.training_stop_requested.clear()
    atomic_json(tmp_path/'abort.json', {})
    with pytest.raises(InterruptedError):
        client.wait('measured_done')


def test_launcher_quoted_paths_and_exec(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path/'not created'
    p = subprocess.run(['bash', str(root/'scripts/benchmark_tcn_concurrency.sh'), '--python', sys.executable,
        '--data', str(tmp_path/'data with spaces'), '--output', str(output), '--dry-run'],
        capture_output=True, text=True, check=True)
    assert json.loads(p.stdout)['max_updates'] == 500
    assert not output.exists()


def test_real_tiny_cpu_worker_end_to_end(tmp_path):
    data, manifest = preparation(tmp_path)
    folder = tmp_path/'phase'; specs = specs_for(folder, count=1, warmup=2, measured=2)
    spec = specs['w0']
    config = generate_trials(data, folder, ['cpu'], 1)['00']
    config['model'] = manifest['model']
    config['training'].update(batch_size=2, pretrain_updates=1, ramp_updates=2)
    worker_dir = Path(spec['worker_dir'])
    config['output'] = str(worker_dir/'run')
    atomic_json(worker_dir/'resolved.json', config)
    spec.update(config_sha256=digest(worker_dir/'resolved.json'), data_manifest_sha256=digest(data/'manifest.json'), barrier_timeout=30)
    result = b.PhaseRunner(folder, specs, 1, 30, shutdown_timeout=5).run()['workers'][0]
    assert result['training']['updates'] == 2
    assert result['validation']['windows'] == manifest['windows']['validation']
    assert result['validation']['start_monotonic'] > result['training']['end_monotonic']
    assert result['checkpoint_seconds'] > 0 and result['startup_seconds'] > 0
    assert set(result['memory']) == {'startup', 'warmup', 'measured', 'validation_setup', 'validation'}
    assert result['device']['logical_device'] == 'cpu'
    assert (worker_dir/'run/latest.pt').exists() and not (worker_dir/'run/best.pt').exists()


def test_main_runs_phases_in_order_and_rejects_reuse(tmp_path, monkeypatch):
    data = tmp_path/'data'; data.mkdir(); atomic_json(data/'manifest.json', {'artifacts': {}})
    calls = []
    def fake(self):
        calls.append(self.folder.name)
        workers = []
        for worker_id in self.specs:
            workers.append(dict(worker_id=worker_id, device={'logical_device': 'cpu'},
                training=dict(start_monotonic=1, end_monotonic=3, seconds=2, updates=300),
                validation=dict(start_monotonic=4, end_monotonic=5, windows=10),
                memory={'measured': b.memory_snapshot('cpu')}))
        return dict(workers=workers, elapsed_seconds=5)
    monkeypatch.setattr(b.PhaseRunner, 'run', fake)
    monkeypatch.setattr(os, 'sched_getaffinity', lambda _: set(range(8)))
    monkeypatch.delenv('SLURM_CPUS_PER_TASK', raising=False)
    output = tmp_path/'benchmark'
    args = ['--data', str(data), '--output', str(output), '--devices', 'cpu']
    assert b.main(args) == 0
    assert calls == ['single', 'four']
    summary = json.loads((output/'summary.json').read_text())
    assert summary['status'] == 'complete' and summary['comparison']['aggregate_speedup'] == 4
    assert (output/'comparison.md').exists()
    assert b.main(args) == 1
    assert calls == ['single', 'four']
    monkeypatch.setenv('SLURM_CPUS_PER_TASK', '7')
    assert b.main(['--data', str(data), '--output', str(tmp_path/'small'), '--devices', 'cpu']) == 0
    assert (tmp_path/'small/summary.json').exists()


def test_main_failure_never_starts_second_phase(tmp_path, monkeypatch):
    data = tmp_path/'data'; data.mkdir(); atomic_json(data/'manifest.json', {'artifacts': {}})
    calls = []
    def fail(self):
        calls.append(self.folder.name)
        raise RuntimeError('mock worker failed')
    monkeypatch.setattr(b.PhaseRunner, 'run', fail)
    monkeypatch.setattr(os, 'sched_getaffinity', lambda _: set(range(8)))
    monkeypatch.delenv('SLURM_CPUS_PER_TASK', raising=False)
    output = tmp_path/'out'
    assert b.main(['--data', str(data), '--output', str(output), '--devices', 'cpu']) == 1
    assert calls == ['single']
    summary = json.loads((output/'summary.json').read_text())
    assert summary['status'] == 'failed' and 'mock worker failed' in summary['error']


def test_measurement_excludes_barrier_time_and_retains_startup_peaks(tmp_path, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(b.time, 'monotonic', lambda: clock[0])
    calls = []
    monkeypatch.setattr(b, 'synchronize', lambda device: calls.append('sync'))
    def snapshot(device):
        calls.append('snapshot')
        return dict(peak_allocated_bytes=100 if calls.count('snapshot') == 1 else 10)
    monkeypatch.setattr(b, 'memory_snapshot', snapshot)
    monkeypatch.setattr(b, 'reset_peaks', lambda device: calls.append('reset'))
    class SlowBarrier(ImmediateBarrier):
        def wait(self, name):
            clock[0] += 10
    class Writer:
        def flush(self):
            pass
    hooks = b.MeasurementHooks(dict(warmup_updates=1, measured_updates=2), SlowBarrier(tmp_path))
    hooks.bind('cpu', Event(), BatchLogger('test', 3))
    hooks.ready(0)
    clock[0] = 1
    hooks.update_end(1, .1, Writer())  # ten-second barrier follows warmup
    assert clock[0] == 11
    clock[0] = 11.2
    hooks.update_end(2, .1, Writer())
    clock[0] = 11.5
    hooks.update_end(3, .1, Writer())  # another ten-second barrier follows measurement
    result = hooks.report()
    assert clock[0] == 21.5
    assert result['seconds'] == .5 and result['updates_per_second'] == 4
    assert result['median_latency_seconds'] == pytest.approx(.25)
    assert result['p95_latency_seconds'] == pytest.approx(.295)
    assert hooks.memory['startup']['peak_allocated_bytes'] == 100
    assert hooks.memory['measured']['peak_allocated_bytes'] == 10
    assert calls.index('snapshot') < calls.index('reset')
