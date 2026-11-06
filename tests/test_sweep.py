"""Sweep orchestration uses mock subprocesses; only tiny fixtures train on CPU."""
import json
import os
from pathlib import Path
import signal
import sys

import numpy as np
import pytest
import torch

from liquid_tcn import sweep
from liquid_tcn.checkpoint import load_checkpoint
from liquid_tcn.ported.storage import atomic_json
from helpers import preparation


def setup_sweep(tmp_path):
    data = tmp_path/'data'
    data.mkdir()
    atomic_json(data/'manifest.json', {'fixture': True})
    root = tmp_path/'sweep'
    root.mkdir()
    manifest, state = sweep.initialize(root, data, ['cpu'], 1, False)
    return root, data, manifest, state


def test_independent_trials_and_baseline():
    baseline = sweep.read_json(sweep.BASELINE)
    trials = sweep.generate_trials('/data', '/output', ['cuda:0', 'cuda:1'], 2)
    assert len(trials) == 16
    common = dict(validation_every=1000, patience=5, checkpoint_every=500,
                  validation_batches=None, num_workers=0, validation_num_workers=0,
                  threads=2, progress=False)
    for i, (trial_id, trial) in enumerate(trials.items()):
        expected = dict(baseline['training'], **sweep.OVERRIDES[i])
        expected.update(common, device=f'cuda:{i % 2}')
        assert trial['training'] == expected
        assert trial['model'] == baseline['model']
        assert trial['training']['updates'] == 20000
        assert trial['max_updates'] is None
        assert trial['training']['batch_size'] == 1024
    trials['01']['model']['dilations'][0] = 99
    assert trials['00']['model'] == baseline['model']
    assert sweep.BASELINE_COMMIT.startswith('5c7792d')


def test_immutable_resume_and_unrelated_directory(tmp_path):
    root, data, manifest, state = setup_sweep(tmp_path)
    assert sweep.initialize(root, data, ['cpu'], 1, True) == (manifest, state)
    with pytest.raises(ValueError, match='already exists'):
        sweep.initialize(root, data, ['cpu'], 1, False)
    with pytest.raises(ValueError, match='identical'):
        sweep.initialize(root, data, ['cpu'], 2, True)
    config = root/'trials/00/resolved.json'
    config.write_text('{}')
    with pytest.raises(ValueError, match='Immutable'):
        sweep.initialize(root, data, ['cpu'], 1, True)
    unrelated = tmp_path/'unrelated'; unrelated.mkdir()
    (unrelated/'important').write_text('keep')
    with pytest.raises(ValueError, match='unrelated'):
        sweep.initialize(unrelated, data, ['cpu'], 1, False)
    assert (unrelated/'important').read_text() == 'keep'


def mock_factory(root, fail=None):
    script = root/'mock_worker.py'
    script.write_text('''import json, os, pathlib, sys, time
root, trial, stage, resume, fail = sys.argv[1:]
with open(pathlib.Path(root)/'events.jsonl', 'a') as f:
    f.write(json.dumps(dict(trial=trial, event='start', time=time.monotonic(), stage=stage, resume=resume))+'\\n')
time.sleep(.25)
run = pathlib.Path(root)/'trials'/trial/'run'
run.mkdir(exist_ok=True)
(run/'latest.pt').write_text('fixture')
with open(pathlib.Path(root)/'events.jsonl', 'a') as f:
    f.write(json.dumps(dict(trial=trial, event='end', time=time.monotonic()))+'\\n')
print('CUDA out of memory' if trial == fail else 'mock worker complete', flush=True)
sys.exit(7 if trial == fail else 0)
''')
    return lambda r, t, stage, resume: [sys.executable, str(script), str(r), t, stage, str(resume), fail or '-']


def fake_result(root, trial, stage, manifest):
    return dict(normalized_mse=float(15-int(trial)), update=sweep.BUDGETS[stage])


def test_process_limit_failure_retry_and_selection(tmp_path, capsys):
    root, _, manifest, state = setup_sweep(tmp_path)
    coordinator = sweep.Coordinator(root, manifest, state, 4, mock_factory(root, '03'), fake_result, .1)
    assert not coordinator.run('screen', manifest['trial_ids'])
    assert state['trials']['03']['stages']['screen']['exit_code'] == 7
    assert state['trials']['03']['attempts'][0]['failure_kind'] == 'out_of_memory'
    with pytest.raises(ValueError, match='no promotion'):
        sweep.select_trials(state)
    events = [json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
    concurrent = maximum = 0
    for entry in sorted(events, key=lambda e: e['time']):
        concurrent += 1 if entry['event'] == 'start' else -1
        maximum = max(maximum, concurrent)
    assert maximum == 4 and concurrent == 0
    assert len([e for e in events if e['event'] == 'start']) == 16
    coordinator.command_factory = mock_factory(root)
    assert coordinator.run('screen', manifest['trial_ids'])
    assert len(state['trials']['03']['attempts']) == 2
    assert all(len(r['attempts']) == 1 for t, r in state['trials'].items() if t != '03')
    assert sweep.select_trials(state) == ['00', '15', '14', '13']
    assert coordinator.run('continue', sweep.select_trials(state))
    events = [json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
    continued = [e for e in events if e.get('stage') == 'continue']
    assert {e['trial'] for e in continued} == {'00', '15', '14', '13'}
    assert all(e['resume'] == 'True' for e in continued)
    count = len(events)
    assert coordinator.run('continue', sweep.select_trials(state))
    assert len((root/'events.jsonl').read_text().splitlines()) == count
    captured = capsys.readouterr().out
    assert 'heartbeat' in captured and '\r' not in captured and '\x1b' not in captured
    with pytest.raises(ValueError):
        sweep.Coordinator(root, manifest, state, 5)


@pytest.mark.parametrize('stage,expected', [('screen', ['screen']), ('auto', ['screen', 'continue'])])
def test_main_stage_modes(tmp_path, monkeypatch, stage, expected):
    data = tmp_path/'data'; data.mkdir(); atomic_json(data/'manifest.json', {})
    calls = []
    def run(self, current, ids):
        calls.append(current)
        for trial in ids:
            self.state['trials'][trial]['stages'][current] = dict(status='complete', result=fake_result(None, trial, current, None))
        self.persist()
        return True
    monkeypatch.setattr(sweep.Coordinator, 'run', run)
    args = ['--data', str(data), '--output', str(tmp_path/'sweep'), '--devices', 'cpu']
    assert sweep.main(args + ['--stage', stage]) == 0
    assert calls == expected
    calls.clear()
    assert sweep.main(args + ['--stage', 'report']) == 0
    assert calls == []
    assert (tmp_path/'sweep/report.json').exists()


def test_no_promotion_after_failed_auto(tmp_path, monkeypatch):
    data = tmp_path/'data'; data.mkdir(); atomic_json(data/'manifest.json', {})
    calls = []
    def fail(self, stage, ids):
        calls.append(stage)
        return False
    monkeypatch.setattr(sweep.Coordinator, 'run', fail)
    assert sweep.main(['--data', str(data), '--output', str(tmp_path/'sweep'), '--devices', 'cpu', '--stage', 'auto']) == 1
    assert calls == ['screen']
    assert not (tmp_path/'sweep/selection.json').exists()


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_nested_equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    else:
        assert a == b


def test_plain_logs_signal_checkpoint_exact_continuation(tmp_path, monkeypatch, capsys):
    import importlib
    training = importlib.import_module('liquid_tcn.train')
    path, _ = preparation(tmp_path)
    cfg = dict(updates=4, pretrain_updates=1, ramp_updates=1, batch_size=2, threads=1,
               validation_every=2, checkpoint_every=2, validation_batches=1, patience=10)
    capsys.readouterr()
    training.train(path, tmp_path/'full', cfg, logging_mode='plain', trial_id='tiny')
    real = training.objective
    signalled = []
    def signal_during_update(*args, **kwargs):
        result = real(*args, **kwargs)
        if not signalled:
            signalled.append(True)
            os.kill(os.getpid(), signal.SIGTERM)
        return result
    # Avoid signaling during the precision probe; this test targets an actual update.
    check = training.precision_check
    def probe(*args):
        result = check(*args)
        monkeypatch.setattr(training, 'objective', signal_during_update)
        return result
    monkeypatch.setattr(training, 'precision_check', probe)
    result = training.train(path, tmp_path/'resumed', cfg, max_updates=2, logging_mode='plain')
    assert result['interrupted'] and result['final_update'] == 1
    interrupted = load_checkpoint(tmp_path/'resumed/latest.pt')
    assert interrupted['update'] == 1 and interrupted['effective_updates'] == 4
    monkeypatch.setattr(training, 'precision_check', check)
    monkeypatch.setattr(training, 'objective', real)
    # Runtime logging options are compatible with a checkpoint that has no such fields.
    training.train(path, tmp_path/'resumed', cfg, resume=tmp_path/'resumed/latest.pt', logging_mode='auto')
    a, b = [load_checkpoint(tmp_path/p/'latest.pt') for p in ('full', 'resumed')]
    for key in ('model', 'optimizer', 'scheduler', 'sampler', 'best', 'stale', 'rng'):
        assert_nested_equal(a[key], b[key])
    records = [json.loads(line) for line in (tmp_path/'resumed/metrics.jsonl').read_text().splitlines()]
    assert [r['update'] for r in records] == [1, 2, 3, 4]
    out = capsys.readouterr()
    assert '\r' not in out.out + out.err and '\x1b' not in out.out + out.err
    events = [json.loads(line) for line in out.out.splitlines()]
    assert {'startup', 'phase', 'validation_start', 'validation_end', 'checkpoint', 'completion', 'interrupted'} <= {e['event'] for e in events}
    assert not any(e['event'] == 'updates' for e in events)  # No per-update batch output.
    assert all({'trial_id', 'budget', 'update', 'supervised', 'short', 'long', 'learning_rate', 'validation_metric', 'updates_per_second'} <= e.keys() for e in events)


def test_checkpoint_result_full_coverage_and_archives(tmp_path, monkeypatch):
    from liquid_tcn.train import train
    path, manifest = preparation(tmp_path)
    root = tmp_path/'sweep'; (root/'trials/00').mkdir(parents=True)
    cfg = sweep.generate_trials(path, root, ['cpu'], 1)['00']
    cfg['model'] = manifest['model']
    cfg['training'].update(updates=4, batch_size=2, pretrain_updates=1, ramp_updates=1,
                           validation_every=2, checkpoint_every=2)
    atomic_json(root/'trials/00/resolved.json', cfg)
    monkeypatch.setitem(sweep.BUDGETS, 'screen', 2)
    provenance = dict(data_manifest_sha256=sweep.digest(path/'manifest.json'), metric='normalized_supervised_mse_all_heads')
    train(path, cfg['output'], cfg['training'], max_updates=2, logging_mode='plain')
    report = sweep.inspect_result(root, '00', 'screen', provenance)
    assert report['coverage']['fraction'] == 1
    assert Path(report['best_checkpoint']).exists()
    assert Path(report['latest_checkpoint']).parent.name == 'screen'
    saved = load_checkpoint(Path(cfg['output'])/'latest.pt')
    saved['update'] = 1
    torch.save(saved, Path(cfg['output'])/'latest.pt')
    with pytest.raises(ValueError, match='Incomplete'):
        sweep.inspect_result(root, '00', 'screen', provenance)


def test_devices_and_dry_run_no_files(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '7')
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 1)
    assert sweep.validate_devices('0') == ['cuda:0']
    with pytest.raises(ValueError, match='only 1'):
        sweep.validate_devices('1')
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '7'
    root = tmp_path/'new'
    assert sweep.main(['--data', str(tmp_path/'absent'), '--output', str(root), '--dry-run']) == 0
    assert not root.exists()
    assert len(json.loads(capsys.readouterr().out)['trials']) == 16


def test_coordinator_stop_drains_workers_without_dispatching_more(tmp_path):
    from threading import Timer
    root, _, manifest, state = setup_sweep(tmp_path)
    script = root/'wait_worker.py'
    script.write_text('''import pathlib, signal, sys, time
stopping = False
def stop(*_):
    global stopping
    stopping = True
signal.signal(signal.SIGTERM, stop)
while not stopping:
    time.sleep(.02)
pathlib.Path(sys.argv[1]).write_text('safe boundary reached')
sys.exit(143)
''')
    def command(root, trial, stage, resume):
        return [sys.executable, str(script), str(root/f'{trial}.stopped')]
    coordinator = sweep.Coordinator(root, manifest, state, 4, command, fake_result)
    timer = Timer(.6, coordinator.stop)
    timer.start()
    try:
        assert not coordinator.run('screen', manifest['trial_ids'])
    finally:
        timer.join()
    attempted = [t for t, r in state['trials'].items() if r['attempts']]
    assert len(attempted) == 4
    assert all((root/f'{t}.stopped').exists() for t in attempted)
    assert all(state['trials'][t]['stages']['screen']['status'] == 'interrupted' for t in attempted)


def test_worker_command_preserves_absolute_budget():
    command = sweep.worker_command(Path('/sweep'), '00', 'screen', False)
    assert command[command.index('--max-updates')+1] == '5000'
    assert '--resume' not in command
    command = sweep.worker_command(Path('/sweep'), '00', 'continue', True)
    assert command[command.index('--max-updates')+1] == '20000'
    assert command[command.index('--resume')+1] == '/sweep/trials/00/run/latest.pt'
    assert command[command.index('--logging-mode')+1] == 'plain'


def test_plain_failure_and_auto_mode(tmp_path, capsys, monkeypatch):
    from liquid_tcn.train import train
    from liquid_tcn.batch_logging import resolve_mode
    monkeypatch.setattr(sys.stderr, 'isatty', lambda: True)
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    assert resolve_mode() == 'progress'
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    assert resolve_mode() == 'plain'
    assert resolve_mode('progress', False) == 'progress'
    with pytest.raises(ValueError, match='Unknown'):
        train(tmp_path/'missing', tmp_path/'run', {'typo': 1}, logging_mode='plain')
    out = capsys.readouterr()
    assert '\r' not in out.out and '\x1b' not in out.out
    assert json.loads(out.out.splitlines()[-1])['event'] == 'failure'
