"""Supervised pretraining, discrete-physics fine-tuning, logging, and resumable checkpoints."""
import copy
import fcntl
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import resource
import signal
from threading import Event
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from .batch_logging import BatchLogger, resolve_mode
from .progress import TrainingProgress
from .model import LiquidTCN, ModelConfig
from .normalization import Normalization
from .dataset import WindowDataset, EpochSampler, make_loader, to_device, close_loader
from .objectives import objective, loss_components
from .gradients import backward_objectives
from .evaluate import evaluate
from .checkpoint import atomic_save, load_checkpoint, rng_state, restore_rng
from .ported.storage import atomic_json, digest

DEFAULTS = dict(seed=1, batch_size=16, num_workers=0, prefetch_factor=2, pin_memory=True,
    cache_size=8, updates=20000, epochs=None, pretrain_updates=5000, ramp_updates=5000,
    pretrain_lr=1e-3, physics_lr=1e-4, weight_decay=1e-4, gradient_clip=1.,
    lambda_short=1., lambda_long=1., supervised='mse', huber_delta=1.,
    physics_dtype='float64', scheduler='cosine', min_lr_ratio=.1,
    validation_every=250, checkpoint_every=250, patience=20, min_delta=0.,
    validation_batches=None, validation_batch_size=None, validation_num_workers=None, amp=False, amp_rtol=.05, precision_rtol=.005,
    device='cpu', threads=4, progress=True, gradient_strategy='pcgrad', pcgrad_eps=1e-12)


def weights(config, update):
    # Supervision always stays active. After pretraining, linearly ramp both
    # physics coefficients from zero to their configured maxima by update.
    ramp = min(1., max(0., (update - config['pretrain_updates'])/max(1, config['ramp_updates'])))
    return config['lambda_short']*ramp, config['lambda_long']*ramp


def precision_check(model, batch, norm, config):
    """Compare network+physics losses and full parameter gradients; gate optional AMP."""
    # Probe copies leave trained parameters untouched; restoring RNG makes
    # this diagnostic neutral to the reproducible training/resume sequence.
    # Compare full parameter gradients too: similar losses can hide bad grads.
    state = rng_state()
    outputs = {}
    modes = ['float64', 'float32'] + (['amp'] if config['amp'] else [])
    try:
        for mode in modes:
            candidate = copy.deepcopy(model).train()
            x, p = norm.normalize('motion', batch['motion']), norm.normalize('properties', batch['properties'])
            if mode == 'float64':
                candidate = candidate.double(); x = x.double(); p = p.double()
            with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=mode == 'amp'):
                prediction = candidate(x, p)
            result = objective(prediction if mode == 'float64' else prediction.float(), batch, norm,
                               lambda_short=config['lambda_short'], lambda_long=config['lambda_long'],
                               supervised=config['supervised'], huber_delta=config['huber_delta'], physics_dtype='float64')
            backward_objectives(result['supervised'],
                result['secondary'] if config['lambda_short'] or config['lambda_long'] else None,
                candidate.parameters(), config['gradient_strategy'], config['pcgrad_eps'])
            gradient = torch.cat([q.grad.detach().flatten().double() for q in candidate.parameters()])
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(f'Nonfinite {mode} gradient')
            outputs[mode] = (float(result['loss'].detach()), gradient)
        reference, grad = outputs['float64']
        report = {}
        for mode in modes[1:]:
            loss, g = outputs[mode]
            loss_error = abs(loss-reference)/max(abs(reference), 1e-12)
            gradient_error = float((g-grad).norm()/grad.norm().clamp_min(1e-12))
            tolerance = config['amp_rtol'] if mode == 'amp' else config['precision_rtol']
            report[mode] = dict(relative_loss_error=loss_error, relative_gradient_error=gradient_error, tolerance=tolerance)
            if max(loss_error, gradient_error) > tolerance:
                raise ValueError(f'{mode} failed float64 precision gate: {report[mode]}')
        return report
    finally:
        restore_rng(state)


def train(data, output, configuration=None, resume=None, max_updates=None,
          expected_model=None, launch_config=None, logging_mode='auto', trial_id='standalone',
          runtime_hooks=None):
    """Logging is a runtime option, deliberately outside checkpoint configuration.

    Optional runtime_hooks observe ready/update/checkpoint boundaries; they are
    never serialized and may defer scheduled maintenance for timing experiments.
    Ordinary training uses no hooks.

    Older checkpoints therefore need no migration: omitted logging_mode means
    auto and omitted trial_id means standalone. SIGTERM/SIGINT only set a flag;
    serialization happens after a complete optimizer/scheduler/sampler update.
    """
    config = dict(DEFAULTS, **(configuration or {}))
    log = BatchLogger(trial_id, max_updates or config['updates'])
    stop_requested = Event()
    received = []
    previous = {}
    run_lock = None
    def request_stop(number, frame):
        received.append(number)
        stop_requested.set()
    try:
        Path(output).mkdir(parents=True, exist_ok=True)
        run_lock = open(Path(output)/'.training.lock', 'a')
        try:
            fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another worker owns this training directory')
        for number in (signal.SIGTERM, signal.SIGINT):
            previous[number] = signal.signal(number, request_stop)
        log.emit('startup', output=str(output), resume=str(resume) if resume else None,
                 scheduler_horizon=config['updates'])
        return _train(data, output, configuration, resume, max_updates, expected_model,
                      launch_config, logging_mode, log, stop_requested, received, runtime_hooks)
    except BaseException as error:
        log.emit('failure', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
        if run_lock is not None:
            run_lock.close()


def _train(data, output, configuration, resume, max_updates, expected_model,
           launch_config, logging_mode, log, stop_requested, received, runtime_hooks):
    mode = resolve_mode(logging_mode, (configuration or {}).get('progress', True))
    unknown = set(configuration or {}) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown training configuration fields: {sorted(unknown)}')
    config = dict(DEFAULTS, **(configuration or {}))
    if config['updates'] < 1 or config['validation_every'] < 1 or config['checkpoint_every'] < 1:
        raise ValueError('Positive update/checkpoint/validation counts required')
    if config['gradient_strategy'] not in ('pcgrad', 'sum'):
        raise ValueError('gradient_strategy must be pcgrad or sum')
    if not math.isfinite(config['pcgrad_eps']) or config['pcgrad_eps'] <= 0:
        raise ValueError('pcgrad_eps must be finite and positive')
    for key, minimum in (('validation_batch_size', 1), ('validation_num_workers', 0)):
        value = config[key]
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < minimum):
            raise ValueError(f'{key} must be null or an integer >= {minimum}')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(config['threads'])
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    torch.use_deterministic_algorithms(True)
    device = torch.device(config['device'])
    if runtime_hooks is not None:
        runtime_hooks.bind(device, stop_requested, log)
    defer_maintenance = runtime_hooks is not None and runtime_hooks.defer_maintenance
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ('latest.pt', 'config.json')) and resume is None:
        raise ValueError('Run exists; supply --resume or use a fresh output')
    training = WindowDataset(data, 'train', config['cache_size'], verify=True)
    validation = WindowDataset(data, 'validation', config['cache_size'])
    if not len(training) or not len(validation):
        raise ValueError('Training and validation must both have eligible windows in separate collection groups')
    if expected_model is not None and ModelConfig(**expected_model) != ModelConfig(**training.manifest['model']):
        raise ValueError('Model configuration differs from prepared window geometry; prepare a matching dataset')
    if config['epochs'] is not None and (not isinstance(config['epochs'], int) or config['epochs'] < 1):
        raise ValueError('epochs must be null or a positive integer')
    # Epoch mode includes the last partial batch. Resolve it once so the
    # learning-rate schedule and checkpoints share the same update horizon.
    total_updates = (config['epochs'] * math.ceil(len(training) / config['batch_size'])
                     if config['epochs'] is not None else config['updates'])
    norm = Normalization(training.manifest['normalization'])
    model = LiquidTCN(ModelConfig(**training.manifest['model'])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['pretrain_lr'], weight_decay=config['weight_decay'])
    # LambdaLR expects a multiplier of the optimizer's base pretrain_lr.
    # At the phase boundary switch to physics_lr, then apply the chosen decay.
    # This LR schedule is independent of the gradual physics-weight ramp.
    def learning_rate(step):
        if step < config['pretrain_updates']:
            return 1.
        if config['scheduler'] not in ('cosine', 'constant'):
            raise ValueError('Scheduler must be cosine or constant')
        fraction = min(1., (step-config['pretrain_updates'])/max(1, total_updates-config['pretrain_updates']))
        decay = 1. if config['scheduler'] == 'constant' else config['min_lr_ratio'] + (1-config['min_lr_ratio'])*.5*(1+math.cos(math.pi*fraction))
        return config['physics_lr']/config['pretrain_lr'] * decay
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate)
    sampler = EpochSampler(len(training), config['seed'])
    provenance = digest(Path(data)/'manifest.json')
    update, best, stale = 0, float('inf'), 0
    # Exact continuation requires more than model weights: optimizer moments,
    # scheduler phase, consumed sampler position, and RNG states must agree.
    # Reject changed data/configuration rather than silently changing the run.
    if resume:
        saved = load_checkpoint(resume, device)
        # Checkpoints predating PCGrad used ordinary summed gradients.
        saved_config = dict(saved['config'])
        saved_config.setdefault('gradient_strategy', 'sum')
        saved_config.setdefault('pcgrad_eps', 1e-12)
        saved_config.setdefault('validation_batch_size', None)
        saved_config.setdefault('validation_num_workers', None)
        if saved['data_manifest_sha256'] != provenance or saved_config != config:
            raise ValueError('Resume requires identical data provenance and configuration')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']); sampler.load_state_dict(saved['sampler'])
        update, best, stale = saved['update'], saved['best'], saved['stale']
        restore_rng(saved['rng'])
    options = {k: config[k] for k in ('batch_size', 'num_workers', 'prefetch_factor', 'pin_memory', 'seed')}
    loader = make_loader(training, sampler=sampler, **options)
    validation_options = dict(options)
    for key in ('batch_size', 'num_workers'):
        if config['validation_' + key] is not None:
            validation_options[key] = config['validation_' + key]
    val_loader = make_loader(validation, **validation_options)
    probe = to_device(next(iter(make_loader(training, batch_size=min(2, config['batch_size']), pin_memory=False))), device)
    checks = precision_check(model, probe, norm, config)
    atomic_json(output / 'precision.json', checks)
    stop = min(total_updates, max_updates) if max_updates is not None else total_updates
    timing = dict(loader_seconds=0., training_seconds=0., validation_seconds=0., samples=0, updates=0, clipped=0)
    start = time.perf_counter()
    last_validation = saved.get('validation') if resume else None
    log.initial = update
    log.budget = stop
    log.state.update(update=update, validation_metric=last_validation['selection_metric'] if last_validation else None)
    atomic_json(output / 'config.json', dict(training=config, effective_updates=total_updates, model=asdict(model.config),
        data_manifest_sha256=provenance, normalization=norm.stats,
        preparation_config=training.manifest['config']))
    atomic_json(output / 'requested_config.json', configuration or {})
    if launch_config is not None:
        atomic_json(output / 'launch_config.json', launch_config)
    if resume and (output/'metrics.jsonl').exists():
        with open(output/'metrics.jsonl') as source, open(output/'metrics.jsonl.tmp', 'w') as target:
            for line in source:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue  # A killed write may leave one partial final line.
                if entry['update'] <= update:
                    target.write(line)
        os.replace(output/'metrics.jsonl.tmp', output/'metrics.jsonl')
    writer = SummaryWriter(log_dir=str(output / 'tensorboard'), purge_step=update + 1 if resume else None)
    writer.add_text('run/configuration', json.dumps(config, indent=2), update)

    # Atomic replacement prevents partially written checkpoints. TensorBoard
    # is flushed alongside model/optimizer state; purge_step above hides stale
    # log events beyond the restored checkpoint when resuming an interrupted run.
    def save(name):
        checkpoint_start = time.monotonic() if runtime_hooks is not None else None
        writer.flush()
        atomic_save(output/name, dict(format_version=1, model=model.state_dict(), model_config=asdict(model.config),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            phase='supervised' if update < config['pretrain_updates'] else 'physics', update=update, effective_updates=total_updates,
            rng=rng_state(), sampler=sampler.state_dict(), best=best, stale=stale,
            normalization=norm.stats, config=config, data_manifest_sha256=provenance,
            data_provenance=dict(sources=training.manifest['sources'], windows=training.manifest['windows'],
                                 preparation_config=training.manifest['config']), validation=last_validation))
        log.emit('checkpoint', update=update, checkpoint=str(output/name))
        if runtime_hooks is not None:
            runtime_hooks.checkpoint(update, name, time.monotonic()-checkpoint_start)

    progress = TrainingProgress(total=stop, initial=update, disable=mode != 'progress')
    previous_phase = None
    try:
        if not resume:
            save('latest.pt')
        if runtime_hooks is not None:
            runtime_hooks.ready(update)
        while update < stop and stale < config['patience'] and not stop_requested.is_set():
            if sampler.position == sampler.length:
                sampler.next_epoch()
            sampler.limit = (stop - update) * config['batch_size']
            tick = time.perf_counter()
            for batch in loader:
                timing['loader_seconds'] += time.perf_counter()-tick
                train_start = time.perf_counter()
                batch = to_device(batch, device)
                # Validation switches to eval mode, so restore training mode each step.
                # Clear previous gradients; each backward here represents one batch update.
                model.train(); optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=config['amp']):
                    pred = model(norm.normalize('motion', batch['motion']), norm.normalize('properties', batch['properties']))
                short, long = weights(config, update)
                phase = 'supervised' if short == long == 0 else 'physics'
                if phase != previous_phase:
                    log.emit('phase', update=update, phase=phase)
                    previous_phase = phase
                result = objective(pred.float(), batch, norm, short, long, config['supervised'],
                                   config['huber_delta'], config['physics_dtype'])
                # Keep supervision primary; project the SUM of weighted physics
                # gradients only on conflict. Clip the merged gradient before AdamW.
                projection = backward_objectives(result['supervised'],
                    result['secondary'] if short or long else None, model.parameters(),
                    config['gradient_strategy'], config['pcgrad_eps'])
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
                used_lr = optimizer.param_groups[0]['lr']
                optimizer.step(); scheduler.step()
                # Acknowledge only optimizer-consumed samples. DataLoader may prefetch
                # future indices; saving its iterator progress would skip data on resume.
                sampler.advance(len(batch['motion'])); update += 1
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                compute_seconds = time.perf_counter()-train_start
                timing['training_seconds'] += compute_seconds
                timing['samples'] += len(batch['motion']); timing['updates'] += 1
                timing['clipped'] += int(grad > config['gradient_clip'])
                record = dict(update=update, loss=float(result['loss'].detach()), supervised=float(result['supervised'].detach()),
                    short=float(result['physics']['short']['loss'].detach()), long=float(result['physics']['long']['loss'].detach()),
                    short_count=int(result['physics']['short']['count']), long_count=int(result['physics']['long']['count']),
                    supervised_counts=result['supervised_counts'].tolist(), lambda_short=short, lambda_long=long,
                    gradient_norm=float(grad), learning_rate=used_lr,
                    secondary=float(result['secondary'].detach()), projection=projection,
                    loss_components=loss_components(result, short, long))
                log.state.update({key: record[key] for key in log.state if key in record})
                if not defer_maintenance and (update % config['validation_every'] == 0 or update == total_updates):
                    log.emit('validation_start', update=update)
                    vstart = time.perf_counter()
                    last_validation = evaluate(model, val_loader, norm, device, config['validation_batches'],
                        supervised=config['supervised'], huber_delta=config['huber_delta'],
                        lambda_short=short, lambda_long=long, physics_dtype=config['physics_dtype'])
                    timing['validation_seconds'] += time.perf_counter()-vstart
                    # Select checkpoints using fixed normalized supervised MSE, independent
                    # of the changing physics weights (and the training MSE/Huber choice).
                    # Patience counts validation events, not individual optimizer updates.
                    metric = last_validation['selection_metric']
                    if metric is None or not math.isfinite(metric):
                        raise ValueError('Empty or nonfinite validation metric')
                    record['validation_metric'] = metric
                    log.emit('validation_end', update=update, validation_metric=metric, seconds=time.perf_counter()-vstart,
                             **{'validation_' + k: (last_validation['losses'] or {}).get(k)
                                for k in ('loss', 'supervised', 'short', 'long')})
                    if metric < best - config['min_delta']:
                        best, stale = metric, 0
                        save('best.pt')
                        atomic_json(output/'best_validation.json', last_validation)
                    else:
                        stale += 1
                    atomic_json(output/'validation.json', last_validation)
                    writer.add_scalar('validation/selection_metric', metric, update)
                    validation_losses = last_validation['losses']
                    for key in ('loss', 'supervised', 'short', 'long', 'secondary', 'lambda_short', 'lambda_long'):
                        writer.add_scalar('validation/' + key, validation_losses[key], update)
                    for key, value in validation_losses['components'].items():
                        writer.add_scalar('validation/' + key, value, update)
                    for head, values in last_validation['heads'].items():
                        for axis, value in zip('xyz', values['rmse_axis']):
                            writer.add_scalar(f'validation/{head}/rmse_{axis}', value, update)
                        writer.add_scalar(f'validation/{head}/maximum_vector_error', values['maximum_vector_error'], update)
                    for horizon, values in last_validation['conservation'].items():
                        for channel, value in zip(('r', 'P', 'H'), values['normalized_rms_vector']):
                            writer.add_scalar(f'validation/{horizon}/normalized_rms_{channel}', value, update)
                for key in ('loss', 'supervised', 'short', 'long', 'lambda_short', 'lambda_long',
                            'gradient_norm', 'learning_rate', 'short_count', 'long_count'):
                    writer.add_scalar('train/' + key, record[key], update)
                writer.add_scalar('train/secondary', record['secondary'], update)
                for key, value in record['loss_components'].items():
                    writer.add_scalar('train/' + key, value, update)
                for key, value in projection.items():
                    writer.add_scalar('train/pcgrad/' + key, value, update)
                for head, count in zip(('force', 'moment', 'r_B', 'v_B', 'S_B'), record['supervised_counts']):
                    writer.add_scalar('train/valid_' + head, count, update)
                writer.add_scalar('train/gradient_clipped', int(grad > config['gradient_clip']), update)
                with open(output/'metrics.jsonl', 'a') as stream:
                    stream.write(json.dumps(record, allow_nan=False)+'\n')
                progress.update(1)
                progress.set_postfix(phase='supervised' if short == long == 0 else 'physics',
                    loss=f"{record['loss']:.4g}", sup=f"{record['supervised']:.4g}",
                    short=f"{record['short']:.3g}", long=f"{record['long']:.3g}",
                    lr=f"{record['learning_rate']:.2g}",
                    val='pending' if last_validation is None else f"{last_validation['selection_metric']:.4g}", refresh=False)
                if update % 250 == 0:
                    log.emit('updates', update=update)
                if runtime_hooks is not None:
                    # End-to-end timing includes data loading and ordinary logging.
                    # Benchmark barriers run here, before any checkpoint I/O.
                    runtime_hooks.update_end(update, compute_seconds, writer)
                if not defer_maintenance and update % config['checkpoint_every'] == 0:
                    save('latest.pt')
                if update >= stop or stale >= config['patience'] or stop_requested.is_set():
                    break
                tick = time.perf_counter()
        save('latest.pt')
    finally:
        progress.close()
        writer.close()
        close_loader(loader)
        close_loader(val_loader)
    timing.update(wall_seconds=time.perf_counter()-start,
        samples_per_training_second=timing['samples']/max(timing['training_seconds'], 1e-12),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_device_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
        clipped_fraction=timing['clipped']/max(timing['updates'], 1), final_update=update,
        stopped_early=stale >= config['patience'], interrupted=stop_requested.is_set(),
        signal=received[-1] if received else None)
    atomic_json(output/'profile.json', timing)
    log.emit('interrupted' if stop_requested.is_set() else 'completion', update=update, stopped_early=timing['stopped_early'])
    return timing
