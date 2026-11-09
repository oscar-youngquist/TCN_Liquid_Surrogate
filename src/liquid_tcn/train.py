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
from .batch_logging import BatchLogger, resolve_mode, MetricWriter, write_scalars
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


def precision_check(model, batch, norm, config, *, diagnostic_path=None, context=None, log=None):
    """Probe copies; finite FP32 differences warn, nonfinite values/AMP still fail."""
    state = rng_state()
    outputs = {}
    report = dict(context=context or {}, physics_dtype=config['physics_dtype'],
                  probe_physics_dtype='float64', training_precision='amp' if config['amp'] else 'float32',
                  status='running')
    modes = ['float64', 'float32'] + (['amp'] if config['amp'] else [])
    current_mode = None
    def persist():
        if diagnostic_path is not None:
            atomic_json(diagnostic_path, report)
    try:
        for mode in modes:
            current_mode = mode
            report[mode] = dict(precision_mode=mode,tolerance=config['amp_rtol'] if mode=='amp' else config['precision_rtol'],
                relative_loss_error=None,absolute_loss_error=None,relative_gradient_error=None,absolute_gradient_error=None,
                gradient_norm=None,reference_gradient_norm=None)
            candidate = copy.deepcopy(model).train()
            x, p = norm.normalize('motion', batch['motion']), norm.normalize('properties', batch['properties'])
            if mode == 'float64':
                candidate = candidate.double(); x = x.double(); p = p.double()
            with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=mode == 'amp'):
                prediction = candidate(x, p)
            result = objective(prediction if mode == 'float64' else prediction.float(), batch, norm,
                               lambda_short=config['lambda_short'], lambda_long=config['lambda_long'],
                               supervised=config['supervised'], huber_delta=config['huber_delta'], physics_dtype='float64')
            if not torch.isfinite(result['loss']).all():
                raise FloatingPointError(f'Nonfinite {mode} loss')
            backward_objectives(result['supervised'],
                result['secondary'] if config['lambda_short'] or config['lambda_long'] else None,
                candidate.parameters(), config['gradient_strategy'], config['pcgrad_eps'])
            gradient = torch.cat([q.grad.detach().flatten().double() for q in candidate.parameters() if q.grad is not None])
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(f'Nonfinite {mode} gradient')
            loss, gradient_norm = float(result['loss'].detach()), float(gradient.norm())
            if not math.isfinite(gradient_norm):
                raise FloatingPointError(f'Nonfinite {mode} gradient norm')
            outputs[mode] = (loss, gradient)
            report[mode].update(precision_mode=mode, loss=loss, gradient_norm=gradient_norm,
                                tolerance=config['amp_rtol'] if mode=='amp' else config['precision_rtol'])
            if mode != 'float64':
                reference, grad = outputs['float64']
                absolute_loss = abs(loss-reference)
                absolute_gradient = float((gradient-grad).norm())
                loss_error = absolute_loss/max(abs(reference), 1e-12)
                gradient_error = absolute_gradient/max(float(grad.norm()), 1e-12)
                if not all(math.isfinite(v) for v in (absolute_loss,absolute_gradient,loss_error,gradient_error)):
                    raise FloatingPointError(f'Nonfinite {mode} precision difference')
                tolerance = report[mode]['tolerance']
                exceeded = max(loss_error,gradient_error)>tolerance
                report[mode].update(relative_loss_error=loss_error,absolute_loss_error=absolute_loss,
                    relative_gradient_error=gradient_error,absolute_gradient_error=absolute_gradient,
                    reference_gradient_norm=float(grad.norm()),reference_loss=reference,
                    tolerance_exceeded=exceeded,status='warning' if exceeded and mode=='float32' else 'failed' if exceeded else 'passed')
                persist()  # Persist evidence before either the warning or rejection.
                if exceeded:
                    if mode == 'amp':
                        raise ValueError(f'amp failed float64 precision gate: {report[mode]}')
                    warning = dict(event='precision_warning',severity='warning',context=report['context'],**report[mode])
                    if log is not None:
                        log.emit('precision_warning',severity='warning',context=report['context'],**report[mode])
                    else:
                        print(json.dumps(warning,allow_nan=False),flush=True)
        report['status'] = 'warning' if report['float32']['tolerance_exceeded'] else 'passed'
        persist()
        return report
    except BaseException as error:
        report.update(status='failed',failure=dict(precision_mode=current_mode,error_type=type(error).__name__,message=str(error)))
        persist()
        raise
    finally:
        restore_rng(state)


def resume_configuration(saved, overrides):
    """Saved settings are defaults; supplied fields take precedence."""
    previous = dict(saved['config']) if saved else {}
    if saved:
        previous.setdefault('gradient_strategy', 'sum')
    return {**DEFAULTS, **previous, **(overrides or {})}


def train(data, output, configuration=None, resume=None, max_updates=None,
          expected_model=None, launch_config=None, logging_mode='auto', trial_id='standalone',
          runtime_hooks=None, verification_receipt=None, staged_policy=None, execution=None):
    """Logging is a runtime option, deliberately outside checkpoint configuration.

    Optional runtime_hooks observe ready/update/checkpoint boundaries; they are
    never serialized and may defer scheduled maintenance for timing experiments.
    Ordinary training uses no hooks.

    Older checkpoints therefore need no migration: omitted logging_mode means
    auto and omitted trial_id means standalone. SIGTERM/SIGINT only set a flag;
    serialization happens after a complete optimizer/scheduler/sampler update.
    """
    saved = load_checkpoint(resume) if resume else None
    config = resume_configuration(saved, configuration)
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
                      launch_config, logging_mode, log, stop_requested, received, runtime_hooks, verification_receipt, staged_policy, execution)
    except BaseException as error:
        log.emit('failure', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)
        if run_lock is not None:
            run_lock.close()


def _train(data, output, configuration, resume, max_updates, expected_model,
           launch_config, logging_mode, log, stop_requested, received, runtime_hooks, verification_receipt, staged_policy, execution):
    mode = resolve_mode(logging_mode, (configuration or {}).get('progress', True))
    unknown = set(configuration or {}) - set(DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown training configuration fields: {sorted(unknown)}')
    saved = load_checkpoint(resume) if resume else None
    config = resume_configuration(saved, configuration)
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
    torch.set_num_threads((execution or {}).get('threads',config['threads']))
    random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
    torch.use_deterministic_algorithms(True)
    device = torch.device((execution or {}).get('device',config['device']))
    if runtime_hooks is not None:
        runtime_hooks.bind(device, stop_requested, log)
    defer_maintenance = staged_policy is not None or (runtime_hooks is not None and runtime_hooks.defer_maintenance)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ('latest.pt', 'config.json')) and resume is None:
        raise ValueError('Run exists; supply --resume or use a fresh output')
    training = WindowDataset(data, 'train', config['cache_size'], verify=False)
    validation = WindowDataset(data, 'validation', config['cache_size'])
    if not len(training) or not len(validation):
        raise ValueError('Training and validation must both have eligible windows in separate collection groups')
    if config['epochs'] is not None and (not isinstance(config['epochs'], int) or config['epochs'] < 1):
        raise ValueError('epochs must be null or a positive integer')
    # Epoch mode includes the last partial batch. Resolve it once so the
    # learning-rate schedule and checkpoints share the same update horizon.
    total_updates = (config['epochs'] * math.ceil(len(training) / config['batch_size'])
                     if config['epochs'] is not None else config['updates'])
    if saved and not {'epochs','updates'}.intersection(configuration or {}):
        total_updates = saved['effective_updates']
    norm = Normalization(saved['normalization'] if saved else training.manifest['normalization'])
    model = LiquidTCN(ModelConfig(**(expected_model or (saved['model_config'] if saved else training.manifest['model'])))).to(device)
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
    if resume:
        saved = load_checkpoint(resume, device)
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']); sampler.load_state_dict(saved['sampler'])
        if 'seed' in (configuration or {}):
            sampler.seed = config['seed']
        for group in optimizer.param_groups:
            if 'weight_decay' in (configuration or {}):
                group['weight_decay'] = config['weight_decay']
        if 'pretrain_lr' in (configuration or {}):
            scheduler.base_lrs = [config['pretrain_lr']] * len(optimizer.param_groups)
        update, best, stale = saved['update'], saved['best'], saved['stale']
        schedule_keys = {'pretrain_lr','physics_lr','scheduler','min_lr_ratio','pretrain_updates','updates','epochs'}
        if any(k in (configuration or {}) and config[k] != saved['config'].get(k) for k in schedule_keys):
            # Keep the resumed scheduler position, applying explicit new LR settings.
            rates = [base * learning_rate(scheduler.last_epoch) for base in scheduler.base_lrs]
            for group, rate in zip(optimizer.param_groups, rates):
                group['lr'] = rate
            scheduler._last_lr = rates
        restore_rng(saved['rng'], source_device=saved.get('execution_device',saved['config']['device']), destination_device=str(device))
    options = {k: config[k] for k in ('batch_size', 'num_workers', 'prefetch_factor', 'pin_memory', 'seed')}
    loader = make_loader(training, sampler=sampler, **options)
    validation_options = dict(options)
    for key in ('batch_size', 'num_workers'):
        if config['validation_' + key] is not None:
            validation_options[key] = config['validation_' + key]
    val_loader = make_loader(validation, **validation_options)
    stop = min(total_updates, max_updates) if max_updates is not None else total_updates
    timing = dict(loader_seconds=0., training_seconds=0., validation_seconds=0., transfer_seconds=0., logging_seconds=0., checkpoint_seconds=0., samples=0, updates=0, clipped=0)
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
    metrics = MetricWriter(output/'metrics.jsonl')
    writer = SummaryWriter(log_dir=str(output / 'tensorboard'), purge_step=update + 1 if resume else None)
    writer.add_text('run/configuration', json.dumps(config, indent=2), update)

    # Atomic replacement prevents partially written checkpoints. TensorBoard
    # is flushed alongside model/optimizer state; purge_step above hides stale
    # log events beyond the restored checkpoint when resuming an interrupted run.
    def save(name):
        checkpoint_start = time.monotonic()
        metrics.flush()
        writer.flush()
        atomic_save(output/name, dict(format_version=1, model=model.state_dict(), model_config=asdict(model.config),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            phase='supervised' if update < config['pretrain_updates'] else 'physics', update=update, effective_updates=total_updates,
            rng=rng_state(), execution_device=str(device), sampler=sampler.state_dict(), best=best, stale=stale,
            normalization=norm.stats, config=config, data_manifest_sha256=provenance,
            data_provenance=dict(sources=training.manifest['sources'], windows=training.manifest['windows'],
                                 preparation_config=training.manifest['config']), validation=last_validation,
            **({'stage_monitoring': staged_policy.state} if staged_policy is not None else {})))
        timing['checkpoint_seconds'] += time.monotonic()-checkpoint_start
        log.emit('checkpoint', update=update, checkpoint=str(output/name))
        if runtime_hooks is not None:
            runtime_hooks.checkpoint(update, name, time.monotonic()-checkpoint_start)

    initial_update = update
    if staged_policy is not None:
        staged_policy.attach(model,norm,validation,val_loader,dict(config,device=str(device)),output,saved if resume else None,update,save,stop_requested,log,writer)
    progress = TrainingProgress(total=stop, initial=update, disable=mode != 'progress')
    previous_phase = None
    try:
        if resume and staged_policy is not None and staged_policy.pending_monitor(update):
            staged_policy.after_update(update)
        if not resume:
            save('latest.pt')
        if runtime_hooks is not None:
            runtime_hooks.ready(update)
        while update < stop and stale < config['patience'] and not stop_requested.is_set() and not (staged_policy is not None and staged_policy.patience_stop):
            if sampler.position == sampler.length:
                sampler.next_epoch()
            sampler.limit = (stop - update) * config['batch_size']
            tick = time.perf_counter()
            for batch in loader:
                timing['loader_seconds'] += time.perf_counter()-tick
                train_start = time.perf_counter()
                batch = to_device(batch, device)
                if device.type == 'cuda' and runtime_hooks is not None:
                    torch.cuda.synchronize(device)
                timing['transfer_seconds'] += time.perf_counter()-train_start
                train_start = time.perf_counter()
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
                if device.type == 'cuda' and runtime_hooks is not None:
                    torch.cuda.synchronize(device)
                compute_seconds = time.perf_counter()-train_start
                timing['training_seconds'] += compute_seconds
                timing['samples'] += len(batch['motion']); timing['updates'] += 1
                logging_start = time.perf_counter()
                scalars = torch.stack([result['loss'].detach().double(), result['supervised'].detach().double(),
                    result['physics']['short']['loss'].detach().double(), result['physics']['long']['loss'].detach().double(),
                    result['physics']['short']['count'].detach().double(), result['physics']['long']['count'].detach().double(),
                    grad.detach().double(), result['secondary'].detach().double()]).cpu().tolist()
                record = dict(update=update, loss=scalars[0], supervised=scalars[1], short=scalars[2], long=scalars[3],
                    short_count=int(scalars[4]), long_count=int(scalars[5]),
                    supervised_counts=result['supervised_counts'].tolist(), lambda_short=short, lambda_long=long,
                    gradient_norm=scalars[6], learning_rate=used_lr, secondary=scalars[7], projection=projection,
                    loss_components=loss_components(result, short, long))
                clipped = int(record['gradient_norm'] > config['gradient_clip'])
                timing['clipped'] += clipped
                log.state.update({key: record[key] for key in log.state if key in record})
                timing['logging_seconds'] += time.perf_counter()-logging_start
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
                logging_start = time.perf_counter()
                scalar_tags = {}
                for key in ('loss', 'supervised', 'short', 'long', 'lambda_short', 'lambda_long',
                            'gradient_norm', 'learning_rate', 'short_count', 'long_count'):
                    scalar_tags['train/' + key] = record[key]
                scalar_tags['train/secondary'] = record['secondary']
                for key, value in record['loss_components'].items():
                    scalar_tags['train/' + key] = value
                for key, value in projection.items():
                    scalar_tags['train/pcgrad/' + key] = value
                for head, count in zip(('force', 'moment', 'r_B', 'v_B', 'S_B'), record['supervised_counts']):
                    scalar_tags['train/valid_' + head] = count
                scalar_tags['train/gradient_clipped'] = clipped
                write_scalars(writer, scalar_tags, update)
                metrics.write(record)
                progress.update(1)
                progress.set_postfix(phase='supervised' if short == long == 0 else 'physics',
                    loss=f"{record['loss']:.4g}", sup=f"{record['supervised']:.4g}",
                    short=f"{record['short']:.3g}", long=f"{record['long']:.3g}",
                    lr=f"{record['learning_rate']:.2g}",
                    val='pending' if last_validation is None else f"{last_validation['selection_metric']:.4g}", refresh=False)
                if update % 250 == 0:
                    log.emit('updates', update=update)
                timing['logging_seconds'] += time.perf_counter()-logging_start
                if runtime_hooks is not None:
                    if hasattr(runtime_hooks, 'costs'):
                        runtime_hooks.costs(timing)
                    # End-to-end timing includes data loading and ordinary logging.
                    # Benchmark barriers run here, before any checkpoint I/O.
                    class FlushLogs:
                        def flush(self):
                            writer.flush()
                            metrics.flush()
                    runtime_hooks.update_end(update, compute_seconds, FlushLogs())
                if staged_policy is not None:
                    staged_policy.after_update(update)
                if not defer_maintenance and update % config['checkpoint_every'] == 0:
                    save('latest.pt')
                if update >= stop or stale >= config['patience'] or stop_requested.is_set() or (staged_policy is not None and staged_policy.patience_stop):
                    break
                tick = time.perf_counter()
        if staged_policy is None or not resume or update != initial_update or stop_requested.is_set():
            save('latest.pt')
        if staged_policy is not None:
            staged_policy.finish(update)
    finally:
        if staged_policy is not None:
            staged_policy.close()
        progress.close()
        writer.close()
        metrics.close()
        close_loader(loader)
        close_loader(val_loader)
    timing.update(timing_definition='Host sections; CUDA synchronized only with benchmark hooks. Ordinary transfer/training timings include asynchronous dispatch; logging includes scalar waits.', wall_seconds=time.perf_counter()-start,
        samples_per_training_second=timing['samples']/max(timing['training_seconds'], 1e-12),
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_device_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
        clipped_fraction=timing['clipped']/max(timing['updates'], 1), final_update=update,
        stopped_early=(staged_policy.patience_stop if staged_policy is not None else stale >= config['patience']), interrupted=stop_requested.is_set(),
        signal=received[-1] if received else None)
    atomic_json(output/'profile.json', timing)
    log.emit('interrupted' if stop_requested.is_set() else 'completion', update=update, stopped_early=timing['stopped_early'])
    return timing
