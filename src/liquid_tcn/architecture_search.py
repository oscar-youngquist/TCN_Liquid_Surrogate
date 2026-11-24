"""Desktop matched-exposure architecture search with durable stage decisions."""
import argparse
import copy
from dataclasses import asdict
import itertools
import json
import math
import multiprocessing
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch

from .checkpoint import atomic_save, load_checkpoint, rng_state, restore_rng
from .episode_chunks import EpisodeChunks
from .model import ModelConfig, build_model, CachedInference
from .normalization import Normalization
from .objectives import objective
from .ported.storage import atomic_json
from .temporal_memory import optimizer_parameters


def emit(event, **values):
    print(json.dumps(dict(timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), event=event, **values)), flush=True)


BASELINE_CONFIG = Path(__file__).resolve().parents[2] / 'configs/pyramid_film.json'


def scaled_pyramid(baseline, seconds):
    """Realize history on the causal convolution grid, with sparse scaled tokens."""
    base = ModelConfig(**baseline)
    baseline_seconds = (base.receptive_field - 1) * base.sample_seconds
    if math.isclose(seconds, baseline_seconds):
        return copy.deepcopy(baseline)
    if seconds <= 0:
        raise ValueError('History must be positive')
    model = copy.deepcopy(baseline)
    # Keep baseline depth through two seconds; longer histories get one block.
    template = list(base.dilations)
    if seconds > 2:
        template.append(2 * template[-1])
    convolution_step = 2 * (base.kernel_size - 1)
    requested_steps = math.ceil(seconds / base.sample_seconds)
    dilation_budget = math.ceil(requested_steps / convolution_step)
    # Remove the deepest scale until the shorter history needs the remaining
    # depth. Keep positive dilations and a local first block.
    while len(template) > 1 and sum(template[:-1]) >= dilation_budget:
        template.pop()
    # Preserve the first local scale and distribute the remaining dilation
    # budget proportionally, using largest remainders for deterministic rounding.
    remaining = dilation_budget - template[0]
    quotas = [remaining * value / sum(template[1:]) for value in template[1:]]
    dilations = [template[0]] + [math.floor(value) for value in quotas]
    order = sorted(range(len(quotas)), key=lambda i: (-(quotas[i] % 1), i))
    for index in order[:dilation_budget-sum(dilations)]:
        dilations[index+1] += 1
    if min(dilations) < 1:
        raise ValueError('History is too short for the requested TCN depth')
    context = convolution_step * sum(dilations)
    model['dilations'] = dilations
    # The endpoint query and every aged token share a sufficient raw history.
    fields = np.cumsum([convolution_step*d for d in dilations]) + 1
    effective_steps = context
    available = [int(effective_steps + 1 - field) for field in fields]
    ages = ([[0] for _ in dilations] if seconds < baseline_seconds else
            [[lag for lag in lags if lag <= available[i]]
             for i, lags in enumerate(base.attention_lags)] +
            [[0] for _ in range(len(dilations)-len(base.attention_lags))])
    baseline_tokens = sum(map(len, base.attention_lags))
    target_tokens = max(len(dilations), round(baseline_tokens * seconds / baseline_seconds))
    # Keep the baseline's near-current samples. Extend each layer to its oldest
    # fully contextual token, then bisect the broadest scale-weighted gap.
    for layer, maximum in enumerate(available):
        if maximum not in ages[layer] and sum(map(len, ages)) < target_tokens:
            ages[layer].append(maximum)
        ages[layer].sort()
    if sum(map(len, ages)) > target_tokens:
        raise ValueError('Requested token budget cannot retain the baseline sampling scales')
    while sum(map(len, ages)) < target_tokens:
        gaps = [( (right-left)/math.sqrt(fields[layer]), -layer, left, right)
                for layer, values in enumerate(ages)
                for left, right in zip(values, values[1:]) if right-left > 1]
        if not gaps:
            raise ValueError('Requested token budget exceeds distinct causal sample positions')
        _, negative_layer, left, right = max(gaps)
        layer = -negative_layer
        ages[layer].append((left+right)//2)
        ages[layer].sort()
    model['attention_ages_ms'] = [[lag * base.sample_seconds * 1000 for lag in values] for values in ages]
    return model


def architectures(histories, baseline=None):
    baseline = (json.loads(BASELINE_CONFIG.read_text())['model'] if baseline is None
                else copy.deepcopy(baseline))
    models = {}
    for seconds in histories:
        model = scaled_pyramid(baseline, seconds)
        for family in ('tcn', 'hybrid'):
            selected = dict(model, episode_training=True, reset_backfill='recorded',
                            temporal_memory='mamba_stateful' if family == 'hybrid' else 'none')
            models[f'{family}-{seconds:g}s'] = dict(family=family, requested_history_seconds=seconds,
                                                    model=asdict(ModelConfig(**selected)))
    return models


def candidate_configs(settings, models):
    return [trial_config(model, settings.get('extra_physics', False), lr, batch, settings['seed'])
            for model in models.values()
            for lr, batch in itertools.product(settings['learning_rates'], settings['effective_batches'])]


def architecture_report(candidate):
    model = build_model(ModelConfig(**candidate['model']))
    config = model.config
    return dict(candidate, requested_history_seconds=candidate.get('requested_history_seconds'),
                actual_history_seconds=(config.receptive_field-1)*config.sample_seconds,
                receptive_field=config.receptive_field, block_count=len(model.blocks),
                dilations=list(config.dilations) if model.blocks else [],
                per_layer_sampling_ages_ms=config.attention_ages_ms if model.blocks else [],
                attention_tokens=sum(map(len,config.attention_lags)) if model.blocks else 0,
                parameters=sum(p.numel() for p in model.parameters()),
                feature_width=config.channels, attention_dim=config.attention_dim,
                attention_heads=config.attention_heads,
                comparison='history-plus-capacity')


def trial_config(architecture, extra, lr, batch, seed):
    cfg = copy.deepcopy(architecture)
    cfg.update(extra_physics=extra, lr=lr, effective_batch=batch, seed=seed)
    cfg['model']['consistency_intervals'] = 200 if extra else 40
    return cfg


def trial_id(cfg):
    import hashlib
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def schedules(settings, reference, family):
    ramp = 1. if settings['physics_ramp']==0 else min(1., max(0., (reference-settings['pretrain'])/settings['physics_ramp']))
    end = settings['pretrain'] + settings['physics_ramp']
    recurrent = min(1., max(0., (reference-end)/settings['recurrent_ramp'])) if family=='hybrid' and settings['recurrent_ramp'] else 1.
    return ramp, recurrent


def components(pred, batch, norm, extra):
    result = objective(pred, batch, norm, long_intervals=40)
    terms = [result['supervised'], result['physics']['short']['loss'], result['physics']['long']['loss']]
    counts = [result['supervised_counts'].sum()*3,
              result['physics']['short']['count'], result['physics']['long']['count']]
    if extra:
        one = objective(pred, batch, norm, long_intervals=200)['physics']['long']
        terms.append(one['loss'])
        counts.append(one['count'])
    return terms, counts


def monitor_population(settings, root, cfg):
    path = root/'monitor_subset.json'
    if path.exists():
        return json.loads(path.read_text())
    chunks = EpisodeChunks(settings['data'], ModelConfig(**cfg['model']), 1, 0, split='validation')
    groups = {}
    total = 0
    for sid, (series, _, end) in chunks.episodes.items():
        rows = chunks.reader.read(series, 0, end)
        count = int(rows['target_valid'].any(-1).sum())
        total += count
        groups.setdefault(series['robot'], []).append((sid, count))
    rng = random.Random(settings['monitor_seed'])
    selected, samples = [], 0
    for values in groups.values():
        rng.shuffle(values)
        target = sum(n for _, n in values) * settings['monitor_fraction']
        current = 0
        for sid, count in values:
            if current < target and (not selected or abs(current+count-target) <= abs(current-target)):
                selected.append(sid)
                current += count
        samples += current
    result = dict(episode_ids=sorted(selected), episodes=len(selected), scoring_samples=samples,
                  total_scoring_samples=total, achieved_fraction=samples/total,
                  requested_fraction=settings['monitor_fraction'], seed=settings['monitor_seed'])
    atomic_json(path, result)
    return result


def new_model(cfg, device):
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    return build_model(ModelConfig(**cfg['model'])).to(device)


def data_stream(settings, cfg, split='train', ids=None):
    return EpisodeChunks(settings['data'], ModelConfig(**cfg['model']),
                         cfg['effective_batch'] if split=='train' else settings['micro_batch'],
                         cfg['seed'], split=split, series_ids=ids)


def next_batch(stream):
    if stream.position == stream.length:
        stream.next_epoch()
    return stream.batch(stream.schedule[stream.position])


def to_device(batch, device):
    return {k:v.to(device) for k,v in batch.items()}


def optimize(model, optimizer, stream, norm, batch, cfg, settings, reference):
    """Accumulate objective numerators, then project the aggregate physics gradient."""
    parameters = list(model.parameters())
    buffers = [[None]*len(parameters) for _ in range(4 if cfg['extra_physics'] else 3)]
    denominators = [0.]*len(buffers)
    numerators = [0.]*len(buffers)
    ramp, recurrent = schedules(settings, reference, cfg['family'])
    if hasattr(model, 'temporal'):
        model.temporal.residual_scale = recurrent
    model.train()
    for start in range(0, len(batch['series']), settings['micro_batch']):
        small = to_device({k:v[start:start+settings['micro_batch']] for k,v in batch.items()}, settings['device'])
        pred = stream.predict(model, norm, small)
        terms, counts = components(pred, small, norm, cfg['extra_physics'])
        for j, (term, count) in enumerate(zip(terms, counts)):
            n = float(count)
            denominators[j] += n
            numerators[j] += float(term.detach())*n
            grads = torch.autograd.grad(term*count, parameters, retain_graph=j<len(terms)-1, allow_unused=True)
            for i,g in enumerate(grads):
                if g is not None:
                    buffers[j][i] = g.detach().clone() if buffers[j][i] is None else buffers[j][i]+g.detach()
    primary, secondary = [], []
    for i,p in enumerate(parameters):
        primary.append(torch.zeros_like(p) if buffers[0][i] is None else buffers[0][i]/max(1.,denominators[0]))
        auxiliary = torch.zeros_like(p)
        for j in range(1,len(buffers)):
            if buffers[j][i] is not None:
                auxiliary += buffers[j][i] * (ramp*(.1 if j==3 else 1.)/max(1.,denominators[j]))
        secondary.append(auxiliary)
    dot = sum((a*b).sum() for a,b in zip(primary,secondary))
    square = sum(a.square().sum() for a in primary)
    projection = torch.minimum(dot, dot.new_zeros(()))/(square+1e-12)
    for p,a,b in zip(parameters,primary,secondary):
        p.grad = a+b-projection*a
    torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    stream.advance(len(batch['series']))
    return dict(losses=[n/max(1.,d) for n,d in zip(numerators,denominators)],
                physics_weight=ramp, recurrent_weight=recurrent)


def episode_metrics(rows, prediction, series, norm):
    from .offline import horizon_errors
    valid = rows['target_valid'][:, :2] & np.isfinite(rows['targets'][:, :2]).all(-1)
    error = prediction[:, :2] - rows['targets'][:, :2]
    scale = np.asarray(norm.stats['targets']['std'])[:2]
    squared = (error/scale)**2
    def score(mask):
        use = valid & mask[:, None]
        return dict(sum=float(np.where(use[..., None], squared, 0).sum()), count=int(use.sum())*3,
                    physical_sum=float(np.where(use[..., None], error**2, 0).sum()),
                    head_squares=np.where(use[...,None],error**2,0).sum((0,2)).tolist(),
                    axis_squares=np.where(use[...,None],error**2,0).sum(0).tolist(),
                    head_counts=(use.sum(0)*3).tolist())
    times = np.arange(len(rows))*.005
    result = dict(wrench=score(np.ones(len(rows), dtype=bool)), startup_200ms=score(times<.2),
                  startup_1240ms=score(times<1.24), post_200ms_to_1240ms=score((times>=.2)&(times<1.24)),
                  after_1240ms=score(times>=1.24), first_step_valid=bool(np.isfinite(prediction[0]).all()),
                  complete_10_seconds=len(rows)>=2001,
                  duration=float(times[-1]), elapsed={str(i):score((times>=i)&(times<i+1)) for i in range(10)} if len(rows)>=2001 else {},
                  horizons={})
    good = (rows['state_valid'] & rows['interval_valid'] & rows['target_valid'].all(-1)
            & np.isclose(rows['h'], .005))
    padded = np.r_[False,good,False].astype(int)
    for start,end in zip(np.flatnonzero(np.diff(padded)==1),np.flatnonzero(np.diff(padded)==-1)):
        if end-start<2:
            continue
        selected=rows[start:end].copy()
        selected['history_valid']=True
        values,_ = horizon_errors(selected, prediction[start:end], series['interior_dimensions'],
                                   selected['x_O_W'][0], horizons=(200,400,1000,2000,4000))
        mass=float(selected['mass'][0])
        gravity=float(np.linalg.norm(selected['gravity_W'][0]))
        length=float(np.linalg.norm(series['interior_dimensions']))
        for steps,value in values.items():
            seconds=steps*.005
            acc=result['horizons'].setdefault(str(seconds), dict(count=0, force=0.,moment=0.,consistency=[0.,0.,0.]))
            acc['count']+=len(value['force'])
            acc['force']+=float(np.sum((value['force']/(mass*gravity*seconds))**2))
            acc['moment']+=float(np.sum((value['moment']/(mass*gravity*length*seconds))**2))
            acc['consistency']=(np.asarray(acc['consistency'])+np.sum(value['predicted_normalized']**2,axis=(0,2))).tolist()
    return result


def summarize(episodes):
    def combine(rows):
        total=sum(e['sum'] for e in rows)
        count=sum(e['count'] for e in rows)
        physical=sum(e['physical_sum'] for e in rows)
        squares=sum((np.asarray(e['head_squares']) for e in rows),np.zeros(2))
        counts=sum((np.asarray(e['head_counts']) for e in rows),np.zeros(2))
        component=np.sqrt(squares/np.maximum(1,counts))
        axes=sum((np.asarray(e['axis_squares']) for e in rows),np.zeros((2,3)))
        return dict(normalized_mse=total/count if count else None, count=count,
                    rms=math.sqrt(physical/count) if count else None,
                    force_moment_rms=component.tolist(),
                    force_moment_component_rms=component.tolist(),
                    force_moment_vector_rms=(component*math.sqrt(3)).tolist(),
                    force_moment_axis_rms=np.sqrt(axes/np.maximum(1,counts[:,None]/3)).tolist())
    def combined(name):
        return combine([e[name] for e in episodes])
    horizons={}
    for key in ('1.0','2.0','5.0','10.0','20.0'):
        terms=[e['horizons'][key] for e in episodes if key in e['horizons']]
        count=sum(t['count'] for t in terms)
        horizons[key]=dict(count=count,
            force=sum(t['force'] for t in terms)/count if count else None,
            moment=sum(t['moment'] for t in terms)/count if count else None,
            consistency=(sum((np.asarray(t['consistency']) for t in terms),np.zeros(3))/count).tolist() if count else None)
    available=[h for k,h in horizons.items() if k!='20.0' and h['count']]
    impulse=float(np.mean([(h['force']+h['moment'])/2 for h in available])) if available else None
    elapsed={}
    for key in sorted({k for e in episodes for k in e['elapsed']},key=int):
        rows=[e['elapsed'][key] for e in episodes if key in e['elapsed']]
        elapsed[key]=combine(rows)
    return dict(wrench=combined('wrench'),startup_200ms=combined('startup_200ms'),
                post_200ms_to_1240ms=combined('post_200ms_to_1240ms'),after_1240ms=combined('after_1240ms'),
                startup_1240ms=combined('startup_1240ms'), horizons=horizons, impulse_score=impulse,
                elapsed=elapsed,elapsed_cohort=dict(episodes=sum(e['complete_10_seconds'] for e in episodes),
                    series=sorted(e['series'] for e in episodes if e['complete_10_seconds']),
                    definition='Same complete 0–10 second episode cohort in all ten [i, i+1) bins'),
                first_step_valid_fraction=float(np.mean([e['first_step_valid'] for e in episodes])),
                maximum_duration=max(e['duration'] for e in episodes))


@torch.no_grad()
def assess(model, cfg, settings, norm, split='validation', ids=None):
    model.eval()
    stream=data_stream(settings,cfg,split,ids)
    predictions={}
    episodes=[]
    for batch in stream:
        prediction=stream.predict(model,norm,to_device(batch,settings['device']))
        prediction=norm.denormalize('targets',prediction[:,stream.horizon:]).cpu().numpy()
        for i,(sid,end,length) in enumerate(zip(batch['series'].tolist(),batch['end'].tolist(),batch['length'].tolist())):
            predictions.setdefault(sid,[]).append(prediction[i,:length])
            series,_,stop=stream.episodes[sid]
            if end+1==stop:
                rows=stream.reader.read(series,0,stop)
                metrics=episode_metrics(rows,np.concatenate(predictions.pop(sid)),series,norm)
                episodes.append(dict(metrics,series=sid,group=series['group']))
    return dict(summary=summarize(episodes),episodes=episodes)


def ranked(results):
    return sorted(results, key=lambda name: (results[name]['summary']['wrench']['normalized_mse'], str(name)))


def run_trial(cfg, settings, root, target, subset, final=False, max_optimizer_updates=None):
    folder=root/'trials'/trial_id(cfg)
    folder.mkdir(parents=True,exist_ok=True)
    atomic_json(folder/'trial.json',cfg)
    endpoint=folder/f'validation-{target:g}.json'
    if endpoint.exists() and (not final or (folder/'selection.json').exists()):
        return json.loads(endpoint.read_text())
    model=new_model(cfg,settings['device'])
    stream=data_stream(settings,cfg)
    norm=Normalization(stream.manifest['normalization'])
    optimizer=torch.optim.AdamW(optimizer_parameters(model,1e-4),lr=cfg['lr'],weight_decay=1e-4)
    update,exposure,next_monitor=0,0,settings['monitor_every']
    shortlist=[]; best=float('inf'); stale=0
    checkpoint=folder/'latest.pt'
    if checkpoint.exists():
        saved=load_checkpoint(checkpoint,settings['device'])
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        stream.load_state_dict(saved['sampler'])
        restore_rng(saved['rng'],saved['device'],settings['device'])
        update,exposure,next_monitor=saved['update'],saved['exposure'],saved['next_monitor']
        shortlist,best,stale=saved['shortlist'],saved['best'],saved['stale']
    stopping={'requested':False}
    def stop(_number,_frame):
        stopping['requested']=True
    previous={number:signal.signal(number,stop) for number in (signal.SIGINT,signal.SIGTERM)}
    def save(path):
        atomic_save(path,dict(model=model.state_dict(), model_config=cfg['model'], config=cfg,
            optimizer=optimizer.state_dict(), sampler=stream.state_dict(), rng=rng_state(), device=settings['device'],
            update=update,exposure=exposure,next_monitor=next_monitor,shortlist=shortlist,best=best,stale=stale,
            normalization=norm.stats, scheduler_horizon=settings['maximum_reference_updates'],
            recurrent_weight=schedules(settings,exposure/6400,cfg['family'])[1]))
    started=time.perf_counter(); initial_exposure=exposure; initial_update=update
    emit('start', trial=trial_id(cfg), target_reference_updates=target,
         reference_updates=exposure/6400, optimizer_steps=update, final=final)
    if str(settings['device']).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats()
    try:
        while (exposure/6400<target or max_optimizer_updates is not None) and not stopping['requested']:
            if final and stale>=settings['patience']:
                break
            if max_optimizer_updates is not None and update-initial_update>=max_optimizer_updates:
                break
            batch=next_batch(stream)
            if max_optimizer_updates is None:
                # Mask excess endpoints in the final optimizer batch so 16/32
                # chunk batches finish at identical supervised exposure.
                remaining=max(0,round(target*6400)-exposure)
                bearing=batch['target_valid'].any(-1)&batch['score_endpoints']
                ordinal=bearing.flatten().cumsum(0).reshape_as(bearing)
                batch['score_endpoints'] &= ordinal<=remaining
            # Actual target-bearing endpoints determine reference exposure.
            scored=int((batch['target_valid'].any(-1)&batch['score_endpoints']).sum())
            ref=exposure/6400
            fraction=min(1.,ref/settings['maximum_reference_updates'])
            lr=cfg['lr']*(.1+.9*.5*(1+math.cos(math.pi*fraction)))
            for group in optimizer.param_groups: group['lr']=lr
            log=optimize(model,optimizer,stream,norm,batch,cfg,settings,ref)
            update+=1;exposure+=scored
            if update%settings.get('log_every',100)==0 or update-initial_update==1:
                emit('update',trial=trial_id(cfg),optimizer_steps=update,reference_updates=exposure/6400,lr=lr,**log)
            if exposure/6400>=next_monitor:
                result=assess(model,cfg,settings,norm,ids=subset['episode_ids'])
                score=result['summary']['wrench']['normalized_mse']
                atomic_json(folder/f'monitor-{update}.json',result)
                if exposure/6400>=settings['pretrain']+settings['physics_ramp']:
                    stale=0 if score<best else stale+1
                    best=min(best,score)
                shortlist.append(dict(update=update,score=score))
                shortlist=sorted(shortlist,key=lambda x:(x['score'],x['update']))[:3]
                while next_monitor<=exposure/6400: next_monitor+=settings['monitor_every']
                if any(x['update']==update for x in shortlist):
                    save(folder/f'shortlist-{update}.pt')
                save(checkpoint)
                emit('monitor',trial=trial_id(cfg),optimizer_steps=update,reference_updates=exposure/6400,score=score,stale=stale)
            if update%settings['checkpoint_every']==0: save(checkpoint)
        save(checkpoint)
        if str(settings['device']).startswith('cuda'): torch.cuda.synchronize()
        seconds=time.perf_counter()-started
        profile=dict(seconds=seconds,updates=update-initial_update,scored_steps=exposure-initial_exposure,
                     optimizer_steps=update,reference_updates=exposure/6400,
                     reference_per_second=(exposure-initial_exposure)/6400/seconds,
                     peak_gpu_bytes=torch.cuda.max_memory_allocated() if str(settings['device']).startswith('cuda') else None)
        atomic_json(folder/'profile.json',profile)
        if stopping['requested']:
            raise InterruptedError('Search interrupted after saving optimizer and streams')
        if max_optimizer_updates is not None:
            return profile
        _,scale=schedules(settings,exposure/6400,cfg['family'])
        if hasattr(model,'temporal'): model.temporal.residual_scale=scale
        report=json.loads(endpoint.read_text()) if endpoint.exists() else assess(model,cfg,settings,norm)
        report.update(update=update,reference=exposure/6400,optimizer_steps=update,reference_updates=exposure/6400,
                      stale=stale,trial=trial_id(cfg))
        if max_optimizer_updates is None:
            atomic_json(endpoint,report)
        if final:
            choices={update:report}
            for row in shortlist:
                if row['update']==update:continue
                saved=load_checkpoint(folder/f"shortlist-{row['update']}.pt",settings['device'])
                model.load_state_dict(saved['model'])
                if hasattr(model,'temporal'):
                    model.temporal.residual_scale=schedules(settings,saved['exposure']/6400,cfg['family'])[1]
                path=folder/f"shortlist-validation-{row['update']}.json"
                if path.exists():
                    choices[row['update']]=json.loads(path.read_text())
                else:
                    choices[row['update']]=assess(model,cfg,settings,norm)
                    choices[row['update']].update(optimizer_steps=saved['update'],reference_updates=saved['exposure']/6400)
                    atomic_json(path,choices[row['update']])
            selected=ranked(choices)[0]
            source=checkpoint if selected==update else folder/f'shortlist-{selected}.pt'
            atomic_save(folder/'best.pt',load_checkpoint(source))
            atomic_json(folder/'selection.json',dict(update=selected,metric='full_validation_normalized_wrench_mse',
                candidates={str(k):v['summary']['wrench']['normalized_mse'] for k,v in choices.items()},report=choices[selected]))
        emit('complete',trial=trial_id(cfg),reference_updates=exposure/6400,optimizer_steps=update,
             validation_mse=report['summary']['wrench']['normalized_mse'],seconds=time.perf_counter()-started)
        return report
    finally:
        for number,handler in previous.items():signal.signal(number,handler)


@torch.no_grad()
def cached_benchmark(model, norm, device, environments=4096, steps=20):
    model.eval()
    motion=torch.zeros(environments,15,device=device)
    motion[:,2]=-9.81
    properties=torch.as_tensor(norm.stats['properties']['mean'],dtype=torch.float32,device=device)[None]
    if str(device).startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    cache=CachedInference(model,norm,environments)
    cache.set_properties(properties)
    def sync():
        if str(device).startswith('cuda'):torch.cuda.synchronize()
    timings={}
    for name,indices in [('full',None),('partial',list(range(max(1,environments//16))))]:
        reset=motion if indices is None else motion[indices]
        sync();started=time.perf_counter()
        cache.initialize_reset(reset,indices)
        sync();timings[name]=time.perf_counter()-started
    for _ in range(5):
        cache.step(motion)
    durations=[]
    for _ in range(steps):
        sync();started=time.perf_counter()
        _,valid=cache.step(motion)
        sync();durations.append(time.perf_counter()-started)
    return dict(environments=environments,mean_seconds=float(np.mean(durations)),
                p95_seconds=float(np.quantile(durations,.95)),environment_steps_per_second=environments/np.mean(durations),
                persistent_bytes=cache.persistent_bytes(),reset_seconds=timings,all_valid=bool(valid.all()),
                peak_gpu_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else None)


def pilot(settings, root, models, subset):
    import gc
    results_path = root/'pilot.json'
    previous = json.loads(results_path.read_text()) if results_path.exists() else {}
    results = {}
    reference_total = (len(candidate_configs(settings,models))*settings['screening']
                       + 2*len(settings['final_seeds'])*settings['maximum_reference_updates'])

    def estimate():
        complete = len(results) == len(models) and all(row.get('status') == 'measured' for row in results.values())
        value = dict(status='measured' if complete else 'incomplete',
                     reference_updates_upper_bound=reference_total,
                     training_hours_range=None, validation_and_inference_time_included=False,
                     workers=1, comparison='history-plus-capacity',
                     concurrency_reason='One-worker pilot only; concurrency has not been benchmarked')
        if complete:
            repeats=len(settings['learning_rates'])*len(settings['effective_batches'])
            screen=sum(repeats*settings['screening']/row['training']['reference_per_second']
                       for row in results.values())
            final_low=final_high=0.
            for family in ('tcn','hybrid'):
                speeds=[row['training']['reference_per_second'] for row in results.values() if row['family']==family]
                exposure=len(settings['final_seeds'])*settings['maximum_reference_updates']
                final_low+=exposure/max(speeds)
                final_high+=exposure/min(speeds)
            value.update(screening_training_hours=screen/3600,
                         training_hours_range=[(screen+final_low)/3600,(screen+final_high)/3600],
                         final_reference_budget_per_run=settings['maximum_reference_updates'],
                         early_stopping_applied=False,
                         note='Four-update pilots include startup effects; LR/batch combinations use the same per-architecture rate estimate.')
        atomic_json(root/'runtime_estimate.json', value)
        return value

    estimate()
    for name, candidate in models.items():
        cfg = trial_config(candidate, settings.get('extra_physics',False), 3e-4, 16, settings['seed'])
        signature = trial_id(dict(candidate=cfg, device=settings['device'],
                                  micro_batch=settings['micro_batch'], pilot_updates=settings['pilot_updates'],
                                  prefill_benchmark_version=2))
        cached = previous.get(name, {})
        if cached.get('signature') == signature and cached.get('status') == 'measured':
            results[name] = cached
            continue
        row = dict(architecture_report(candidate), signature=signature, status='pending',
                   training=None, cached=None)
        destination = root/'pilot'/name/signature
        options = dict(settings, pretrain=0, physics_ramp=0, recurrent_ramp=0)
        try:
            run_trial(cfg, options, destination, 1, subset,
                      max_optimizer_updates=settings['pilot_updates'])
            folder = destination/'trials'/trial_id(cfg)
            row['training'] = json.loads((folder/'profile.json').read_text())
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            row['training_error'] = str(error)
        gc.collect()
        if str(settings['device']).startswith('cuda'):
            torch.cuda.empty_cache()
        try:
            model = new_model(cfg, settings['device'])
            manifest = json.loads((Path(settings['data'])/'manifest.json').read_text())
            norm = Normalization(manifest['normalization'])
            row['cached'] = cached_benchmark(model, norm, settings['device'])
            del model
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            row['cached_error'] = str(error)
        row['status'] = 'measured' if row['training'] is not None and row['cached'] is not None else 'failed'
        results[name] = row
        atomic_json(results_path, results)
        emit('pilot', architecture=name, status=row['status'], training=row['training'])
        gc.collect()
        if str(settings['device']).startswith('cuda'):
            torch.cuda.empty_cache()
        estimate()
    atomic_json(results_path, results)
    return estimate()


def paired_summary(reports,seed=1729,repetitions=1000):
    keys=list(reports)
    output={}
    for key,values in reports.items():
        output[key]=dict(seeds=[v['summary'] for v in values],
                        wrench_mean=float(np.mean([v['summary']['wrench']['normalized_mse'] for v in values])),
                        wrench_seed_sd=float(np.std([v['summary']['wrench']['normalized_mse'] for v in values],ddof=1)),
                        impulse_mean=float(np.mean([v['summary']['impulse_score'] for v in values])) if all(v['summary']['impulse_score'] is not None for v in values) else None,
                        impulse_seed_sd=float(np.std([v['summary']['impulse_score'] for v in values],ddof=1)) if all(v['summary']['impulse_score'] is not None for v in values) else None)
        def aggregate(items):
            if all(isinstance(v,dict) for v in items):
                return {k:aggregate([v[k] for v in items]) for k in items[0]
                        if k not in ('count','elapsed_cohort','maximum_duration')}
            if all(isinstance(v,(int,float,list)) for v in items):
                array=np.asarray(items,dtype=float)
                return dict(mean=np.mean(array,axis=0).tolist(),
                            seed_sd=np.std(array,axis=0,ddof=1 if len(items)>1 else 0).tolist())
            return None
        output[key]['metrics']=aggregate([v['summary'] for v in values])
    groups=sorted(set(e['group'] for values in reports.values() for v in values for e in v['episodes']))
    rng=np.random.default_rng(seed)
    paired=[]
    impulse_paired=[]
    totals={}
    impulses={}
    for key,values in reports.items():
        totals[key]=[]
        for group in groups:
            rows=[e for v in values for e in v['episodes'] if e['group']==group]
            totals[key].append((sum(e['wrench']['sum'] for e in rows),sum(e['wrench']['count'] for e in rows)))
        totals[key]=np.asarray(totals[key])
        impulses[key]=[]
        for group in groups:
            rows=[e for v in values for e in v['episodes'] if e['group']==group]
            impulses[key].append([[sum(e['horizons'].get(h,{}).get('force',0)+e['horizons'].get(h,{}).get('moment',0) for e in rows),
                                   sum(e['horizons'].get(h,{}).get('count',0) for e in rows)]
                                  for h in ('1.0','2.0','5.0','10.0')])
        impulses[key]=np.asarray(impulses[key])
    for _ in range(repetitions):
        indices=rng.integers(len(groups),size=len(groups))
        scores=[]
        for key in keys:
            sums=totals[key][indices].sum(0)
            scores.append(sums[0]/sums[1])
        paired.append(scores[0]-scores[1])
        impulse_scores=[]
        for key in keys:
            summed=impulses[key][indices].sum(0)
            eligible=summed[:,1]>0
            impulse_scores.append(float(np.mean(summed[eligible,0]/(2*summed[eligible,1]))) if eligible.any() else None)
        if all(value is not None for value in impulse_scores):
            impulse_paired.append(impulse_scores[0]-impulse_scores[1])
    return dict(seed_variation=output,paired_group_bootstrap=dict(comparison=keys,groups=len(groups),
        repetitions=repetitions,wrench_difference_ci95=np.quantile(paired,[.025,.975]).tolist(),
        impulse_difference_ci95=np.quantile(impulse_paired,[.025,.975]).tolist() if impulse_paired else None,
        definition='Paired resampling of collection groups, with seeds pooled within each group'))


def _trial_worker(cfg, settings, root, target, subset, final):
    """A spawned worker owns one trial and appends resumable, line-based logs."""
    import contextlib
    import traceback
    folder=root/'trials'/trial_id(cfg)
    folder.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(settings['threads'])
    with (folder/'worker.log').open('a',buffering=1) as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                run_trial(cfg,settings,root,target,subset,final=final)
            except BaseException:
                traceback.print_exc()
                raise


def run_stage(configs, settings, root, target, subset, final=False):
    """Bounded independent processes; interrupted workers save consumed streams."""
    concurrency=settings.get('concurrency',1)
    pending=[]
    reports={}
    for cfg in configs:
        key=trial_id(cfg)
        folder=root/'trials'/key
        path=folder/f'validation-{target:g}.json'
        if path.exists() and (not final or (folder/'selection.json').exists()):
            reports[key]=json.loads(path.read_text())
        else:
            pending.append(cfg)
    if concurrency==1:
        for cfg in pending:
            reports[trial_id(cfg)]=run_trial(cfg,settings,root,target,subset,final=final)
        return reports
    context=multiprocessing.get_context('spawn')
    active={}
    def stop(_number,_frame):
        raise KeyboardInterrupt('Coordinator interrupted; saving active workers')
    previous={number:signal.signal(number,stop) for number in (signal.SIGINT,signal.SIGTERM)}
    try:
        while pending or active:
            while pending and len(active)<concurrency:
                cfg=pending.pop(0);key=trial_id(cfg)
                process=context.Process(target=_trial_worker,args=(cfg,settings,root,target,subset,final))
                process.start();active[key]=process
                emit('worker_start',trial=key,pid=process.pid,final=final,log=str(root/'trials'/key/'worker.log'))
            for key,process in list(active.items()):
                if process.is_alive():
                    continue
                process.join()
                del active[key]
                if process.exitcode:
                    raise RuntimeError(f'Trial {key} exited {process.exitcode}; see {root / "trials" / key / "worker.log"}. Relaunch to resume.')
                reports[key]=json.loads((root/'trials'/key/f'validation-{target:g}.json').read_text())
                emit('worker_complete',trial=key,reference_updates=reports[key]['reference'],
                     optimizer_steps=reports[key]['update'],validation_mse=reports[key]['summary']['wrench']['normalized_mse'])
            if active:
                time.sleep(.2)
    finally:
        # SIGTERM is handled by run_trial after its current optimizer operation.
        for process in active.values():
            if process.is_alive():process.terminate()
        for process in active.values():process.join()
        for number,handler in previous.items():signal.signal(number,handler)
    return reports


def execute(settings,root,models,subset):
    decisions=root/'decisions.json'
    state=json.loads(decisions.read_text()) if decisions.exists() else {}
    candidates=candidate_configs(settings,models)
    configs={trial_id(cfg):cfg for cfg in candidates}
    screening=run_stage(candidates,settings,root/'screening',settings['screening'],subset)
    winners={family:ranked({k:v for k,v in screening.items() if configs[k]['family']==family})[0]
             for family in ('tcn','hybrid')}
    state['screening']=dict(trials=list(configs),ranking=ranked(screening),winners=winners)
    atomic_json(decisions,state)
    final_configs=[];finalists={}
    for family,winner in winners.items():
        finalists[family]=[]
        for seed in settings['final_seeds']:
            cfg=copy.deepcopy(configs[winner]);cfg['seed']=seed
            final_configs.append(cfg);finalists[family].append(trial_id(cfg))
    state['finalists']=finalists
    atomic_json(decisions,state)
    run_stage(final_configs,settings,root/'final',settings['maximum_reference_updates'],subset,final=True)
    validation={family:[json.loads((root/'final'/'trials'/key/'selection.json').read_text())['report']
                        for key in ids] for family,ids in finalists.items()}
    atomic_json(root/'final_validation_comparison.json',paired_summary(validation))
    # Every validation-selected checkpoint is durable before any test assessment.
    reports={family:[] for family in finalists}
    for cfg in final_configs:
        key=trial_id(cfg);folder=root/'final'/'trials'/key
        result_path=folder/'test.json'
        if result_path.exists():
            report=json.loads(result_path.read_text())
        else:
            saved=load_checkpoint(folder/'best.pt',settings['device'])
            model=new_model(cfg,settings['device']);model.load_state_dict(saved['model'])
            if hasattr(model,'temporal'):model.temporal.residual_scale=saved['recurrent_weight']
            norm=Normalization(saved['normalization'])
            report=assess(model,cfg,settings,norm,'test')
            report.update(seed=cfg['seed'],trial=key,selected_optimizer_steps=saved['update'],
                          selected_reference_updates=saved['exposure']/6400,architecture=architecture_report(cfg))
            # Save test results before attempting an optional large-batch benchmark.
            atomic_json(result_path,report)
        if 'cached' not in report:
            saved=load_checkpoint(folder/'best.pt',settings['device'])
            model=new_model(cfg,settings['device']);model.load_state_dict(saved['model'])
            if hasattr(model,'temporal'):model.temporal.residual_scale=saved['recurrent_weight']
            norm=Normalization(saved['normalization'])
            try:
                report['cached']=cached_benchmark(model,norm,settings['device'],
                    environments=settings.get('benchmark_environments',4096),steps=settings.get('benchmark_steps',20))
            except (torch.cuda.OutOfMemoryError,RuntimeError) as error:
                report['cached']=dict(status='unavailable',environments=settings.get('benchmark_environments',4096),error=str(error))
                if str(settings['device']).startswith('cuda'):torch.cuda.empty_cache()
            atomic_json(result_path,report)
        reports[cfg['family']].append(report)
    atomic_json(root/'final_comparison.json',paired_summary(reports))
    emit('search_complete',output=str(root),final_runs=len(final_configs))


def holdout(settings, root, data):
    state=json.loads((root/'decisions.json').read_text())
    options=dict(settings,data=data)
    results={}
    for candidate,ids in state['finalists'].items():
        results[candidate]=[]
        for key in ids:
            folder=root/'final'/'trials'/key
            saved=load_checkpoint(folder/'best.pt',settings['device'])
            cfg=json.loads((folder/'trial.json').read_text())
            model=new_model(cfg,settings['device']);model.load_state_dict(saved['model'])
            if hasattr(model,'temporal'):
                model.temporal.residual_scale=saved['recurrent_weight']
            report=assess(model,cfg,options,Normalization(saved['normalization']),'test')
            results[candidate].append(report)
            atomic_json(folder/'holdout.json',report)
    atomic_json(root/'holdout_comparison.json',paired_summary(results))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/architecture_search.json')
    parser.add_argument('--mode',choices=('plan','smoke','pilot','run','holdout'),default='plan')
    parser.add_argument('--holdout-data')
    parser.add_argument('--data');parser.add_argument('--output');parser.add_argument('--device')
    parser.add_argument('--concurrency',type=int)
    parser.add_argument('--micro-batch',type=int)
    parser.add_argument('--threads',type=int)
    parser.add_argument('--log-every',type=int)
    args=parser.parse_args()
    settings=json.loads(Path(args.config).read_text())
    for key in ('data','output','device','concurrency','micro_batch','threads','log_every'):
        if getattr(args,key) is not None:settings[key]=getattr(args,key)
    for key in ('concurrency','micro_batch','threads','log_every'):
        if settings[key]<1:parser.error(f'{key} must be positive')
    if settings.get('search_kind')!='focused_history_lr_batch_v1':
        parser.error('Use the focused search configuration and a separate output directory')
    root=Path(settings['output']);root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(settings['threads'])
    models=architectures(settings['histories_seconds'])
    import fcntl
    with (root/'.search.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        saved_settings=root/'search.json'
        if saved_settings.exists():
            previous=json.loads(saved_settings.read_text())
            # Execution resources can move while scientific settings remain fixed.
            for key in ('device','threads','micro_batch','concurrency','log_every'):previous[key]=settings[key]
            if previous!=settings:raise ValueError('Use a new output folder for changed search settings')
        atomic_json(saved_settings,settings)
        atomic_json(root/'ranking_protocol.json',dict(
            wrench='Mean squared force and moment errors normalized by training-only per-axis target standard deviations',
            selection='Minimum full-validation normalized wrench MSE; trial ID breaks exact ties',
            finalists='Best TCN and best hybrid across histories, learning rates and effective batches',
            checkpoint_selection='Minimum full-validation normalized wrench MSE among top three monitor checkpoints and stopping checkpoint',
            rms='force_moment_component_rms pools xyz squared error per component; vector_rms is sqrt(3) times this; axis_rms reports each xyz component',
            impulse='Equal mean across supported 1, 2, 5 and 10 second horizons of (linear squared error + angular squared error) / 2',
            linear_scale='mass * gravity magnitude * horizon seconds',
            angular_scale='mass * gravity magnitude * cavity diagonal length * horizon seconds',
            absent_horizons='Unavailable, excluded equally for every candidate on the same episode population',
            reference_update_scored_endpoints=6400,
            bootstrap='Paired collection-group resampling with seeds pooled; seed SD reported separately',
            reset_prefill='Repeat the full recorded reset input, including gravity, linear and angular velocity, and held acceleration'))
        subset=monitor_population(settings,root,next(iter(models.values())))
        spec={}
        for name,cfg in models.items():
            spec[name] = architecture_report(cfg)
        atomic_json(root/'architectures.json',spec)
        candidates={trial_id(cfg):dict(cfg,actual_history_seconds=spec[f"{cfg['family']}-{cfg['requested_history_seconds']:g}s"]['actual_history_seconds'],
                    parameters=spec[f"{cfg['family']}-{cfg['requested_history_seconds']:g}s"]['parameters'])
                    for cfg in candidate_configs(settings,models)}
        atomic_json(root/'candidate_manifest.json',dict(settings=settings,candidates=candidates,
            screening_reference_updates=settings['screening'],scheduler_horizon=settings['maximum_reference_updates'],
            reference_update_scored_endpoints=6400,chunk_new_steps=400,
            optimizer=dict(name='AdamW',weight_decay=1e-4,gradient_clip=1.,lr_batch_scaling=False),
            objectives=dict(reconstruction=1.,physics_5ms=1.,physics_200ms=1.,extra_1s_physics=settings['extra_physics'])))
        audit=data_stream(settings,trial_config(next(iter(models.values())),False,3e-4,16,1),'test')
        durations=[(end-1)*.005 for _,_,end in audit.episodes.values()]
        atomic_json(root/'duration_audit.json',dict(episodes=len(durations),minimum=min(durations),maximum=max(durations),
            episodes_supporting_20_seconds=sum(t>=20 for t in durations),note='Each horizon stays within one continuous recorded episode.'))
        emit('plan',architectures=len(models),candidates=len(candidates),monitor_episodes=subset['episodes'],
             monitor_fraction=subset['achieved_fraction'],maximum_episode_seconds=max(durations),output=str(root))
        if args.mode=='smoke':
            cfg=trial_config(models['hybrid-1.24s'],settings['extra_physics'],3e-4,16,settings['seed'])
            options=dict(settings,pretrain=0,physics_ramp=0,recurrent_ramp=0)
            profile=run_trial(cfg,options,root/'smoke',1,subset,max_optimizer_updates=settings['pilot_updates'])
            emit('smoke',**profile)
        if args.mode=='pilot':emit('runtime_estimate',**pilot(settings,root,models,subset))
        if args.mode=='run':execute(settings,root,models,subset)
        if args.mode=='holdout':
            if not args.holdout_data:parser.error('--holdout-data is required')
            holdout(settings,root,args.holdout_data)


if __name__=='__main__':
    main()
