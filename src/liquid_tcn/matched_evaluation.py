"""Matched reset-inclusive, episode-long evaluation of frozen selected checkpoints."""
import argparse
from collections import defaultdict
import csv
from dataclasses import asdict
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import time
import zipfile

import numpy as np
import torch

from .checkpoint import load_checkpoint
from .episode_chunks import episode_start, episode_end
from .metadata import PROPERTY_FIELDS
from .model import ModelConfig, build_model, CachedInference, HEADS
from .normalization import Normalization
from .offline import horizon_errors
from .ported.storage import atomic_json
from .trajectories import Trajectories

HORIZONS = (1, 40, 200, 400, 1000)
UNITS = ('N', 'N m', 'm', 'm/s', 'kg m^2/s')
MOTION_ORDER = ('g_B', 'u_O_B', 'omega_B', 'held_a_B', 'held_alpha_B')
VIEWS = ('reset_inclusive', 'after_1p24s', 'startup_0_0p2s', 'middle_0p2_1p24s',
         'legacy_eligible', 'legacy_window_weighted') + tuple(f'elapsed_{i}_{i+1}s' for i in range(10))


def emit(event, **values):
    print(json.dumps(dict(event=event, **values)), flush=True)


def extract_archive(archive, destination):
    """Validate every member before extracting; install the new directory atomically."""
    archive, destination = Path(archive).resolve(), Path(destination).resolve()
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if destination.exists():
        marker = destination / '.extracted.json'
        if not marker.exists() or json.loads(marker.read_text())['archive'] != str(archive):
            raise FileExistsError(f'Choose an unused --older-dir; refusing to overwrite {destination}')
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        names = set()
        for info in z.infolist():
            path = PurePosixPath(info.filename)
            mode = info.external_attr >> 16
            if (path.is_absolute() or '..' in path.parts or '\\' in info.filename
                    or info.filename in names or stat.S_ISLNK(mode)
                    or (stat.S_IFMT(mode) and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)))):
                raise ValueError(f'Unsafe archive member: {info.filename}')
            names.add(info.filename)
        temporary = Path(tempfile.mkdtemp(prefix=destination.name+'-', dir=destination.parent))
        try:
            for info in z.infolist():
                output = temporary / info.filename
                if info.is_dir():
                    output.mkdir(parents=True, exist_ok=True)
                else:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(info) as src, output.open('xb') as dst:
                        shutil.copyfileobj(src, dst)
            atomic_json(temporary/'.extracted.json', dict(archive=str(archive), members=sorted(names)))
            temporary.rename(destination)
        finally:
            if temporary.exists():shutil.rmtree(temporary)
    return destination


def discover(older, focused):
    entries = []
    for path in sorted(older.rglob('best.pt')):
        selection = path.parent/'selection.json'
        if not selection.is_file():raise FileNotFoundError(selection)
        saved = load_checkpoint(path)
        entries.append(dict(family='older_tcn', seed=int(saved['config']['seed']), checkpoint=str(path.resolve()),
                            selection=str(selection.resolve())))
    if not entries:raise FileNotFoundError(f'No selected best.pt checkpoints in {older}')
    decisions = json.loads((focused/'decisions.json').read_text())
    for family, ids in decisions['finalists'].items():
        for key in ids:
            folder = focused/'final'/'trials'/key
            for name in ('best.pt', 'selection.json', 'trial.json'):
                if not (folder/name).is_file():raise FileNotFoundError(folder/name)
            cfg = json.loads((folder/'trial.json').read_text())
            entries.append(dict(family='focused_'+family, seed=int(cfg['seed']),
                                checkpoint=str((folder/'best.pt').resolve()), selection=str((folder/'selection.json').resolve())))
    for entry in entries:entry['name'] = f"{entry['family']}_seed{entry['seed']}"
    if len({e['name'] for e in entries}) != len(entries):raise ValueError('Duplicate family/seed checkpoint names')
    return entries


def load_selected(entry, manifest, device):
    saved = load_checkpoint(entry['checkpoint'])
    for key in ('model_config', 'model', 'normalization'):
        if key not in saved:raise ValueError(f"{entry['checkpoint']} is missing {key}")
    original = saved['model_config']
    config = dict(original)
    # Old format-1 TCNs predate these inference flags. No parameter or architecture
    # is replaced: enable the requested recorded-reset cache prefill explicitly.
    config.setdefault('temporal_memory', 'none')
    config.update(episode_training=True, reset_backfill='recorded')
    model = build_model(ModelConfig(**config)).to(device).eval()
    model.load_state_dict(saved['model'], strict=True)
    norm = Normalization(saved['normalization'])
    for field, shape in [('motion',(15,)), ('properties',(model.config.property_dim,)), ('targets',(5,3))]:
        for key in ('mean','std'):
            value = np.asarray(norm.stats[field][key])
            if value.shape != shape or not np.isfinite(value).all() or (key=='std' and (value<=0).any()):
                raise ValueError(f"Invalid checkpoint {field}/{key}: {entry['checkpoint']}")
    if hasattr(model, 'temporal'):
        model.temporal.residual_scale = saved['recurrent_weight']
    ordering = saved.get('property_fields', PROPERTY_FIELDS)
    indices = [manifest['property_fields'].index(field) for field in ordering]
    if len(indices) != model.config.property_dim:raise ValueError('Checkpoint property ordering has the wrong length')
    if saved.get('input_order', list(MOTION_ORDER)) != list(MOTION_ORDER):
        raise ValueError('Unsupported explicitly stored motion ordering')
    metadata = dict(entry, selected_optimizer_step=saved['update'], original_model_config=original,
                    selected_reference_updates=saved['exposure']/6400 if 'exposure' in saved else None,
                    training_configuration=saved['config'],
                    evaluation_model_config=asdict(model.config), normalization=norm.stats,
                    input_order=list(MOTION_ORDER), property_order=list(ordering), output_order=list(HEADS),
                    ordering_source='Explicit checkpoint metadata when present; otherwise fixed repository format-1 feature/head contract',
                    parameters=sum(p.numel() for p in model.parameters()))
    return model, norm, indices, metadata


def legacy_weights(rows, rf, length, stride):
    """Reproduce SeriesWriter's original eligible-window rule and label multiplicity."""
    run = 0; delta = np.zeros(len(rows)+1, dtype=np.int64)
    for i,row in enumerate(rows):
        contiguous = i>0 and int(row['tick']-rows['tick'][i-1])==stride
        run = run+1 if contiguous and row['interval_valid'] and row['state_valid'] and run else int(row['state_valid'])
        if run>=rf+length:
            delta[i-length]+=1;delta[i+1]-=1
    return np.cumsum(delta[:-1])


def prepare_common(reader, root, legacy_config):
    directory=root/'common';directory.mkdir(parents=True,exist_ok=True)
    episodes=[];excluded=[]
    for series in reader.series:
        reset=episode_start(reader,series)
        if reset is None:
            excluded.append(dict(series=series['id'],reason='Missing recorded reset or first sample is not the true reset'));continue
        end=episode_end(reader,series)
        if not end:
            excluded.append(dict(series=series['id'],reason='No valid contiguous input prefix / invalid physical assumptions'));continue
        rows=reader.read(series,0,end)
        native=reader.native_dt(series);stride=round(.005/native)
        times=(rows['tick']-rows['tick'][0])*native
        eligible=rows['target_valid'] & np.isfinite(rows['targets']).all(-1) & rows['state_valid'][:,None]
        eligible[:,:2] &= rows['interval_valid'][:,None]
        state=rows['state_valid'] & rows['target_valid'][:,2:].all(-1) & np.isfinite(rows['targets'][:,2:]).all((1,2))
        for field in ('x_O_W','u_O_W','omega_B','q_WB_wxyz','gravity_W'):
            state &= np.isfinite(rows[field]).all(-1)
        state &= np.isfinite(rows['mass']) & (rows['mass']>0) & (np.sum(rows['q_WB_wxyz']**2,-1)>1e-12)
        edges=np.zeros(end,dtype=bool)
        edges[1:] = (state[1:] & state[:-1] & rows['interval_valid'][1:] & rows['target_valid'][1:,:2].all(-1)
                     & np.isfinite(rows['targets'][1:,:2]).all((1,2)) & np.isclose(rows['h'][1:],.005,rtol=0,atol=1e-8)
                     & (np.diff(rows['tick'])==stride) & (rows['episode_id'][1:]==rows['episode_id'][:-1]))
        legacy=legacy_weights(rows,legacy_config.receptive_field,legacy_config.long_intervals,stride)
        path=directory/f"series-{series['id']:06d}.npz"
        if not path.exists():
            temporary=path.with_suffix('.tmp.npz')
            np.savez_compressed(temporary,ticks=rows['tick'],times=times,eligible=eligible,edges=edges,legacy_weights=legacy)
            temporary.replace(path)
        episodes.append(dict(series=series['id'],group=series['group'],end=end,original_rows=series['rows'],
                             reset_motion=reset.tolist(),complete_10_seconds=bool(times[-1]>=10-1e-8),
                             start_tick=int(rows['tick'][0]),end_tick=int(rows['tick'][-1]),native_dt=native))
    if not episodes:raise ValueError('No test episodes with true reset metadata and contiguous motion')
    result=dict(episodes=episodes,excluded=excluded,groups=len({e['group'] for e in episodes}),
                gap_or_invalid_input_tail_rows=sum(e['original_rows']-e['end'] for e in episodes),
                shared_target_scale=reader.manifest['normalization']['targets'],
                legacy_mask='Reconstructed original SeriesWriter RF+40 eligibility; union plus original overlapping-window endpoint multiplicities',
                prediction_policy='No prediction-dependent mask intersection. Missing eligible predictions invalidate the affected aggregate metric and are counted.',
                inference='Recorded-motion reset prefill; continuous cached 200 Hz inference; no resets at chunk boundaries',
                physics='Existing horizon_errors/discrete_constraints; endpoint frame interval wrenches; angular impulses/residuals shifted to each horizon-start base origin')
    atomic_json(directory/'cohort.json',result)
    return result


def reconstruction_acc():
    return dict(count=np.zeros(5,dtype=np.int64),missing=np.zeros(5,dtype=np.int64),
                squares=np.zeros((5,3)),normalized=np.zeros((5,3)),own_normalized=np.zeros((5,3)))


def physics_acc():
    return dict(count=0,missing=0,**{k:np.zeros(shape) for k,shape in
        [('predicted',(3,3)),('sph',(3,3)),('predicted_normalized',(3,3)),('sph_normalized',(3,3)),
         ('impulse',(2,3)),('impulse_normalized',(2,3))]})


def serializable(value):
    if isinstance(value,dict):return {k:serializable(v) for k,v in value.items()}
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    return value


def score_episode(reader, episode, prediction, common_path, scale, own_scale, chunk):
    series=reader.dataset.series[episode['series']];end=episode['end']
    with np.load(common_path) as m:mask={k:m[k] for k in m.files}
    rec={key:reconstruction_acc() for key in VIEWS}
    phys={view:{str(n*.005):physics_acc() for n in HORIZONS} for view in ('reset_inclusive','after_1p24s')}
    for pos in range(0,end,chunk):
        stop=min(end,pos+chunk);begin=max(0,pos-max(HORIZONS))
        rows=reader.read(series,begin,stop);p=np.asarray(prediction[begin:stop],dtype=np.float64)
        local=pos-begin;time_values=mask['times'][pos:stop]
        weights=dict(reset_inclusive=np.ones(stop-pos),after_1p24s=time_values>=1.24-1e-10,
                     startup_0_0p2s=time_values<.2-1e-10,middle_0p2_1p24s=(time_values>=.2-1e-10)&(time_values<1.24-1e-10),
                     legacy_eligible=mask['legacy_weights'][pos:stop]>0,legacy_window_weighted=mask['legacy_weights'][pos:stop])
        for i in range(10):weights[f'elapsed_{i}_{i+1}s']=(time_values>=i-1e-10)&(time_values<i+1-1e-10)&episode['complete_10_seconds']
        valid=mask['eligible'][pos:stop];finite=np.isfinite(p[local:]).all(-1)
        error=np.where(valid[...,None]&finite[...,None],p[local:]-rows['targets'][local:],0.)
        for key,w in weights.items():
            use=valid*w[:,None];a=rec[key]
            a['count']+=use.sum(0).astype(np.int64);a['missing']+=(use*~finite).sum(0).astype(np.int64)
            a['squares']+=(error**2*use[...,None]).sum(0)
            a['normalized']+=((error/scale)**2*use[...,None]).sum(0)
            a['own_normalized']+=((error/own_scale)**2*use[...,None]).sum(0)
        edges=mask['edges'][begin:stop].copy();edges[0]=False
        padded=np.r_[False,edges,False].astype(int)
        for a,b in zip(np.flatnonzero(np.diff(padded)==1),np.flatnonzero(np.diff(padded)==-1)):
            first=a-1;part=rows[first:b].copy();part['history_valid']=True
            estimates=p[first:b];bad=~np.isfinite(estimates).all((1,2))
            values,_=horizon_errors(part,np.nan_to_num(estimates,nan=0.,posinf=0.,neginf=0.),series['interior_dimensions'],part['x_O_W'][0],horizons=HORIZONS)
            prefix=np.r_[0,np.cumsum(bad)]
            for n,v in values.items():
                ends=v['ends'];global_ends=begin+first+ends;global_starts=global_ends-n
                missing=prefix[ends+1]-prefix[ends-n]>0
                for view in phys:
                    keep=global_ends>=pos
                    if view=='after_1p24s':keep &= mask['times'][global_starts]>=1.24-1e-10
                    dest=phys[view][str(n*.005)];dest['count']+=int(keep.sum());dest['missing']+=int((keep&missing).sum())
                    for key in ('predicted','sph','predicted_normalized','sph_normalized'):
                        dest[key]+=(v[key][keep]**2).sum(0)
                    impulse=np.stack((v['force'],v['moment']),axis=1)
                    f0=float(part['mass'][0])*np.linalg.norm(part['gravity_W'][0]);tau0=f0*np.linalg.norm(series['interior_dimensions'])
                    dest['impulse']+=(impulse[keep]**2).sum(0)
                    dest['impulse_normalized']+=((impulse[keep]/(v['duration'][keep,None,None]*np.array([f0,tau0])[None,:,None]))**2).sum(0)
    return serializable(dict(series=episode['series'],group=episode['group'],reconstruction=rec,physics=phys,
                             predicted_rows=end,missing_prediction_rows=int((~np.isfinite(prediction).all((1,2))).sum())))


def sum_acc(rows):
    return {k:sum((np.asarray(r[k]) for r in rows),np.zeros_like(np.asarray(rows[0][k]))) for k in rows[0]}


def aggregate(episodes):
    result=dict(reconstruction={},physics={},episodes=len(episodes),groups=len({e['group'] for e in episodes}),
                missing_prediction_rows=sum(e['missing_prediction_rows'] for e in episodes))
    for view in VIEWS:
        a=sum_acc([e['reconstruction'][view] for e in episodes]);heads={}
        for i,name in enumerate(HEADS):
            n=int(a['count'][i]);missing=int(a['missing'][i]);ok=n>0 and missing==0
            participating=[e for e in episodes if e['reconstruction'][view]['count'][i]>0]
            heads[name]=dict(units=UNITS[i],eligible_samples=n,missing_predictions=missing,
                             episodes=len(participating),groups=len({e['group'] for e in participating}),
                             axis_rms=np.sqrt(a['squares'][i]/n).tolist() if ok else None,
                             vector_rms=float(np.sqrt(a['squares'][i].sum()/n)) if ok else None)
        n=int(a['count'][:2].sum()*3);all_n=int(a['count'].sum()*3)
        result['reconstruction'][view]=dict(heads=heads,
            normalized_wrench_mse=float(a['normalized'][:2].sum()/n) if n and not a['missing'][:2].any() else None,
            shared_normalized_all_head_mse=float(a['normalized'].sum()/all_n) if all_n and not a['missing'].any() else None,
            own_normalized_all_head_mse=float(a['own_normalized'].sum()/all_n) if all_n and not a['missing'].any() else None)
    for view in ('reset_inclusive','after_1p24s'):
        result['physics'][view]={}
        for n in HORIZONS:
            key=str(n*.005);a=sum_acc([e['physics'][view][key] for e in episodes]);count=int(a['count']);missing=int(a['missing'])
            participating=[e for e in episodes if e['physics'][view][key]['count']>0]
            values=dict(count=count,missing_predictions=missing,episodes=len(participating),groups=len({e['group'] for e in participating}),
                        status='unavailable' if not count else 'missing_predictions' if missing else 'available')
            for field in ('predicted','sph','predicted_normalized','sph_normalized','impulse','impulse_normalized'):
                ok=count>0 and (not missing or field.startswith('sph'))
                values[field]=dict(axis_rms=np.sqrt(a[field]/count).tolist(),vector_rms=np.sqrt(a[field].sum(-1)/count).tolist()) if ok else None
            result['physics'][view][key]=values
    return result


@torch.inference_mode()
def smoke(entry, reader, episodes, manifest, device, chunk):
    model,norm,order,_=load_selected(entry,manifest,device)
    episode=next(e for e in episodes if e['end']>chunk+2)
    series=reader.dataset.series[episode['series']];length=min(episode['end'],chunk+17)
    rows=reader.read(series,0,length)
    props=torch.tensor(np.asarray(series['properties'])[order],dtype=torch.float32,device=device)[None]
    motion=torch.tensor(rows['motion'].copy(),dtype=torch.float32,device=device)[None]
    reset=torch.tensor(episode['reset_motion'],dtype=torch.float32,device=device)[None]
    normalized_props=norm.normalize('properties',props);normalized_reset=norm.normalize('motion',reset)
    history=reset[:,None].expand(-1,model.config.receptive_field-1,-1)
    inputs=torch.cat((history,motion),1)
    initial=model.temporal.initial_state(normalized_props,normalized_reset) if hasattr(model,'temporal') else ()
    expected,last=model(norm.normalize('motion',inputs),normalized_props,memory_state=initial,return_state=True)
    expected=norm.denormalize('targets',expected)
    state=initial;parts=[]
    for pos in range(0,length,chunk):
        sub=inputs[:,pos:pos+model.config.receptive_field-1+min(chunk,length-pos)]
        out,state=model(norm.normalize('motion',sub),normalized_props,memory_state=state,return_state=True)
        parts.append(norm.denormalize('targets',out))
    torch.testing.assert_close(torch.cat(parts,1),expected,rtol=3e-4,atol=3e-4)
    for a,b in zip(state,last):torch.testing.assert_close(a,b,rtol=3e-4,atol=3e-4)
    cache=CachedInference(model,norm,1);cache.set_properties(props);cache.initialize_reset(reset)
    actual=[];cursors=[]
    for pos in range(0,length,chunk):
        cursors.append(cache.cursor)
        for j in range(pos,min(length,pos+chunk)):
            value,valid=cache.step(motion[:,j],return_all=True)
            assert bool(valid.all());actual.append(value.clone())
    actual=torch.stack(actual,1)
    torch.testing.assert_close(actual,expected,rtol=3e-4,atol=3e-4)
    assert cursors[1]-cursors[0]==chunk
    np.testing.assert_array_equal(rows['tick'],episode['start_tick']+np.arange(length)*round(.005/episode['native_dt']))
    return dict(checkpoint=entry['name'],steps=length,chunk_boundary=chunk,
                maximum_absolute_difference=float((actual-expected).abs().max()),state_preserved=True,timestamps_aligned=True)


@torch.inference_mode()
def evaluate_checkpoint(entry, reader, cohort, output, args):
    directory=output/entry['name'];directory.mkdir(parents=True,exist_ok=True)
    per_episode=directory/'episodes';per_episode.mkdir(exist_ok=True)
    model,norm,order,metadata=load_selected(entry,reader.manifest,args.device)
    atomic_json(directory/'checkpoint.json',metadata)
    pending=[e for e in cohort['episodes'] if not (per_episode/f"{e['series']:06d}.json").exists()]
    started=time.perf_counter()
    emit('checkpoint_start',name=entry['name'],episodes_remaining=len(pending))
    scale=np.asarray(cohort['shared_target_scale']['std']);own_scale=np.asarray(norm.stats['targets']['std'])
    for offset in range(0,len(pending),args.batch_size):
        batch=pending[offset:offset+args.batch_size]
        properties=torch.tensor(np.array([np.asarray(reader.dataset.series[e['series']]['properties'])[order] for e in batch]),dtype=torch.float32,device=args.device)
        reset=torch.tensor(np.array([e['reset_motion'] for e in batch]),dtype=torch.float32,device=args.device)
        cache=CachedInference(model,norm,len(batch));cache.set_properties(properties);cache.initialize_reset(reset)
        # Per-episode disk-backed arrays bound RAM independently of episode length.
        with tempfile.TemporaryDirectory(prefix='predictions-',dir=directory) as work:
            predictions=[np.lib.format.open_memmap(Path(work)/f"{e['series']}.npy",mode='w+',dtype='float32',shape=(e['end'],5,3)) for e in batch]
            for pos in range(0,max(e['end'] for e in batch),args.chunk_size):
                width=min(args.chunk_size,max(e['end'] for e in batch)-pos)
                motions=np.empty((len(batch),width,15),dtype=np.float32)
                for i,e in enumerate(batch):
                    if pos<e['end']:
                        series=reader.dataset.series[e['series']]
                        values=reader.read(series,pos,min(e['end'],pos+width))['motion']
                        motions[i,:len(values)]=values;motions[i,len(values):]=values[-1]
                    else:motions[i]=e['reset_motion']  # Finished slots never contribute or get reused.
                device_motion=torch.tensor(motions,device=args.device)
                values=torch.empty((len(batch),width,5,3),device=args.device)
                for j in range(width):values[:,j]=cache.step(device_motion[:,j],return_all=True)[0]
                values=values.cpu().numpy()
                for i,e in enumerate(batch):
                    length=min(width,max(0,e['end']-pos))
                    if length:predictions[i][pos:pos+length]=values[i,:length]
            for e,prediction in zip(batch,predictions):
                result=score_episode(reader,e,prediction,output/'common'/f"series-{e['series']:06d}.npz",scale,own_scale,args.chunk_size)
                atomic_json(per_episode/f"{e['series']:06d}.json",result)
            for prediction in predictions:prediction.flush()
            del predictions
        emit('checkpoint_progress',name=entry['name'],completed=len(cohort['episodes'])-len(pending)+offset+len(batch),total=len(cohort['episodes']),seconds=round(time.perf_counter()-started,1))
    episodes=[json.loads((per_episode/f"{e['series']:06d}.json").read_text()) for e in cohort['episodes']]
    summary=aggregate(episodes)
    report=dict(checkpoint=metadata,summary=summary)
    atomic_json(directory/'report.json',report)
    write_csv(directory/'report.csv',[dict(metric=k,value=v) for k,v in flatten(summary).items()])
    emit('checkpoint_complete',name=entry['name'],mse=summary['reconstruction']['reset_inclusive']['normalized_wrench_mse'],missing=summary['missing_prediction_rows'])
    return report,episodes


def flatten(value, prefix=''):
    result={}
    if isinstance(value,dict):
        for k,v in value.items():result.update(flatten(v,f'{prefix}/{k}'.strip('/')))
    elif isinstance(value,list):
        for i,v in enumerate(value):result.update(flatten(v,f'{prefix}/{i}'))
    elif value is None or isinstance(value,(float,int)):
        result[prefix]=value
    return result


def write_csv(path, rows):
    if not rows:return
    with Path(path).with_suffix('.tmp.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    Path(path).with_suffix('.tmp.csv').replace(path)


def comparison(reports, episode_reports, root, repetitions):
    grouped=defaultdict(list)
    for report in reports:grouped[report['checkpoint']['family']].append(report)
    metrics={};table=[]
    for family,items in grouped.items():
        flat=[flatten(item['summary']) for item in items];values={}
        for key in flat[0]:
            xs=[f[key] for f in flat]
            if any(x is None for x in xs):continue
            a=np.asarray(xs,dtype=float)
            values[key]=dict(mean=float(a.mean()),seed_sd=float(a.std(ddof=1)) if len(a)>1 else None,n_seeds=len(a))
            table.append(dict(family=family,metric=key,**values[key]))
        metrics[family]=values
    atomic_json(root/'comparison.json',dict(families=metrics,seed_sd='Sample SD across the selected checkpoints; not a confidence interval'))
    write_csv(root/'comparison.csv',table)
    specs={}
    for view in VIEWS:
        if view=='legacy_window_weighted':continue
        specs[f'reconstruction/{view}/normalized_wrench_mse']=('reconstruction',view,'normalized',None,False)
        for i,name in enumerate(HEADS[:2]):specs[f'reconstruction/{view}/heads/{name}/vector_rms']=('reconstruction',view,'squares',i,True)
    for view in ('reset_inclusive','after_1p24s'):
        for h in HORIZONS:
            for field,names in [('predicted_normalized',('kinematic','linear','angular')),('impulse_normalized',('force','moment'))]:
                for i,name in enumerate(names):
                    specs[f'physics/{view}/{h*.005:g}s/{field}/{name}']=('physics',view,str(h*.005),field,i)
    groups=sorted({e['group'] for entries in episode_reports.values() for e in entries})
    rng=np.random.default_rng(1729)
    draws=rng.multinomial(len(groups),np.ones(len(groups))/len(groups),size=repetitions)
    lookup={}
    for family in grouped:
        by_group=defaultdict(list)
        for report in grouped[family]:
            for episode in episode_reports[report['checkpoint']['name']]:by_group[episode['group']].append(episode)
        lookup[family]=by_group
    boot={};boot_csv=[]
    for first,second in itertools.combinations(sorted(grouped),2):
        pair=f'{second}_minus_{first}';boot[pair]={}
        for key,spec in specs.items():
            outputs=[];invalid=False
            for family in (first,second):
                nums=[];dens=[];sqrt=False
                for group in groups:
                    entries=lookup[family][group]
                    if spec[0]=='reconstruction':
                        _,view,field,index,sqrt=spec
                        a=sum_acc([e['reconstruction'][view] for e in entries])
                        if index is None:
                            num=a[field][:2].sum();den=a['count'][:2].sum()*3;missing=a['missing'][:2].sum()
                        else:num=a[field][index].sum();den=a['count'][index];missing=a['missing'][index]
                    else:
                        _,view,h,field,index=spec;sqrt=True
                        a=sum_acc([e['physics'][view][h] for e in entries])
                        num=a[field][index].sum();den=a['count'];missing=a['missing']
                    invalid |= bool(missing);nums.append(num);dens.append(den)
                nums=np.asarray(nums);dens=np.asarray(dens)
                denominator=draws@dens
                if not dens.sum() or np.any(denominator==0):invalid=True
                values=(draws@nums)/np.maximum(1,denominator);point=nums.sum()/max(1,dens.sum())
                outputs.append((np.sqrt(values) if sqrt else values,math.sqrt(point) if sqrt else point))
            if invalid:boot[pair][key]=dict(status='unavailable');continue
            difference=outputs[1][0]-outputs[0][0];ci=np.quantile(difference,[.025,.975]).tolist()
            item=dict(difference=outputs[1][1]-outputs[0][1],ci95=ci)
            boot[pair][key]=item;boot_csv.append(dict(comparison=pair,metric=key,difference=item['difference'],ci95_low=ci[0],ci95_high=ci[1]))
    method=('Paired nonparametric collection-group bootstrap, 2000 draws by default (actual count below), seed 1729. '
            'Resample collection groups with replacement using the same draw for every model; retain all episodes and all three fixed seeds per group. '
            'Pool squared-error numerators and eligible counts before MSE/RMS; compute second-minus-first differences within each draw. '
            'Percentile 2.5/97.5 bounds. This measures held-out group variation conditional on the trained seeds; seed sample SD is reported separately. '
            'Groups contain all robot/environment episodes in that collection group. No test-based checkpoint selection.')
    atomic_json(root/'bootstrap.json',dict(method=method,draws=repetitions,groups=len(groups),differences=boot))
    write_csv(root/'bootstrap.csv',boot_csv)
    return metrics,boot


def plots(metrics, root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
    for ax,tail,label in zip(axes,['normalized_wrench_mse','heads/force/vector_rms','heads/moment/vector_rms'],['Normalized wrench MSE','Force vector RMS [N]','Moment vector RMS [N m]']):
        for family,values in metrics.items():
            keys=[f'reconstruction/elapsed_{i}_{i+1}s/{tail}' for i in range(10)]
            mean=[values.get(k,{}).get('mean',np.nan) for k in keys];sd=[values.get(k,{}).get('seed_sd',0.) for k in keys]
            ax.errorbar(np.arange(10)+.5,mean,yerr=sd,label=family,marker='.',capsize=2)
        ax.set(xlabel='Seconds from reset',ylabel=label);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.savefig(root/'reconstruction_vs_time.png',dpi=170);fig.savefig(root/'reconstruction_vs_time.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,5,figsize=(16,3.5),constrained_layout=True)
    descriptions=[('predicted_normalized',i,n) for i,n in enumerate(['Kinematic residual','Linear residual','Angular residual'])]+[('impulse_normalized',i,n) for i,n in enumerate(['Force impulse error','Angular impulse error'])]
    x=np.array(HORIZONS)*.005
    for ax,(field,i,label) in zip(axes,descriptions):
        for family,values in metrics.items():
            keys=[f'physics/reset_inclusive/{h}/'+field+f'/vector_rms/{i}' for h in map(str,x)]
            mean=[values.get(k,{}).get('mean',np.nan) for k in keys];sd=[values.get(k,{}).get('seed_sd',0.) for k in keys]
            ax.errorbar(x,mean,yerr=sd,label=family,marker='.',capsize=2)
        if field=='predicted_normalized':
            values=next(iter(metrics.values()))
            floor=[values.get(f'physics/reset_inclusive/{h}/sph_normalized/vector_rms/{i}',{}).get('mean',np.nan) for h in map(str,x)]
            ax.plot(x,floor,'k--',label='SPH consistency floor')
        ax.set(xlabel='Horizon [s]',ylabel='Normalized vector RMS',title=label,xscale='log');ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.savefig(root/'physics_vs_horizon.png',dpi=170);fig.savefig(root/'physics_vs_horizon.pdf');plt.close(fig)


def assessment(metrics, boots, cohort, root):
    def cell(family,key):
        v=metrics[family].get(key)
        return 'unavailable' if not v else f"{v['mean']:.6g} ± {v['seed_sd']:.3g}" if v['seed_sd'] is not None else f"{v['mean']:.6g}"
    lines=['# Matched sequential selected-checkpoint evaluation','',
           'All checkpoints were selected on validation before this evaluation. No retraining or test-based selection.',
           '',f"Common test cohort: {len(cohort['episodes'])} episodes in {cohort['groups']} collection groups. "
           f"Complete 10-second cohort: {sum(e['complete_10_seconds'] for e in cohort['episodes'])} episodes. "
           f"Excluded episodes: {len(cohort['excluded'])}; excluded input-gap/invalid-input tail rows: {cohort['gap_or_invalid_input_tail_rows']}.",
           '', 'Normalized wrench MSE uses the same prepared-training target scales for every model. Values are mean ± sample seed SD.',
           '', '| Model | Reset-inclusive | 0–200 ms | 200 ms–1.24 s | After 1.24 s | Older eligible timestamps |',
           '|---|---:|---:|---:|---:|---:|']
    views=['reset_inclusive','startup_0_0p2s','middle_0p2_1p24s','after_1p24s','legacy_eligible']
    for family in metrics:
        lines.append('| '+family+' | '+' | '.join(cell(family,f'reconstruction/{view}/normalized_wrench_mse') for view in views)+' |')
    lines+=['','| Model | Force axis x/y/z RMS [N] | Force vector RMS [N] | Moment axis x/y/z RMS [N m] | Moment vector RMS [N m] |','|---|---|---:|---|---:|']
    for family in metrics:
        prefix='reconstruction/reset_inclusive/heads/'
        lines.append('| '+family+' | '+' / '.join(cell(family,prefix+f'force/axis_rms/{i}') for i in range(3))+' | '+cell(family,prefix+'force/vector_rms')+' | '+' / '.join(cell(family,prefix+f'moment/axis_rms/{i}') for i in range(3))+' | '+cell(family,prefix+'moment/vector_rms')+' |')
    lines+=['','## Interpretation','']
    for newer in ('focused_tcn','focused_hybrid'):
        statements=[]
        for view,label in [('reset_inclusive','overall'),('startup_0_0p2s','first 200 ms'),('after_1p24s','after 1.24 s')]:
            key=f'reconstruction/{view}/normalized_wrench_mse'
            old=metrics['older_tcn'].get(key);new=metrics[newer].get(key)
            if old and new:
                relative=(old['mean']/new['mean']-1)*100
                statements.append(f"{label}: older TCN MSE is {abs(relative):.1f}% {'lower' if relative<0 else 'higher'}")
        lines.append(f"Against {newer}, "+'; '.join(statements)+'.')
    lines+=['','Paired collection-group bootstrap intervals for overall MSE (difference in the named order):','']
    key='reconstruction/reset_inclusive/normalized_wrench_mse'
    for pair,values in boots.items():
        item=values[key]
        if 'ci95' in item:lines.append(f"- {pair}: {item['difference']:.6g}, 95% CI [{item['ci95'][0]:.6g}, {item['ci95'][1]:.6g}].")
    lines+=['','Older archived selection reports are full-validation, overlapping-window, all-five-head objectives. '
            'They are not the same population, weighting, scale, or head subset as chronological test wrench MSE. '
            'The legacy_eligible view uses the union of original eligible windows; legacy_window_weighted reproduces '
            'endpoint multiplicity and also reports each checkpoint’s own normalized all-head MSE. These are diagnostics, not checkpoint selection.',
            '', 'This matches evaluation conditions, not training exposure. Older checkpoints used the original overlapping-window training procedure; '
            'the focused finalists used chronological exposure-normalized training. Selected steps and saved training settings appear in each checkpoint report. '
            'The result does not isolate architecture from training procedure or budget.',
            '', 'Physical self-consistency is separate from accuracy against SPH. Residual units are m, kg m/s and kg m²/s; '
            'force/angular impulse errors use N s and N m s. The SPH consistency floor uses the identical recorded intervals and frame conventions. '
            'No unsupported horizon is extrapolated. Residual normalization uses T·sqrt(gL), T·mg, T·mgL; impulse normalization uses T·mg and T·mgL.',
            '', 'Every model uses the same prediction-independent masks. Missing eligible predictions make affected aggregate scores unavailable; '
            'counts are retained, never reduced to improve a score. All five heads and validity/sample/episode/group counts are in each report.json/report.csv. '
            'Elapsed bins always use the same complete 10-second cohort; the exact 10 s endpoint is outside the [9,10) bin.',
            '', '[Reconstruction plot](reconstruction_vs_time.png) · [Physics/impulse plot](physics_vs_horizon.png) · '
            '[Mean/seed-SD CSV](comparison.csv) · [Paired bootstrap CSV](bootstrap.csv).',
            '', 'Inference is rerun for all checkpoints because archived newer evaluation files contain summary statistics, not compatible all-head predictions. '
            'Motion/property ordering and the original normalization are recorded per checkpoint. Only the explicit episode-training/reset flags are '
            'enabled for legacy cache prefill; strict weight loading remains unchanged. Completed per-episode metrics are resumable; an interrupted '
            'episode is replayed from its actual reset. No existing checkpoint or search result is modified.']
    (root/'assessment.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-zip',type=Path,default=Path('runs/pyramid_film_full_selected_checkpoints.zip'))
    parser.add_argument('--older-dir',type=Path,default=Path('runs/imported_pyramid_film_selected'))
    parser.add_argument('--focused-root',type=Path,default=Path('runs/focused_architecture_search'))
    parser.add_argument('--data',type=Path)
    parser.add_argument('--output',type=Path,default=Path('runs/matched_sequential_evaluation'))
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--chunk-size',type=int,default=400)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--bootstrap',type=int,default=2000)
    parser.add_argument('--smoke-only',action='store_true')
    args=parser.parse_args()
    if min(args.batch_size,args.chunk_size,args.threads,args.bootstrap)<1:parser.error('Sizes must be positive')
    torch.set_num_threads(args.threads)
    settings=json.loads((args.focused_root/'search.json').read_text())
    data=args.data or Path(settings['data'])
    older=extract_archive(args.legacy_zip,args.older_dir)
    entries=discover(older,args.focused_root)
    reader=Trajectories(data,'test',block_rows=args.chunk_size)
    root=args.output;root.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (root/'.evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=dict(data=str(data.resolve()),checkpoints=entries,protocol='matched-recorded-reset-v1')
        path=root/'plan.json'
        if path.exists() and json.loads(path.read_text())!=plan:
            raise ValueError('Use a new output directory for a different checkpoint/data selection')
        atomic_json(path,plan)
        legacy_model,_,_,_=load_selected(entries[0],reader.manifest,'cpu')
        cohort=prepare_common(reader,root,legacy_model.config);del legacy_model
        emit('cohort',episodes=len(cohort['episodes']),groups=cohort['groups'],excluded=len(cohort['excluded']),checkpoints=len(entries))
        if not (root/'smoke.json').exists():
            seen=set();checks=[]
            for entry in entries:
                if entry['family'] in seen:continue
                checks.append(smoke(entry,reader,cohort['episodes'],reader.manifest,args.device,args.chunk_size))
                seen.add(entry['family']);emit('smoke',**checks[-1])
            atomic_json(root/'smoke.json',checks)
        if args.smoke_only:return
        reports=[];episodes={}
        for entry in entries:
            report,records=evaluate_checkpoint(entry,reader,cohort,root,args)
            reports.append(report);episodes[entry['name']]=records
        metrics,boots=comparison(reports,episodes,root,args.bootstrap)
        plots(metrics,root);assessment(metrics,boots,cohort,root)
        atomic_json(root/'complete.json',dict(checkpoints=len(reports),episodes=len(cohort['episodes']),groups=cohort['groups']))
        emit('complete',output=str(root),checkpoints=len(reports))


if __name__=='__main__':
    main()
