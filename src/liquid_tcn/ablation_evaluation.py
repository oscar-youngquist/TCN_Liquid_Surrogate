"""Matched reset-inclusive cached inference and sliding Newton–Euler assessment."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import torch
from .checkpoint import load_checkpoint
from .model import build_model, ModelConfig, HEADS, CachedInference
from .normalization import Normalization
from .trajectories import Trajectories
from .offline import horizon_errors
from .ported.storage import atomic_json
from .sweep import event

HORIZONS = (1, 40, 200, 400, 1000)
SCOPES = ('reset_inclusive', 'startup', 'after_history')
PERIODS = ('overall', '0_200ms', '200ms_history', 'after_history')


def load_model(checkpoint, device):
    saved = load_checkpoint(checkpoint)
    model = build_model(ModelConfig(**saved['model_config'])).to(device).eval()
    model.load_state_dict(saved['model'], strict=True)
    return model, Normalization(saved['normalization'])


def input_context(rows, stride, receptive_field):
    """Backfill is allowed only at the initial verified reset, never after a gap."""
    result = np.zeros(len(rows), dtype=bool)
    prefix = True; run = 0
    for i, row in enumerate(rows):
        good = bool(np.isfinite(row['motion']).all())
        contiguous = i == 0 or (row['tick']-rows['tick'][i-1] == stride and row['episode_id']==rows['episode_id'][i-1])
        prefix = prefix and good and contiguous
        run = run+1 if good and contiguous else int(good)
        result[i] = good and (prefix or run >= receptive_field)
    return result


def prepare_protocol(data, startup_index, destination, model_config, normalization, historical=()):
    """Freeze model-independent episodes, timestamps and masks once for all runs."""
    destination = Path(destination); path = destination/'protocol.json'
    if path.exists(): return json.loads(path.read_text())
    destination.mkdir(parents=True, exist_ok=True)
    reader = Trajectories(data, 'test')
    resets = json.loads((Path(startup_index)/'index.json').read_text())['resets']
    model = ModelConfig(**model_config)
    rf = max([model.receptive_field] + [ModelConfig(**load_checkpoint(p)['model_config']).receptive_field for p in historical])
    entries = []; exclusions = []; cohort = []
    for series in reader.series:
        sid = series['id']; reset = resets.get(str(sid))
        if reset is None:
            exclusions.append(dict(series=sid, reason='no verified recorded reset with available startup endpoints')); continue
        rows = reader.read(series, 0, series['rows'])
        if not len(rows) or int(rows['tick'][0]) != reset['tick']:
            raise ValueError(f'Series {sid}: reset endpoint is absent')
        stride = round(model.sample_seconds/reset['native_dt'])
        times = (rows['tick']-reset['tick'])*reset['native_dt']
        context = input_context(rows, stride, rf)
        targets = rows['target_valid'] & np.isfinite(rows['targets']).all(-1) & context[:,None]
        states = rows['state_valid'] & targets[:,2:].all(-1)
        for name in ('x_O_W','u_O_W','omega_B','q_WB_wxyz','mass','gravity_W'):
            finite = np.isfinite(rows[name]); states &= finite.reshape(len(rows),-1).all(-1)
        states &= (rows['mass']>0) & (np.linalg.norm(rows['q_WB_wxyz'],axis=-1)>1e-6)
        edges = np.zeros(len(rows),dtype=bool)
        edges[1:] = (states[1:] & states[:-1] & targets[1:,:2].all(-1) & rows['interval_valid'][1:]
            & (np.diff(rows['tick'])==stride) & (rows['episode_id'][1:]==rows['episode_id'][:-1])
            & np.isclose(rows['h'][1:],model.sample_seconds,rtol=0,atol=1e-8)
            & np.isclose(rows['mass'][1:], rows['mass'][:-1],rtol=1e-7,atol=1e-9)
            & np.isclose(rows['gravity_W'][1:],rows['gravity_W'][:-1],rtol=1e-7,atol=1e-9).all(-1))
        if not series.get('assumptions_valid',True): edges[:]=False
        reaches = bool(times[-1]>=10.-1e-9 and np.isfinite(rows['motion'][times<=10.+1e-9]).all()
                       and np.all(np.diff(rows['tick'][times<=10.+1e-9])==stride)
                       and np.all(rows['episode_id'][times<=10.+1e-9]==rows['episode_id'][0]))
        if reaches: cohort.append(sid)
        name = f'series-{sid:06d}.npz'
        np.savez_compressed(destination/name,ticks=rows['tick'],times=times,context=context,target_valid=targets,edges=edges)
        entries.append(dict(series=sid,group=series['group'],rows=len(rows),mask=name,reset=reset,stride=stride,
            profile_cohort=reaches,valid_head_counts=targets.sum(0).tolist(),valid_intervals=int(edges.sum()),
            excluded_input_context=int((~context).sum()),excluded_intervals=int(len(rows)-1-edges.sum())))
    if not entries: raise ValueError('No verified-reset test episodes available')
    value = dict(data=str(Path(data).resolve()),episodes=entries,exclusions=exclusions,profile_cohort=cohort,
        comparison_normalization=normalization,history_seconds=(model.receptive_field-1)*model.sample_seconds,
        context_receptive_field=rf,sample_seconds=model.sample_seconds,horizons_steps=list(HORIZONS),
        convention='Recorded reset input repeated before normalization; persistent cached inference; fixed endpoint timestamps and masks; sliding horizons once per endpoint; no gaps bridged',
        profile_definition='Same recorded episodes reaching 10 seconds with continuous finite input through 10 seconds; shorter episodes remain in overall metrics')
    atomic_json(path,value);return value


def segments(edges):
    """Half-open row runs; each true edge connects row i-1 to i."""
    ids=np.flatnonzero(edges)
    if not len(ids): return
    for group in np.split(ids,np.flatnonzero(np.diff(ids)!=1)+1):
        yield int(group[0]-1),int(group[-1]+1)


def metric_specs():
    result={}
    for period in (*PERIODS,*(f'profile_{i}_{i+1}s' for i in range(10))):
        for head in HEADS:
            for axis in ('x','y','z','vector'):result[f'reconstruction/{period}/{head}/{axis}']='rms'
        for kind in ('wrench','all_heads'):result[f'reconstruction/{period}/normalized_{kind}_mse']='mse'
    for scope in SCOPES:
        for h in HORIZONS:
            for source in ('predicted','sph'):
                for channel in ('r','P','H'):
                    for axis in ('x','y','z','vector','normalized'):
                        result[f'consistency/{scope}/{h*5}ms/{source}/{channel}/{axis}']='rms'
            for head in ('force','moment'):
                for axis in ('x','y','z','vector'):result[f'impulse/{scope}/{h*5}ms/{head}/{axis}']='rms'
    return result


def empty_metrics():return {key:dict(n=0,square=0.,reduction=value) for key,value in metric_specs().items()}


def add_square(metrics,key,values):
    values=np.asarray(values,dtype=np.float64)
    if not np.isfinite(values).all():raise FloatingPointError(f'Nonfinite scored values: {key}')
    row=metrics[key];row['n']+=len(values);row['square']+=float(np.square(values).sum())


def add_vector(metrics,key,values):
    add_square(metrics,key+'/vector',values)
    for i,axis in enumerate('xyz'):add_square(metrics,key+'/'+axis,values[:,i])


def merge_metrics(destination,source):
    for key,row in source.items():
        target=destination.setdefault(key,dict(n=0,square=0.,reduction=row['reduction']))
        target['n']+=row['n'];target['square']+=row['square']


def metric_value(row):
    if not row['n']:return None
    mean=row['square']/row['n']
    return mean if row['reduction']=='mse' else float(np.sqrt(mean))


def reconstruct(metrics,rows,prediction,mask,times,history,scale,cohort):
    bins={'overall':np.ones(len(rows),bool),'0_200ms':times<.2,
          '200ms_history':(times>=.2)&(times<history),'after_history':times>=history}
    if cohort:
        bins.update({f'profile_{i}_{i+1}s':(times>=i)&(times<(i+1) if i<9 else times<=10.+1e-9) for i in range(10)})
    for period,selected in bins.items():
        for h,head in enumerate(HEADS):
            use=mask[:,h]&selected
            add_vector(metrics,f'reconstruction/{period}/{head}',prediction[use,h]-rows['targets'][use,h])
        for name,stop in (('wrench',2),('all_heads',5)):
            use=mask[:,:stop]&selected[:,None]
            error=(prediction[:,:stop]-rows['targets'][:,:stop])/scale[:stop]
            add_square(metrics,f'reconstruction/{period}/normalized_{name}_mse',error[use].reshape(-1))


@torch.inference_mode()
def infer_batch(reader,entries,model,norm,protocol,output,device,chunk_size=400):
    """Persistent per-environment rings survive chunk boundaries and independent gaps."""
    n=len(entries);cache=CachedInference(model,norm,n)
    series=[reader.dataset.series[e['series']] for e in entries]
    props=torch.tensor([s['properties'] for s in series],dtype=torch.float32,device=device)
    cache.set_properties(props)
    reset=torch.tensor([e['reset']['motion'] for e in entries],dtype=torch.float32,device=device)
    cache.initialize_reset(reset)
    previous=[None]*n;outputs=[]
    for e in entries:
        outputs.append(np.lib.format.open_memmap(output/f"prediction-{e['series']:06d}.tmp.npy",mode='w+',dtype='float32',shape=(e['rows'],5,3)))
    for begin in range(0,max(e['rows'] for e in entries),chunk_size):
        blocks=[reader.read(s,begin,min(e['rows'],begin+chunk_size)) if begin<e['rows'] else None for s,e in zip(series,entries)]
        length=max(len(b) for b in blocks if b is not None)
        motions=np.zeros((length,n,15),np.float32);bad=np.zeros((length,n),bool);reset_flags=np.zeros_like(bad)
        for env,block in enumerate(blocks):
            if block is None:continue
            for j,row in enumerate(block):
                prior=previous[env]
                reset_flags[j,env]=prior is not None and (row['tick']-prior[0]!=entries[env]['stride'] or row['episode_id']!=prior[1])
                bad[j,env]=not np.isfinite(row['motion']).all()
                if not bad[j,env]:motions[j,env]=row['motion']
                previous[env]=(int(row['tick']),int(row['episode_id']))
        motion=torch.tensor(motions,device=device);predicted=torch.empty((length,n,5,3),device=device)
        for j in range(length):
            resets=np.flatnonzero(reset_flags[j]|bad[j])
            if len(resets):cache.reset(resets.tolist())
            value,_=cache.step(motion[j],return_all=True);predicted[j]=value
            invalid=np.flatnonzero(bad[j])
            if len(invalid):cache.reset(invalid.tolist())  # Synthetic placeholder never becomes real history.
        values=predicted.cpu().numpy()
        for env,block in enumerate(blocks):
            if block is not None:outputs[env][begin:begin+len(block)]=values[:len(block),env]
    for output_array in outputs:output_array.flush()
    outputs.clear()
    for entry in entries:
        sid=entry['series'];(output/f'prediction-{sid:06d}.tmp.npy').replace(output/f'prediction-{sid:06d}.npy')


def score_episode(reader,entry,prediction,protocol,protocol_root,chunk_size=400):
    sid=entry['series'];series=reader.dataset.series[sid]
    metrics=empty_metrics();history=protocol['history_seconds']
    scale=np.asarray(protocol['comparison_normalization']['targets']['std'])
    with np.load(Path(protocol_root)/entry['mask']) as masks:
        times=masks['times'];valid=masks['target_valid'];edges=masks['edges'];ticks=masks['ticks'];context=masks['context']
    missing=np.flatnonzero(context & ~np.isfinite(prediction).all((1,2)))
    if len(missing):raise FloatingPointError(f'Series {sid}: {len(missing)} missing/nonfinite predictions at common scoring timestamps; first row {missing[0]}')
    for begin in range(0,entry['rows'],chunk_size):
        end=min(entry['rows'],begin+chunk_size);rows=reader.read(series,begin,end)
        if not np.array_equal(rows['tick'],ticks[begin:end]):raise ValueError(f'Series {sid}: scoring timestamps changed')
        reconstruct(metrics,rows,np.asarray(prediction[begin:end],dtype=float),valid[begin:end],times[begin:end],history,scale,entry['profile_cohort'])
    for start,end in segments(edges):
        for cursor in range(start+1,end,chunk_size):
            begin=max(start,cursor-max(HORIZONS));stop=min(end,cursor+chunk_size)
            rows=reader.read(series,begin,stop).copy();rows['history_valid']=True
            predicted=np.asarray(prediction[begin:stop],dtype=float)
            horizons,_=horizon_errors(rows,predicted,series['interior_dimensions'],rows['x_O_W'][0],HORIZONS)
            for h,terms in horizons.items():
                global_end=begin+terms['ends'];once=global_end>=cursor
                for scope in SCOPES:
                    use=once.copy()
                    if scope=='after_history':use &= times[global_end-h]>=history
                    elif scope=='startup':use &= times[global_end]<history
                    for source in ('predicted','sph'):
                        for c,channel in enumerate(('r','P','H')):
                            key=f'consistency/{scope}/{h*5}ms/{source}/{channel}'
                            add_vector(metrics,key,terms[source][use,c])
                            add_square(metrics,key+'/normalized',terms[source+'_normalized'][use,c])
                    for head in ('force','moment'):add_vector(metrics,f'impulse/{scope}/{h*5}ms/{head}',terms[head][use])
    return dict(series=sid,group=entry['group'],rows=entry['rows'],profile_cohort=entry['profile_cohort'],
        valid_head_counts=entry['valid_head_counts'],valid_intervals=entry['valid_intervals'],missing_predictions=0,
        excluded_input_context=entry['excluded_input_context'],excluded_intervals=entry['excluded_intervals'],metrics=metrics)


def evaluate_checkpoint(data,checkpoint,protocol_root,output,device='cpu',batch_size=64,chunk_size=400,threads=2):
    """Resume at completed episodes; incomplete inference batches restart from true resets."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    summary=output/'summary.json'
    if summary.exists():return json.loads(summary.read_text())
    torch.set_num_threads(threads)
    protocol=json.loads((Path(protocol_root)/'protocol.json').read_text())
    reader=Trajectories(data,'test');model,norm=load_model(checkpoint,device)
    pending=[e for e in protocol['episodes'] if not (output/f"episode-{e['series']:06d}.json").exists()]
    for begin in range(0,len(pending),batch_size):
        entries=pending[begin:begin+batch_size]
        inference=[e for e in entries if not (output/f"prediction-{e['series']:06d}.npy").exists()]
        if inference:infer_batch(reader,inference,model,norm,protocol,output,device,chunk_size)
        for entry in entries:
            path=output/f"prediction-{entry['series']:06d}.npy"
            predictions=np.load(path,mmap_mode='r')
            try:record=score_episode(reader,entry,predictions,protocol,protocol_root,chunk_size)
            except Exception as error:
                atomic_json(output/'failure.json',dict(series=entry['series'],error=str(error),status='failed'));raise
            atomic_json(output/f"episode-{entry['series']:06d}.json",record)
            del predictions;path.unlink()
        event('episode_evaluation',checkpoint=str(checkpoint),completed=min(begin+batch_size,len(pending)),remaining=max(0,len(pending)-begin-batch_size))
    overall=empty_metrics();groups={};coverage=defaultdict(int)
    for entry in protocol['episodes']:
        record=json.loads((output/f"episode-{entry['series']:06d}.json").read_text())
        merge_metrics(overall,record['metrics']);merge_metrics(groups.setdefault(record['group'],{}),record['metrics'])
        for key in ('rows','valid_intervals','missing_predictions','excluded_input_context','excluded_intervals'):coverage[key]+=record[key]
    result=dict(status='complete',checkpoint=str(checkpoint),split='test',metrics={k:dict(v,value=metric_value(v)) for k,v in overall.items()},groups=groups,
        coverage=dict(coverage,episodes=len(protocol['episodes']),collection_groups=len(groups),profile_episodes=len(protocol['profile_cohort'])),
        exclusions=protocol['exclusions'],history_seconds=protocol['history_seconds'],model_history_seconds=(model.config.receptive_field-1)*model.config.sample_seconds,
        parameters=sum(p.numel() for p in model.parameters()),
        convention=protocol['convention'],profile_definition=protocol['profile_definition'],horizons_steps=list(HORIZONS),
        physics_definition='All sliding horizons, once per end timestamp; interval averages rotated to world; angular impulse shifted to window-start origin; same valid intervals for predictions and SPH; after_history requires window start at/after history boundary',
        normalization='One frozen training-derived target standard deviation for scoring all models; each checkpoint retains its own internal normalization')
    atomic_json(summary,result);return result
