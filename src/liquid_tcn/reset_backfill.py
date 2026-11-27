"""Opt-in reset-context windows, consumed-position sampling and startup diagnostics."""
import argparse
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from .dataset import WindowDataset, EpochSampler, INDEX_DTYPE, prepared_model_config, make_loader, close_loader, to_device
from .model import ModelConfig, HEADS
from .ported.storage import atomic_json


def recorded_resets(path):
    """Read explicit reset events, never infer a reset from a file/episode start."""
    path=Path(path)
    if not path.is_dir():raise FileNotFoundError(f'Reset metadata source is missing: {path}')
    boundary=sorted((path/'boundaries').glob('chunk-*.npz'))
    primary=sorted(path.glob('chunk-*.npz'))
    records={}
    # Completed collection boundaries retain reset events; older streams without
    # that side stream require scanning their primary chunks.
    for chunk in [*(primary[:1] if boundary else primary),*boundary]:
        with np.load(chunk) as values:
            if 'reset' not in values:continue
            for i,env in zip(*np.nonzero(values['reset'] & values['episode_selected'])):
                records[(int(env),int(values['episode_id'][i,env]))]=int(values['native_step'][i])
    native=json.loads((path/'resolved.json').read_text())['native_dt']
    return records,native


def build_startup_index(data, destination=None):
    """Write indices/reset metadata only; numerical shards and normalization are reused."""
    from .trajectories import Trajectories
    data=Path(data);destination=Path(destination) if destination else data/'reset_backfill'
    if (destination/'index.json').exists():return json.loads((destination/'index.json').read_text())
    destination.mkdir(parents=True,exist_ok=True)
    reader=Trajectories(data,'train')
    model=prepared_model_config(reader.manifest);r=model.receptive_field;k=model.long_intervals
    records={};episodes={s:[] for s in ('train','validation','test')};excluded=[];resets={}
    sources={}
    for series in reader.manifest['series']:
        sid=series['id'];source=series['source']
        if not series.get('assumptions_valid',True):
            excluded.append(dict(series=sid,reason='invalid physical assumptions'));continue
        reset=series.get('recorded_reset')
        if reset is None:
            if source not in sources:
                try:sources[source]=recorded_resets(reader.manifest['sources'][source]['path'])
                except FileNotFoundError as error:sources[source]=str(error)
            value=sources[source]
            if isinstance(value,str):
                excluded.append(dict(series=sid,reason=value));continue
            ticks,native=value;tick=ticks.get((series['environment'],series['episode']))
            if tick is None:
                excluded.append(dict(series=sid,reason='no recorded reset event'));continue
            reset=dict(tick=tick,native_dt=native)
        width=min(series['rows'],r-1+k)
        rows=reader.read(series,0,width)
        if not len(rows) or int(rows['tick'][0])!=reset['tick']:
            excluded.append(dict(series=sid,reason='first prepared endpoint is not the recorded reset'));continue
        motion=np.asarray(reset.get('motion',rows['motion'][0]),dtype=np.float64)
        if not np.isfinite(motion).all():
            excluded.append(dict(series=sid,reason='nonfinite reset input'));continue
        good=np.isfinite(rows['motion']).all(-1)
        good[1:] &= (np.diff(rows['tick'])==round(model.sample_seconds/reset['native_dt'])) & (rows['episode_id'][1:]==rows['episode_id'][:-1])
        broken=np.flatnonzero(~good);end=int(broken[0]) if len(broken) else len(rows)
        ages=np.arange(max(0,min(r-1,end-k)))
        # Keep partial target masks, but omit windows with no learnable target at all.
        present=rows['target_valid'].any(-1)
        totals=np.r_[0,np.cumsum(present)]
        ages=ages[(totals[ages+k+1]-totals[ages])>0]
        if not len(ages):
            excluded.append(dict(series=sid,reason='insufficient real endpoints before a gap/end'));continue
        entries=np.zeros(len(ages),dtype=INDEX_DTYPE);entries['series']=sid;entries['end']=ages+k
        records.setdefault(series['split'],[]).append(entries)
        offset=sum(x['count'] for x in episodes[series['split']])
        episodes[series['split']].append(dict(series=sid,offset=offset,count=len(ages),group=series['group']))
        resets[str(sid)]=dict(tick=int(reset['tick']),motion=motion.tolist(),native_dt=reset['native_dt'])
    counts={}
    for split in episodes:
        entries=np.concatenate(records[split]) if split in records else np.empty(0,dtype=INDEX_DTYPE)
        temp=destination/f'{split}.tmp.idx';entries.tofile(temp);temp.replace(destination/f'{split}.idx');counts[split]=len(entries)
    result=dict(receptive_field=r,physics_intervals=k,sample_seconds=model.sample_seconds,
                windows=counts,episodes=episodes,resets=resets,excluded=excluded,
                definition='First supervised age s < R-1; repeat reset motion only at negative input indices; all K+1 endpoints are real')
    atomic_json(destination/'index.json',result)
    return result


class BackfilledWindows(WindowDataset):
    def __init__(self,data,split='train',cache_size=8,startup_index=None,startup_only=False):
        super().__init__(data,split,cache_size)
        self.ordinary_length=0 if startup_only else self.length
        self.startup_path=Path(startup_index) if startup_index else self.path/'reset_backfill'
        info=self.startup_path/'index.json'
        if not info.exists():
            raise FileNotFoundError(f'{info} is missing. Run python -m liquid_tcn.reset_backfill prepare --data {data}')
        self.startup_info=json.loads(info.read_text())
        if (self.startup_info['receptive_field'],self.startup_info['physics_intervals'])!=(self.rf,self.input_length-self.rf):
            raise ValueError('Startup index must match the prepared receptive field and physics interval count')
        self.startup_length=self.startup_info['windows'][split]
        if not self.startup_length:raise ValueError(f'No genuine-reset startup windows in {split}; see {info}')
        self.length=self.ordinary_length+self.startup_length
        self.startup_index=None
        self.resets={int(s):np.asarray(v['motion'],dtype=np.float32) for s,v in self.startup_info['resets'].items()}
        self.episodes=self.startup_info['episodes'][split]

    def _local(self):
        previous=self.pid
        super()._local()
        if self.pid!=previous:self.startup_index=None
        if self.startup_index is None:
            self.startup_index=np.memmap(self.startup_path/f'{self.split}.idx',dtype=INDEX_DTYPE,mode='r')

    def __getstate__(self):
        state=super().__getstate__();state['startup_index']=None;return state

    def __getitems__(self,indices):
        self._local()
        indices=np.asarray(indices,dtype=np.int64)
        if not len(indices) or np.any((indices<0)|(indices>=self.length)):raise IndexError(indices)
        startup=indices>=self.ordinary_length
        entries=np.empty(len(indices),dtype=INDEX_DTYPE)
        entries[~startup]=self.index[indices[~startup]]
        entries[startup]=self.startup_index[indices[startup]-self.ordinary_length]
        result=self._gather_entries(entries,self.resets)
        batch=result.batch
        batch['startup']=torch.from_numpy(startup)
        ages=entries['end']-(self.input_length-self.rf)
        batch['reset_age']=torch.from_numpy(np.where(startup,ages,-1).copy())
        context=batch['history_valid'].clone();context[startup]=True
        batch['input_context_valid']=context
        batch['supervised_age']=torch.from_numpy(ages[:,None]+np.arange(self.input_length-self.rf+1))
        return result

    def validation_indices(self,ages_per_episode=16):
        return [self.ordinary_length+e['offset']+int(i) for e in self.episodes
                for i in np.unique(np.linspace(0,e['count']-1,min(e['count'],ages_per_episode),dtype=int))]


def affine(n,seed):
    rng=random.Random(seed);a=rng.randrange(1,n+1)
    while math.gcd(a,n)!=1:a=a%n+1
    return a,rng.randrange(n)


class MixedWindowSampler(EpochSampler):
    """Fixed per-batch mix, shuffled slots, episode-balanced random startup ages.

    All iterator cursors are local. Only advance()/observe() mutate consumed
    state, so DataLoader prefetch never advances a saved sampling position.
    """
    def __init__(self,dataset,fraction,batch_size,seed):
        if not 0<fraction<1:raise ValueError('startup_fraction must be between zero and one')
        self.ordinary=dataset.ordinary_length;self.episodes=dataset.episodes
        if not self.ordinary:raise ValueError('Mixed sampling requires ordinary windows')
        self.fraction,self.batch_size=fraction,batch_size
        super().__init__(math.ceil(self.ordinary/(1-fraction)),seed)
        self.startup_total=self.length-self.ordinary
        self.samples_seen=self.startup_seen=0;self.age_histogram=np.zeros(8,dtype=np.int64);self.history=dataset.rf-1

    def __iter__(self):
        stop=self.length if self.limit is None else min(self.length,self.position+self.limit)
        base=self.seed+self.epoch*1000003
        a,b=affine(self.ordinary,base)
        ea,eb=affine(len(self.episodes),base+17)
        for start in range(self.position//self.batch_size*self.batch_size,stop,self.batch_size):
            width=min(self.batch_size,self.length-start)
            prior=start*self.startup_total//self.length
            count=(start+width)*self.startup_total//self.length-prior
            slots=set(random.Random(base+start*1000033).sample(range(width),count))
            mature=start-prior;early=prior
            for j in range(width):
                if j in slots:
                    episode=self.episodes[(ea*early+eb)%len(self.episodes)]
                    age=random.Random(base+early*1000037+97).randrange(episode['count'])
                    value=self.ordinary+episode['offset']+age;early+=1
                else:value=(a*mature+b)%self.ordinary;mature+=1
                if self.position<=start+j<stop:yield value

    def observe(self,batch):
        ages=batch['reset_age'].detach().cpu().numpy();ages=ages[ages>=0]
        self.samples_seen+=len(batch['motion']);self.startup_seen+=len(ages)
        self.age_histogram+=np.bincount(np.minimum(7,ages*8//self.history),minlength=8)
        return dict(batch_fraction=len(ages)/len(batch['motion']),startup_samples=len(ages),
                    consumed_fraction=self.startup_seen/self.samples_seen,
                    age_histogram=self.age_histogram.tolist(),age_bin_edges_steps=np.linspace(0,self.history,9).tolist())

    def state_dict(self):
        return dict(super().state_dict(),startup_fraction=self.fraction,batch_size=self.batch_size,
                    samples_seen=self.samples_seen,startup_seen=self.startup_seen,age_histogram=self.age_histogram.tolist())

    def load_state_dict(self,state):
        if (state.get('startup_fraction'),state.get('batch_size'),state['length'])!=(self.fraction,self.batch_size,self.length):
            raise ValueError('Mixed-window resume requires the saved startup_fraction, batch size and index sizes')
        super().load_state_dict(state)
        self.samples_seen=state['samples_seen'];self.startup_seen=state['startup_seen'];self.age_histogram=np.asarray(state['age_histogram'],dtype=np.int64)


class PeriodMetrics:
    """Endpoint/interval reductions with explicit counts, including empty bands."""
    def __init__(self, history_seconds):
        self.bounds = [('0_200ms', 0., .2), ('200ms_history', .2, history_seconds),
                       ('after_history', max(.2, history_seconds), float('inf'))]
        self.values = {name: dict(count=np.zeros(5,dtype=np.int64), sse=np.zeros((5,3)),
            normalized=np.zeros(5), episodes=set(), groups=set(), physics={}) for name,_,_ in self.bounds}

    def add(self, physical, batch, physics, norm, series, times, reconstruction=None):
        p=physical.detach().cpu().numpy();y=batch['targets'].detach().cpu().numpy()
        valid=batch['target_valid'].cpu().numpy() & np.isfinite(y).all(-1)
        error=np.where(valid[...,None],p-y,0.)
        std=np.asarray(norm.stats['targets']['std'])
        for name,lo,hi in self.bounds:
            acc=self.values[name];mask=(times>=lo)&(times<hi)
            if reconstruction is not None:mask &= reconstruction
            v=valid & mask[...,None]
            acc['count']+=v.sum((0,1));acc['sse']+=np.where(v[...,None],error**2,0).sum((0,1))
            acc['normalized']+=np.where(v[...,None],(error/std)**2,0).sum((0,1,3))
            for i,sid in enumerate(batch['series'].tolist()):
                if v[i].any():
                    acc['episodes'].add(sid);acc['groups'].add(series[sid]['group'])
            for horizon in ('short','long'):
                terms=physics[horizon]
                # Attribute a constraint to its completed endpoint's period;
                # equations always use real preceding endpoints, even at a bin edge.
                endtimes=times[:,1:] if horizon=='short' else times[:,-1]
                selected=terms['valid'].detach().cpu().numpy() & (endtimes>=lo)&(endtimes<hi)
                raw=terms['residual'].detach().cpu().numpy()[selected]
                scaled=terms['normalized'].detach().cpu().numpy()[selected]
                a=acc['physics'].setdefault(horizon,dict(count=0,raw=np.zeros(3),normalized=np.zeros(3)))
                a['count']+=int(selected.sum());a['raw']+=(raw**2).sum((0,2));a['normalized']+=(scaled**2).sum((0,2))

    def report(self):
        result={}
        for name,lo,hi in self.bounds:
            a=self.values[name];n=a['count'];den=np.maximum(n,1)
            result[name]=dict(start_seconds=lo,end_seconds=hi if math.isfinite(hi) else None,
                episodes=len(a['episodes']),collection_groups=len(a['groups']),
                normalized_mse_all_heads=float(a['normalized'].sum()/(3*n.sum())) if n.sum() else None,
                normalized_wrench_mse=float(a['normalized'][:2].sum()/(3*n[:2].sum())) if n[:2].sum() else None,
                heads={head:dict(count=int(n[i]),rmse_axis=np.sqrt(a['sse'][i]/den[i]).tolist() if n[i] else None,
                    rms_3d_vector=float(np.sqrt(a['sse'][i].sum()/den[i])) if n[i] else None) for i,head in enumerate(HEADS)},
                physics={h:dict(count=b['count'],rms_vector=np.sqrt(b['raw']/b['count']).tolist() if b['count'] else None,
                    normalized_rms_vector=np.sqrt(b['normalized']/b['count']).tolist() if b['count'] else None)
                    for h,b in a['physics'].items()})
        return result


def elapsed_seconds(batch,dataset):
    # Recorded native ticks, not shard/row positions, define episode age.
    values=[]
    for i,sid in enumerate(batch['series'].tolist()):
        reset=dataset.startup_info['resets'][str(sid)]
        values.append((batch['tick'][i].cpu().numpy()-reset['tick'])*reset['native_dt'])
    return np.stack(values)


@torch.no_grad()
def validation_diagnostics(model,norm,data,config,device,split='validation',writer=None,update=0):
    """Fixed age coverage supplements, but never changes, mature checkpoint selection."""
    from .objectives import objective
    dataset=BackfilledWindows(data,split,config.get('cache_size',8),config.get('startup_index'))
    dataset.check_model(model.config);dataset._local()
    ages=config.get('startup_validation_ages',16)
    if not isinstance(ages,int) or ages<1:raise ValueError('startup_validation_ages must be positive')
    indices=dataset.validation_indices(ages)
    # Prepared indices are emitted in series order. Uniform positions per true
    # reset episode give mature coverage with no random resampling between checks.
    sids=dataset.index['series']
    for episode in dataset.episodes:
        left=np.searchsorted(sids,episode['series'],side='left')
        right=np.searchsorted(sids,episode['series'],side='right')
        if right>left:indices.extend(np.unique(np.linspace(left,right-1,min(ages,right-left),dtype=int)).tolist())
    loader=make_loader(dataset,batch_size=config.get('validation_batch_size') or config['batch_size'],sampler=indices,num_workers=0)
    metric=PeriodMetrics((model.config.receptive_field-1)*model.config.sample_seconds)
    model.eval()
    try:
        for cpu in loader:
            times=elapsed_seconds(cpu,dataset);batch=to_device(cpu,device)
            prediction=model(norm.normalize('motion',batch['motion']),norm.normalize('properties',batch['properties']))
            result=objective(prediction,batch,norm,physics_dtype=config.get('physics_dtype','float64'))
            metric.add(result['physical'],batch,result['physics'],norm,dataset.series,times)
    finally:close_loader(loader)
    report=dict(periods=metric.report(),windows=len(indices),history_seconds=(model.config.receptive_field-1)*model.config.sample_seconds,
        split=split,selection='diagnostic only; existing full-history validation selection unchanged',
        coverage='deterministic uniform startup and mature window ages per verified-reset episode',
        weighting='window-weighted endpoints; overlapping endpoints repeated; physics attributed by completed endpoint time',
        physics_channels=['kinematic','linear_momentum','angular_momentum'],
        excluded_series=dataset.startup_info['excluded'])
    if writer is not None:
        for period,values in report['periods'].items():
            for key in ('normalized_mse_all_heads','normalized_wrench_mse','episodes'):
                if values[key] is not None:writer.add_scalar(f'reset_validation/{period}/{key}',values[key],update)
            for horizon,terms in values['physics'].items():
                writer.add_scalar(f'reset_validation/{period}/{horizon}_count',terms['count'],update)
                if terms['normalized_rms_vector'] is not None:
                    for channel,value in zip(('r','P','H'),terms['normalized_rms_vector']):
                        writer.add_scalar(f'reset_validation/{period}/{horizon}_{channel}',value,update)
    return report


@torch.no_grad()
def episode_evaluation(model,norm,data,config,device,split='test'):
    """Unique endpoints on verified-reset episodes; bounded causal input overlap.

    Finite-RF nonrecurrent chunk inference is identical to persistent cached
    stepping. Only raw motion overlaps; each scored endpoint occurs once.
    Physics uses consecutive real 40-interval windows (no duplicated intervals).
    """
    from .trajectories import Trajectories
    from .ported.discrete.losses import discrete_constraints
    reader=Trajectories(data,split);reader.dataset.check_model(model.config)
    path=Path(config.get('startup_index') or Path(data)/'reset_backfill')
    info=json.loads((path/'index.json').read_text())
    rf=model.config.receptive_field;k=model.config.long_intervals;dt=model.config.sample_seconds
    metrics=PeriodMetrics((rf-1)*dt);series_by_id={s['id']:s for s in reader.series}
    model.eval();episodes=0;excluded=[]
    for series in reader.series:
        sid=series['id'];reset=info['resets'].get(str(sid))
        if reset is None:
            excluded.append(sid);continue
        episodes+=1;stride=round(dt/reset['native_dt']);properties=torch.tensor(series['properties'],device=device,dtype=torch.float32)[None]
        # Prefix continuity determines where synthetic reset context is allowed;
        # after a gap only the original real-history eligibility can recover.
        prefix=True;previous_tick=None
        for start in range(0,series['rows']-1,k):
            end=min(series['rows'],start+k+1);begin=max(0,start-rf+1)
            rows=reader.read(series,begin,end);endpoints=rows[start-begin:]
            motion=rows['motion'].copy()
            if start<rf-1:motion=np.concatenate((np.repeat(np.asarray(reset['motion'])[None],rf-1-start,axis=0),motion))
            prediction=model(norm.normalize('motion',torch.tensor(np.nan_to_num(motion),device=device,dtype=torch.float32)[None]),norm.normalize('properties',properties))
            physical=norm.denormalize('targets',prediction.double())
            batch={key:torch.as_tensor(endpoints[key].copy(),device=device)[None] for key in endpoints.dtype.names if key not in ('motion','held_acceleration_W')}
            context=endpoints['history_valid'].copy()
            for j,row in enumerate(endpoints):
                if start and j==0:continue  # Already consumed anchor, retain its eligibility below.
                continuous=previous_tick is None or int(row['tick'])-previous_tick==stride
                prefix &= continuous and bool(np.isfinite(row['motion']).all())
                context[j] |= prefix
                previous_tick=int(row['tick'])
            if start:context[0]=anchor_context
            anchor_context=bool(context[-1])
            batch['target_valid'] &= torch.as_tensor(context,device=device)[None,:,None]
            batch['series']=torch.tensor([sid]);batch['interior_dimensions']=torch.tensor(series['interior_dimensions'],device=device,dtype=torch.float64)[None]
            args={key:batch[key].double() for key in ('x_O_W','u_O_W','omega_B','q_WB_wxyz','mass','gravity_W','interior_dimensions')}
            physics=discrete_constraints(**args,r_B=physical[:,:,2],v_B=physical[:,:,3],S_B=physical[:,:,4],
                F_average_B=physical[:,1:,0],tau_average_O_B=physical[:,1:,1],h=batch['h'][:,1:].double(),
                episode_id=batch['episode_id'],state_valid=batch['state_valid'],interval_valid=batch['interval_valid'][:,1:],
                history_valid=torch.tensor(context,device=device)[None],window_intervals=k,window_seconds=k*dt)
            times=(endpoints['tick'][None]-reset['tick'])*reset['native_dt']
            unique=np.ones(times.shape,dtype=bool)
            if start:unique[:,0]=False
            metrics.add(physical,batch,physics,norm,series_by_id,times,unique)
    return dict(periods=metrics.report(),episodes=episodes,excluded_series=excluded,
        weighting='unique real endpoints; nonoverlapping 5 ms intervals and 200 ms windows; constraints attributed by end time',
        inference='bounded ordinary causal forward with raw RF-1 context overlap, matching continuous cached inference',
        history_seconds=(rf-1)*dt,split=split)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--data',required=True);prep.add_argument('--output')
    ev=sub.add_parser('evaluate');ev.add_argument('--data',required=True);ev.add_argument('--checkpoint',required=True)
    ev.add_argument('--output',required=True);ev.add_argument('--device',default='cpu');ev.add_argument('--batch-size',type=int,default=64)
    ev.add_argument('--startup-index');ev.add_argument('--split',choices=('validation','test'),default='test')
    args=parser.parse_args()
    if args.command=='prepare':
        result=build_startup_index(args.data,args.output)
        print(json.dumps(dict(windows=result['windows'],verified_reset_episodes={k:len(v) for k,v in result['episodes'].items()},excluded=len(result['excluded']))));return
    from .checkpoint import load_checkpoint
    from .model import build_model
    from .normalization import Normalization
    from .evaluate import evaluate
    torch.set_num_threads(2)
    saved=load_checkpoint(args.checkpoint)
    model=build_model(ModelConfig(**saved['model_config'])).to(args.device);model.load_state_dict(saved['model'],strict=True)
    norm=Normalization(saved['normalization']);config=dict(saved['config'],batch_size=args.batch_size,validation_batch_size=args.batch_size)
    if args.startup_index:config['startup_index']=args.startup_index
    print(f'Evaluating selected checkpoint on {args.split}; no checkpoint selection',flush=True)
    loader=make_loader(WindowDataset(args.data,args.split),batch_size=args.batch_size)
    try:mature=evaluate(model,loader,norm,args.device)
    finally:close_loader(loader)
    report=dict(checkpoint=str(args.checkpoint),mature_full_history=mature,
        reset_windows=validation_diagnostics(model,norm,args.data,config,args.device,args.split),
        matched_episodes=episode_evaluation(model,norm,args.data,config,args.device,args.split))
    atomic_json(args.output,report);print(args.output,flush=True)


if __name__=='__main__':main()
