"""Bounded-memory chronological analyses of selected models; never used for selection."""
from pathlib import Path
from collections import defaultdict
import json
import math
import os
import sqlite3
import numpy as np
import torch
from .trajectories import Trajectories, motion_features, event_signals, cache_predictions
from .ported.discrete.losses import quaternion_matrix, discrete_constraints
from .ported.storage import atomic_json
from .metadata import PROPERTY_FIELDS

HORIZONS=(10,40,100,200,400)  # 50, 200, 500, 1000, 2000 ms


class Stats:
    """Exact moments/counts plus a bounded uniform priority reservoir for quantiles."""
    def __init__(self, capacity=2048, seed=1729):
        self.capacity=capacity;self.n=0;self.total=0.;self.square=0.
        self.values=np.empty(0);self.keys=np.empty(0);self.rng=np.random.default_rng(seed)

    def add(self, values):
        v=np.asarray(values,dtype=float).reshape(-1)
        if not np.isfinite(v).all(): raise FloatingPointError('Nonfinite offline metric')
        self.n+=len(v);self.total+=float(v.sum());self.square+=float(v@v)
        self._sample(v,self.rng.random(len(v)))

    def _sample(self,v,k):
        self.values=np.r_[self.values,v];self.keys=np.r_[self.keys,k]
        if len(self.values)>self.capacity:
            keep=np.argpartition(self.keys,-self.capacity)[-self.capacity:]
            self.values,self.keys=self.values[keep],self.keys[keep]

    def merge(self, other):
        self.n+=other['n'];self.total+=other['total'];self.square+=other['square']
        self._sample(np.asarray(other['values']),np.asarray(other['keys']))

    def state(self):
        return dict(n=self.n,total=self.total,square=self.square,values=self.values.tolist(),keys=self.keys.tolist())

    def result(self):
        if not self.n: return dict(status='unavailable',count=0)
        q=np.percentile(self.values,[50,95,99])
        tail=self.values[self.values>=q[2]]
        return dict(count=self.n,mean=self.total/self.n,rms=math.sqrt(self.square/self.n),
            median=float(q[0]),p95=float(q[1]),p99=float(q[2]),worst_1pct_mean=float(tail.mean()),
            percentile_method='uniform bounded reservoir',reservoir_count=len(self.values))


def nearest(features, reference, exclude_self=False):
    """Euclidean nearest-reference distance; bounded blocks, no pairwise 3-D tensor."""
    distance=np.maximum(0.,(features**2).sum(1)[:,None]+(reference**2).sum(1)[None]-2*features@reference.T)
    if exclude_self: distance[distance<1e-12]=np.inf
    return np.sqrt(distance.min(1))


def fit_reference(data, path, cancelled=lambda:False):
    """Fit bounded training-only feature/event reference; persist before reading test."""
    path=Path(path)
    if path.exists(): return json.loads(path.read_text())
    reader=Trajectories(data,'train');sample=np.empty((0,44));keys=np.empty(0)
    rng=np.random.default_rng(1729);count=0
    for series in reader.series:
        for start,end in reader.segments(series):
            for pos,begin,rows in reader.blocks(series,start,end,1):
                if cancelled(): raise InterruptedError('Reference fitting interrupted')
                skip=max(1,pos-begin)
                values=np.column_stack((motion_features(rows,series['properties']),event_signals(rows)))[skip:]
                count+=len(values)
                sample=np.concatenate((sample,values));keys=np.r_[keys,rng.random(len(values))]
                if len(keys)>4096:
                    keep=np.argpartition(keys,-4096)[-4096:];sample,keys=sample[keep],keys[keep]
    if len(sample)<2: raise ValueError('Training reference needs at least two eligible intervals')
    features=sample[:,:39];mean=features.mean(0);std=features.std(0);std[std<1e-12]=1.
    z=(features-mean)/std;reference=z[:min(1024,len(z))]
    distances=nearest(z,reference,True)
    finite=distances[np.isfinite(distances)]
    signals=sample[:,39:]
    result=dict(training_intervals=count,sampled_intervals=len(sample),mean=mean.tolist(),std=std.tolist(),
        reference=reference.tolist(),sparsity_edges=np.percentile(finite,[50,90]).tolist() if len(finite) else [0.,0.],
        property_edges=np.percentile(features[:,15:],[33.333,66.667],axis=0).T.tolist(),
        intensity_edges=np.percentile(np.linalg.norm(features[:,9:15],axis=1),[50,90]).tolist(),
        acceleration_change=float(np.percentile(signals[:,0],95)),stop_speed=float(np.percentile(signals[:,1],10)),
        turn_rate=float(np.percentile(signals[:,2],95)),force_event=float(np.percentile(signals[:,3],99)),
        moment_event=float(np.percentile(signals[:,4],99)),
        settling_force=max(1e-9,float(np.percentile(signals[:,3],95))*.05),
        settling_moment=max(1e-9,float(np.percentile(signals[:,4],95))*.05),
        feature_fields=['motion[0:15]',*PROPERTY_FIELDS],
        sparsity_definition='distance to nearest of <=1024 sampled training references in training-standardized motion/property coordinates; not an extrapolation label')
    atomic_json(path,result);return result


def shared_events(data, root, cancelled=lambda:False):
    """One persisted event/trajectory table, independent of model predictions."""
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    ref=fit_reference(data,root/'reference.json',cancelled)
    path=root/'events.sqlite'
    if path.exists(): return ref,path
    temp=root/'events.tmp.sqlite'
    if temp.exists(): temp.unlink()
    db=sqlite3.connect(temp)
    db.executescript('CREATE TABLE segments(sid INT,start INT,end INT); CREATE TABLE events(sid INT,start INT,end INT,first INT,last INT,kind TEXT);')
    reader=Trajectories(data)
    for series in reader.series:
        for start,end in reader.segments(series):
            db.execute('INSERT INTO segments VALUES(?,?,?)',(series['id'],start,end))
            pending=None
            for pos,begin,rows in reader.blocks(series,start,end,1):
                if cancelled(): db.close();raise InterruptedError('Event construction interrupted')
                sig=event_signals(rows)
                for j in range(max(1,pos-begin),len(rows)):
                    kinds=[]
                    if sig[j,0]>ref['acceleration_change']: kinds.append('acceleration_change')
                    if sig[j-1,1]>ref['stop_speed'] and sig[j,1]<=ref['stop_speed']: kinds.append('stop')
                    if sig[j,2]>ref['turn_rate']: kinds.append('turn')
                    if sig[j,3]>ref['force_event'] or sig[j,4]>ref['moment_event']: kinds.append('large_sph_wrench')
                    if not kinds: continue
                    at=begin+j;left=max(start+1,at-40);right=min(end,at+81)
                    if pending and left<=pending[1]:
                        pending[1]=right;pending[3]=at;pending[4].update(kinds)
                    else:
                        if pending: db.execute('INSERT INTO events VALUES(?,?,?,?,?,?)',(series['id'],*pending[:4],','.join(sorted(pending[4]))))
                        pending=[left,right,at,at,set(kinds)]
            if pending: db.execute('INSERT INTO events VALUES(?,?,?,?,?,?)',(series['id'],*pending[:4],','.join(sorted(pending[4]))))
        db.commit()
    db.executescript('CREATE INDEX segment_sid ON segments(sid); CREATE INDEX event_sid ON events(sid);')
    db.close();os.replace(temp,path)
    return ref,path


def physical_arrays(rows, heads, dimensions, reference):
    """Reuse the validated discrete equations, with a fixed recorded world origin."""
    tensors={k:torch.from_numpy(np.asarray(rows[k]).copy())[None] for k in
        ('x_O_W','u_O_W','omega_B','q_WB_wxyz','mass','gravity_W','episode_id','state_valid','history_valid')}
    h=torch.tensor(np.asarray(heads),dtype=torch.float64)[None]
    out=discrete_constraints(**tensors,r_B=h[:,:,2],v_B=h[:,:,3],S_B=h[:,:,4],
        F_average_B=h[:,1:,0],tau_average_O_B=h[:,1:,1],h=torch.tensor(rows['h'][1:].copy())[None],
        interval_valid=torch.tensor(rows['interval_valid'][1:].copy())[None],
        interior_dimensions=torch.tensor(dimensions,dtype=torch.float64)[None],reference_W=torch.tensor(reference,dtype=torch.float64)[None],
        interval_seconds=None,window_seconds=None,window_intervals=None)
    if not bool(out['short']['valid'].all()): raise ValueError('Invalid interval entered chronological physics')
    R=quaternion_matrix(tensors['q_WB_wxyz'])[0].numpy()
    force=np.einsum('nij,nj->ni',R,np.asarray(heads)[:,0])
    moment=np.einsum('nij,nj->ni',R,np.asarray(heads)[:,1])
    power=(force*rows['u_O_W']).sum(1)+(np.asarray(heads)[:,1]*rows['omega_B']).sum(1)
    return dict(force=force,moment=moment+np.cross(rows['x_O_W']-reference,force),power=power,
        residual=out['short']['residual'][0].numpy(),
        rates=np.array([out['u0'].item(),out['F0'].item(),out['tau0'].item()]))


def horizon_errors(rows, predicted, dimensions, reference, horizons=HORIZONS):
    """All ending horizons in one bounded block; index j denotes its endpoint.

    Reaction wrenches use endpoint rotations and actual h, never trapezoids of
    adjacent force labels. Moment sums are shifted from fixed reference A to the
    recorded window-start origin B by J_B = J_A - (B-A) x I.
    Work uses endpoint base velocities for both signals (interval approximation).
    """
    true=physical_arrays(rows,rows['targets'],dimensions,reference)
    pred=physical_arrays(rows,predicted,dimensions,reference)
    h=rows['h'][1:]
    def prefix(v): return np.concatenate((np.zeros((1,)+v.shape[1:]),np.cumsum(v,axis=0)))
    ferr=prefix((pred['force'][1:]-true['force'][1:])*h[:,None])
    merr=prefix((pred['moment'][1:]-true['moment'][1:])*h[:,None])
    work=prefix((pred['power'][1:]-true['power'][1:])*h)
    residuals=[prefix(v['residual']) for v in (pred,true)]
    duration=prefix(h)
    result={}
    for n in horizons:
        if len(rows)<=n:continue
        ends=np.arange(n,len(rows));starts=ends-n
        dt=duration[ends]-duration[starts]
        f=ferr[ends]-ferr[starts];m=merr[ends]-merr[starts]-np.cross(rows['x_O_W'][starts]-reference,f)
        item=dict(ends=ends,force=f,moment=m,work=work[ends]-work[starts],duration=dt)
        for name,p in zip(('predicted','sph'),residuals):
            r=p[ends]-p[starts]
            r[:,2]-=np.cross(rows['x_O_W'][starts]-reference,r[:,1])
            item[name]=r;item[name+'_normalized']=r/(dt[:,None,None]*pred['rates'][None,:,None])
        result[n]=item
    return result,pred['power']-true['power']


def add_vector(stats,name,values):
    stats[name+'/norm'].add(np.linalg.norm(values,axis=-1))
    for i,axis in enumerate('xyz'): stats[name+'/'+axis].add(values[:,i])


def analyze_events(reader, series, predictions, db, stats, ref, plot_path, cancelled):
    """Stream even long merged events; representative traces use fixed decimation."""
    counts=0;unavailable=defaultdict(int);representative=None
    for start,end,first,last,kind in db.execute('SELECT start,end,first,last,kind FROM events WHERE sid=? ORDER BY start',(series['id'],)):
        if cancelled(): raise InterruptedError('Event analysis interrupted')
        counts+=1;integrals=np.zeros((2,3));peak=np.full((2,2),-np.inf);peak_at=np.zeros((2,2))
        settle=[None,None];run=[0,0];native=reader.native_dt(series)
        reference=reader.read(series,start-1,start)['x_O_W'][0]
        trace=[];stride=max(1,math.ceil((end-start)/512))
        for pos,begin,rows in reader.blocks(series,start-1,end,1):
            if cancelled():raise InterruptedError('Event analysis interrupted')
            if len(rows)<2:continue
            p=np.asarray(predictions[begin:begin+len(rows)],dtype=float)
            true=physical_arrays(rows,rows['targets'],series['interior_dimensions'],reference)
            pred=physical_arrays(rows,p,series['interior_dimensions'],reference)
            skip=max(1,pos-begin);indices=np.arange(begin+skip,begin+len(rows))
            for head,name in enumerate(('force','moment')):
                error=p[skip:,head]-rows['targets'][skip:,head]
                norms=np.linalg.norm(error,axis=1)
                for label,mask in (('pre',indices<first),('during',(indices>=first)&(indices<last+40)),('post',indices>=last+40)):
                    stats[f'events/{name}/{label}_error'].add(norms[mask])
                integrals[head]+=((pred[name][skip:]-true[name][skip:])*rows['h'][skip:,None]).sum(0)
                for signal,values in enumerate((np.linalg.norm(p[skip:,head],axis=1),np.linalg.norm(rows['targets'][skip:,head],axis=1))):
                    at=int(np.argmax(values))
                    if values[at]>peak[head,signal]:
                        peak[head,signal]=values[at];peak_at[head,signal]=rows['tick'][skip+at]*native
                for idx,value in zip(indices,norms):
                    if idx>=last+40:
                        run[head]=run[head]+1 if value<=ref['settling_'+name] else 0
                        if run[head]>=20 and settle[head] is None: settle[head]=(idx-19-(last+40))*.005
            if representative is None:
                for j in range(skip,len(rows)):
                    if (begin+j-start)%stride==0:
                        trace.append([float(rows['tick'][j]*native),*np.linalg.norm(p[j,:2],axis=1).tolist(),*np.linalg.norm(rows['targets'][j,:2],axis=1).tolist()])
        for head,name in enumerate(('force','moment')):
            stats[f'events/{name}/peak_signed_error'].add([peak[head,0]-peak[head,1]])
            stats[f'events/{name}/peak_timing_error_s'].add([peak_at[head,0]-peak_at[head,1]])
            add_vector(stats,f'events/{name}/impulse_error',integrals[head:head+1])
            if settle[head] is None: unavailable[name+'_settling_censored']+=1
            else: stats[f'events/{name}/settling_error_s'].add([settle[head]])
        if representative is None:representative=dict(kind=kind,start=start,end=end,trace=trace)
    if representative is not None: atomic_json(plot_path,representative)
    return dict(events=counts,unavailable=dict(unavailable))


def assess(data, checkpoint, output, shared, device='cpu', block_rows=2048, cancelled=lambda:False, inference_mode='full'):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if (output/'summary.json').exists():return json.loads((output/'summary.json').read_text())
    ref,event_path=shared_events(data,shared,cancelled)
    reader=cache_predictions(data,checkpoint,output/'predictions',device,block_rows,cancelled,inference_mode)
    db=sqlite3.connect(event_path)
    global_stats=defaultdict(Stats);groups={};conditions=defaultdict(Stats);condition_groups=defaultdict(set);coverage=defaultdict(int);censored=defaultdict(int);exclusions=defaultdict(int)
    for series in reader.series:
        sid=series['id'];path=output/f'series-{sid:06d}.json'
        if path.exists(): record=json.loads(path.read_text())
        else:
            stats=defaultdict(lambda:Stats(seed=1729+sid));local_conditions=defaultdict(lambda:Stats(seed=1729+sid));intervals=0;segments=0
            rejected=defaultdict(int)
            if not series.get('assumptions_valid',True):rejected['series_physical_assumptions']=series['rows']
            if reader.native_dt(series) is None:rejected['no_certified_clock_pair']=series['rows']
            for pos,begin,raw in reader.blocks(series,0,series['rows']):
                if cancelled():raise InterruptedError('Coverage scan interrupted')
                for field in ('history_valid','interval_valid','state_valid'):
                    rejected[field+'_false']+=int((~raw[field]).sum())
                rejected['incomplete_target_mask']+=int((~raw['target_valid'].all(1)).sum())
                rejected['nonfinite_targets']+=int((~np.isfinite(raw['targets']).all((1,2))).sum())
            predictions=np.load(output/'predictions'/f'series-{sid:06d}.npy',mmap_mode='r')
            for start,end in db.execute('SELECT start,end FROM segments WHERE sid=? ORDER BY start',(sid,)):
                segments+=1;reference=reader.read(series,start,start+1)['x_O_W'][0]
                for pos,begin,rows in reader.blocks(series,start,end,max(HORIZONS)):
                    if cancelled():raise InterruptedError('Offline assessment interrupted')
                    if len(rows)<2:continue
                    p=np.asarray(predictions[begin:begin+len(rows)],dtype=float)
                    skip=max(1,pos-begin);intervals+=len(rows)-skip
                    for head,name in enumerate(('force','moment','r_B','v_B','S_B')):
                        add_vector(stats,'endpoint/'+name,p[skip:,head]-rows['targets'][skip:,head])
                    intensity=np.linalg.norm(rows['motion'][skip:,9:15],axis=1)
                    z=(motion_features(rows[skip:],series['properties'])-ref['mean'])/ref['std']
                    distances=nearest(z,np.asarray(ref['reference']))
                    bins=dict(intensity=np.searchsorted(ref['intensity_edges'],intensity),sparsity=np.searchsorted(ref['sparsity_edges'],distances))
                    for head,name in enumerate(('force','moment')):
                        errors=np.linalg.norm(p[skip:,head]-rows['targets'][skip:,head],axis=1)
                        for label,value in (('robot',series.get('robot')),('payload',series.get('payload_id'))):
                            if value is not None:local_conditions[f'{label}/{value}/{name}'].add(errors)
                        for field,value,edges in zip(PROPERTY_FIELDS,series['properties'],ref['property_edges']):
                            local_conditions[f'property/{field}/bin{np.searchsorted(edges,value)}/{name}'].add(errors)
                        for label,values in bins.items():
                            for k in range(3):local_conditions[f'{label}/{k}/{name}'].add(errors[values==k])
                    horizons,power=horizon_errors(rows,p,series['interior_dimensions'],reference)
                    stats['power/signed_error_W'].add(power[skip:]);stats['power/absolute_error_W'].add(abs(power[skip:]))
                    for n,item in horizons.items():
                        keep=item['ends']>=skip;tag=f'{n*5}ms'
                        for name in ('force','moment'):add_vector(stats,f'impulse/{tag}/{name}',item[name][keep])
                        stats[f'work/{tag}/signed_error_J'].add(item['work'][keep]);stats[f'work/{tag}/absolute_error_J'].add(abs(item['work'][keep]))
                        if n>=100:
                            for source in ('predicted','sph'):
                                for axis,name in enumerate(('r','P','H')):
                                    add_vector(stats,f'consistency/{tag}/{source}/{name}',item[source][keep,axis])
                                    stats[f'consistency/{tag}/{source}/{name}/normalized'].add(np.linalg.norm(item[source+'_normalized'][keep,axis],axis=1))
            events=analyze_events(reader,series,predictions,db,stats,ref,output/f'transient-{sid:06d}.json',cancelled)
            record=dict(series=sid,group=series['group'],episode=series['episode'],environment=series['environment'],
                source=series['source'],robot=series.get('robot'),payload_id=series.get('payload_id'),
                payload_properties=dict(zip(PROPERTY_FIELDS,series['properties'])),intervals=intervals,
                segments=segments,rows=series['rows'],events=events,exclusions=dict(rejected),
                metrics={k:v.state() for k,v in stats.items()},conditions={k:v.state() for k,v in local_conditions.items()})
            atomic_json(path,record)
        coverage['series']+=1;coverage['eligible_series']+=int(record['intervals']>0)
        coverage['rows']+=record['rows'];coverage['intervals']+=record['intervals'];coverage['segments']+=record['segments'];coverage['events']+=record['events']['events']
        for name,n in record['exclusions'].items():exclusions[name]+=n
        group=groups.setdefault(record['group'],{})
        for name,value in record['metrics'].items():
            global_stats[name].merge(value)
            acc=group.setdefault(name,dict(n=0,total=0.,square=0.))
            for k in acc:acc[k]+=value[k]
        for name,value in record['conditions'].items():
            conditions[name].merge(value)
            if value['n']:condition_groups[name].add(record['group'])
        for name,n in record['events']['unavailable'].items():censored[name]+=n
    db.close()
    metrics={k:v.result() for k,v in global_stats.items() if v.n}
    unavailable=[]
    for n in HORIZONS:
        if f'impulse/{n*5}ms/force/norm' not in metrics:unavailable.append(f'No continuous eligible {n*5} ms horizons')
    if any(s.get('robot') is None for s in reader.series):unavailable.append('Robot breakdown missing for series without robot metadata')
    if any(s.get('payload_id') is None for s in reader.series):unavailable.append('Payload identity unavailable for some series; numeric properties remain reported')
    if not coverage['events']:unavailable.append('No events meet the frozen training thresholds')
    coverage['collection_groups']=len(groups)
    coverage['excluded_rows_or_segment_anchors']=coverage['rows']-coverage['intervals']
    summary=dict(status='complete',inference_mode=inference_mode,weighting='unique chronological intervals; rolling horizons counted once per end timestamp',
        checkpoint=str(checkpoint),coverage=dict(coverage),exclusions_nonexclusive=dict(exclusions),metrics=metrics,conditions={k:dict(v.result(),groups=len(condition_groups[k])) for k,v in conditions.items() if v.n},
        censored_events=dict(censored),
        groups=groups,unavailable=unavailable,
        definitions=dict(moment='endpoint-base wrench rotated to world and shifted to fixed window-start base origin',
            work='interval-data approximation: sum h*(F_W dot u_O_W + tau_B dot omega_B) at interval endpoints; not exact fluid energy or a passivity guarantee',
            consistency='ported discrete_constraints summed over adjacent intervals; SPH floor uses the same quadrature',
            tails='exact moments; approximate percentiles and worst-1% mean from <=2048 uniform reservoir entries',
            events='training-only thresholds; 200 ms pre, through 200 ms after last detection during, 200 ms post; overlapping ranges merged',
            settling='first 100 ms sustained wrench-error tolerance in post phase; censored events retained in per-series counts',
            sparsity=ref['sparsity_definition']))
    atomic_json(output/'summary.json',summary);return summary
