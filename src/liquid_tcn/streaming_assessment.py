"""Reuse chronological streaming predictions for all held-out assessment views."""
from collections import OrderedDict
import csv
import json
import math
from pathlib import Path
import numpy as np
import torch
from .ported.storage import atomic_json


class CachedWindowPredictions:
    """Read endpoint ranges from bounded mmaps; never repeat inference per window."""
    def __init__(self, directory, normalization, long_intervals, cache_size=8):
        self.directory=Path(directory);self.norm=normalization
        self.long_intervals=long_intervals;self.cache_size=cache_size;self.cache=OrderedDict()

    def __call__(self, batch):
        values=np.empty((len(batch['series']),self.long_intervals+1,5,3),dtype=np.float32)
        for i,(sid,end) in enumerate(zip(batch['series'].tolist(),batch['end'].tolist())):
            if sid not in self.cache:
                self.cache[sid]=np.load(self.directory/f'series-{sid:06d}.npy',mmap_mode='r')
                if len(self.cache)>self.cache_size:self.cache.popitem(last=False)
            self.cache.move_to_end(sid)
            values[i]=self.cache[sid][end-self.long_intervals:end+1]
        if not np.isfinite(values).all():
            raise ValueError('Streaming cache lacks valid predictions for an eligible test window')
        return self.norm.normalize('targets',torch.from_numpy(values))


def compare_assessments(directory, window_only=False):
    """Full and streaming use identical checkpoints, windows, segments and events."""
    directory=Path(directory)
    def flatten(value,prefix=''):
        result={}
        if isinstance(value,dict):
            for key,item in value.items():result.update(flatten(item,prefix+'/'+str(key)))
        elif isinstance(value,(list,tuple)):
            for i,item in enumerate(value):result.update(flatten(item,prefix+'/'+str(i)))
        elif isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value):
            result[prefix]=value
        return result
    differences={}
    for filename in (['window_test.json'] if window_only else ['window_test.json','summary.json']):
        a=json.loads((directory/filename).read_text());b=json.loads((directory/'streaming'/filename).read_text())
        x,y=flatten(a),flatten(b)
        differences[filename]={k:dict(full=x[k],streaming=y[k],difference=y[k]-x[k]) for k in x.keys() & y.keys()}
    result=dict(status='complete',definition='streaming minus full; same selected checkpoint and held-out data',
        differences=differences,
        notes=['Window metrics remain overlapping-window weighted; chronological metrics use unique physical intervals.',
               'Streaming window metrics reuse physical-unit float32 prediction files, then re-normalize for supervision.',
               'Assessment timing includes loading and metrics; use benchmark_film_streaming.py for CUDA step latency.'])
    if not window_only:
        squares=np.zeros((5,3));maximum=np.zeros((5,3));count=0
        for path in sorted((directory/'predictions').glob('series-*.npy')):
            a=np.load(path,mmap_mode='r');b=np.load(directory/'streaming/predictions'/path.name,mmap_mode='r')
            for start in range(0,len(a),8192):
                aa=np.asarray(a[start:start+8192],dtype=float);bb=np.asarray(b[start:start+8192],dtype=float)
                keep=np.isfinite(aa).all((1,2)) & np.isfinite(bb).all((1,2))
                error=bb[keep]-aa[keep]
                if len(error):
                    squares+=(error**2).sum(0);maximum=np.maximum(maximum,abs(error).max(0));count+=len(error)
        result['prediction_difference']=dict(endpoints=count,
            weighting='jointly finite history-valid chronological endpoints, including segment anchors',
            head_order=['force','moment','r_B','v_B','S_B'],units=['N','N m','m','m/s','kg m^2/s'],
            rms_by_head_coordinate=np.sqrt(squares/count).tolist() if count else None,
            max_abs_by_head_coordinate=maximum.tolist() if count else None)
    atomic_json(directory/'streaming_comparison.json',result)
    with (directory/'streaming_comparison.csv.tmp').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['report','metric','full','streaming','streaming_minus_full'])
        for report,items in differences.items():
            for key,values in sorted(items.items()):writer.writerow([report,key,values['full'],values['streaming'],values['difference']])
    (directory/'streaming_comparison.csv.tmp').replace(directory/'streaming_comparison.csv')
    metric=differences['window_test.json']['/selection_metric']
    (directory/'streaming_comparison.md.tmp').write_text(
        '# Cached streaming comparison\n\nSame selected checkpoint and test data; no test-based checkpoint selection.\n\n'
        '| Window-weighted normalized supervised MSE | Value |\n|---|---:|\n'
        f'| Full sequence | {metric["full"]:.10g} |\n| Cached streaming | {metric["streaming"]:.10g} |\n'
        f'| Streaming minus full | {metric["difference"]:.10g} |\n\n'
        'All numeric metric differences: [CSV](streaming_comparison.csv), [JSON](streaming_comparison.json).\n\n'
        'Streaming physical analyses are in `streaming/summary.json`, with matching per-series and transient files.\n'
        'The same frozen event definitions and training-only reference statistics are reused.\n')
    (directory/'streaming_comparison.md.tmp').replace(directory/'streaming_comparison.md')
    return result


def window_streamer(model, normalization, device):
    """Bounded partial-evaluation fallback: replay each independent test window."""
    from .model import CachedInference
    from .normalization import Normalization
    identity=Normalization({name:dict(mean=np.zeros(shape).tolist(),std=np.ones(shape).tolist())
        for name,shape in [('motion',(15,)),('properties',(model.config.property_dim,)),('targets',(5,3))]})
    @torch.inference_mode()
    def predict(batch):
        motion=normalization.normalize('motion',batch['motion'].to(device))
        properties=normalization.normalize('properties',batch['properties'].to(device))
        cache=CachedInference(model,identity,len(motion));cache.set_properties(properties)
        start=model.config.receptive_field-1
        out=motion.new_empty(len(motion),motion.shape[1]-start,5,3)
        for i in range(motion.shape[1]):
            value,_=cache.step(motion[:,i],return_all=True)
            if i>=start:out[:,i-start].copy_(value)
        return out
    return predict
