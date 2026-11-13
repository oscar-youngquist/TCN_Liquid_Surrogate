"""Equivalent one-endpoint full-history / reference-cache / ring-cache benchmark.

FP32, no compilation, no training. GPU inputs and CUDA events are preallocated.
The full-history path consumes RF real samples and emits ONE endpoint, not the
41 endpoints of a training window. All paths return the same 12 physical values.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from liquid_tcn.model import ModelConfig, build_model, CachedInference, ReferenceCachedInference
from liquid_tcn.normalization import Normalization


def sync(device):
    torch.cuda.synchronize(device)


def storage_bytes(cache):
    if hasattr(cache,'persistent_bytes'):return cache.persistent_bytes()
    tensors=[cache.age,*cache.caches,*cache.histories]
    stores={t.untyped_storage().data_ptr():t.untyped_storage().nbytes() for t in tensors if t is not None}
    return sum(stores.values())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda:1');parser.add_argument('--batch-size',type=int,default=4096)
    parser.add_argument('--threads',type=int,default=2);parser.add_argument('--warmup',type=int,default=8)
    parser.add_argument('--repeats',type=int,default=5);parser.add_argument('--steps',type=int,default=20)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if min(args.batch_size,args.threads,args.warmup,args.repeats,args.steps)<1:parser.error('Counts must be positive')
    device=torch.device(args.device);torch.set_num_threads(args.threads)
    if device.type!='cuda' or not torch.cuda.is_available():
        raise RuntimeError('This benchmark requires an available CUDA GPU; batch size will not be reduced')
    torch.cuda.set_device(device)
    # Compare actual FP32 arithmetic across different convolution shapes; TF32
    # kernel selection otherwise introduces precision differences unrelated to caches.
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    norm=Normalization({k:dict(mean=np.zeros(s).tolist(),std=np.ones(s).tolist())
        for k,s in [('motion',(15,)),('properties',(24,)),('targets',(5,3))]})
    report=dict(gpu=torch.cuda.get_device_name(device),device=str(device),dtype='float32',batch_size=args.batch_size,
        torch=torch.__version__,cuda=torch.version.cuda,python=platform.python_version(),threads=args.threads,
        warmup=args.warmup,repeats=args.repeats,steps_per_repeat=args.steps,compiled=False,
        compilation_seconds=0.,tf32=False,outputs='12 physical-unit values per environment per step',
        scope=__doc__,results=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for architecture in ('tcn','tcn_pyramid_attention'):
        for conditioning in ('existing','film'):
            torch.manual_seed(17)
            c=ModelConfig(architecture=architecture,property_conditioning=conditioning)
            model=build_model(c).to(device).eval()
            if model.film is not None:
                # Nonidentity modulation tests the trained-FiLM path, not only zeros.
                with torch.no_grad():
                    model.film.weight.normal_(0,.01);model.film.bias.normal_(0,.01)
            n=args.warmup+args.repeats*args.steps
            inputs=torch.randn(args.batch_size,c.receptive_field+n,15,device=device)
            props=torch.randn(1,24,device=device)  # One shared liquid configuration.
            starts=[torch.cuda.Event(enable_timing=True) for _ in range(n)]
            ends=[torch.cuda.Event(enable_timing=True) for _ in range(n)]
            row=dict(architecture=architecture,conditioning=conditioning,config=asdict(c),paths={})
            expected=None;prior_value=None
            for path in ('full_history','reference_cached','optimized_cached'):
                sync(device);torch.cuda.empty_cache();begin=time.perf_counter()
                cache=None
                if path!='full_history':
                    cls=ReferenceCachedInference if path=='reference_cached' else CachedInference
                    cache=cls(model,norm,args.batch_size)
                    if path=='optimized_cached':cache.set_properties(props)
                sync(device);initialization=time.perf_counter()-begin
                with torch.inference_mode():
                    begin=time.perf_counter()
                    if cache is not None:
                        for t in range(c.receptive_field):
                            if path=='optimized_cached':cache.step(inputs[:,t])
                            else:cache.step(inputs[:,t],props)
                    sync(device);prefill=time.perf_counter()-begin
                    def step(t):
                        if path=='full_history':
                            return model(inputs[:,t-c.receptive_field+1:t+1],props)[:, -1,:4].flatten(1)
                        return cache.step(inputs[:,t],None if path=='optimized_cached' else props)[0]
                    for t in range(c.receptive_field,c.receptive_field+args.warmup):step(t)
                    sync(device);torch.cuda.reset_peak_memory_stats(device)
                    values=[]
                    for repeat in range(args.repeats):
                        for j in range(args.steps):
                            i=repeat*args.steps+j
                            starts[i].record()
                            value=step(c.receptive_field+args.warmup+i)
                            ends[i].record()
                        # Synchronization is outside the sustained multi-step loop.
                        sync(device)
                        values.extend(starts[repeat*args.steps+j].elapsed_time(ends[repeat*args.steps+j])
                                      for j in range(args.steps))
                    result=dict(mean_ms=float(np.mean(values)),median_ms=float(np.median(values)),
                        p95_ms=float(np.percentile(values,95)),environment_predictions_per_second=args.batch_size*1000/float(np.mean(values)),
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                        persistent_cache_bytes=storage_bytes(cache) if cache is not None else 0,
                        initialization_seconds=initialization,prefill_seconds=prefill,
                        prefill_samples=c.receptive_field if cache is not None else 0)
                    if expected is None:expected=value.clone()
                    else:
                        error=(value-expected).abs()
                        result['max_abs_difference_vs_full']=float(error.max())
                        result['rms_difference_vs_full']=float(error.square().mean().sqrt())
                        result['allclose_vs_full']=bool(torch.allclose(value,expected,atol=2e-5,rtol=2e-4))
                    if path=='reference_cached':prior_value=value.clone()
                    elif path=='optimized_cached':
                        result['max_abs_difference_vs_prior']=float((value-prior_value).abs().max())
                    row['paths'][path]=result
                    print(json.dumps(dict(architecture=architecture,conditioning=conditioning,path=path,**result)),flush=True)
                    del value
                del cache
            full=row['paths']['full_history']['mean_ms'];prior=row['paths']['reference_cached']['mean_ms']
            for result in row['paths'].values():
                result.update(speedup_vs_full=full/result['mean_ms'],speedup_vs_prior=prior/result['mean_ms'])
            report['results'].append(row)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            del model,inputs,props,expected,prior_value,starts,ends
            torch.cuda.empty_cache()
    print(args.output)


if __name__=='__main__':main()
