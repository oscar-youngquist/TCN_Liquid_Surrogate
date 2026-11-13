"""Short synthetic network benchmark; no dataset, training run, or admission check.

Forward/backward includes gradient clearing and a scalar squared-output loss,
not PCGrad, physics, loading, optimizer, or checkpoint I/O. Forward-only uses
eval/inference_mode on a full normalized input window already on the device.
Cached inference measures one sample per environment, including normalization
and validity checks, after filling all history buffers.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import time
import numpy as np
import torch
from liquid_tcn.model import ModelConfig, build_model, CachedInference
from liquid_tcn.normalization import Normalization


def measure(call, device, warmup, iterations):
    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    for _ in range(warmup):
        call()
    synchronize()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    elapsed = []
    for _ in range(iterations):
        synchronize()
        start = time.perf_counter()
        call()
        synchronize()
        elapsed.append(time.perf_counter()-start)
    return dict(mean_ms=1000*float(np.mean(elapsed)),median_ms=1000*float(np.median(elapsed)),
        p95_ms=1000*float(np.percentile(elapsed,95)),calls_per_second=iterations/sum(elapsed),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=2)
    p.add_argument('--train-batches',type=int,nargs='+',default=[16,64,1024])
    p.add_argument('--skip-training',action='store_true')
    p.add_argument('--forward-batches',type=int,nargs='+',default=[])
    p.add_argument('--inference-batches',type=int,nargs='+',default=[1,16,64])
    p.add_argument('--warmup',type=int,default=3);p.add_argument('--iterations',type=int,default=10)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if min(args.threads,args.warmup,args.iterations,*args.train_batches,*args.forward_batches,*args.inference_batches)<1:
        p.error('All counts must be positive')
    device=torch.device(args.device);torch.set_num_threads(args.threads)
    report=dict(device=str(device),device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else platform.processor(),
        torch=torch.__version__,cuda_runtime=torch.version.cuda,python=platform.python_version(),
        threads=args.threads,warmup=args.warmup,iterations=args.iterations,compiled=False,
        compilation_seconds=0.,scope=__doc__,cpu_peak_memory='unavailable',results=[])
    norm=Normalization({k:dict(mean=np.zeros(s).tolist(),std=np.ones(s).tolist())
        for k,s in [('motion',(15,)),('properties',(24,)),('targets',(5,3))]})
    for architecture,backend in [('tcn','custom'),('tcn_pyramid_attention','custom'),('tcn_pyramid_attention','sdpa')]:
        torch.manual_seed(1)
        c=ModelConfig(architecture=architecture,attention_backend=backend)
        model=build_model(c).to(device)
        result=dict(architecture=architecture,backend=backend,parameters=sum(p.numel() for p in model.parameters()),
            config=asdict(c),training=[],forward_only=[],inference=[])
        for batch in ([] if args.skip_training else args.train_batches):
            x=torch.randn(batch,c.input_length,15,device=device);props=torch.randn(batch,24,device=device)
            def forward_backward():
                model.zero_grad(set_to_none=True)
                model(x,props).square().mean().backward()
            model.train()
            timing=measure(forward_backward,device,args.warmup,args.iterations)
            timing.update(batch_size=batch,windows_per_second=batch*timing['calls_per_second'])
            result['training'].append(timing)
            del x,props
        model.zero_grad(set_to_none=True)
        model.eval()
        for batch in args.forward_batches:
            x=torch.randn(batch,c.input_length,15,device=device);props=torch.randn(batch,24,device=device)
            with torch.inference_mode():
                timing=measure(lambda:model(x,props),device,args.warmup,args.iterations)
            timing.update(batch_size=batch,input_samples=c.input_length,
                predicted_endpoints=c.long_intervals+1,windows_per_second=batch*timing['calls_per_second'])
            result['forward_only'].append(timing)
            del x,props
        # Clear unused allocator blocks so reserved memory in cached timing does
        # not include a preceding large full-window benchmark's workspace.
        if device.type=='cuda':torch.cuda.empty_cache()
        for batch in args.inference_batches:
            cache=CachedInference(model,norm,batch)
            x=torch.randn(batch,15,device=device);props=torch.randn(batch,24,device=device)
            with torch.inference_mode():
                for _ in range(c.receptive_field):
                    cache.step(x,props)
                timing=measure(lambda:cache.step(x,props),device,args.warmup,args.iterations)
            timing.update(environments=batch,samples_per_second=batch*timing['calls_per_second'])
            result['inference'].append(timing)
            del cache,x,props
        report['results'].append(result)
        print(f'{architecture}/{backend}: {result["parameters"]} parameters',flush=True)
        del model
        if device.type=='cuda':torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(args.output)


if __name__=='__main__':main()
