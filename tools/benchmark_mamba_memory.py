"""Warm CUDA cached one-step comparison: FiLM pyramid versus FiLM pyramid+Mamba."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import subprocess
import numpy as np
import torch
from benchmark_film_streaming import sync, storage_bytes
from liquid_tcn.model import ModelConfig,build_model,CachedInference
from liquid_tcn.normalization import Normalization
from liquid_tcn.ported.storage import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda:1')
    parser.add_argument('--batch-size',type=int,default=4096)
    parser.add_argument('--steps',type=int,default=20)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if min(args.batch_size,args.steps,args.repeats)<1:parser.error('Counts must be positive')
    torch.set_num_threads(2);torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    norm=Normalization({k:dict(mean=np.zeros(s).tolist(),std=np.ones(s).tolist())
        for k,s in [('motion',(15,)),('properties',(24,)),('targets',(5,3))]})
    try:
        background = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
                                               '--format=csv'],text=True).strip()
    except (OSError,subprocess.SubprocessError):
        background = 'unavailable'
    report=dict(background_compute_processes=background, gpu=torch.cuda.get_device_name(args.device),torch=torch.__version__,cuda=torch.version.cuda,
        device=args.device,batch_size=args.batch_size,threads=2,precision='FP32, TF32 disabled',
        compilation=False,steps=args.steps,repeats=args.repeats,results={},
        scope='CUDA event latency of cached 12-output step; shared payload; no data loading/training')
    for memory in ('none','mamba'):
        torch.manual_seed(17)
        config=ModelConfig(architecture='tcn_pyramid_attention',property_conditioning='film',temporal_memory=memory)
        model=build_model(config).to(args.device).eval()
        with torch.inference_mode():
            model.film.weight.normal_(0,.01)
            motion=torch.randn(args.batch_size,15,device=args.device)
            props=torch.randn(1,24,device=args.device)
            sync(args.device);start=time.perf_counter()
            cache=CachedInference(model,norm,args.batch_size);cache.set_properties(props)
            for _ in range(config.receptive_field+208):cache.step(motion)
            sync(args.device);prefill=time.perf_counter()-start
            startup_peak=torch.cuda.max_memory_allocated(args.device)
            starts=[torch.cuda.Event(enable_timing=True) for _ in range(args.steps)]
            ends=[torch.cuda.Event(enable_timing=True) for _ in range(args.steps)]
            torch.cuda.reset_peak_memory_stats(args.device)
            values=[];begin=time.perf_counter()
            for _ in range(args.repeats):
                for i in range(args.steps):
                    starts[i].record();output,valid=cache.step(motion);ends[i].record()
                sync(args.device)
                values.extend(a.elapsed_time(b) for a,b in zip(starts,ends))
            elapsed=time.perf_counter()-begin
            assert bool(valid.all()) and bool(torch.isfinite(output).all())
            result=dict(config=asdict(config),parameters=sum(p.numel() for p in model.parameters()),
                mean_ms=float(np.mean(values)),median_ms=float(np.median(values)),p95_ms=float(np.percentile(values,95)),
                predictions_per_second=args.batch_size*1000/float(np.mean(values)),
                host_seconds=elapsed,setup_and_prefill_seconds=prefill,startup_peak_allocated_bytes=startup_peak,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device),persistent_cache_bytes=storage_bytes(cache))
            report['results'][memory]=result
            print(json.dumps(dict(memory=memory,**result)),flush=True)
        del model,cache,motion,props,output,valid,starts,ends
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(args.device)
    report['latency_ratio']=report['results']['mamba']['mean_ms']/report['results']['none']['mean_ms']
    import mamba_ssm
    report['mamba_ssm']=mamba_ssm.__version__
    args.output.parent.mkdir(parents=True,exist_ok=True);atomic_json(args.output,report)


if __name__=='__main__':main()
