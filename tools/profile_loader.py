"""Bounded loader throughput and process memory benchmark, with optional workers."""
import argparse
import json
import resource
import time
from liquid_tcn.dataset import WindowDataset, make_loader, close_loader
from liquid_tcn.ported.storage import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=0);p.add_argument('--batches',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=16);args=p.parse_args()
    ds=WindowDataset(args.data,cache_size=8,verify=True)
    loader=make_loader(ds,batch_size=args.batch_size,num_workers=args.workers,pin_memory=False,
                       sampler=range(min(len(ds),args.batches*args.batch_size)))
    start=time.perf_counter();count=0;first=None
    for i,batch in enumerate(loader):
        if first is None:first=time.perf_counter()-start
        count+=len(batch['motion'])
    elapsed=time.perf_counter()-start
    report=dict(workers=args.workers,windows=count,seconds=elapsed,first_batch_seconds=first,
        windows_per_second=count/elapsed,main_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        cache_maps_per_process=ds.cache_size,prefetch_batches_per_worker=2,shard_rows=ds.manifest['shard_rows'],
        note='Includes worker startup; OS page cache shared; main RSS excludes spawned workers')
    close_loader(loader)
    atomic_json(args.output,report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
