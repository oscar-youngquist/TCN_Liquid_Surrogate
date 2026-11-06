"""Small CPU before/after comparison against source files from a local git commit.

Uses synthetic fixtures from tests; does not read production data or run a sweep.
Model, physics and normalization are held at their current implementation. Only
loader/trainer/gradient and scalar-logging implementations come from the baseline.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'tests'))
from helpers import preparation, make_attempt
from liquid_tcn import dataset, train
from liquid_tcn.checkpoint import load_checkpoint
from liquid_tcn.preprocess import prepare
from liquid_tcn.ported.storage import atomic_json
from liquid_tcn.sweep import runtime_versions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',default='25f8382')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='liquid-compare-') as temp:
        root=Path(temp)
        def old_module(name):
            source=subprocess.check_output(['git','show',f'{args.baseline}:src/liquid_tcn/{name}.py'],cwd=REPO,text=True)
            path=root/(name+'.py');path.write_text(source)
            spec=importlib.util.spec_from_file_location('liquid_tcn._before_'+name,path)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            return module
        old_data,old_train,old_grad,old_objectives=[old_module(n) for n in ('dataset','train','gradients','objectives')]
        old_train.WindowDataset=old_data.WindowDataset;old_train.make_loader=old_data.make_loader
        old_train.backward_objectives=old_grad.backward_objectives
        old_train.loss_components=old_objectives.loss_components
        attempt=make_attempt(root/'raw',samples=650)
        prepare(dict(shard_rows=71,sources=[dict(path=str(attempt),group='a',split='train')]),root/'loader_data')
        loaders={}
        for name,module in [('before',old_data),('after',dataset)]:
            ds=module.WindowDataset(root/'loader_data');times=[]
            for _ in range(4):
                start=time.perf_counter();count=0
                for batch in module.make_loader(ds,batch_size=64,pin_memory=False):count+=len(batch['motion'])
                times.append(time.perf_counter()-start)
            loaders[name]=dict(windows=count,seconds=times,windows_per_second=count/min(times))
        small=root/'small';small.mkdir();data,_=preparation(small)
        config=dict(updates=60,batch_size=8,device='cpu',threads=1,pretrain_updates=10,ramp_updates=20,
                    validation_every=60,checkpoint_every=60,progress=False,gradient_strategy='pcgrad')
        training={}
        for name,module in [('before',old_train),('after',train)]:
            training[name]=module.train(data,root/name,config)
        a,b=[load_checkpoint(root/name/'latest.pt') for name in ('before','after')]
        assert all(torch.equal(v,b['model'][k]) for k,v in a['model'].items())
        assert a['scheduler']==b['scheduler'] and a['sampler']==b['sampler']
        for param,values in a['optimizer']['state'].items():
            for k,v in values.items():assert torch.equal(v,b['optimizer']['state'][param][k])
        result=dict(baseline=args.baseline,runtime=runtime_versions(),loader=loaders,training=training,
            model_optimizer_sampler_scheduler_equal=True,
            note='CPU synthetic fixtures; loader fastest of four cache-warm passes; sequential same-process RSS is cumulative; training wall excludes initialization/precision checks')
        atomic_json(args.output,result)
        print(json.dumps(result,indent=2))


if __name__=='__main__':main()
