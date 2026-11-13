"""Explicit FP32/optional BF16 check against float64 loss and full parameter gradients."""
import argparse
import json
import torch
from liquid_tcn.dataset import WindowDataset, make_loader, to_device
from liquid_tcn.model import build_model, ModelConfig
from liquid_tcn.normalization import Normalization
from liquid_tcn.train import precision_check, DEFAULTS


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--amp',action='store_true');args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(1)
    ds=WindowDataset(args.data)
    model=build_model(ModelConfig(**ds.manifest['model'])).to(args.device)
    batch=to_device(next(iter(make_loader(ds,batch_size=2,pin_memory=False))),args.device)
    config=dict(DEFAULTS,amp=args.amp)
    report=precision_check(model,batch,Normalization(ds.manifest['normalization']),config,
                           diagnostic_path=args.output,context=dict(device=args.device,update=0,checkpoint=None))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
