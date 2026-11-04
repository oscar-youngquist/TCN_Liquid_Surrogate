"""Explicit FP32/optional BF16 check against float64 loss and full parameter gradients."""
import argparse
import json
import torch
from liquid_tcn.dataset import WindowDataset, make_loader, to_device
from liquid_tcn.model import LiquidTCN, ModelConfig
from liquid_tcn.normalization import Normalization
from liquid_tcn.train import precision_check, DEFAULTS
from liquid_tcn.ported.storage import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--amp',action='store_true');args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(1)
    ds=WindowDataset(args.data)
    model=LiquidTCN(ModelConfig(**ds.manifest['model'])).to(args.device)
    batch=to_device(next(iter(make_loader(ds,batch_size=2,pin_memory=False))),args.device)
    config=dict(DEFAULTS,amp=args.amp)
    try:
        checks=precision_check(model,batch,Normalization(ds.manifest['normalization']),config)
        report=dict(status='passed',device=args.device,checks=checks)
    except (ValueError,FloatingPointError) as exc:
        report=dict(status='rejected',device=args.device,reason=str(exc))
        atomic_json(args.output,report)
        raise
    atomic_json(args.output,report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
