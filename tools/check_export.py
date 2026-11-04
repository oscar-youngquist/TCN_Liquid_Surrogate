"""Check a frozen artifact against one fully contextual recorded window."""
import argparse
import json
import torch
from liquid_tcn.dataset import WindowDataset
from liquid_tcn.inference import load_export
from liquid_tcn.ported.storage import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--export',required=True)
    p.add_argument('--output',required=True);args=p.parse_args()
    torch.set_num_threads(2)
    item=WindowDataset(args.data)[0]
    cached=load_export(args.export)
    x=item['motion'][None];prop=item['properties'][None]
    with torch.no_grad():
        batch=cached.norm.denormalize('targets',cached.model(cached.norm.normalize('motion',x),cached.norm.normalize('properties',prop)))
        actual=[]
        for i in range(x.shape[1]):
            output,valid=cached.step(x[:,i],prop)
            if valid.all():actual.append(output)
        actual=torch.stack(actual,1)
        expected=batch[:,:,:4].flatten(-2)
        torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
    report=dict(contextual_predictions=actual.shape[1],output_values=actual.shape[2],
                maximum_absolute_difference=float((actual-expected).abs().max()),status='passed')
    atomic_json(args.output,report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
