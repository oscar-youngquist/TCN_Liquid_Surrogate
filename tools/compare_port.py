"""Compare a read-only baseline snapshot with the port on synthetic and recorded rows.

Usage: python tools/compare_port.py --source /tmp/liquid-source-300d3288
       --attempt /path/to/completed/attempt --output artifacts/port-equivalence.json
The source path is used only by this acceptance tool, never the installed package.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from liquid_tcn.ported.discrete import data as port_data
from liquid_tcn.ported.discrete import losses as port_loss
from liquid_tcn.preprocess import iter_merged_rows, verify_attempt
from liquid_tcn.ported.storage import atomic_json


def compare(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:compare(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y)
    elif isinstance(a,torch.Tensor):
        torch.testing.assert_close(a,b,atol=0,rtol=0,equal_nan=True)
    else:
        np.testing.assert_equal(a,b)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--attempt');p.add_argument('--output',required=True)
    args=p.parse_args()
    sys.path.insert(0,str(Path(args.source).resolve()))
    original=importlib.import_module('liquid_collection.discrete.data')
    loss=importlib.import_module('liquid_collection.discrete.losses')
    # Independent random endpoints exercise gradients through every head.
    torch.manual_seed(17);B,N=2,41
    vec=lambda: torch.randn(B,N,3,dtype=torch.float64)
    q=torch.randn(B,N,4,dtype=torch.float64)
    g=torch.zeros(B,N,3,dtype=torch.float64);g[...,2]=-9.81
    data=dict(r_B=vec(),v_B=vec(),S_B=vec(),F_average_B=vec()[:,1:],tau_average_O_B=vec()[:,1:],
        x_O_W=vec(),u_O_W=vec(),omega_B=vec(),q_WB_wxyz=q,h=torch.full((B,N-1),.005,dtype=torch.float64),
        mass=torch.full((B,N),2.,dtype=torch.float64),gravity_W=g,interior_dimensions=torch.ones(B,3,dtype=torch.float64),
        episode_id=torch.zeros(B,N,dtype=torch.long),state_valid=torch.ones(B,N,dtype=torch.bool),
        history_valid=torch.ones(B,N,dtype=torch.bool),interval_valid=torch.ones(B,N-1,dtype=torch.bool))
    heads=('r_B','v_B','S_B','F_average_B','tau_average_O_B')
    results=[];gradients=[]
    for module in (loss,port_loss):
        inp={k:v.clone().requires_grad_(k in heads) for k,v in data.items()}
        out=module.discrete_constraints(**inp)
        (out['short']['loss']+out['long']['loss']).backward()
        results.append(out);gradients.append([inp[k].grad for k in heads])
    compare(*results);compare(*gradients)
    report=dict(synthetic_physics='bitwise equal including all-head gradients', recorded=None)
    if args.attempt:
        identities=verify_attempt(args.attempt)
        rows=original.merged_rows(args.attempt)
        compare(rows,port_data.merged_rows(args.attempt));compare(rows,list(iter_merged_rows(args.attempt)))
        dt=json.loads((Path(args.attempt)/'resolved.json').read_text())['native_dt']
        counts=[]
        for env in range(len(rows[0]['episode_id'])):
            a=original.environment_episodes(rows,env,dt);b=port_data.environment_episodes(rows,env,dt);compare(a,b)
            for ep in a:
                x=original.grid_intervals(a[ep],dt);y=port_data.grid_intervals(b[ep],dt);compare(x,y)
                aw=original.eligible_windows(x[0],a[ep]['nodes'],round(.005/dt))
                bw=port_data.eligible_windows(y[0],b[ep]['nodes'],round(.005/dt))
                signature=lambda w: ([[p['end'] for p in window] for window in w[0]], [[p['end'] for p in window] for window in w[1]], w[2])
                compare(signature(aw),signature(bw))
                counts.append(dict(environment=env,episode=ep,short=len(aw[0]),long=len(aw[1]),histories=aw[2]))
        report['recorded']=dict(path=str(Path(args.attempt).resolve()),rows=len(rows),counts=counts,
            hashes=identities,comparison='bitwise equal stream merge, episode extraction, grids, windows')
    atomic_json(args.output,report)
    print(json.dumps({k:v for k,v in report.items() if k!='recorded'},indent=2))
    if report['recorded']:print('Recorded rows and eligible windows match:',report['recorded']['counts'])


if __name__=='__main__':main()
