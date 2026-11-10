"""One small smoke scenario: alignment, physical quadrature and resumed dispatch."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from helpers import preparation
from liquid_tcn.checkpoint import atomic_save
from liquid_tcn.model import LiquidTCN,ModelConfig
from liquid_tcn.normalization import Normalization
from liquid_tcn.dataset import WindowDataset,make_loader
from liquid_tcn.preprocess import ROW_DTYPE
from liquid_tcn.trajectories import Trajectories,cache_predictions
from liquid_tcn.offline import horizon_errors,shared_events,assess,fit_reference
from liquid_tcn import experiment
from liquid_tcn.ported.storage import atomic_json


def test_chronological_experiment_smoke(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    # Translating base and constant wrench offsets: impulse must include origin
    # transport, while work uses the same endpoint velocity quadrature for both.
    rows=np.zeros(451,dtype=ROW_DTYPE)
    rows['q_WB_wxyz'][:,0]=1;rows['mass']=2;rows['gravity_W'][:,2]=-9.81
    rows['h']=.005;rows['tick']=np.arange(len(rows))*5
    rows['state_valid']=rows['history_valid']=rows['interval_valid']=True;rows['target_valid']=True
    rows['targets'][:,0,2]=-19.62
    rows['x_O_W'][:,1]=np.arange(len(rows))*.005*3;rows['u_O_W'][:,1]=3
    predicted=rows['targets'].copy();predicted[:,0,:2]+=[1,2];predicted[:,1,2]+=2
    horizons,power=horizon_errors(rows,predicted,[.3,.2,.4],np.zeros(3))
    np.testing.assert_allclose(power,6.)
    for n,v in horizons.items():
        T=n*.005
        np.testing.assert_allclose(v['force'],np.tile([T,2*T,0],(len(v['ends']),1)),atol=1e-12)
        np.testing.assert_allclose(v['moment'][:,2],2*T-3*(T*T/2+T*.005/2),atol=1e-11)
        np.testing.assert_allclose(v['work'],6*T,atol=1e-12)
    data,manifest=preparation(tmp_path)
    cfg=ModelConfig(**manifest['model']);model=LiquidTCN(cfg).eval()
    checkpoint=tmp_path/'model.pt'
    atomic_save(checkpoint,dict(model=model.state_dict(),model_config=manifest['model'],normalization=manifest['normalization']))
    reader=cache_predictions(data,checkpoint,tmp_path/'cache',block_rows=17)
    ds=WindowDataset(data,'test');batch=next(iter(make_loader(ds,batch_size=1)))
    norm=Normalization(manifest['normalization'])
    with torch.no_grad():expected=norm.denormalize('targets',model(norm.normalize('motion',batch['motion']),norm.normalize('properties',batch['properties']))).numpy()[0]
    sid=int(batch['series'][0]);end=int(batch['end'][0]);cache=np.load(tmp_path/'cache'/f'series-{sid:06d}.npy')
    np.testing.assert_allclose(cache[end-cfg.long_intervals:end+1],expected,rtol=2e-5,atol=2e-5)
    series=reader.dataset.series[sid];segments=list(reader.segments(series))
    intervals=[i for start,end in segments for i in range(start+1,end)]
    assert len(intervals)==len(set(intervals)) and len(intervals)<len(ds)*(cfg.long_intervals+1)
    # Insert a reset and rebuilt history into one prepared test series.
    for shard in range(series['shards']):
        path=data/f'series-{sid:06d}-{shard:06d}.npy';part=np.load(path)
        ids=np.arange(len(part))+shard*manifest['shard_rows']
        part['episode_id'][ids>=20]+=1
        part['history_valid'][(ids>=20)&(ids<20+cfg.receptive_field)]=False
        np.save(path,part)
    reader=Trajectories(data,block_rows=17);segments=list(reader.segments(series))
    assert len(segments)==2
    assert all(not (start<20<end) for start,end in segments)
    # Tiny assessment exercises cached reuse and reports horizons too long for
    # these trajectories as unavailable instead of reporting zero residuals.
    (tmp_path/'shared').mkdir()
    ref=fit_reference(data,tmp_path/'shared/reference.json')
    ref['force_event']=0.  # Exercise event merging/streaming with a deterministic fixture.
    atomic_json(tmp_path/'shared/reference.json',ref)
    ref,events=shared_events(data,tmp_path/'shared')
    report=assess(data,checkpoint,tmp_path/'assessment',tmp_path/'shared',block_rows=17)
    assert report['coverage']['intervals']>0 and report['unavailable']
    assert report['coverage']['events']>0
    assert (tmp_path/'assessment'/f'transient-{sid:06d}.json').exists()
    assert assess(data,checkpoint,tmp_path/'assessment',tmp_path/'shared')==report
    # Resume orchestration uses the existing Coordinator; fake workers keep this
    # a dispatch test, without twelve training runs or any GPU probe.
    root=tmp_path/'experiment';root.mkdir()
    selected=experiment.generate_trials(data,root,['cpu'],1)['00']
    selected['model']=manifest['model'];selected['training']['batch_size']=2
    atomic_json(tmp_path/'selected.json',selected)
    args=SimpleNamespace(data=data,resume=False,assess_only=False,config=tmp_path/'selected.json',threads=1,concurrency=2,max_updates=2,block_rows=17)
    manifest,state=experiment.initialize(root,args,['cpu'])
    full=json.loads((root/'trials/full-s1/resolved.json').read_text())
    short=json.loads((root/'trials/short-s1/resolved.json').read_text())
    assert full['training']['seed']==short['training']['seed']==1
    assert full['training']['pretrain_updates']==500 and full['training']['ramp_updates']==1500
    assert full['training']['physics_lr']==short['training']['physics_lr']==.001
    assert short['training']['lambda_long']==0 and short['training']['lambda_short']==1
    assert experiment.run_worker(root,'full-s1','train')==0
    selection=experiment.inspect(root,'full-s1','train',manifest)
    assert selection['selected_update']==2
    state['trials']['full-s1']['stages']['train']=dict(status='complete',result=selection)
    atomic_json(root/'state.json',state)
    args.resume=True;args.threads=2
    manifest,state=experiment.initialize(root,args,['cpu'])
    launched=[]
    def command(root,trial,stage,resume):
        launched.append(trial)
        return ['/bin/true']
    coordinator=experiment.Coordinator(root,manifest,state,2,command,lambda *a:dict(update=2))
    assert coordinator.run('train',['full-s1','short-s1'])
    assert launched==['short-s1'] and state['trials']['full-s1']['stages']['train']['result']['update']==2

    # Report aggregation uses cached assessment data; no additional model runs.
    from liquid_tcn.experiment_report import write_report,bootstrap_difference
    for trial in manifest['configs']:
        folder=root/'trials'/trial/'assessment';folder.mkdir(parents=True,exist_ok=True)
        atomic_json(folder/'summary.json',report)
        atomic_json(folder/'window_test.json',{'selection_metric':1.})
    atomic_json(root/'selected.json',{t:selection for t in manifest['configs']})
    write_report(root,manifest,state,make_plots=False)
    assert (root/'metrics.csv').exists() and (root/'report.md').exists()
    summary=json.loads((root/'report.json').read_text())
    key='endpoint/force/norm'
    assert summary['paired_full_minus_ablation'][key+'/rms']['short']['mean']==0.
    # Two whole groups, identical paired models: the cluster CI is exactly zero.
    import copy
    paired={t:copy.deepcopy(report) for t in manifest['configs']}
    for r in paired.values():r['groups']['second-group']=copy.deepcopy(next(iter(r['groups'].values())))
    assert bootstrap_difference(paired,'short',key,20)['ci95']==[0.,0.]
