import json
import numpy as np
import torch
from dataclasses import asdict
from helpers import make_attempt
from liquid_tcn.preprocess import prepare
from liquid_tcn.model import build_model
from liquid_tcn.checkpoint import atomic_save
from liquid_tcn.assessment import assess_checkpoint
from liquid_tcn.train import DEFAULTS
from test_film_streaming import config


def test_same_test_intervals_and_offline_metrics(tmp_path,monkeypatch):
    model=build_model(config()).eval()
    with torch.no_grad():model.film.weight.normal_(0,.03)
    sources=[]
    for i,split in enumerate(('train','validation','test')):
        path=make_attempt(tmp_path/split,samples=80,reset_at=40,shift=float(i))
        sources.append(dict(path=str(path),group=split,split=split))
    data=tmp_path/'data'
    manifest=prepare(dict(model=asdict(model.config),sources=sources,shard_rows=17),data)
    checkpoint=tmp_path/'model.pt'
    atomic_save(checkpoint,dict(model=model.state_dict(),model_config=asdict(model.config),
        normalization=manifest['normalization'],config=DEFAULTS,update=1,data_manifest_sha256='fixture'))
    output=tmp_path/'assessment'
    assess_checkpoint(data,checkpoint,output,tmp_path/'shared',DEFAULTS,'cpu',block_rows=17)
    full=json.loads((output/'summary.json').read_text())
    stream=json.loads((output/'streaming/summary.json').read_text())
    assert full['coverage']==stream['coverage']
    assert full['exclusions_nonexclusive']==stream['exclusions_nonexclusive']
    assert full['metrics'].keys()==stream['metrics'].keys()
    for key in full['metrics']:
        assert full['metrics'][key]['count']==stream['metrics'][key]['count']
        np.testing.assert_allclose(full['metrics'][key]['rms'],stream['metrics'][key]['rms'],rtol=2e-5,atol=2e-5)
    comparison=json.loads((output/'streaming_comparison.json').read_text())
    assert comparison['prediction_difference']['endpoints']>0
    assert np.max(comparison['prediction_difference']['max_abs_by_head_coordinate'])<2e-5
    assert (output/'streaming_comparison.csv').exists()
    # All completed predictions/metrics survive a repeated assessment invocation.
    from liquid_tcn import assessment
    monkeypatch.setattr(assessment,'evaluate',lambda *a,**kw:(_ for _ in ()).throw(AssertionError('must reuse reports')))
    assess_checkpoint(data,checkpoint,output,tmp_path/'shared',DEFAULTS,'cpu',block_rows=17)
    from liquid_tcn.cli import main
    import sys
    path=tmp_path/'partial.json'
    monkeypatch.setattr(sys,'argv',['liquid-tcn','evaluate','--data',str(data),'--checkpoint',str(checkpoint),
        '--output',str(path),'--max-batches','1','--batch-size','2'])
    main()
    result=json.loads(path.read_text())
    assert result['inference_mode']=='both'
    assert result['streaming']['coverage']==result['coverage']
    assert abs(result['streaming_mse_difference'])<2e-5
