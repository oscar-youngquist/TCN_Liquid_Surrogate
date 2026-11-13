"""Standalone CLI uses the same subset and shortlist policy as experiments."""
import json
import sys
from pathlib import Path
from helpers import preparation
from liquid_tcn.cli import main
from liquid_tcn.checkpoint import load_checkpoint
from liquid_tcn.staged_validation import monitoring_subset


def test_cli_shortlist_resume(tmp_path, monkeypatch):
    import liquid_tcn.staged_validation as validation
    # Exercise monitoring/selection with two updates rather than 2,000.
    monkeypatch.setattr(validation, 'MONITOR_EVERY', 1)
    path, manifest = preparation(tmp_path, model=dict(channels=8,embedding_dim=4,
        kernel_size=3,dilations=[1,2],long_intervals=4,
        architecture='tcn_pyramid_attention',attention_ages_ms=[[0,20],[0]]))
    settings=dict(policy='shortlist',seed=1729,subset=None)
    training=dict(updates=2,pretrain_updates=0,ramp_updates=1,batch_size=2,
        threads=2,device='cpu',num_workers=0,progress=False,
        validation_every=1000,checkpoint_every=200,patience=40001)
    output=tmp_path/'run'
    config=tmp_path/'config.json'
    config.write_text(json.dumps(dict(data=str(path),output=str(output),model=manifest['model'],
                                    training=training,monitoring=settings)))
    def invoke(*args):
        monkeypatch.setattr(sys,'argv',['liquid-tcn','train',*args])
        main()
    invoke('--config',str(config),'--max-updates','1')
    saved=load_checkpoint(output/'latest.pt')
    assert saved['update']==1 and saved['stage_monitoring']['checks'][0]['update']==1
    subset=(output/'monitor_subset.json').read_bytes()
    reference=monitoring_subset(path,tmp_path/'experiment_subset.json')
    assert json.loads(subset)['indices']==reference['indices']
    assert not (output/'selection/selection.json').exists()
    # No config argument: checkpoint settings and persisted monitoring survive.
    invoke('--data',str(path),'--output',str(output),'--resume',str(output/'latest.pt'))
    saved=load_checkpoint(output/'latest.pt')
    assert saved['update']==2 and len(saved['stage_monitoring']['checks'])==2
    assert (output/'monitor_subset.json').read_bytes()==subset
    selection=json.loads((output/'selection/selection.json').read_text())
    assert selection['selection_complete'] and len(selection['evaluations'])==2
    assert (output/'best.pt').exists() and not (tmp_path/'6').exists()
    assert selection['selected_update']==load_checkpoint(output/'best.pt')['update']
    assert all(json.loads(Path(r['report_path']).read_text())['report']['coverage']['fraction']==1
               for r in selection['evaluations'])
    from liquid_tcn import assessment
    assert assessment.main(['--run',str(output),'--device','cpu','--threads','2','--window-only'])==0
    assert (output/'assessment/window_test.json').exists()


def test_pyramid_config_matches_full_training_settings():
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/'configs/pyramid_attention.json').read_text())
    t=config['training']
    assert t['updates']==40000 and t['checkpoint_every']==200 and t['validation_every']==1000
    assert t['threads']==2 and t['validation_batch_size']==1024 and t['validation_num_workers']==0
    assert t['patience']==40001 and not t['progress']
    assert config['monitoring']==dict(policy='shortlist',seed=1729,subset=None)
