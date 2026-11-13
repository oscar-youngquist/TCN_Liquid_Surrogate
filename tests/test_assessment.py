"""Single-run assessment reuses shortlist selection without updating training state."""
from pathlib import Path
from threading import Event
import torch
from helpers import preparation
from liquid_tcn import assessment
from liquid_tcn.batch_logging import BatchLogger
from liquid_tcn.ported.storage import digest
from liquid_tcn.staged_validation import StageValidation,monitoring_subset
from liquid_tcn.train import train


def test_select_and_assess_local_run(tmp_path,monkeypatch):
    import liquid_tcn.staged_validation as validation
    torch.set_num_threads(1)
    data,manifest=preparation(tmp_path)
    run=tmp_path/'run';subset=tmp_path/'monitor_subset.json'
    monitoring_subset(data,subset)
    monkeypatch.setattr(validation,'MONITOR_EVERY',1)
    train(data,run,dict(updates=2,batch_size=2,threads=1,progress=False),
        expected_model=manifest['model'],staged_policy=StageValidation('6',2,subset))
    latest=digest(run/'latest.pt');events=sorted((run/'tensorboard').iterdir())
    original=assessment.read_json(tmp_path/'6/selection.json')
    # Simulate interruption after cached full validations but before selection commit.
    (tmp_path/'6/selection.json').unlink()
    def unexpected(*a,**kw):raise AssertionError('Completed evaluation should be reused')
    monkeypatch.setattr(validation,'evaluate',unexpected)
    result=assessment.select_checkpoint(run,data,subset,'cpu',1,Event(),BatchLogger('test',2))
    assert result['selected_update']==original['selected_update']
    assert digest(run/'latest.pt')==latest
    assert sorted((run/'tensorboard').iterdir())==events
    args=['--run',str(run),'--data',str(data),'--device','cpu','--threads','1']
    assert assessment.main(args)==0
    output=tmp_path/'assessment'
    assert assessment.read_json(output/'window_test.json')['coverage']['fraction']==1
    assert assessment.read_json(output/'summary.json')['status']=='complete'
    assert (output/'report.md').exists()
    monkeypatch.setattr(assessment,'evaluate',unexpected)
    assert assessment.main(args)==0
    assert digest(run/'latest.pt')==latest
