import numpy as np
import pytest
import torch
from helpers import preparation
from liquid_tcn.dataset import WindowDataset, make_loader
from liquid_tcn.evaluate import evaluate, wrench_metrics
from liquid_tcn.model import LiquidTCN, ModelConfig
from liquid_tcn.normalization import Normalization
from liquid_tcn.objectives import objective


def reference_lags(p, y, valid):
    out = np.zeros((len(p), 2))
    available = np.zeros((len(p), 2), dtype=bool)
    for i in range(len(p)):
        for head in (0, 1):
            a, b = p[i,:,head], y[i,:,head]
            if valid[i,:,head].all() and len(a) >= 10:
                a, b = a-a.mean(0), b-b.mean(0)
                if np.linalg.norm(a)>1e-10 and np.linalg.norm(b)>1e-10:
                    lags = list(range(-min(10,len(a)//4), min(10,len(a)//4)+1))
                    correlations=[]
                    for lag in lags:
                        aa=a[max(lag,0):len(a)+min(lag,0)]
                        bb=b[max(-lag,0):len(b)+min(-lag,0)]
                        correlations.append((aa*bb).sum()/max(np.linalg.norm(aa)*np.linalg.norm(bb),1e-30))
                    out[i,head]=lags[int(np.argmax(correlations))]*.005
                    available[i,head]=True
    return out, available


def test_vectorized_lags_peaks_and_masking():
    rng=np.random.default_rng(4)
    y=rng.normal(size=(12,41,5,3));p=np.roll(y,3,axis=1)
    mask=np.ones((12,41,5),dtype=bool)
    mask[0,:,0]=False;mask[1,4,1]=False
    p[2,:,0]=0.;y[3,:,1]=0.
    result=wrench_metrics(p,y,mask)
    expected,valid=reference_lags(p,y,mask)
    np.testing.assert_array_equal(result['lag_valid'],valid)
    np.testing.assert_array_equal(result['lag_seconds'][valid],expected[valid])
    assert not result['peak_valid'][0,0]
    for i in range(len(p)):
        for h in (0,1):
            if mask[i,:,h].any():
                peak=np.argmax(np.where(mask[i,:,h],np.linalg.norm(y[i,:,h],axis=-1),-np.inf))
                assert result['peaks'][i,h] == pytest.approx(np.linalg.norm(p[i,peak,h]-y[i,peak,h]))


@pytest.mark.parametrize('supervised', ['mse','huber'])
def test_validation_losses_use_dataset_counts_and_fixed_mse(tmp_path, supervised):
    path,manifest=preparation(tmp_path)
    torch.manual_seed(12)
    model=LiquidTCN(ModelConfig(**manifest['model']))
    norm=Normalization(manifest['normalization'])
    dataset=WindowDataset(path,'validation')
    # Unequal label coverage, plus an incomplete final batch, exercises the
    # distinction between batch means and dataset-level valid-count weighting.
    class Loader:
        def __init__(self): self.dataset=dataset
        def __iter__(self):
            for batch in make_loader(dataset,batch_size=7):
                batch['target_valid'][0,:2,0]=False
                yield batch
    sums=torch.zeros(5,dtype=torch.float64);counts=torch.zeros(5,dtype=torch.int64)
    mse=torch.zeros(5,dtype=torch.float64)
    physics={h:[torch.zeros(3,dtype=torch.float64),0] for h in ('short','long')}
    with torch.no_grad():
        for batch in Loader():
            pred=model(norm.normalize('motion',batch['motion']),norm.normalize('properties',batch['properties']))
            result=objective(pred,batch,norm,supervised=supervised,huber_delta=.4)
            sums+=result['supervised_sums'].double();counts+=result['supervised_counts']
            mse+=objective(pred,batch,norm)['supervised_sums'].double()
            for h in physics:
                r=result['physics'][h];physics[h][0]+=r['loss_by_channel']*r['count'];physics[h][1]+=r['count'].item()
    report=evaluate(model,Loader(),norm,supervised=supervised,huber_delta=.4,lambda_short=.3,lambda_long=.7)
    losses=report['losses']
    assert losses['supervised']==pytest.approx(float(sums.sum()/(3*counts.sum())))
    assert report['selection_metric']==pytest.approx(float(mse.sum()/(3*counts.sum())))
    for h in physics:
        assert losses[h]==pytest.approx(float(physics[h][0].sum()/physics[h][1]))
    assert losses['loss']==pytest.approx(losses['supervised']+.3*losses['short']+.7*losses['long'])
    assert sum(v for k,v in losses['components'].items() if k.startswith('supervised_contribution/'))==pytest.approx(losses['supervised'])
    assert losses['supervised_counts']==counts.tolist()
    assert report['coverage']['fraction']==1
