from dataclasses import asdict
import copy
import json

import torch

from helpers import make_attempt
from test_mamba_memory import config
from liquid_tcn.architecture_search import execute, monitor_population, architectures
from liquid_tcn.preprocess import prepare


def test_tiny_search_and_resume(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    cfg=config('mamba_stateful')
    sources=[]
    for i, split in enumerate(('train','validation','test')):
        path=make_attempt(tmp_path/split,samples=65,initial_reset=True,shift=i)
        sources.append(dict(path=str(path),group=split,split=split))
    data=tmp_path/'data'
    prepare(dict(model=asdict(cfg),sources=sources),data)
    settings=json.load(open('configs/architecture_search.json'))
    settings.update(data=str(data),device='cpu',micro_batch=1,concurrency=2,screening=.02,
        maximum_reference_updates=.04,monitor_every=.01,checkpoint_every=1,benchmark_environments=2,benchmark_steps=2,
        learning_rates=[.0003],effective_batches=[16],pretrain=0,physics_ramp=0,recurrent_ramp=0)
    models={}
    for family in ('tcn','hybrid'):
        c=asdict(cfg)
        c.update(episode_training=True,reset_backfill='recorded')
        if family=='tcn':c['temporal_memory']='none'
        if family=='recurrent':c['architecture']='recurrent_only'
        models[family]=dict(family=family,model=c)
    root=tmp_path/'search';root.mkdir()
    subset=monitor_population(settings,root,models['tcn'])
    import liquid_tcn.architecture_search as search
    monkeypatch.setattr(search,'cached_benchmark',lambda *a,**k: {})
    execute(settings,root,models,subset)
    before=(root/'decisions.json').read_bytes()
    with monkeypatch.context() as guard:
        def unexpected_assessment(*args,**kwargs):
            raise AssertionError('Completed validation/test assessments must be reused on resume')
        guard.setattr(search,'assess',unexpected_assessment)
        execute(settings,root,models,subset)
    assert (root/'decisions.json').read_bytes()==before
    assert (root/'final_comparison.json').exists()
    assert len(architectures([.2,.5,1.24]))==6
    decisions=json.loads(before)
    assert set(decisions)=={'screening','finalists'}
    assert set(decisions['screening']['winners'])=={'tcn','hybrid'}
    for phase,budget in [('screening',.02),('final',.04)]:
        for folder in (root/phase/'trials').iterdir():
            saved=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False)
            assert saved['exposure']==round(budget*6400)
            assert saved['sampler']['position']>0
            assert (folder/f'validation-{budget:g}.json').exists()
            if phase=='final':
                assert (folder/'selection.json').exists()
                assert (folder/'test.json').exists()
    # A worker interrupted after an optimizer operation resumes its stream.
    candidate=search.trial_config(models['hybrid'],False,.0003,16,1)
    interrupted=root/'interrupted'
    search.run_trial(candidate,settings,interrupted,.04,subset,max_optimizer_updates=1)
    path=interrupted/'trials'/search.trial_id(candidate)/'latest.pt'
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    assert 0<checkpoint['exposure']<round(.04*6400)
    result=search.run_trial(candidate,settings,interrupted,.04,subset)
    assert result['reference_updates']==.04
    assert result['optimizer_steps']>checkpoint['update']


def test_search_startup_and_one_second_boundaries(tmp_path):
    import numpy as np
    from test_pyramid import identity_norm
    from liquid_tcn.model import ModelConfig, build_model, CachedInference
    from liquid_tcn.episode_chunks import EpisodeChunks
    from liquid_tcn.objectives import objective
    norm=identity_norm()
    base=asdict(config('mamba_stateful'))
    base.update(episode_training=True,reset_backfill='recorded',consistency_intervals=200)
    for family in ('tcn','hybrid','recurrent'):
        cfg=copy.deepcopy(base)
        if family=='tcn':cfg['temporal_memory']='none'
        if family=='recurrent':cfg['architecture']='recurrent_only'
        model=build_model(ModelConfig(**cfg)).eval()
        props=torch.randn(2,24)
        motion=torch.randn(2,45,15)
        reset=motion[:,0]
        backfill=reset.clone()
        cache=CachedInference(model,norm,2)
        cache.set_properties(props);cache.initialize_reset(reset)
        state=model.temporal.initial_state(props,reset) if hasattr(model,'temporal') else ()
        with torch.no_grad():
            history=backfill[:,None].expand(-1,model.config.receptive_field-1,-1)
            expected=model(torch.cat((history,motion),1),props,memory_state=state)
            for i in range(45):
                actual,valid=cache.step(motion[:,i],return_all=True)
                assert valid.all()
                torch.testing.assert_close(actual,expected[:,i],atol=2e-5,rtol=2e-4)
    source=make_attempt(tmp_path/'raw',samples=450,initial_reset=True)
    data=tmp_path/'data'
    prepare(dict(model=base,sources=[dict(path=str(source),group='train',split='train')]),data)
    stream=EpisodeChunks(data,ModelConfig(**base),2,1,steps=400)
    short_count=long_count=0
    for batch in stream:
        # Ground-truth predictions give the same integrated constraints regardless of chunk partition.
        prediction=batch['targets'].float()
        result=objective(prediction,batch,norm,long_intervals=200)
        short_count+=int(result['physics']['short']['count'])
        long_count+=int(result['physics']['long']['count'])
    assert short_count==2*449
    assert long_count==2*250
    # Disjoint startup bands, physical RMS definitions, and a fixed elapsed cohort.
    from liquid_tcn.architecture_search import episode_metrics, summarize
    series=next(iter(stream.episodes.values()))[0]
    rows=np.repeat(stream.reader.read(series,0,1),2001)
    rows['interval_valid']=False
    rows['target_valid']=True
    prediction=rows['targets']+1.
    complete=dict(episode_metrics(rows,prediction,series,norm),series=1)
    short=dict(episode_metrics(rows[:2000],prediction[:2000],series,norm),series=2)
    summary=summarize([complete,short])
    assert summary['elapsed_cohort']['series']==[1]
    assert len(summary['elapsed'])==10
    assert all(b['count']==1200 for b in summary['elapsed'].values())
    assert sum(summary[k]['count'] for k in ('startup_200ms','post_200ms_to_1240ms','after_1240ms'))==summary['wrench']['count']
    np.testing.assert_allclose(summary['wrench']['force_moment_component_rms'],[1.,1.])
    np.testing.assert_allclose(summary['wrench']['force_moment_vector_rms'],np.sqrt(3)*np.ones(2))


def test_six_histories_and_cached_startup():
    from liquid_tcn.architecture_search import BASELINE_CONFIG, architecture_report, candidate_configs, ranked
    from liquid_tcn.model import ModelConfig, build_model, CachedInference
    from test_pyramid import identity_norm
    torch.set_num_threads(2)
    baseline=json.loads(BASELINE_CONFIG.read_text())['model']
    base=ModelConfig(**baseline)
    candidates=architectures([.2,.5,1.24])
    assert len(candidates)==6
    settings=json.load(open('configs/architecture_search.json'))
    assert len(candidate_configs(settings,candidates))==24
    assert ranked({'a':{'summary':{'wrench':{'normalized_mse':1.01},'impulse_score':0}},
                   'b':{'summary':{'wrench':{'normalized_mse':1.},'impulse_score':100}}})==['b','a']
    for requested,actual,blocks,tokens in [( .2,.2,3,5),(.5,.52,4,11),(1.24,1.24,5,28)]:
        model=ModelConfig(**candidates[f'tcn-{requested:g}s']['model'])
        hybrid=ModelConfig(**candidates[f'hybrid-{requested:g}s']['model'])
        assert model.reset_backfill==hybrid.reset_backfill=='recorded'
        assert model.dilations==hybrid.dilations
        assert model.attention_lags==hybrid.attention_lags
        assert (model.channels,model.attention_dim,model.attention_heads)==(base.channels,base.attention_dim,base.attention_heads)
        assert len(model.dilations)==blocks
        assert abs((model.receptive_field-1)*model.sample_seconds-actual)<1e-9
        assert sum(map(len,model.attention_lags))==tokens
        assert model.attention_lags[-1]==(0,)
        covered=1
        for dilation in model.dilations:
            assert dilation<=covered
            covered+=2*(model.kernel_size-1)*dilation
        for field,lags in zip(model.block_receptive_fields,model.attention_lags):
            assert list(lags)==sorted(set(lags))
            assert len(lags)>0 and max(lags)+field<=model.receptive_field
        if requested==1.24:
            assert model.dilations==base.dilations
            assert model.attention_lags==base.attention_lags
        for family in ('tcn','hybrid'):
            cfg=candidates[f'{family}-{requested:g}s']
            net=build_model(ModelConfig(**cfg['model'])).eval()
            props=torch.randn(2,24);motion=torch.randn(2,12,15)
            reset=motion[:,0];history=reset[:,None].expand(-1,net.config.receptive_field-1,-1)
            cache=CachedInference(net,identity_norm(),2)
            cache.set_properties(props);cache.initialize_reset(reset)
            state=net.temporal.initial_state(props,reset) if family=='hybrid' else ()
            with torch.no_grad():
                expected=net(torch.cat((history,motion),1),props,memory_state=state)
                for i in range(motion.shape[1]):
                    actual_prediction,valid=cache.step(motion[:,i],return_all=True)
                    assert valid.all()
                    torch.testing.assert_close(actual_prediction,expected[:,i],atol=2e-5,rtol=2e-4)
            assert architecture_report(cfg)['actual_history_seconds']==actual
