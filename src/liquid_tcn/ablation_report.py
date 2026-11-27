"""Seed variability, paired collection bootstrap, compact tables and shareable plots."""
import csv
import json
from pathlib import Path
import numpy as np
from .ported.storage import atomic_json
from .ablation_evaluation import metric_specs, HORIZONS
from .experiment_report import unit


def read(path):return json.loads(Path(path).read_text())


def bootstrap_pair(reports,variant,seeds,key,repetitions=2000,seed=1729):
    """Paired whole-group draws shared by variants/seeds; training seeds stay fixed."""
    pairs=[(reports[f'full-s{s}'],reports[f'{variant}-s{s}']) for s in seeds]
    groups=sorted(pairs[0][0]['groups'])
    if any(set(r['groups'])!=set(groups) for pair in pairs for r in pair):
        raise ValueError('Paired evaluation collection groups differ')
    counts=np.array([[[r['groups'][g][key]['n'] for g in groups] for r in pair] for pair in pairs],float)
    squares=np.array([[[r['groups'][g][key]['square'] for g in groups] for r in pair] for pair in pairs],float)
    if not np.all(counts==counts[0,0]):raise ValueError(f'Common scoring counts differ: {key}')
    eligible=counts[0,0]>0;count=int(eligible.sum())
    if count<2:return dict(status='unavailable',groups=count,reason='Fewer than two collection groups with valid observations')
    counts=counts[:,:,eligible];squares=squares[:,:,eligible]
    weights=np.random.default_rng(seed).multinomial(count,np.full(count,1/count),size=repetitions)
    reduction=pairs[0][0]['metrics'][key]['reduction']
    reduce=lambda x:x if reduction=='mse' else np.sqrt(x)
    values=reduce(np.einsum('rg,svg->svr',weights,squares)/np.einsum('rg,svg->svr',weights,counts))
    differences=(values[:,0]-values[:,1]).mean(0)
    point=reduce(squares.sum(2)/counts.sum(2))
    return dict(status='measured',groups=count,zero_count_groups=len(groups)-count,repetitions=repetitions,
        mean_difference=float((point[:,0]-point[:,1]).mean()),
        per_seed_differences=(point[:,0]-point[:,1]).tolist(),
        seed_difference_sd=float(np.std(point[:,0]-point[:,1],ddof=1)) if len(seeds)>1 else None,ci95=np.percentile(differences,[2.5,97.5]).tolist(),
        method='paired collection-group percentile bootstrap; same draws for both variants and all fixed training seeds; recompute count-weighted metric for each seed, then average paired differences')


def csv_file(path,rows,fields):
    with open(path,'w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def make_plots(destination,statistics,historical):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(supervised='C0',short='C1',long='C2',full='C3')
    def plot(ax,keys,x):
        for variant,color in colors.items():
            values=[statistics.get(k,{}).get(variant,{}) for k in keys]
            y=[v.get('mean',np.nan) if v.get('mean') is not None else np.nan for v in values]
            sd=[v.get('sample_sd') or 0 for v in values]
            if np.isfinite(y).any():ax.errorbar(x,y,yerr=sd,label=variant,color=color,marker='o',capsize=2)
        for label,report in historical.items():
            y=[report['metrics'].get(k,{}).get('value') for k in keys]
            if any(v is not None for v in y):ax.plot(x,[v if v is not None else np.nan for v in y],linestyle='--',label=label)
    def save(fig,name):
        fig.tight_layout()
        for suffix in ('png','pdf'):fig.savefig(destination/f'{name}.{suffix}',dpi=160)
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for head,ax,unit in zip(('force','moment'),axes,('N','N m')):
        plot(ax,[f'reconstruction/profile_{i}_{i+1}s/{head}/vector' for i in range(10)],np.arange(10)+.5)
        ax.set(xlabel='Episode age (s)',ylabel=f'{head} 3D vector RMS ({unit})',title='Common cohort reaching 10 s')
    axes[0].legend();save(fig,'reconstruction_vs_episode_age')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for head,ax,unit in zip(('force','moment'),axes,('N s','N m s')):
        plot(ax,[f'impulse/reset_inclusive/{h*5}ms/{head}/vector' for h in HORIZONS],np.asarray(HORIZONS)*.005)
        ax.axvline(.2,color='gray',linestyle=':',label='200 ms');ax.set_xscale('log')
        ax.set(xlabel='Horizon (s)',ylabel=f'{head} impulse error vector RMS ({unit})')
    axes[0].legend();save(fig,'impulse_error_vs_horizon')
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for head,ax,unit in zip(('force','moment'),axes[0],('N','N m')):
        plot(ax,[f'reconstruction/{p}/{head}/vector' for p in ('0_200ms','200ms_history','after_history')],np.arange(3))
        ax.set(xticks=np.arange(3),xticklabels=['0–200 ms','200 ms–history','After history'],ylabel=f'{head} vector RMS ({unit})')
    for channel,ax in zip(('P','H'),axes[1]):
        plot(ax,[f'consistency/{scope}/200ms/predicted/{channel}/normalized' for scope in ('startup','after_history')],np.arange(2))
        floors=[statistics.get(f'consistency/{scope}/200ms/sph/{channel}/normalized',{}).get('full',{}).get('mean') for scope in ('startup','after_history')]
        if any(v is not None for v in floors):ax.plot(np.arange(2),[v if v is not None else np.nan for v in floors],color='black',linestyle=':',label='SPH floor')
        ax.legend()
        ax.set(xticks=np.arange(2),xticklabels=['Startup','After history'],ylabel=f'Normalized {channel} consistency RMS (200 ms)')
    axes[0,0].legend();save(fig,'startup_vs_mature')


def write_report(root):
    root=Path(root);manifest=read(root/'ablation.json');destination=root/'report';destination.mkdir(exist_ok=True)
    reports={};historical={};selections=[]
    for trial in manifest['trials']:
        complete=root/'trials'/trial/'complete.json'
        if not complete.exists():continue
        done=read(complete);report=read(done['evaluation'])
        if done['kind']=='historical':historical[trial]=report
        else:reports[trial]=report
        selected=done['selection']
        selections.append(dict(trial=trial,variant=done['variant'],seed=done['seed'],
            selected_update=selected.get('selected_update'),stopping_update=selected.get('update'),
            stopping_reason=selected.get('stopping_reason','historical'),validation_mse=selected.get('normalized_mse'),
            monitor_windows=selected.get('monitor_windows'),effective_patience=selected.get('effective_patience'),
            parameters=report['parameters'],episodes=report['coverage']['episodes'],groups=report['coverage']['collection_groups']))
    individual=[];statistics={};seed_rows=[]
    for key in metric_specs():
        by_variant={}
        for trial,report in {**reports,**historical}.items():
            value=report['metrics'][key]
            individual.append(dict(trial=trial,metric=key,unit=unit(key),value=value['value'],count=value['n']))
        for variant in ('supervised','short','long','full'):
            values=[reports[f'{variant}-s{s}']['metrics'][key]['value'] for s in manifest['seeds'] if f'{variant}-s{s}' in reports]
            valid=[v for v in values if v is not None]
            row=dict(n=len(valid),mean=float(np.mean(valid)) if valid else None,
                     sample_sd=float(np.std(valid,ddof=1)) if len(valid)>1 else None)
            by_variant[variant]=row;seed_rows.append(dict(variant=variant,metric=key,unit=unit(key),**row))
        statistics[key]=by_variant
    bootstrap={};bootstrap_rows=[]
    for variant in ('supervised','short','long'):
        if not all(f'{v}-s{s}' in reports for v in ('full',variant) for s in manifest['seeds']):continue
        bootstrap[variant]={}
        for key in metric_specs():
            if '/sph/' in key:continue
            value=bootstrap_pair(reports,variant,manifest['seeds'],key,
                manifest['settings'].get('bootstrap_repetitions',2000),manifest['settings'].get('bootstrap_seed',1729))
            bootstrap[variant][key]=value
            bootstrap_rows.append(dict(comparison=f'full-minus-{variant}',metric=key,unit=unit(key),status=value['status'],groups=value['groups'],seed_difference_sd=value.get('seed_difference_sd'),
                difference=value.get('mean_difference'),lower95=(value.get('ci95') or [None,None])[0],upper95=(value.get('ci95') or [None,None])[1]))
    summary=dict(completed_runs=len(reports),expected_runs=4*len(manifest['seeds']),historical_references=list(historical),
        seed_statistics=statistics,paired_collection_bootstrap=bootstrap,selections=selections,
        uncertainty='sample_sd is across training seeds (ddof=1). Bootstrap CIs resample paired collection groups conditional on those fixed seeds; these are different sources of uncertainty.',
        definitions=dict(vector_rms='sqrt(mean(sum_xyz(error^2))); per-axis RMS reported separately',
            normalized_wrench_mse='mean squared force/moment coordinate error divided by one shared training-derived per-axis scale',
            physics='self-consistency is distinct from predicted-versus-recorded impulse error; SPH floors use identical integration/masks',
            selection='mature all-head validation reconstruction MSE only; test has no selection role'))
    atomic_json(destination/'summary.json',summary)
    csv_file(destination/'per_run_metrics.csv',individual,['trial','metric','unit','value','count'])
    csv_file(destination/'seed_summary.csv',seed_rows,['variant','metric','unit','n','mean','sample_sd'])
    csv_file(destination/'paired_bootstrap.csv',bootstrap_rows,['comparison','metric','unit','status','groups','seed_difference_sd','difference','lower95','upper95'])
    if selections:csv_file(destination/'selections.csv',selections,list(selections[0]))
    lines=['# Reset-backfill physics ablation','',f"Completed: {len(reports)}/{4*len(manifest['seeds'])} runs. Historical references: {len(historical)}.",'',
        'All comparisons use fixed episode/timestamp masks and training-derived evaluation scales. Checkpoints are selected using mature validation reconstruction only.',
        '', '| Variant | Whole-episode wrench MSE | 0–200 ms wrench MSE | After-history wrench MSE |', '|---|---:|---:|---:|']
    for variant in ('supervised','short','long','full'):
        cells=[]
        for period in ('overall','0_200ms','after_history'):
            v=statistics[f'reconstruction/{period}/normalized_wrench_mse'][variant]
            cells.append('unavailable' if v['mean'] is None else f"{v['mean']:.6g}"+(f" ± {v['sample_sd']:.3g}" if v['sample_sd'] is not None else ' (one seed)'))
        lines.append('| '+variant+' | '+' | '.join(cells)+' |')
    lines+=['','Values are means ± sample SD across training seeds. Zero-count horizons/bins remain unavailable. Per-run CSV includes valid counts; per-episode reports retain exclusions.',
        '', '## Paired collection-group uncertainty','',
        'Whole collection groups are resampled with replacement. Each draw is shared across variants and all fixed seeds; metrics are recomputed from summed squared errors and counts before averaging seed-paired differences. Groups with zero valid observations for a metric are listed as zero-count groups. This measures collection sampling uncertainty, not training-seed variability.']
    for variant,metrics in bootstrap.items():
        for period in ('overall','0_200ms','after_history'):
            value=metrics[f'reconstruction/{period}/normalized_wrench_mse']
            if value['status']=='measured':
                lo,hi=value['ci95'];interpret='lower error for full' if hi<0 else 'higher error for full' if lo>0 else 'difference unresolved by this interval'
                lines.append(f"- Full minus {variant}, {period}: {value['mean_difference']:.6g}, paired 95% CI [{lo:.6g}, {hi:.6g}]; {interpret}.")
    lines+=['','Reconstruction and impulse errors assess prediction accuracy. Physics residuals assess self-consistency; the SPH floor is reported separately. No single metric should substitute for the others.',
        'Plots show reconstruction versus age on the common 10-second cohort, impulse error versus horizon (200 ms marked), and startup versus mature reconstruction/consistency. Error bars represent seed SD, not bootstrap CIs.']
    (destination/'assessment.md').write_text('\n'.join(lines)+'\n')
    if reports or historical:make_plots(destination,statistics,historical)
    return summary
