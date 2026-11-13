"""Seed-paired summaries, collection-group bootstrap, CSV tables and offline plots."""
import csv
import json
import math
from pathlib import Path
import numpy as np
from .ported.storage import atomic_json
from .sweep import read_json

VARIANTS=('full','supervised','short','long')
KEY_METRICS=('endpoint/force/norm','endpoint/moment/norm','impulse/200ms/force/norm',
    'impulse/200ms/moment/norm','consistency/1000ms/predicted/P/norm',
    'consistency/1000ms/predicted/H/norm','work/200ms/signed_error_J')


def unit(name):
    if 'timing' in name or 'settling' in name:return 's'
    if name.startswith('power'):return 'W'
    if name.startswith('work'):return 'J'
    if 'normalized' in name:return 'dimensionless'
    if name.startswith('consistency'):
        return 'm' if '/r/' in name else 'N s' if '/P/' in name else 'N m s'
    if 'impulse' in name:return 'N m s' if 'moment' in name else 'N s'
    for label,value in (('force','N'),('moment','N m'),('r_B','m'),('v_B','m/s'),('S_B','kg m^2/s')):
        if label in name:return value
    return 'unspecified'


def bootstrap_difference(reports, variant, metric, repetitions=1000):
    """Resample whole collection groups; all seeds and variants share each draw.

    Recompute each seed's count-weighted RMS, then average its paired difference.
    This conditions on the three trained seeds and estimates collection sampling
    uncertainty. It does not treat rolling horizons or overlapping windows as IID.
    """
    group_sets=[]
    for seed in (1,2,3):
        for v in ('full',variant):
            group_sets.append({g for g,m in reports[f'{v}-s{seed}']['groups'].items() if metric in m and m[metric]['n']})
    groups=sorted(set.intersection(*group_sets))
    if len(groups)<2:return dict(status='unavailable',reason='Fewer than two shared collection groups',groups=len(groups))
    counts=np.empty((3,2,len(groups)));squares=np.empty_like(counts)
    for i,seed in enumerate((1,2,3)):
        for j,v in enumerate(('full',variant)):
            for k,g in enumerate(groups):
                row=reports[f'{v}-s{seed}']['groups'][g][metric]
                counts[i,j,k]=row['n'];squares[i,j,k]=row['square']
    rng=np.random.default_rng(1729);values=[]
    for _ in range(repetitions):
        ids=rng.integers(len(groups),size=len(groups))
        rms=np.sqrt(squares[:,:,ids].sum(2)/counts[:,:,ids].sum(2))
        values.append(float((rms[:,0]-rms[:,1]).mean()))
    return dict(status='measured',groups=len(groups),repetitions=repetitions,
        ci95=np.percentile(values,[2.5,97.5]).tolist(),unit=unit(metric),
        definition='paired full-minus-ablation RMS; collection-group bootstrap, fixed three seeds')


def summarize(reports):
    names=sorted(set.intersection(*(set(v['metrics']) for v in reports.values())))
    means={};paired={}
    for name in names:
        for stat in ('mean','rms','median','p95','p99','worst_1pct_mean'):
            key=name+'/'+stat
            means[key]={};paired[key]={}
            full=np.array([reports[f'full-s{s}']['metrics'][name][stat] for s in (1,2,3)])
            for variant in VARIANTS:
                values=np.array([reports[f'{variant}-s{s}']['metrics'][name][stat] for s in (1,2,3)])
                means[key][variant]=dict(mean=float(values.mean()),sample_sd=float(values.std(ddof=1)),n=3)
                if variant!='full':
                    difference=full-values
                    paired[key][variant]=dict(mean=float(difference.mean()),sample_sd=float(difference.std(ddof=1)),per_seed=difference.tolist())
    bootstrap={name:{v:bootstrap_difference(reports,v,name) for v in VARIANTS[1:]} for name in KEY_METRICS}
    return dict(seed_statistics=means,paired_full_minus_ablation=paired,bootstrap=bootstrap)


def plots(root,reports,summary,source_root=None,inference_mode='full'):
    source_root=Path(source_root or root)
    assessment='assessment/streaming' if inference_mode=='streaming' else 'assessment'
    # Plotting is an installation dependency, imported only by the final report.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination=root/'plots';destination.mkdir(exist_ok=True)
    for kind,name,units in [('impulse','force','N s'),('impulse','moment','N m s'),('work','signed_error_J','J')]:
        fig,ax=plt.subplots()
        for variant in VARIANTS:
            x=[];y=[];sd=[]
            for ms in (50,200,500,1000,2000):
                key=f'{kind}/{ms}ms/{name}'+('/norm/rms' if kind=='impulse' else '/rms')
                if key in summary['seed_statistics']:
                    value=summary['seed_statistics'][key][variant]
                    x.append(ms);y.append(value['mean']);sd.append(value['sample_sd'])
            if x:ax.errorbar(x,y,yerr=sd,marker='o',label=variant,capsize=3)
        ax.set(xlabel='Horizon (ms)',ylabel=f'RMS error ({units})',title='Mean ± sample SD across three seeds')
        if ax.lines:ax.legend()
        fig.tight_layout();fig.savefig(destination/f'{kind}-{name}.png',dpi=160);plt.close(fig)
    for channel,units in (('r','m'),('P','N s'),('H','N m s')):
        fig,ax=plt.subplots()
        for variant in VARIANTS:
            for source in ('predicted','sph'):
                x=[];y=[]
                for ms in (500,1000,2000):
                    key=f'consistency/{ms}ms/{source}/{channel}/norm/rms'
                    if key in summary['seed_statistics']:
                        x.append(ms);y.append(summary['seed_statistics'][key][variant]['mean'])
                if x and (source=='predicted' or variant=='full'):
                    ax.plot(x,y,marker='o',linestyle='--' if source=='sph' else '-',label='SPH floor' if source=='sph' else variant)
        ax.set(xlabel='Horizon (ms)',ylabel=f'RMS residual ({units})',title=f'Discrete consistency: {channel}')
        if ax.lines:ax.legend()
        fig.tight_layout();fig.savefig(destination/f'consistency-{channel}.png',dpi=160);plt.close(fig)
    for name in ('force','moment'):
        fig,ax=plt.subplots();x=np.arange(4)
        for i,variant in enumerate(VARIANTS):
            values=[]
            for stat in ('median','p95','p99','worst_1pct_mean'):
                key=f'endpoint/{name}/norm/{stat}'
                values.append(summary['seed_statistics'].get(key,{}).get(variant,{}).get('mean',np.nan))
            ax.plot(x,values,marker='o',label=variant)
        ax.set(xticks=x,xticklabels=['Median','95th','99th','Worst 1% mean'],ylabel=f'Vector error ({unit(name)})',title=f'{name}: reservoir tail estimates')
        ax.legend();fig.tight_layout();fig.savefig(destination/f'tails-{name}.png',dpi=160);plt.close(fig)
    # Earliest detected event in the lowest series ID, seed 1, for every variant.
    paths=sorted((source_root/'trials/full-s1'/assessment).glob('transient-*.json'))
    if paths:
        filename=paths[0].name;fig,axes=plt.subplots(2,1,sharex=True)
        for variant in VARIANTS:
            path=source_root/'trials'/f'{variant}-s1'/assessment/filename
            if not path.exists():continue
            trace=np.asarray(read_json(path)['trace'])
            for h,ax in enumerate(axes):
                ax.plot(trace[:,0]-trace[0,0],trace[:,h+1],label=variant)
                if variant=='full':ax.plot(trace[:,0]-trace[0,0],trace[:,h+3],color='black',linestyle='--',label='SPH')
        axes[0].set(ylabel='Force magnitude (N)');axes[1].set(xlabel='Time from event start (s)',ylabel='Moment magnitude (N m)')
        axes[0].legend();fig.tight_layout();fig.savefig(destination/'transient-seed1.png',dpi=160);plt.close(fig)


def write_report(root,manifest,state,make_plots=True,inference_mode='full'):
    source_root=Path(root)
    assessment='assessment/streaming' if inference_mode=='streaming' else 'assessment'
    root=source_root/'streaming_report' if inference_mode=='streaming' else source_root
    root.mkdir(parents=True,exist_ok=True)
    reports={t:read_json(source_root/'trials'/t/assessment/'summary.json') for t in manifest['configs']}
    summary=summarize(reports)
    summary.update(inference_mode=inference_mode,individual=reports,selection=read_json(source_root/'selected.json'),
        window_weighted_reports={t:str(source_root/'trials'/t/assessment/'window_test.json') for t in reports})
    from .staged_sweep import summarize_seeds
    window_reports=[]
    for trial in reports:
        variant,seed=trial.rsplit('-s',1)
        window_reports.append(dict(selection=dict(label=variant+'_seed'+seed,seed=int(seed)),
            report=read_json(source_root/'trials'/trial/assessment/'window_test.json')))
    summary['window_weighted_seed_statistics']=summarize_seeds(window_reports)
    atomic_json(root/'report.json',summary)
    with open(root/'metrics.csv.tmp','w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['trial','metric','unit','count','mean','rms','median','p95','p99','worst_1pct_mean'])
        for trial,report in reports.items():
            for name,values in sorted(report['metrics'].items()):
                writer.writerow([trial,name,unit(name),*[values[k] for k in ('count','mean','rms','median','p95','p99','worst_1pct_mean')]])
    (root/'metrics.csv.tmp').replace(root/'metrics.csv')
    with open(root/'conditions.csv.tmp','w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['trial','condition','count','groups','median','p95','p99','worst_1pct_mean'])
        for trial,report in reports.items():
            for name,v in report['conditions'].items():writer.writerow([trial,name,v['count'],v.get('groups'),*[v[k] for k in ('median','p95','p99','worst_1pct_mean')]])
    (root/'conditions.csv.tmp').replace(root/'conditions.csv')
    if make_plots:plots(root,reports,summary,source_root,inference_mode)
    lines=['# Full model and matched-seed ablations','',
        'Checkpoints were selected on full validation from the monitoring shortlist. Test results do not affect selection.',
        '', '| Run | Stopping update | Selected update | Intervals | Events |','|---|---:|---:|---:|---:|']
    for trial,r in reports.items():
        chosen=summary['selection'][trial]
        lines.append(f"| {trial} | {chosen['update']} | {chosen['selected_update']} | {r['coverage']['intervals']} | {r['coverage']['events']} |")
    lines+=['','## Chronological RMS errors','', '| Metric | Full | Supervised | Short | Long |','|---|---:|---:|---:|---:|']
    for name in KEY_METRICS:
        values=summary['seed_statistics'].get(name+'/rms')
        if values:lines.append('| '+name+' ('+unit(name)+') | '+' | '.join(f"{values[v]['mean']:.5g} ± {values[v]['sample_sd']:.3g}" for v in VARIANTS)+' |')
    lines+=['','## Paired full-minus-ablation RMS differences','', '| Metric | Ablation | Difference ± seed SD | Group-bootstrap 95% CI |','|---|---|---:|---|']
    for name in KEY_METRICS:
        for variant in VARIANTS[1:]:
            value=summary['paired_full_minus_ablation'].get(name+'/rms',{}).get(variant)
            if value:
                ci=summary['bootstrap'][name][variant]
                lines.append(f"| {name} | {variant} | {value['mean']:.5g} ± {value['sample_sd']:.3g} | {ci.get('ci95',ci.get('reason'))} |")
    lines+=['','Negative paired differences favor the full model. Bootstrap draws whole collection groups jointly across models and seeds; horizons are not independent bootstrap samples.',
        '', 'Work is an interval-data approximation, not exact fluid energy conservation or a passivity guarantee. Moments are shifted to fixed world reference points before summation.',
        '', 'Tail quantiles and worst-1% means use bounded reservoirs. Counts, means and RMS use every eligible interval. Sparsity is nearest sampled training-reference distance, not a claim of extrapolation.',
        '', 'Existing overlapping-window results are in each `assessment/window_test.json`. Metric definitions, exclusions, per-group moments, per-seed results and paired comparisons are in `report.json`; CSV tables are `metrics.csv` and `conditions.csv`.',
        '', 'Plots: horizon-error curves and wrench tail estimates in `plots/`; the transient plot uses the first detected event by series ID, for seed 1 across all variants.']
    for trial,r in reports.items():
        if r['unavailable']:lines+=['',trial+': '+'; '.join(r['unavailable'])]
    temp=root/'report.md.tmp';temp.write_text('\n'.join(lines)+'\n');temp.replace(root/'report.md')
