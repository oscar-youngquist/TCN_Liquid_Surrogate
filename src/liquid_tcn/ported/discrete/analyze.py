"""Versioned, read-only endpoint/impulse validation. No continuous reconstruction."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from .losses import discrete_constraints,quaternion_matrix
from .data import merged_rows,environment_episodes,grid_intervals,eligible_windows,state_ok
from liquid_tcn.ported.storage import atomic_json,digest

VERSION='discrete_newton_euler_v1'
CRITERIA=dict(version=VERSION,provisional=True,normalized_rms=.10,additional_rms=.05,normalized_p95=.10,
    minimum_histories=100,minimum_short_intervals=100,minimum_long_windows=40,
    history_taps=249,grid_dt=.005,long_intervals=40,long_dt=.2,
    L='Euclidean diagonal of nominal interior dimensions',
    scales='r: sqrt(|g|*L)*T; P: m*|g|*T; H: m*|g|*L*T',
    statistics='Euclidean vector norm; RMS=sqrt(mean(sum(normalized_residual**2))); p95=percentile(vector norm,95)',
    eligibility='249 causal held-input-valid endpoints at window start and throughout, same episode; full 5 ms intervals only',
    reference='fixed recorded base origin at window start',
    native_gravity='kick then drift: native interval-start recorded CoM cross m*g times native dt',
    empty_verdict='inconclusive; never pass',mass_rtol=1e-7,gravity_rtol=1e-7)
CHANNELS=('kinematics','linear_momentum','angular_momentum')


def rotation(state):
    return quaternion_matrix(torch.from_numpy(state['q_WB_wxyz'])).numpy()


def statistics(values,scales):
    values=np.asarray(values,dtype=np.float64);scales=np.asarray(scales,dtype=np.float64)
    if not len(values):return dict(count=0)
    norm=np.linalg.norm(values,axis=-1);normalized=values/scales[:,None];nn=np.linalg.norm(normalized,axis=-1)
    return dict(count=len(values),bias=values.mean(0).tolist(),rms=float(np.sqrt(np.mean(norm**2))),
        p95=float(np.percentile(norm,95)),maximum=float(norm.max()),normalized_bias=normalized.mean(0).tolist(),
        normalized_rms=float(np.sqrt(np.mean(nn**2))),normalized_p95=float(np.percentile(nn,95)),normalized_maximum=float(nn.max()))


def verdict(total,excess,enough,criteria):
    if not enough or not total.get('count') or not excess.get('count'):
        return dict(status='inconclusive',reasons=['insufficient valid coverage'])
    reasons=[]
    for metric,key,limit in ((total,'normalized_rms',criteria['normalized_rms']),
                              (total,'normalized_p95',criteria['normalized_p95']),
                              (excess,'normalized_rms',criteria['additional_rms']),
                              (excess,'normalized_p95',criteria['normalized_p95'])):
        if metric[key]>limit:reasons.append(f"{'total' if metric is total else 'additional'} {key} {metric[key]:.6g} > {limit}")
    return dict(status='fail' if reasons else 'pass',reasons=reasons)


def assumed_constant(episode,criteria):
    states=[s for s in episode['nodes'].values() if bool(s['episode_selected']) and bool(s['episode_supported'])]
    if not states:return False
    m=np.array([s['mass'] for s in states]);g=np.stack([s['gravity_W'] for s in states])
    return bool(np.isfinite(m).all() and (m>0).all() and np.isfinite(g).all() and np.linalg.norm(g[0])>0
        and np.allclose(m,m[0],rtol=criteria['mass_rtol'],atol=1e-9)
        and np.allclose(g,g[0],rtol=criteria['gravity_rtol'],atol=1e-9))


def prepare_intervals(compact,native,dt,criteria):
    grid,rejected=grid_intervals(compact,dt,criteria['grid_dt'])
    native_by_start={p['start']:p for p in native['pieces']};nn=native['nodes'];cn=compact['nodes']
    first=cn[min(cn)];anchor=first['x_O_W'];m=float(first['mass']);g=first['gravity_W'];mg=m*g
    for c in grid:
        start,end=c['start'],c['end'];parts=[];pos=start
        while pos<end and pos in native_by_start:
            p=native_by_start[pos]
            if p['end']>end:break
            parts.append(p);pos=p['end']
        aligned=pos==end and start in nn and end in nn and all(p['start'] in nn and p['end'] in nn for p in parts)
        matched=aligned and all(p['valid'] and state_ok(nn[p['start']]) and state_ok(nn[p['end']]) for p in parts)
        c['valid'] &= matched
        if not matched:
            reason='native_invalid_or_held_input' if aligned else 'native_timing_match_failed'
            rejected[reason]=rejected.get(reason,0)+1
            continue
        # Compact model-style labels: reconstruct each split piece's impulse in a
        # common nearby world reference, then express the sum at the final base.
        IF=np.zeros(3);J=np.zeros(3)
        for p in c['parts']:
            s=p['endpoint'];R=rotation(s)
            force=p['h']*(R@p['force_interval_average_B'])
            IF+=force
            J+=p['h']*(R@p['moment_interval_average_O_B'])+np.cross(s['x_O_W']-anchor,force)
        s=cn[end];R=rotation(s)
        c['F_B']=R.T@IF/c['h'];c['tau_B']=R.T@(J-np.cross(s['x_O_W']-anchor,IF))/c['h']
        # Native independent measured coupling. Never infer it from delta P/H.
        force=np.stack([p['coupling_force_impulse_W'] for p in parts])
        moment=np.stack([p['coupling_moment_impulse_world_origin_W'] for p in parts])
        hs=np.array([p['h'] for p in parts])
        X=np.stack([nn[p['start']]['com_W'] for p in parts])
        c['native_IF']=force.sum(0)
        c['native_J']=(moment-np.cross(anchor,force)).sum(0)
        c['native_G']=(np.cross(X-anchor,mg)*hs[:,None]).sum(0)
        c['native_dr']=sum((nn[p['start']]['v_B']+nn[p['end']]['v_B'])*p['h']/2 for p in parts)
        # Diagnostic ONLY: gravity recovered from precomputed balance fields.
        recovered=sum(p['momentum_balance_moment_impulse_world_origin_W']+
            nn[p['end']]['angular_momentum_world_origin_W']-nn[p['start']]['angular_momentum_world_origin_W']
            -np.cross(anchor,mg)*p['h'] for p in parts)
        c['gravity_recovery_crosscheck']=recovered-c['native_G']
        c['stored_compact_IF']=sum(p['coupling_force_impulse_W'] for p in c['parts'])
        c['stored_compact_J']=sum(p['coupling_moment_impulse_world_origin_W']-
                                  np.cross(anchor,p['coupling_force_impulse_W']) for p in c['parts'])
    return grid,rejected,anchor


def evaluate_windows(windows,compact,native,anchor,dimensions,horizon):
    if not windows:return None
    cn,nn=compact['nodes'],native['nodes']
    keys=[[w[0]['start']]+[p['end'] for p in w] for w in windows]
    def field(key):return torch.tensor(np.stack([[cn[t][key] for t in ticks] for ticks in keys]),dtype=torch.float64)
    r=field('r_B');B,N,_=r.shape;q=field('q_WB_wxyz');R=quaternion_matrix(q)
    velocity=field('velocity_W');omega=(R.transpose(-1,-2)@velocity[...,3:,None])[...,0]
    args=dict(r_B=r,v_B=field('v_B'),S_B=field('S_B'),F_average_B=torch.tensor(np.stack([[p['F_B'] for p in w] for w in windows])),
        tau_average_O_B=torch.tensor(np.stack([[p['tau_B'] for p in w] for w in windows])),
        x_O_W=field('x_O_W'),u_O_W=velocity[...,:3],omega_B=omega,q_WB_wxyz=q,
        h=torch.tensor([[p['h'] for p in w] for w in windows],dtype=torch.float64),mass=field('mass'),gravity_W=field('gravity_W'),
        interior_dimensions=torch.tensor(np.broadcast_to(dimensions,(B,3)).copy()),
        episode_id=field('episode_id').long(),state_valid=torch.ones(B,N,dtype=torch.bool),
        interval_valid=torch.ones(B,N-1,dtype=torch.bool),history_valid=torch.ones(B,N,dtype=torch.bool))
    o=discrete_constraints(**args)
    result=o['short'] if horizon=='short' else o['long']
    if not result['valid'].all():raise ValueError('Window eligibility disagrees with physics module')
    total=result['residual'].numpy()
    if horizon=='short':total=total[:,0]
    A=o['reference_W'].numpy();delta=A-anchor
    m=args['mass'][:,0].numpy();g=args['gravity_W'][:,0].numpy();mg=m[:,None]*g
    T=args['h'].sum(1).numpy()
    rates=np.stack((o['u0'].numpy(),o['F0'].numpy(),o['tau0'].numpy()),-1);scales=rates*T[:,None]
    start=[nn[ts[0]] for ts in keys];end=[nn[ts[-1]] for ts in keys]
    dp=np.stack([b['momentum_W']-a['momentum_W'] for a,b in zip(start,end)])
    dh=np.stack([b['angular_momentum_world_origin_W']-a['angular_momentum_world_origin_W'] for a,b in zip(start,end)])-np.cross(A,dp)
    def summed(key):return np.stack([sum(p[key] for p in w) for w in windows])
    IF=summed('native_IF');J=summed('native_J')-np.cross(delta,IF)
    G=summed('native_G')-np.cross(delta,mg)*T[:,None]
    dr=np.stack([b['r_B']-a['r_B'] for a,b in zip(start,end)])
    floor=np.stack((dr-summed('native_dr'),dp-mg*T[:,None]+IF,dh-G+J),1)
    gravity=np.zeros_like(total);gravity[:,2]=G-o['gravity_impulse_A_W'].sum(1).numpy()
    additional=total-floor
    frame=additional-gravity
    model_dp=(o['P_W'][:,-1]-o['P_W'][:,0]).numpy()
    model_dh=(o['H_A_W'][:,-1]-o['H_A_W'][:,0]).numpy()
    model_IF=o['force_impulse_W'].sum(1).numpy();model_J=o['moment_impulse_A_W'].sum(1).numpy()
    # Independent algebraic decomposition: no residual-derived force/moment labels.
    independent_frame=np.stack(((r[:,-1]-r[:,0]).numpy()-dr+summed('native_dr')-
        (args['h'][...,None]*(args['v_B'][:,:-1]+args['v_B'][:,1:])/2).sum(1).numpy(),
        model_dp-dp+model_IF-IF,model_dh-dh+model_J-J),1)
    if not np.allclose(frame,independent_frame,rtol=1e-7,atol=1e-10):raise ValueError('Error decomposition mismatch')
    endpoint=np.zeros_like(total);endpoint[:,1]=model_dp-dp;endpoint[:,2]=model_dh-dh
    impulse=np.zeros_like(total);impulse[:,1]=model_IF-IF;impulse[:,2]=model_J-J
    quadrature=np.zeros_like(total);quadrature[:,0]=frame[:,0]
    arrays=dict(total=total,native_floor=floor,additional=additional,aggregation_frame=frame,
                gravity_quadrature=gravity,endpoint_momentum=endpoint,wrench_aggregation_frame=impulse,
                endpoint_velocity_quadrature=quadrature)
    return dict(channels={ch:{name:statistics(value[:,i],scales[:,i]) for name,value in arrays.items()}
                          for i,ch in enumerate(CHANNELS)},
        count=B,gravity_recovery_crosscheck=statistics(summed('gravity_recovery_crosscheck'),scales[:,2]),
        times_s=[cn[t[-1]]['time_s'] for t in keys],normalized_total=total/scales[...,None],
        normalized_native=floor/scales[...,None],normalized_frame=frame/scales[...,None],normalized_gravity=gravity/scales[...,None])


def analyze_case(path,criteria):
    resolved=json.loads((path/'resolved.json').read_text());dt=resolved['native_dt'];compact_path=path/'compact_200hz'
    if not (compact_path/'source.json').exists():raise ValueError('Matching compact replay missing; do not alter source data')
    if json.loads((compact_path/'source.json').read_text())['chunks_sha256']!=digest(path/'chunks.json'):
        raise ValueError('Compact/native source identity differs')
    if resolved['cavity_coupling']!='planes' or not np.isclose(dt,.0005):raise ValueError('Unexpected contact/time configuration')
    native_rows=merged_rows(path);compact_rows=merged_rows(compact_path)
    env_count=len(native_rows[0]['episode_id']);results=[];curves=[]
    for env in range(env_count):
        ne=environment_episodes(native_rows,env,dt);ce=environment_episodes(compact_rows,env,dt)
        for ep,c in ce.items():
            if not any(bool(s['episode_selected']) for s in c['nodes'].values()):continue
            # A reset-only episode with no advancing pieces is not an attempted trajectory.
            if not c['pieces']:continue
            r=dict(environment=env,episode=ep,assumptions_valid=False,histories=0,horizons={},rejected={})
            if ep not in ne or not assumed_constant(c,criteria) or not assumed_constant(ne[ep],criteria):
                r.update(status='inconclusive',reason='missing native episode or nonconstant mass/gravity');results.append(r);continue
            if not np.isclose(float(c['nodes'][min(c['nodes'])]['mass']),float(ne[ep]['nodes'][min(ne[ep]['nodes'])]['mass']),rtol=criteria['mass_rtol'],atol=1e-9) or not np.allclose(c['nodes'][min(c['nodes'])]['gravity_W'],ne[ep]['nodes'][min(ne[ep]['nodes'])]['gravity_W'],rtol=criteria['gravity_rtol'],atol=1e-9):
                r.update(status='inconclusive',reason='compact/native mass or gravity mismatch');results.append(r);continue
            r['assumptions_valid']=True
            grid,rejected,anchor=prepare_intervals(c,ne[ep],dt,criteria)
            short,long,histories=eligible_windows(grid,c['nodes'],round(criteria['grid_dt']/dt),criteria['history_taps'],criteria['long_intervals'])
            r.update(histories=histories,rejected=rejected,mass_kg=float(c['nodes'][min(c['nodes'])]['mass']),
                     gravity_W=c['nodes'][min(c['nodes'])]['gravity_W'].tolist())
            r['rejected']['history_or_invalid_short']=len(grid)-len(short)
            r['rejected']['history_incomplete_or_invalid_long']=max(0,len(grid)-criteria['long_intervals']+1)-len(long)
            for horizon,windows,minimum in (('short',short,criteria['minimum_short_intervals']),('long',long,criteria['minimum_long_windows'])):
                measured=evaluate_windows(windows,c,ne[ep],anchor,np.asarray(resolved['geometry']['interior'],dtype=np.float64),horizon)
                if measured is None:
                    r['horizons'][horizon]=dict(count=0,status='inconclusive',reason='no eligible windows');continue
                enough=histories>=criteria['minimum_histories'] and len(windows)>=minimum
                for ch,values in measured['channels'].items():
                    values['verdict']=verdict(values['total'],values['additional'],enough,criteria)
                    # Native floor has no added-error criterion; report the measured floor separately.
                    values['native_floor_verdict']=verdict(values['native_floor'],statistics(np.zeros((len(windows),3)),np.ones(len(windows))),enough,criteria)
                    if values['verdict']['status']=='fail':
                        components=('native_floor','aggregation_frame','gravity_quadrature')
                        values['dominant_error_component']=max(components,key=lambda k:values[k]['normalized_rms'])
                curves.append(dict(environment=env,episode=ep,horizon=horizon,**{k:measured.pop(k) for k in ('times_s','normalized_total','normalized_native','normalized_frame','normalized_gravity')}))
                measured['status']='inconclusive' if not enough else 'fail' if any(v['verdict']['status']=='fail' for v in measured['channels'].values()) else 'pass'
                r['horizons'][horizon]=measured
            statuses=[h['status'] for h in r['horizons'].values()]
            r['status']='inconclusive' if 'inconclusive' in statuses else 'fail' if 'fail' in statuses else 'pass'
            results.append(r)
    status='inconclusive' if len({r['environment'] for r in results})!=env_count or any(r['status']=='inconclusive' for r in results) else 'fail' if any(r['status']=='fail' for r in results) else 'pass'
    identities={str(p.relative_to(path)):digest(p) for p in path.rglob('chunks.json') if 'compact_prior_' not in str(p)}
    for name in ('job.json','resolved.json','provenance.json','compact_200hz/source.json'):
        identities[name]=digest(path/name)
    # Retain historical flags with their exact original meaning, not updated readiness.
    old=json.loads((path/'validation.json').read_text()) if (path/'validation.json').exists() else {}
    return dict(status=status,environments=results,data_path=str(path.resolve()),data_identities=identities,
                historical_instantaneous_diagnostics=old.get('status'),
                stored_instantaneous_wrench_valid=sorted({bool(x) for row in compact_rows if 'instantaneous_wrench_valid' in row for x in row['instantaneous_wrench_valid']}),
                stored_derivative_reconstruction_valid=sorted({bool(x) for row in compact_rows if 'derivative_reconstruction_valid' in row for x in row['derivative_reconstruction_valid']})),curves


def plot_case(output,name,curves):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True)
    for c in curves:
        row=0 if c['horizon']=='short' else 1
        for j,ch in enumerate(CHANNELS):
            axes[row,j].plot(c['times_s'],np.linalg.norm(c['normalized_total'][:,j],axis=-1),lw=.7,label=f'env {c["environment"]}')
            axes[row,j].axhline(.1,color='black',linestyle='--',lw=.5)
            axes[row,j].set_title(c['horizon']+' '+ch.replace('_',' '));axes[row,j].set_ylabel('Normalized vector residual')
    for ax in axes[1]:ax.set_xlabel('Episode time (s)')
    axes[0,0].legend();fig.suptitle(name);fig.tight_layout();fig.savefig(output/(name+'.png'),dpi=130);plt.close(fig)


def write_report(output,report):
    lines=['# Discrete Newton–Euler validation v1','',
        f"Revised discrete objective verdict: **{report['revised_discrete_verdict']}** within the tested scope.",
        '', 'This evaluates recorded SPH targets against the revised endpoint/interval-average objective. It does not validate a trained surrogate, deployment-time alignment, closed-loop control or large-scale reliability.', '',
        'All arithmetic is float64. Stored float32 precision cannot be recovered by casting. The native angular floor uses recorded world-origin angular momentum shifted to a nearby fixed point, so it includes its recording precision floor. No instantaneous reconstruction, acceleration derivatives or JVPs enter this analysis.', '',
        '## Method and scales','',
        'Rows are completed (t[k-1],t[k]] intervals, with endpoint state at t[k]. Force and moment are liquid-on-robot average reactions in B[k]; the moment is about the fixed endpoint base origin. For split intervals, impulses are combined in a common world reference before conversion to the endpoint frame. Terminal endpoints are retained; histories and windows never cross per-environment resets.', '',
        'L is the Euclidean diagonal of the nominal interior. F0=m|g|, tau0=F0 L, u0=sqrt(|g| L). Residual scales are u0 T, F0 T, tau0 T, with T=5 ms or 200 ms. RMS and p95 use Euclidean vector norms. These differ from the historical continuous-report scales.', '',
        'Criteria frozen before evaluation: total normalized RMS ≤10%, total p95 ≤10%, additional RMS over matched native ≤5%, additional p95 ≤10%. Each environment needs ≥100 valid 249-sample histories, ≥100 short intervals and ≥40 long windows. Windows overlap and are not independent statistical trials.', '',
        'Native coupling impulses are measured independently by the recorder. The inspected SPH order kicks velocity (including gravity), applies coupling, then drifts position. Native gravity angular impulse therefore uses native interval-start CoM. Recovery from precomputed balance fields is reported only as a cross-check. Native kinematics uses native endpoint-velocity trapezoids; the additional kinematic error includes coarse velocity quadrature.', '',
        '## Per-environment decisions','',
        '| Case / env / episode | Histories | 5 ms intervals | 200 ms windows | Short r / P / H | Long r / P / H |',
        '|---|---:|---:|---:|---|---|']
    for name,case in report['cases'].items():
        for e in case.get('environments',[]):
            def statuses(h):return ' / '.join(v['verdict']['status'] for v in e['horizons'].get(h,{}).get('channels',{}).values()) or 'inconclusive'
            lines.append(f"| {name} / {e['environment']} / {e['episode']} | {e['histories']} | {e['horizons'].get('short',{}).get('count',0)} | {e['horizons'].get('long',{}).get('count',0)} | {statuses('short')} | {statuses('long')} |")
    lines+=['','## Error decomposition','',
        'All values below are normalized percentages. Each row is one environment and channel: total RMS / p95 / max, matched native RMS, additional RMS, aggregation/frame RMS, gravity-quadrature RMS. Bias vectors, dimensional statistics, recovery cross-checks, source identities and rejection counts are in `summary.json`. Additional = aggregation/frame + gravity quadrature as vectors; RMS magnitudes are not additive.', '',
        '| Case / env | Horizon / channel | Total RMS / p95 / max | Native RMS | Additional RMS | Frame RMS | Gravity RMS |',
        '|---|---|---|---:|---:|---:|---:|']
    for name,case in report['cases'].items():
        for e in case.get('environments',[]):
            for horizon,h in e['horizons'].items():
                for ch,v in h.get('channels',{}).items():
                    t=v['total'];triple=' / '.join(f"{100*t[k]:.4g}" for k in ('normalized_rms','normalized_p95','normalized_maximum'))
                    cols=' | '.join(f"{100*v[k]['normalized_rms']:.4g}" for k in ('native_floor','additional','aggregation_frame','gravity_quadrature'))
                    lines.append(f"| {name} / {e['environment']} | {horizon} / {ch} | {triple} | {cols} |")
    lines+=['','## Interpretation','',
        'The revised objective is direct SPH supervision of r, v, S and the two interval-average wrench heads, plus short and long discrete consistency. It intentionally replaces the pointwise Newton/Euler JVP objective. Historical instantaneous-wrench/JVP flags remain unchanged and are included separately in the machine-readable summary.', '',
        'For failure attribution, compare the measured native floor, aggregation/frame contribution (including endpoint momentum reconstruction and velocity quadrature), and gravity quadrature. Small differences comparable to float32 recording precision should not be described as additional physical forces. Empty or insufficient valid sets are inconclusive, never a pass.', '',
        '## Reproduction','', '```bash',
        'SIMULATOR=genesis_pact_water MPLCONFIGDIR=/tmp/liquid-mpl \\',
        '/home/oyoungquist/.conda/envs/genesis_lr/bin/python -m liquid_tcn.ported.discrete.analyze \\',
        '  --source runs/liquid_validation_planes10_4_20261103 --output runs/discrete_newton_euler_v1_NEW',
        '```','',
        'Use a fresh output directory. `--prepare-only` freezes criteria without evaluating data; rerun with `--evaluate-prepared` to evaluate that frozen directory. Raw trajectories and previous reports are read-only. Each case has a diagnostic PNG. See `provenance.json` and `checksums.json` for source/data identities. No simulation or training is launched by this command.','']
    (output/'report.md').write_text('\n'.join(lines))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--evaluate-prepared',action='store_true');a=p.parse_args()
    if a.output.exists() and not a.evaluate_prepared:p.error('Use a fresh versioned directory')
    if a.evaluate_prepared and (not (a.output/'criteria.json').exists() or (a.output/'summary.json').exists()):p.error('Need frozen unevaluated criteria')
    a.output.mkdir(parents=True,exist_ok=True);f=a.output/'criteria.json'
    if not f.exists():atomic_json(f,CRITERIA)
    criteria=json.loads(f.read_text())
    if criteria!=CRITERIA:raise ValueError('v1 frozen criteria do not match implementation')
    if a.prepare_only:return
    torch.set_num_threads(1);started=time.monotonic();cases={}
    source_snapshot=a.output/'sources';source_snapshot.mkdir(exist_ok=True)
    identities={}
    for src in [*Path(__file__).parent.glob('*.py'),Path('liquid_collection/recorder.py'),Path('liquid_collection/cadence.py'),Path('tests/test_liquid_discrete.py')]:
        relative=src.resolve().relative_to(Path.cwd());dst=source_snapshot/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst);identities[str(relative)]=digest(src)
    for name in sorted(json.loads((a.source/'experiment.json').read_text())['plans']):
        paths=sorted((a.source/name).glob('round-*/collection/attempt-*'))
        if not paths:cases[name]=dict(status='inconclusive',reason='missing attempt');continue
        path=paths[-1]
        cases[name],curves=analyze_case(path,criteria);plot_case(a.output,name,curves)
        atomic_json(a.output/(name+'.json'),cases[name]);print(name,cases[name]['status'],flush=True)
    statuses=[c['status'] for c in cases.values()]
    overall='inconclusive' if len(cases)!=6 or 'inconclusive' in statuses else 'fail' if 'fail' in statuses else 'pass'
    report=dict(version=VERSION,revised_discrete_verdict=overall,criteria=criteria,cases=cases,
                trained_surrogate_validated=False,deployment_alignment_validated=False,large_scale_reliability_validated=False)
    atomic_json(a.output/'summary.json',report);write_report(a.output,report)
    atomic_json(a.output/'provenance.json',dict(sources=identities,criteria_sha256=digest(f),
        baseline=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_experiment_sha256=digest(a.source/'experiment.json'),command=[sys.executable,*sys.argv],wall_seconds=time.monotonic()-started,
        native_convention='SPH non-pressure gravity kick before coupling; drift after coupling',
        recorded_sph_source_sha256={name:digest(sorted((a.source/name).glob('round-*/collection/attempt-*/sources/genesis/engine/solvers/sph_solver.py'))[-1]) for name in cases}))
    atomic_json(a.output/'checksums.json',{str(f.relative_to(a.output)):digest(f) for f in a.output.rglob('*') if f.is_file()})
    return 0

if __name__=='__main__':raise SystemExit(main())
