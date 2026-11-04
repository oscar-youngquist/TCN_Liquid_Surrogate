"""Read-only merge/alignment of primary, boundary and exact-event streams.

Integer native ticks identify endpoints; repeated timestamps are ordered by
record_sequence. Episode identity is PER ENVIRONMENT. An unaffected environment
may join pieces split by another environment's reset; impulses are kept in common
world references. Partial leading/trailing intervals are counted, never padded.
"""
from collections import defaultdict
from pathlib import Path
import numpy as np
from liquid_tcn.ported.audit import stream_rows


STATE_KEYS=('x_O_W','q_WB_wxyz','velocity_W','r_B','v_B','S_B','mass','gravity_W',
            'com_W','com_vel_W','momentum_W','angular_momentum_world_origin_W',
            'state_valid','held_valid','episode_supported','episode_selected',
            'episode_id','episode_native_step')
IMPULSE_KEYS=('coupling_force_impulse_W','coupling_moment_impulse_world_origin_W',
              'force_interval_average_B','moment_interval_average_O_B',
              'momentum_balance_moment_impulse_world_origin_W')


def merged_rows(path):
    """Deduplicate stream copies by sequence, verifying any overlapping fields."""
    records={}
    for sub in ('','boundaries','events'):
        for row in stream_rows(Path(path)/sub):
            key=(int(row['native_step']),int(row['record_sequence']))
            if key not in records:
                records[key]=dict(row)
            else:
                for field,value in row.items():
                    if field in records[key] and not np.array_equal(value,records[key][field],equal_nan=True):
                        raise ValueError(f'Conflicting copies at {key}: {field}')
                    records[key][field]=value
    return [records[key] for key in sorted(records)]


def environment_episodes(rows, env, native_dt):
    """Extract endpoint states and nonoverlapping advancing pieces, retaining terminals."""
    episodes=defaultdict(lambda:dict(nodes={},pieces=[],events=[],terminals=[]))
    previous_episode=None
    for row in rows:
        tick=int(row['native_step']);ep=int(row['episode_id'][env])
        if previous_episode is not None and ep<previous_episode:
            raise ValueError('Decreasing per-environment episode identity')
        previous_episode=ep;out=episodes[ep]
        if not np.isclose(float(row['time_s']),tick*native_dt,rtol=0,atol=1e-9):
            raise ValueError('Timestamp/native tick mismatch')
        out['events'].append(dict(tick=tick,held_valid=bool(row['held_valid'][env]),
            reset=bool(row['reset'][env]),terminal=bool(row['terminal'][env])))
        if row['terminal'][env]:out['terminals'].append(tick)
        if 'x_O_W' not in row:
            # Exact sparse events have no invented physical endpoint state.
            if tick in out['nodes']:
                for k in ('held_valid','episode_selected'):
                    out['nodes'][tick][k]=bool(row[k][env])
            continue
        state={k:np.asarray(row[k][env],dtype=np.float64) if np.asarray(row[k][env]).dtype.kind=='f'
               else np.asarray(row[k][env]).copy() for k in STATE_KEYS}
        state.update(tick=tick,time_s=tick*native_dt)
        # Terminal endpoint precedes reset; a new episode gets a distinct key.
        out['nodes'][tick]=state
        n=int(row['interval_native_steps'][env])
        h=float(row['interval_duration_s'][env])
        if n==0:
            if h!=0:raise ValueError('Zero-step event has positive duration')
            continue
        if n<0 or not np.isclose(h,n*native_dt,rtol=0,atol=1e-8):
            raise ValueError('Recorded interval duration does not match tick span')
        piece=dict(start=tick-n,end=tick,h=n*native_dt,stored_h=h,
                   valid=bool(row['interval_valid'][env]),endpoint=state)
        piece.update({k:np.asarray(row[k][env],dtype=np.float64) for k in IMPULSE_KEYS})
        out['pieces'].append(piece)
    for out in episodes.values():
        last=None
        for p in out['pieces']:
            if last is not None and p['start']<last:raise ValueError('Overlapping advancing records')
            p['valid'] &= p['start'] in out['nodes'] and p['end'] in out['nodes']
            last=p['end']
    return dict(episodes)


def state_ok(s):
    return all(bool(s[k]) for k in ('state_valid','episode_supported','episode_selected','held_valid'))


def grid_intervals(episode,native_dt,grid_dt=.005):
    """Join valid same-episode pieces into full grid intervals; do not cross gaps."""
    stride=round(grid_dt/native_dt)
    if not np.isclose(stride*native_dt,grid_dt,rtol=0,atol=1e-12):raise ValueError('Nonintegral grid')
    nodes=episode['nodes'];by_start={p['start']:p for p in episode['pieces']}
    if len(by_start)!=len(episode['pieces']):raise ValueError('Duplicate interval starts')
    results=[];used=set();rejected=defaultdict(int)
    for start in sorted(nodes):
        if start%stride or start+stride not in nodes:continue
        end=start+stride;pos=start;parts=[]
        while pos<end and pos in by_start:
            p=by_start[pos]
            if p['end']>end:break
            parts.append(p);pos=p['end']
        if pos!=end:
            rejected['missing_or_misaligned_interval']+=1;continue
        valid=all(p['valid'] for p in parts) and state_ok(nodes[start]) and state_ok(nodes[end])
        # Every intervening state/event must remain supported and held-input valid.
        valid &= all(state_ok(nodes[t]) for t in nodes if start<=t<=end)
        valid &= all(e['held_valid'] and not (e['reset'] and e['tick']>start)
                     for e in episode['events'] if start<=e['tick']<=end)
        results.append(dict(start=start,end=end,h=grid_dt,parts=parts,valid=valid))
        used.update((p['start'],p['end']) for p in parts)
        if not valid:rejected['invalid_full_interval']+=1
    unused=[p for p in episode['pieces'] if (p['start'],p['end']) not in used]
    rejected['unmerged_partial_pieces']=len(unused)
    rejected['unmerged_partial_duration_s']=sum(p['h'] for p in unused)
    rejected['merged_split_intervals']=sum(len(p['parts'])>1 for p in results)
    return results,dict(rejected)


def eligible_windows(intervals,nodes,stride,history_taps=249,long_intervals=40):
    """History at window START and every endpoint, with valid intervening intervals."""
    by_end={p['end']:p for p in intervals};history={}
    for tick in sorted(nodes):
        taps=[tick-i*stride for i in range(history_taps)]
        history[tick]=all(t in nodes and state_ok(nodes[t]) for t in taps)
        history[tick] &= all(t in by_end and by_end[t]['valid'] for t in taps[:-1])
    short=[];long=[]
    for p in intervals:
        if not p['valid'] or not history.get(p['start'],False) or not history.get(p['end'],False):continue
        short.append([p])
        start=p['end']-long_intervals*stride
        ticks=list(range(start+stride,p['end']+1,stride))
        if history.get(start,False) and all(t in by_end and by_end[t]['valid'] and history.get(t,False) for t in ticks):
            long.append([by_end[t] for t in ticks])
    return short,long,sum(history.values())
