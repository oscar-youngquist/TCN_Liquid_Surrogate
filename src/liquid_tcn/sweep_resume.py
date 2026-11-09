"""Resume recorded progress using supplied paths and current execution resources."""
import copy
import json
from pathlib import Path
import time
from .ported.storage import atomic_json
from .sweep import read_json


def relocate(root, data, manifest, state):
    """Rebase run-owned paths when a sweep directory has been moved or copied."""
    root = Path(root).resolve()
    ids = list(state['trials'])
    old_roots = set()
    for trial in ids:
        config = read_json(root/'trials'/trial/'resolved.json')
        old_roots.add(str(Path(config['output']).parents[2]))
    def rebase(value):
        if isinstance(value, dict):
            return {k:rebase(v) for k,v in value.items()}
        if isinstance(value, list):
            return [rebase(v) for v in value]
        if isinstance(value, str):
            for old in old_roots:
                if value.startswith(old+'/'):
                    return str(root)+value[len(old):]
        return value
    for path in [root/'final_selections.json', *root.glob('trials/*/*/selection.json'), *root.glob('trials/*/test.json')]:
        if path.exists():
            atomic_json(path,rebase(read_json(path)))
    for trial in ids:
        path = root/'trials'/trial/'resolved.json'
        config = rebase(read_json(path))
        config.update(data=str(Path(data).resolve()),output=str(root/'trials'/trial/'run'))
        atomic_json(path,config)
    updated = rebase(state)
    state.clear(); state.update(updated)
    manifest['data'] = str(Path(data).resolve())
    if 'registry' in state:
        from .staged_plan import identity
        state['registry'] = {identity(read_json(root/'trials'/trial/'resolved.json')):trial for trial in ids}


def resume_session(root, args, devices, started, inspect=None):
    manifest, state = read_json(root/'staged_sweep.json'), read_json(root/'state.json')
    relocate(root,args.data,manifest,state)
    manifest.setdefault('sessions',[]).append(dict(started=started,devices=devices,
        threads=args.threads,requested_concurrency=args.concurrency,status='initializing'))
    atomic_json(root/'staged_sweep.json',manifest)
    atomic_json(root/'state.json',state)
    return manifest,state


def assign_devices(root, manifest, slots, threads):
    previous = manifest.get('execution',{}).get('assignments',{})
    assignments, load = {},{d:0 for d in slots}
    for trial in manifest['configs']:
        config = read_json(root/'trials'/trial/'resolved.json')
        preferred = previous.get(trial,config['training']['device'])
        device = preferred if preferred in slots else min(slots,key=lambda d:(load[d]/slots[d],d))
        assignments[trial] = device
        load[device] += 1
    manifest['execution'] = dict(assignments=assignments,threads=threads,slots=slots)
    manifest['sessions'][-1]['execution'] = copy.deepcopy(manifest['execution'])
    return assignments
