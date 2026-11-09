"""Audited allocation sessions; immutable scientific configs and checkpoint checks."""
import copy
import json
import os
from pathlib import Path
import shutil
import socket
import time
from .ported.storage import atomic_json, digest
from .sweep import code_provenance, runtime_versions, read_json


def validate_checkpoint(path, config, data_manifest, manifest_hash):
    from .checkpoint import load_checkpoint
    from .model import LiquidTCN, ModelConfig
    import torch
    saved = load_checkpoint(path)
    if (saved.get('format_version') != 1 or saved['config'] != config['training']
            or saved['data_manifest_sha256'] != manifest_hash
            or saved['normalization'] != data_manifest['normalization']
            or ModelConfig(**saved['model_config']) != ModelConfig(**config['model'])
            or saved['effective_updates'] != config['training']['updates']):
        raise ValueError(f'Checkpoint scientific configuration/normalization/serialization mismatch: {path}')
    for key in ('optimizer','scheduler','rng','sampler','update'):
        if key not in saved:
            raise ValueError(f'Checkpoint missing {key}: {path}')
    # A code-change acknowledgement cannot bypass an implementation architecture
    # change hidden behind an unchanged model-config JSON.
    with torch.random.fork_rng(devices=[]):
        model = LiquidTCN(ModelConfig(**config['model']))
        model.load_state_dict(saved['model'], strict=True)
    return saved


def resume_session(root, args, devices, started, inspect):
    manifest, state = read_json(root/'staged_sweep.json'), read_json(root/'state.json')
    current_code, runtime = code_provenance(), runtime_versions()
    if manifest['data'] != str(args.data.resolve()) or manifest['data_manifest_sha256'] != digest(args.data/'manifest.json'):
        raise ValueError('Resume requires identical dataset path and manifest (including normalization)')
    if runtime != manifest['runtime']:
        raise ValueError(f'Runtime-version incompatibility; --allow-code-change does not override it. Saved={manifest["runtime"]}; current={runtime}')
    old_code = manifest.get('sessions', [{}])[-1].get('code',manifest['code'])
    changed = old_code['source_sha256'] != current_code['source_sha256']
    if changed and not args.allow_code_change:
        raise ValueError('Package source changed; inspect the patch and use --resume --allow-code-change to acknowledge it')
    data_manifest = read_json(args.data/'manifest.json')
    checkpoints = {}
    for trial, expected in manifest['configs'].items():
        config_path = root/'trials'/trial/'resolved.json'
        if digest(config_path) != expected:
            raise ValueError(f'Resolved configuration changed: {trial}')
        cfg = read_json(config_path)
        row = state['trials'][trial]
        for stage, result in row['stages'].items():
            if result['status']=='complete':
                checked = inspect(root,trial,stage,manifest)
                if checked != result['result']:
                    raise ValueError(f'Completed result changed: {trial}/{stage}')
                if stage!='test':
                    for kind in ('best','latest'):
                        validate_checkpoint(checked[kind+'_checkpoint'],cfg,data_manifest,manifest['data_manifest_sha256'])
        latest = Path(cfg['output'])/'latest.pt'
        if latest.exists():
            saved = validate_checkpoint(latest,cfg,data_manifest,manifest['data_manifest_sha256'])
            checkpoints[trial] = dict(path=str(latest),sha256=digest(latest),update=saved['update'])
    # Archive original metadata before adding session/allocation/migration fields.
    session_id = f'session-{time.time_ns()}'
    archive = root/'metadata_history'/session_id
    archive.mkdir(parents=True)
    for path in list(root.glob('*.json'))+list(root.glob('trials/*/resolved.json')):
        destination = archive/path.relative_to(root)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,destination)
    session = dict(id=session_id,started=started,hostname=socket.gethostname(),slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        devices=devices,threads=args.threads,requested_concurrency=args.concurrency,
        code=current_code,previous_code=old_code,code_changed=changed,allow_code_change=bool(args.allow_code_change),
        runtime=runtime,metadata_archive=str(archive),resume_checkpoints=checkpoints,status='initializing',
        previous_execution=copy.deepcopy(manifest.get('execution')),
        reproducibility='Restored state; bitwise equality across hardware or thread allocations is not guaranteed')
    manifest.setdefault('sessions',[]).append(session)
    # Original code/runtime/devices/threads/deadline remain historical provenance.
    atomic_json(root/'staged_sweep.json',manifest)
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
