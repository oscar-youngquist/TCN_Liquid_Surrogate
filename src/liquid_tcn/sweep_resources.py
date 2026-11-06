"""Allocation limits and conservative, measured-wave wall-clock forecasts."""
import math
import os
from pathlib import Path
import time


def available_resources():
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)
    if os.environ.get('SLURM_CPUS_PER_TASK'):
        cpus = min(cpus, int(os.environ['SLURM_CPUS_PER_TASK']))
    memory = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
    # cgroup v2 job/container constraints can be tighter than node-wide memory.
    groups = [Path('/sys/fs/cgroup')]
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        if line.startswith('0::'):
            groups.append(Path('/sys/fs/cgroup')/line[3:].lstrip('/'))
    for group in groups:
        try:
            limit = (group/'memory.max').read_text().strip()
            if limit != 'max':
                memory = min(memory, int(limit)-int((group/'memory.current').read_text()))
        except (OSError, ValueError):
            pass
    if os.environ.get('SLURM_MEM_PER_NODE'):
        memory = min(memory, int(os.environ['SLURM_MEM_PER_NODE'])*1024**2)
    if os.environ.get('SLURM_MEM_PER_CPU'):
        memory = min(memory, int(os.environ['SLURM_MEM_PER_CPU'])*cpus*1024**2)
    return dict(cpus=cpus, available_ram_bytes=max(0, memory))


def cpu_limit(requested, threads, resources, loader_workers=0):
    # Reserve one CPU for coordinator/checkpoint inspection, except single-CPU tests.
    slots = max(1, resources['cpus']-1)//(threads+loader_workers)
    if slots < 1:
        raise ValueError('Worker threads exceed allocated CPUs')
    return min(requested, slots)


def capacity(requested, threads, resources, probes):
    """Use process RSS and max CUDA reserved memory across all measured phases."""
    slots = cpu_limit(requested, threads, resources)
    rss = max(w['peak_rss_kib']*1024 for w in probes.values())
    slots = min(slots, int(resources['available_ram_bytes']*.8/max(1, rss*1.25)))
    limits = {}
    for device, worker in probes.items():
        if device == 'cpu':
            limits[device] = slots
            continue
        scopes = [m for m in worker['memory'].values() if m.get('peak_reserved_bytes') is not None]
        peak = max(m['peak_reserved_bytes'] for m in scopes)
        free = min(m.get('free_visible_device_bytes', 0) for m in scopes)
        # Measured free excludes this process's pool. Account for it once and
        # leave 20% headroom for contexts, allocator growth and other job activity.
        limits[device] = int(.8*(free+peak)/max(1, peak*1.2))
    if slots < 1 or any(v < 1 for v in limits.values()):
        raise ValueError('Insufficient measured RAM/GPU capacity for one worker per requested device')
    allocation = {d:0 for d in limits}
    for _ in range(min(slots, sum(limits.values()))):
        choices = [d for d in limits if allocation[d] < limits[d]]
        d = min(choices, key=lambda d:(allocation[d], d))
        allocation[d] += 1
    return {d:n for d,n in allocation.items() if n}


def original_deadline(start, hours, environment=None):
    env = os.environ if environment is None else environment
    deadline = start+hours*3600
    if env.get('SLURM_JOB_END_TIME'):
        deadline = min(deadline, float(env['SLURM_JOB_END_TIME']))
    return deadline


def forecast(jobs, slots, measurements, test_windows, validation_windows):
    """Greedy per-device wave scheduling, with measured concurrent worker costs.

    Each job is (device, remaining_updates, remaining_validation_events). Costs
    use the slowest cohort worker, not a single-worker extrapolation to twelve.
    """
    workers = measurements['workers']
    update = max(w['training']['seconds']/w['training']['updates'] for w in workers)
    validation = max(w['validation']['seconds'] for w in workers)
    setup = max(w['startup_including_initial_checkpoint_seconds']+w['checkpoint_seconds'] for w in workers)
    queues = {d:[0.]*n for d,n in slots.items()}
    for device, updates, validations in jobs:
        cost = setup + updates*update + validations*validation + math.ceil(updates/500)*max(w['checkpoint_seconds'] for w in workers)
        queue = queues[device]
        i = min(range(len(queue)), key=queue.__getitem__)
        queue[i] += cost*1.25
    training = max((max(v) for v in queues.values()), default=0.)
    tests = math.ceil(12/sum(slots.values()))*(setup+validation*test_windows/max(1,validation_windows))*1.25
    return dict(training_seconds=training, test_seconds=tests, shutdown_reporting_seconds=1800,
                total_seconds=training+tests+1800, per_update_seconds=update,
                full_validation_seconds=validation, safety_factor=1.25)


def profile_loader_settings(data, baseline, slots, resources, deadline):
    """Small CPU/RAM-bounded loader matrix; measurements never mutate configs."""
    from .dataset import WindowDataset, make_loader, close_loader
    import torch
    torch.set_num_threads(1)
    dataset = WindowDataset(data)
    batch_size = baseline['training']['batch_size']
    one = dataset[0]
    sample_bytes = sum(v.numel()*v.element_size() for v in one.values() if isinstance(v,torch.Tensor))
    processes = sum(slots.values())
    records=[]
    for workers,prefetch in ((0,2),(1,2),(1,4),(2,2)):
        required_cpu=processes*(baseline['training']['threads']+workers)+1
        required_ram=processes*sample_bytes*batch_size*(1+workers*prefetch)
        if required_cpu > resources['cpus'] or required_ram > resources['available_ram_bytes']*.4:
            records.append(dict(workers=workers,prefetch=prefetch,status='skipped',reason='CPU/RAM limit',estimated_buffer_bytes=required_ram))
            continue
        if time.time() >= deadline:
            raise TimeoutError('Deadline during loader profiling')
        loader=make_loader(dataset,batch_size=batch_size,num_workers=workers,prefetch_factor=prefetch,
            pin_memory=False,sampler=range(min(len(dataset),16*batch_size)))
        start=time.monotonic(); windows=0
        try:
            for batch in loader:
                windows += len(batch['motion'])
                if time.time() >= deadline:
                    raise TimeoutError('Deadline during loader profiling')
        finally:
            close_loader(loader)
        elapsed=time.monotonic()-start
        records.append(dict(workers=workers,prefetch=prefetch,status='measured',windows=windows,
            seconds=elapsed,windows_per_second=windows/elapsed,estimated_buffer_bytes=required_ram))
    return dict(settings=records,selected=dict(workers=baseline['training']['num_workers'],prefetch=baseline['training']['prefetch_factor']),
        note='Includes startup/collation; bounded 16 batches per setting; settings retained for checkpoint parity, no automatic loader tuning')
