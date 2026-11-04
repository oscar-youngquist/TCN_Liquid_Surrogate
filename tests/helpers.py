import json
from pathlib import Path
import numpy as np
from liquid_tcn.ported.storage import ChunkWriter, atomic_json
from test_ported_discrete import row


def make_attempt(path, samples=330, shift=0., reset_at=None, split_event=False):
    path = Path(path)
    primary = ChunkWriter(path, chunk_size=37)
    events = ChunkWriter(path/'events', chunk_size=23)
    boundaries = ChunkWriter(path/'boundaries', chunk_size=23)
    seq = 0
    for i in range(samples):
        ep = int(reset_at is not None and i >= reset_at)
        r = row(i*5, seq, ep=(ep, 0), n=5 if i else 0, reset=(i==reset_at, False))
        seq += 1
        r['held_acceleration_W'] = np.zeros((2, 6))
        r['z'] = np.zeros((2, 15)); r['z'][:, 2] = -9.81
        r['z'][:, 3] = shift + np.sin(i*.03)
        r['velocity_W'][:, 0] = r['z'][:, 3]
        r['force_interval_average_B'][:, 2] = -19.62
        r['moment_interval_average_O_B'] = np.cross(r['r_B'], r['force_interval_average_B'])
        r['coupling_force_impulse_W'] = r['force_interval_average_B'] * .005
        r['coupling_moment_impulse_world_origin_W'] = r['moment_interval_average_O_B'] * .005
        primary.append(r)
    events.finish(dict(status='complete')); boundaries.finish(dict(status='complete'))
    primary.finish(dict(status='complete', auxiliary_streams=['events','boundaries']))
    geometry = dict(interior=[.3,.2,.4], fill_size=[.2,.1,.3], fill_box_volume=.006,
                    mount_translation_B=[0,0,.05], mount_quaternion_wxyz=[1,0,0,0], scale=[1,1,1])
    payload = dict(rho=1000., mu=.01, gamma=.005, offset=.02)
    atomic_json(path/'liquid.json', dict(geometry=geometry, payload=payload, mass_per_env=[2.,2.], rest_density_volume_per_env=[.002,.002]))
    atomic_json(path/'resolved.json', dict(native_dt=.001, save_mode='200hz'))
    atomic_json(path/'job.json', dict(schema_version=2, particle_size=.01, selected=dict(seed=int(shift*100+10), round=0, policy=dict(robot='go1'))))
    return path


def preparation(tmp_path, model=None):
    from liquid_tcn.preprocess import prepare
    model = model or dict(channels=8, embedding_dim=4, kernel_size=3, dilations=[1,2], long_intervals=4)
    a=make_attempt(tmp_path/'a', samples=50)
    b=make_attempt(tmp_path/'b', samples=50, shift=2.)
    c=make_attempt(tmp_path/'c', samples=50, shift=4.)
    cfg=dict(model=model, shard_rows=7, sources=[dict(path=str(p),group=g,split=s)
        for p,g,s in [(a,'a','train'),(b,'b','validation'),(c,'c','test')]])
    manifest=prepare(cfg,tmp_path/'prepared')
    return tmp_path/'prepared',manifest
