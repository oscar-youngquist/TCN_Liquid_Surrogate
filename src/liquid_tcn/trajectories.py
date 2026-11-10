"""Chronological prepared-data access, with no overlapping-window weighting."""
from pathlib import Path
import os
import numpy as np
import torch
from .dataset import WindowDataset
from .model import LiquidTCN, ModelConfig
from .normalization import Normalization
from .checkpoint import load_checkpoint
from .ported.storage import atomic_json


class Trajectories:
    def __init__(self, data, split='test', block_rows=2048):
        self.dataset = WindowDataset(data, split)
        self.dataset._local()
        self.manifest = self.dataset.manifest
        self.block_rows = block_rows
        self.clocks = {}
        self.series = [s for s in self.manifest['series'] if s['split']==split]

    def read(self, series, start, end):
        """Copy at most the requested block from bounded worker-local mmap shards."""
        chunks=[]
        while start<end:
            shard,offset=divmod(start,self.manifest['shard_rows'])
            rows=self.dataset._shard(series['id'],shard)
            count=min(end-start,len(rows)-offset)
            if count<=0:raise ValueError(f'Truncated series {series["id"]} at row {start}')
            chunks.append(rows[offset:offset+count])
            start+=count
        return np.concatenate(chunks) if chunks else np.empty(0)

    def native_dt(self, series):
        sid=series['id']
        if sid not in self.clocks:
            source=self.manifest['sources'][series['source']]
            value=source.get('native_dt')
            if value is None:
                previous=None
                for pos in range(0,series['rows'],self.block_rows):
                    for row in self.read(series,pos,min(series['rows'],pos+self.block_rows)):
                        if (previous is not None and previous['history_valid'] and row['history_valid']
                                and row['interval_valid'] and row['episode_id']==previous['episode_id']
                                and row['tick']>previous['tick']):
                            # Certified adjacent history rows encode the exact
                            # clock conversion: physical h / native tick delta.
                            value=float(row['h'])/int(row['tick']-previous['tick'])
                            break
                        previous=row.copy()
                    if value is not None: break
            self.clocks[sid]=value
        return self.clocks[sid]

    def segments(self, series):
        """Yield half-open endpoint runs with real RF history and valid 5 ms intervals.

        The preparation history mask already certifies RF inputs without padding.
        A fresh run after a gap must rebuild that history. Wrenches belong to the
        interval ending at their row; the first endpoint of a run is an anchor,
        not an interval to integrate. Series IDs distinguish retries/environments.
        """
        if not series.get('assumptions_valid',True):
            return
        native=self.native_dt(series)
        if native is None: return
        previous=None; start=None; last=None
        for begin in range(0,series['rows'],self.block_rows):
            rows=self.read(series,begin,min(series['rows'],begin+self.block_rows))
            for j,row in enumerate(rows):
                index=begin+j
                good=bool(row['history_valid'] and row['state_valid'] and row['interval_valid']
                    and row['target_valid'].all() and np.isfinite(row['targets']).all()
                    and np.isfinite(row['motion']).all() and np.isclose(row['h'],.005,rtol=0,atol=1e-8))
                # Ticks are native ticks, whose stride depends on the source.
                contiguous=(previous is not None and row['episode_id']==previous['episode_id']
                    and int(row['tick'])-int(previous['tick'])==round(.005/native))
                if not good or (start is not None and not contiguous):
                    if start is not None and last-start>=1:
                        yield start,last+1
                    start=None
                if good and start is None:
                    start=index
                previous=row.copy();last=index
        if start is not None and last-start>=1:
            yield start,last+1

    def blocks(self, series, start, end, overlap=0):
        for pos in range(start,end,self.block_rows):
            begin=max(start,pos-overlap)
            yield pos,begin,self.read(series,begin,min(end,pos+self.block_rows))


def motion_features(rows, properties):
    return np.concatenate((rows['motion'],np.broadcast_to(properties,(len(rows),len(properties)))),axis=1)


def event_signals(rows):
    """Recorded held acceleration changes, speed, turn rate and SPH wrench norms."""
    return np.column_stack((np.r_[0.,np.linalg.norm(np.diff(rows['motion'][:,9:15],axis=0),axis=1)],
        np.linalg.norm(rows['u_O_W'],axis=1),np.linalg.norm(rows['omega_B'],axis=1),
        np.linalg.norm(rows['targets'][:,0],axis=1),np.linalg.norm(rows['targets'][:,1],axis=1)))


@torch.no_grad()
def cache_predictions(data, checkpoint, destination, device='cpu', block_rows=2048, cancelled=lambda:False):
    """Atomic per-series caches; each endpoint is predicted only once per cache.

    Chunk inference overlaps only the RF-1 *inputs*. Its contextual outputs are
    disjoint. No training-window index is expanded or duplicated here.
    """
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    reader=Trajectories(data,block_rows=block_rows)
    saved=load_checkpoint(checkpoint)
    model=LiquidTCN(ModelConfig(**saved['model_config'])).to(device).eval()
    model.load_state_dict(saved['model'])
    norm=Normalization(saved['normalization']);rf=model.config.receptive_field
    for series in reader.series:
        path=destination/f"series-{series['id']:06d}.npy"
        if path.exists():
            continue
        temp=path.with_suffix('.tmp.npy')
        out=np.lib.format.open_memmap(temp,mode='w+',dtype='float32',shape=(series['rows'],5,3))
        out[:min(rf-1,series['rows'])]=np.nan
        properties=torch.tensor(series['properties'],device=device,dtype=torch.float32)[None]
        for pos in range(rf-1,series['rows'],block_rows):
            if cancelled():
                raise InterruptedError('Trajectory inference interrupted')
            end=min(series['rows'],pos+block_rows)
            rows=reader.read(series,pos-rf+1,end)
            motion=torch.tensor(rows['motion'],dtype=torch.float32,device=device)[None]
            # Invalid inputs never enter an eligible output's causal history.
            motion=torch.nan_to_num(motion)
            prediction=model(norm.normalize('motion',motion),norm.normalize('properties',properties))
            physical=norm.denormalize('targets',prediction).cpu().numpy()[0]
            eligible=rows['history_valid'][rf-1:]
            if not np.isfinite(physical[eligible]).all():
                raise FloatingPointError('Nonfinite chronological predictions')
            out[pos:end]=physical
        out.flush();del out
        os.replace(temp,path)
    atomic_json(destination/'complete.json',dict(checkpoint=str(checkpoint),split='test',
        weighting='unique chronological intervals',series=len(reader.series)))
    return reader
