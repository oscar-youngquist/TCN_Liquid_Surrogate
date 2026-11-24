"""Contiguous clip indices over existing prepared shards, independent of windows."""
import bisect
import numpy as np
import torch
from .dataset import WindowDataset, FetchedBatch
from .preprocess import INDEX_DTYPE


def contextual_segments(reader, series):
    """Half-open runs of real contextual features; never bridge a reset or gap.

    Target masks do not reset memory: missing supervision is different from
    missing causal input. Prepared history_valid certifies the preceding RF.
    """
    if not series.get('assumptions_valid', True):
        return
    native = reader.native_dt(series)
    if native is None:
        return
    stride = round(.005/native)
    start = None
    previous = None
    for begin in range(0, series['rows'], reader.block_rows):
        rows = reader.read(series, begin, min(series['rows'], begin+reader.block_rows))
        for j, row in enumerate(rows):
            i = begin+j
            good = bool(row['history_valid'] and row['state_valid'] and row['interval_valid']
                        and np.isfinite(row['motion']).all())
            adjacent = previous is not None and row['episode_id'] == previous['episode_id'] and int(row['tick'])-int(previous['tick']) == stride
            if start is not None and (not good or not adjacent):
                yield start, i
                start = None
            if good and start is None:
                start = i
            previous = row.copy()
    if start is not None:
        yield start, series['rows']


class ClipIndex:
    """O(segments) storage; generate endpoint entries only for the sampled batch."""
    def __init__(self, spans, length):
        self.spans = spans
        self.length = length
        self.ends = np.cumsum([end-start-length+1 for sid,start,end in spans]).tolist()

    def __getitem__(self, indices):
        out = np.empty(len(indices), dtype=INDEX_DTYPE)
        for j, index in enumerate(indices):
            k = bisect.bisect_right(self.ends, index)
            sid, start, end = self.spans[k]
            offset = index-(self.ends[k-1] if k else 0)
            out[j] = (sid, start+offset+self.length-1)
        return out


class SequenceDataset(WindowDataset):
    def __init__(self, path, config, burn_in=200, scored_intervals=400, cache_size=8):
        super().__init__(path, 'train', cache_size)
        super().check_model(config)
        if burn_in < 0 or scored_intervals < 40:
            raise ValueError('Sequence burn-in must be nonnegative and scored_intervals >= 40')
        from .trajectories import Trajectories
        reader = Trajectories(path, 'train')
        count = burn_in+scored_intervals+1
        self.spans = [(s['id'], a, b) for s in reader.series for a,b in contextual_segments(reader,s) if b-a >= count]
        self.clip_index = ClipIndex(self.spans, count)
        self.length = self.clip_index.ends[-1] if self.spans else 0
        self.burn_in = burn_in
        self.input_length = self.rf+burn_in+scored_intervals

    def _local(self):
        super()._local()
        self.index = self.clip_index

    def check_model(self, config):
        if config.receptive_field != self.rf:
            raise ValueError('Clip and model causal histories differ')

    def __getitems__(self, indices):
        batch = super().__getitems__(indices).batch
        # Motion retains RF history and recurrent burn-in. All endpoint labels
        # begin at the first scored state. The boundary interval is not scored.
        for key in batch:
            if key not in ('motion','properties','interior_dimensions','series','end'):
                batch[key] = batch[key][:, self.burn_in:]
        return FetchedBatch(batch)


@torch.inference_mode()
def recurrent_series(reader, series, model, norm, out, device, block_rows=2048, cancelled=lambda:False, streaming=False):
    """Bounded disjoint output chunks with persistent state within each segment.

    Only RF-1 raw input samples overlap between chunks. Those samples supply
    convolution context, never recurrent tokens. Recurrent updates start at the
    first certified feature and stop at the segment boundary.
    """
    import torch
    from .model import CachedInference
    if model.config.temporal_memory == 'mamba_stateful':
        from .episode_chunks import episode_start, episode_end, chunk_motion
        out[:] = np.nan
        reset = episode_start(reader, series)
        end = episode_end(reader, series) if reset is not None else 0
        if not end:
            return
        properties = torch.as_tensor(series['properties'], dtype=torch.float32, device=device)[None]
        normalized = norm.normalize('properties', properties)
        reset_input = torch.as_tensor(reset, dtype=torch.float32, device=device)[None]
        if streaming:
            cache = CachedInference(model, norm, 1)
            cache.set_properties(properties)
            cache.initialize_reset(reset_input)
        else:
            state = model.temporal.initial_state(normalized, norm.normalize('motion', reset_input))
        for start in range(0, end, min(block_rows, 400)):
            if cancelled():
                raise InterruptedError('Recurrent trajectory inference interrupted')
            stop = min(start + min(block_rows, 400), end)
            if streaming:
                rows = reader.read(series, start, stop)
                motion = torch.as_tensor(rows['motion'].copy(), dtype=torch.float32, device=device)
                values = torch.cat([cache.step(row[None], return_all=True)[0] for row in motion])
            else:
                motion = chunk_motion(reader, series, start, stop, model.config.receptive_field - 1,
                                      reset)
                motion = torch.as_tensor(motion, device=device)[None]
                values, state = model(norm.normalize('motion', motion), normalized,
                                      memory_state=state, return_state=True)
                values = norm.denormalize('targets', values[0])
            out[start:stop] = values.cpu().numpy()
        return
    rf = model.config.receptive_field
    out[:] = np.nan
    props = torch.tensor(series['properties'], device=device, dtype=next(model.parameters()).dtype)[None]
    for start,end in contextual_segments(reader,series):
        state = model.temporal.inference_state()
        cache = CachedInference(model,norm,1) if streaming else None
        if cache is not None:
            cache.set_properties(props)
            raw_start = start-rf+1
        else:
            raw_start = start
        for pos in range(raw_start,end,block_rows):
            if cancelled():raise InterruptedError('Recurrent trajectory inference interrupted')
            stop = min(end,pos+block_rows)
            rows = reader.read(series,pos if streaming else pos-rf+1,stop)
            motion = torch.as_tensor(rows['motion'].copy(),device=device,dtype=props.dtype)[None]
            if streaming:
                values = motion.new_empty(stop-pos,5,3)
                for i in range(stop-pos):
                    if i % 128 == 0 and cancelled():raise InterruptedError('Recurrent streaming interrupted')
                    values[i] = cache.step(motion[:,i],return_all=True)[0][0]
                begin = max(start,pos)
                out[begin:stop] = values[begin-pos:].cpu().numpy()
            else:
                prediction = model(norm.normalize('motion',motion),norm.normalize('properties',props),memory_state=state)
                out[pos:stop] = norm.denormalize('targets',prediction)[0].cpu().numpy()
