"""Low-volume, single-line training events for terminals and batch jobs."""
from datetime import datetime, timezone
import json
import os
import sys
import time


def resolve_mode(mode='auto', progress=True):
    if mode not in ('auto', 'plain', 'progress'):
        raise ValueError('logging mode must be auto, plain, or progress')
    if mode == 'auto':
        return 'progress' if progress and sys.stderr.isatty() and not os.environ.get('SLURM_JOB_ID') else 'plain'
    return mode


class BatchLogger:
    def __init__(self, trial_id, budget):
        self.trial_id = trial_id
        self.budget = budget
        self.start = time.monotonic()
        self.initial = 0
        self.state = dict(update=0, phase='initializing', loss=None, supervised=None, short=None,
                          long=None, learning_rate=None, validation_metric=None)

    def emit(self, event, **values):
        self.state.update({k: v for k, v in values.items() if k in self.state})
        record = dict(timestamp=datetime.now(timezone.utc).isoformat(), event=event,
                      trial_id=self.trial_id, budget=self.budget, **self.state,
                      elapsed_seconds=time.monotonic()-self.start,
                      updates_per_second=(self.state['update']-self.initial)/max(time.monotonic()-self.start, 1e-9))
        record.update(values)
        # JSON escapes newlines/control characters in exceptions and identifiers.
        print(json.dumps(record, allow_nan=False), flush=True)


class MetricWriter:
    """Persistent JSONL stream, bounded buffering, flushed with checkpoints."""
    def __init__(self, path, flush_seconds=10, flush_records=100):
        self.stream = open(path, 'a', buffering=65536)
        self.flush_seconds, self.flush_records = flush_seconds, flush_records
        self.last_flush, self.pending = time.monotonic(), 0

    def write(self, record):
        self.stream.write(json.dumps(record, allow_nan=False)+'\n')
        self.pending += 1
        if self.pending >= self.flush_records or time.monotonic()-self.last_flush >= self.flush_seconds:
            self.flush()

    def flush(self):
        self.stream.flush()
        self.pending, self.last_flush = 0, time.monotonic()

    def close(self):
        self.flush()
        self.stream.close()


def write_scalars(writer, values, step):
    """One TensorBoard event per update, retaining every scalar tag and step."""
    from tensorboard.compat.proto.summary_pb2 import Summary
    writer.file_writer.add_summary(Summary(value=[Summary.Value(tag=k, simple_value=float(v))
                                                  for k,v in values.items()]), step)
