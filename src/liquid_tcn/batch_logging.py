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
                      updates_per_second=(self.state['update']-self.initial)/max(time.monotonic()-self.start, 1e-9))
        record.update(values)
        # JSON escapes newlines/control characters in exceptions and identifiers.
        print(json.dumps(record, allow_nan=False), flush=True)
