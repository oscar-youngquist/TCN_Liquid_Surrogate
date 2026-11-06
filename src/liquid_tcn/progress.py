"""Terminal-width-aware training progress with metrics on separate rows."""
import os
import shutil
import sys
import textwrap
import time
from tqdm import tqdm


def terminal_width(stream):
    try:
        return max(10, os.get_terminal_size(stream.fileno()).columns)
    except (AttributeError, OSError, ValueError):
        return max(10, shutil.get_terminal_size(fallback=(80, 24)).columns)


def metric_lines(metrics, width):
    text = '  '.join(f'{key}={value}' for key, value in metrics.items())
    return textwrap.wrap(text, width=max(1, width), break_long_words=True,
                         break_on_hyphens=False)


class TrainingProgress:
    """Keep the bar on one row and wrap metrics underneath it on each refresh.

    Explicit row positions let tqdm clear/update each line without appending a
    fresh log entry every step. Metrics refresh at most five times per second.
    Width is read again on refresh, including after a terminal resize.
    """
    def __init__(self, total, initial=0, disable=False, file=None):
        self.file = file if file is not None else sys.stderr
        self.disabled = disable
        self.metrics = {}
        self.rows = []
        self.last_refresh = 0.
        self.bar = tqdm(total=total, initial=initial, file=self.file, disable=disable,
                        position=0, unit='update', mininterval=.2,
                        ncols=terminal_width(self.file), bar_format=self._format())

    def _format(self):
        width = terminal_width(self.file)
        if width < 40:
            return '{percentage:3.0f}% {n_fmt}/{total_fmt}'
        if width < 65:
            return '{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{remaining}]'
        return '{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'

    def update(self, count=1):
        self.bar.ncols = terminal_width(self.file)
        self.bar.bar_format = self._format()
        self.bar.update(count)

    def set_postfix(self, refresh=False, **metrics):
        self.metrics = metrics
        now = time.monotonic()
        if self.disabled or (not refresh and now - self.last_refresh < .2 and self.bar.n < self.bar.total):
            return
        self.last_refresh = now
        width = terminal_width(self.file)
        lines = metric_lines(metrics, width)
        # Clear obsolete lines before retiring their cursor positions.
        while len(self.rows) > len(lines):
            row = self.rows.pop()
            row.clear()
            row.close()
        while len(self.rows) < len(lines):
            self.rows.append(tqdm(total=0, position=len(self.rows)+1, file=self.file,
                                  bar_format='{desc}', ncols=width, leave=False))
        for row, line in zip(self.rows, lines):
            row.ncols = width
            row.set_description_str(line, refresh=True)

    def close(self):
        for row in reversed(self.rows):
            row.close()
        self.bar.close()
