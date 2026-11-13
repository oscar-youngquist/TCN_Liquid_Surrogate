"""Merge completed TensorBoard sessions, honoring checkpoint-resume purge markers.

Original files are read-only. A temporary SQLite spool bounds memory while events
are sorted by step. At each step, the last value for each tag wins. Event wall times
and summary metadata are preserved; session-start markers are consumed rather than
copied because their rollback has already been applied to the consolidated history.
"""
import argparse
from itertools import groupby
import json
from pathlib import Path
import sqlite3
import tempfile

from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
from tensorboard.compat.proto import event_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter


def merge(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    files = sorted(source.glob('events.out.tfevents.*'))
    if not files:
        raise FileNotFoundError(f'No TensorBoard event files in {source}')
    destination.mkdir(parents=True, exist_ok=False)
    report = dict(source=str(source), files=[p.name for p in files],
                  source_events=0, purged_summary_events=0, resume_markers=[],
                  output_events=0, output_summary_values=0, duplicate_values_removed=0)
    with tempfile.TemporaryDirectory(prefix='tensorboard-merge-') as temporary:
        db = sqlite3.connect(Path(temporary)/'events.sqlite')
        db.execute('CREATE TABLE events(seq INTEGER PRIMARY KEY, step INTEGER, summary INTEGER, payload BLOB)')
        db.execute('CREATE INDEX event_steps ON events(step)')
        for path in files:
            for event in LegacyEventFileLoader(str(path)).Load():
                report['source_events'] += 1
                if event.HasField('session_log'):
                    if event.session_log.status == event_pb2.SessionLog.START:
                        removed = db.execute('DELETE FROM events WHERE summary=1 AND step>=?', (event.step,)).rowcount
                        report['purged_summary_events'] += removed
                        report['resume_markers'].append(dict(file=path.name, step=event.step, purged_events=removed))
                    continue
                if event.HasField('file_version'):
                    continue  # EventFileWriter creates one header for the merged log.
                db.execute('INSERT INTO events(step,summary,payload) VALUES(?,?,?)',
                           (event.step, int(event.HasField('summary')), event.SerializeToString()))
            db.commit()
        writer = EventFileWriter(str(destination), filename_suffix='.merged')
        first_step, last_step = None, None
        try:
            cursor = db.execute('SELECT step,seq,payload FROM events ORDER BY step,seq')
            for step, rows in groupby(cursor, key=lambda row: row[0]):
                originals = {}; latest = {}
                for _, seq, payload in rows:
                    event = event_pb2.Event.FromString(payload)
                    if not event.HasField('summary'):
                        writer.add_event(event); report['output_events'] += 1
                        continue
                    originals[seq] = event
                    for value in event.summary.value:
                        if value.tag in latest:
                            report['duplicate_values_removed'] += 1
                            prior = latest[value.tag][1]
                            if not value.HasField('metadata') and prior.HasField('metadata'):
                                value.metadata.CopyFrom(prior.metadata)
                        latest[value.tag] = (seq, value)
                grouped = {}
                for seq, value in latest.values():
                    grouped.setdefault(seq, []).append(value)
                for seq in sorted(grouped):
                    event = event_pb2.Event()
                    event.CopyFrom(originals[seq])
                    event.ClearField('summary')
                    event.summary.value.extend(grouped[seq])
                    writer.add_event(event)
                    report['output_events'] += 1
                    report['output_summary_values'] += len(grouped[seq])
                    first_step = step if first_step is None else first_step
                    last_step = step
            writer.flush()
        finally:
            writer.close(); db.close()
    report.update(first_step=first_step, last_step=last_step)
    (destination/'merge.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(merge(args.source, args.output), indent=2))
