"""Snapshot completed attempts beneath a live collection tree, without modifying it."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from .metadata import read_metadata


class IncompleteAttemptError(ValueError):
    """A recording or auxiliary stream has not published a complete manifest."""


def collection_group(metadata):
    """Stable across retries, robots, root moves and additions to the campaign.

    The collector shares selected round seed and payload across platforms. Do not
    include policy/robot/attempt or directory names, which would split relatives.
    Repeated collections with the same identity are conservatively kept together.
    """
    selected = metadata['job']['selected']
    if any(key not in selected for key in ('seed', 'round')):
        raise ValueError('Discovery requires job.selected.seed and job.selected.round; use explicit sources for legacy metadata')
    identity = [selected['seed'], selected['round'], metadata['payload_id']]
    encoded = json.dumps(identity, sort_keys=True).encode()
    return 'collection-' + hashlib.sha256(encoded).hexdigest()


def discover_sources(root, verify, exclude=()):
    """Return verified sources and an auditable scan report.

    Missing/non-complete manifests are expected during live collection. A corrupt
    completed attempt is an error, never silently treated as an unfinished job.
    No locks are taken from the collector, and this scan never watches for updates.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f'Collection root is not a directory: {root}')
    excluded = {Path(p).resolve() for p in exclude}
    report = dict(root=str(root), started_utc=datetime.now(timezone.utc).isoformat(),
                  grouping='sha256([selected.seed, selected.round, payload_id])',
                  included=[], skipped=[], errors=[])
    # Freeze candidate membership and completion decisions before expensive hash
    # verification. A completion arriving later waits for the next preparation.
    candidates = []
    def walk_error(error):
        raise error  # Never silently omit an unreadable subtree.
    for directory, children, files in os.walk(root, topdown=True, followlinks=False, onerror=walk_error):
        path = Path(directory)
        children[:] = sorted(c for c in children if (path / c).resolve() not in excluded
                             and not (path / c).is_symlink())
        attempt = path.name.startswith('attempt-') or ('job.json' in files and
                    ('liquid.json' in files or 'resolved.json' in files))
        if not attempt:
            continue
        children[:] = []  # Do not descend into auxiliary streams or source snapshots.
        result_path = path / 'result.json'
        if not result_path.exists():
            report['skipped'].append(dict(path=str(path), reason='missing_result', status='unfinished'))
            continue
        try:
            result = json.loads(result_path.read_text())
            status = result.get('status')
            if status != 'complete':
                report['skipped'].append(dict(path=str(path), reason='not_complete', status=status))
                continue
            candidates.append(path)
        except (OSError, ValueError, AttributeError) as exc:
            report['errors'].append(dict(path=str(path), reason=str(exc)))
    sources = []
    for path in sorted(candidates):
        try:
            # Missing auxiliary completion markers are still unfinished. Hash or
            # schema errors in a claimed-complete recording must fail preparation.
            hashes = verify(path)
            metadata = read_metadata(path)
            group = collection_group(metadata)
            sources.append(dict(path=str(path), group=group, payload_id=metadata['payload_id'],
                                robot=metadata['robot'], hashes=hashes, metadata=metadata))
            report['included'].append(dict(path=str(path), group=group,
                                           payload_id=metadata['payload_id'], robot=metadata['robot']))
        except IncompleteAttemptError:
            report['skipped'].append(dict(path=str(path), reason='incomplete_auxiliary_stream', status='unfinished'))
        except (ValueError, OSError, KeyError, TypeError, AttributeError) as exc:
            report['errors'].append(dict(path=str(path), reason=str(exc)))
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    report['counts'] = {key: len(report[key]) for key in ('included', 'skipped', 'errors')}
    report['counts']['groups'] = len({s['group'] for s in sources})
    return sources, report
