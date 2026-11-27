"""Read-only ablation progress; reads small JSON files and bounded log tails only."""
import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def tail_records(path, size=65536):
    try:
        with path.open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell()-size))
            lines = stream.read().splitlines()
    except FileNotFoundError:
        return []
    records = []
    for line in lines:
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                records.append(value)
        except (ValueError, UnicodeDecodeError):
            pass  # A truncated first line or an in-progress final write.
    return records


def status(root):
    root = Path(root)
    manifest, state = read(root/'ablation.json'), read(root/'state.json')
    if not manifest:
        raise FileNotFoundError(f'Experiment manifest not found: {root / "ablation.json"}')
    entries = []
    for trial, config in manifest['trials'].items():
        folder = root/'trials'/trial
        row = state.get('trials', {}).get(trial, {})
        stage = row.get('stages', {}).get('ablation', {}).get('status', 'pending')
        attempts = row.get('attempts', [])
        events = tail_records(Path(attempts[-1]['console'])) if attempts and attempts[-1].get('console') else []
        metrics = tail_records(folder/'run/metrics.jsonl')
        latest = metrics[-1] if metrics else {}
        selection = read(folder/'run/selection/selection.json')
        update = max([latest.get('update', 0), selection.get('update', 0),
                      *(e.get('update') or 0 for e in events)])
        episodes = sum(1 for _ in (folder/'evaluation').glob('episode-*.json'))
        complete = (folder/'complete.json').exists() or stage == 'complete'
        if complete:
            phase = 'complete'
        elif stage in ('failed', 'interrupted'):
            phase = stage
        elif stage == 'running':
            if (folder/'evaluation/summary.json').exists():
                phase = 'finalizing'
            elif selection or config.get('kind') == 'historical':
                phase = 'test evaluation'
            elif (folder/'run/selection').exists():
                phase = 'full validation'
            else:
                phase = 'training/monitor'
        else:
            phase = 'pending'
        entries.append(dict(trial=trial, phase=phase, complete=complete, active=stage=='running' and not complete,
                            update=update, maximum=config['training']['updates'],
                            training=config.get('kind', 'ablation')=='ablation', episodes=episodes,
                            selected=selection.get('selected_update'), loss=latest.get('loss')))
    return manifest, entries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest, entries = status(args.output)
    except FileNotFoundError as error:
        parser.exit(1, str(error)+'\n')
    counts = Counter(e['phase'] for e in entries)
    training = [e for e in entries if e['training']]
    completed = sum(e['complete'] for e in training)
    active = sum(e['active'] for e in entries)
    protocol = read(args.output/'evaluation_protocol/protocol.json')
    total_episodes = len(protocol.get('episodes', []))
    print('Physics ablation — '+datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z'))
    print(f"Completed: {completed}/{len(training)} training + evaluation runs | Active workers: {active}/{manifest.get('concurrency', '?')} | "
          f"Pending: {counts['pending']} | Failed/interrupted: {counts['failed']+counts['interrupted']}")
    print(f"Recorded training updates: {sum(e['update'] for e in training):,} / {sum(e['maximum'] for e in training):,} maximum (early stopping may reduce total)")
    print()
    print(f"{'Run':<18} {'Phase':<18} {'Updates':>15} {'Test episodes':>14} {'Selected':>9} {'Last loss':>11}")
    print('-'*91)
    for entry in sorted(entries, key=lambda e: (not e['active'], e['complete'], e['trial'])):
        updates = f"{entry['update']:,}/{entry['maximum']:,}" if entry['training'] else '—'
        episodes = f"{entry['episodes']}/{total_episodes}" if total_episodes else str(entry['episodes'])
        selected = str(entry['selected']) if entry['selected'] is not None else '—'
        loss = f"{entry['loss']:.5g}" if entry['loss'] is not None else '—'
        print(f"{entry['trial']:<18} {entry['phase']:<18} {updates:>15} {episodes:>14} {selected:>9} {loss:>11}")
    print('\nStatus comes from saved files; buffered logs can lag live updates. Selected = validation-selected update.')
    if manifest.get('historical_checkpoints'):
        print('Historical references are additional evaluation jobs, excluded from the training-run completion count.')


if __name__ == '__main__':
    main()
