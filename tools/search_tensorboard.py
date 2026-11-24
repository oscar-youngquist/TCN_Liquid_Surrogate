"""View architecture-search JSON metrics in TensorBoard, optionally refreshing live."""
import argparse
import json
import math
from pathlib import Path
import time

from torch.utils.tensorboard import SummaryWriter


def scalars(value, prefix=''):
    if isinstance(value, dict):
        for key, child in value.items():
            if key != 'elapsed_cohort':
                yield from scalars(child, f'{prefix}/{key}'.strip('/'))
    elif isinstance(value, list):
        # Force and moment have distinct physical units, so label them explicitly.
        labels = ['force_N', 'moment_Nm'] if prefix.split('/')[-1].startswith('force_moment_') else range(len(value))
        for key, child in zip(labels, value):
            yield from scalars(child, f'{prefix}/{key}')
    elif isinstance(value, (int, float)) and math.isfinite(value):
        yield prefix, float(value)


def export(sources, destination):
    destination.mkdir(parents=True, exist_ok=True)
    index_path = destination / 'export_index.json'
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    updated = 0
    runs = set()
    for source in sources:
        for config_path in sorted(source.glob('**/trial.json')):
            relative = config_path.relative_to(source)
            if any(part in ('pilot', 'smoke') for part in relative.parts):
                continue
            cfg = json.loads(config_path.read_text())
            stage = relative.parts[0] if relative.parts[0] in ('screening', 'final') else 'search'
            name = (f"{cfg['family']}_H{cfg.get('requested_history_seconds', 0):g}s"
                    f"_lr{cfg['lr']:g}_batch{cfg['effective_batch']}_seed{cfg['seed']}"
                    f"_extra1s{int(cfg['extra_physics'])}_{config_path.parent.name}")
            run = destination / source.name / stage / name
            writer = None
            try:
                for report_path in sorted(config_path.parent.glob('*.json')):
                    filename = report_path.name
                    if not (filename.startswith(('monitor-', 'validation-', 'shortlist-validation-')) or filename == 'test.json'):
                        continue
                    stamp = report_path.stat()
                    signature = [stamp.st_mtime_ns, stamp.st_size]
                    key = str(report_path.resolve())
                    runs.add(str(run))
                    if index.get(key) == signature:
                        continue
                    try:
                        report = json.loads(report_path.read_text())
                    except (OSError, json.JSONDecodeError):
                        continue  # Retry reports being written by a live worker.
                    if 'summary' not in report:
                        continue
                    step = report.get('optimizer_steps', report.get('update', report.get('selected_optimizer_steps')))
                    if step is None and filename.startswith(('monitor-', 'shortlist-validation-')):
                        step = int(report_path.stem.rsplit('-', 1)[1])
                    if step is None:
                        continue
                    if writer is None:
                        writer = SummaryWriter(str(run))
                        writer.add_text('configuration', '```json\n' + json.dumps(cfg, indent=2) + '\n```', 0)
                    category = ('monitor' if filename.startswith('monitor-') else
                                'test' if filename == 'test.json' else 'full_validation')
                    for tag, value in scalars(report['summary']):
                        writer.add_scalar(f'{category}/{tag}', value, int(step), walltime=stamp.st_mtime)
                    reference = report.get('reference_updates', report.get('reference', report.get('selected_reference_updates')))
                    if reference is not None:
                        writer.add_scalar('progress/reference_updates', reference, int(step), walltime=stamp.st_mtime)
                    index[key] = signature
                    updated += 1
            finally:
                if writer is not None:
                    writer.close()
    temporary = index_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(index, indent=2) + '\n')
    temporary.replace(index_path)
    return dict(runs=len(runs), reports_updated=updated, logdir=str(destination))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, action='append', help='Repeat to include multiple search roots')
    parser.add_argument('--logdir', type=Path, default=Path('runs/search_tensorboard'))
    parser.add_argument('--serve', action='store_true', help='Launch TensorBoard and refresh reports continuously')
    parser.add_argument('--port', type=int, default=6006)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--refresh-seconds', type=float, default=15)
    args = parser.parse_args()
    if args.refresh_seconds <= 0:
        parser.error('--refresh-seconds must be positive')
    sources = args.source or [Path('runs/architecture_search'), Path('runs/focused_architecture_search')]
    print(json.dumps(export(sources, args.logdir)), flush=True)
    if args.serve:
        from tensorboard import program
        board = program.TensorBoard()
        board.configure(argv=[None, '--logdir', str(args.logdir.resolve()), '--port', str(args.port), '--host', args.host])
        print(f'TensorBoard: {board.launch()}', flush=True)
        print('X-axis: actual optimizer steps. Historical monitor reports do not store exact reference exposure.', flush=True)
        try:
            while True:
                time.sleep(args.refresh_seconds)
                result = export(sources, args.logdir)
                if result['reports_updated']:
                    print(json.dumps(result), flush=True)
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
