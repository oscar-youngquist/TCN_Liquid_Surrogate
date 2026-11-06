"""Benchmark preprocessing worker counts and verify identical prepared artifacts.

Each run prepares the entire configured source snapshot into a fresh directory.
Use explicit completed sources when collection is live so membership stays fixed.
"""
import argparse
import json
from pathlib import Path
from liquid_tcn.preprocess import prepare
from liquid_tcn.ported.storage import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', nargs='+', type=int, default=[0, 2, 4])
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(args.config).read_text())
    reports = []
    reference = None
    for run, workers in enumerate(args.workers):
        manifest = prepare(dict(config, num_workers=workers, progress=False), root/f'run-{run}')
        comparable = {k: manifest[k] for k in ('artifacts', 'series', 'normalization', 'counts', 'windows', 'sources')}
        if reference is not None and reference != comparable:
            raise ValueError('Prepared output differs across runs; check source snapshot membership and content')
        reference = comparable
        reports.append(dict(workers=workers, **manifest['profile']))
        atomic_json(root/'profile.json', dict(identical_outputs=True, runs=reports))
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
