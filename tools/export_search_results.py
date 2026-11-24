"""Copy search reports into a standalone, timestamped results snapshot."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path


def export(source, destination):
    source = source.resolve()
    destination = destination.resolve()
    if destination == source or source in destination.parents:
        raise ValueError('Choose a results directory outside the live search directory')
    paths = sorted(source.rglob('*.json'))
    if not (source / 'search.json').is_file():
        raise ValueError(f'No architecture search configuration in {source}')
    destination.mkdir(parents=True, exist_ok=True)
    reports = {}
    for path in paths:
        content = path.read_bytes()
        relative = path.relative_to(source)
        reports[relative] = json.loads(content)
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix('.json.tmp')
        temporary.write_bytes(content)
        temporary.replace(output)

    rows = []
    for relative, report in reports.items():
        if relative.name.startswith('monitor-') or 'summary' not in report:
            continue
        config = reports.get(relative.parent / 'trial.json')
        if config is None:
            continue
        summary = report['summary']
        row = dict(
            trial=relative.parent.name,
            stage=relative.parts[0] if relative.parts[0] in ('final','screening') else 'search',
            report=relative.name,
            family=config['family'],
            requested_history_seconds=config.get('requested_history_seconds'),
            extra_one_second_physics=config['extra_physics'],
            learning_rate=config['lr'],
            effective_batch=config['effective_batch'],
            seed=config['seed'],
            reference_updates=report.get('reference_updates',report.get('reference',report.get('selected_reference_updates'))),
            optimizer_updates=report.get('optimizer_steps',report.get('update',report.get('selected_optimizer_steps'))),
            normalized_wrench_mse=summary['wrench']['normalized_mse'],
            normalized_impulse_score=summary['impulse_score'],
            force_rms_N=summary['wrench']['force_moment_rms'][0],
            moment_rms_Nm=summary['wrench']['force_moment_rms'][1],
            force_vector_rms_N=summary['wrench'].get('force_moment_vector_rms',[None,None])[0],
            moment_vector_rms_Nm=summary['wrench'].get('force_moment_vector_rms',[None,None])[1],
            startup_200ms_normalized_mse=summary['startup_200ms']['normalized_mse'],
            startup_1240ms_normalized_mse=summary['startup_1240ms']['normalized_mse'],
            post_200ms_to_1240ms_normalized_mse=summary.get('post_200ms_to_1240ms',{}).get('normalized_mse'),
            after_1240ms_normalized_mse=summary.get('after_1240ms',{}).get('normalized_mse'),
            first_step_valid_fraction=summary['first_step_valid_fraction'],
            maximum_episode_seconds=summary['maximum_duration'],
            report_path=relative.as_posix(),
        )
        rows.append(row)
    if rows:
        with (destination / 'summary.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    snapshot = dict(exported_utc=datetime.now(timezone.utc).isoformat(),
                    source=str(source), json_reports=len(reports), summary_rows=len(rows),
                    stage_decisions=reports.get(Path('decisions.json'), {}))
    (destination / 'snapshot.json').write_text(json.dumps(snapshot, indent=2) + '\n')
    (destination / 'README.txt').write_text(
        'Architecture search results\n\n'
        f'Snapshot exported at {snapshot["exported_utc"]}.\n'
        f'Live search directory: {source}\n\n'
        'Start with summary.csv for completed full-validation and test results.\n'
        'Rows at different reference-update budgets represent different data exposure.\n'
        'decisions.json records stage promotion and ranking.\n'
        'architectures.json contains resolved model dimensions and attention schedules.\n'
        'ranking_protocol.json defines the comparison metrics.\n'
        'monitor_subset.json lists the fixed validation monitoring episodes.\n'
        'duration_audit.json records the supported episode durations.\n'
        'search.json contains the sweep settings.\n\n'
        'candidate_manifest.json contains all resolved focused-search candidates.\n'
        'RMS CSV columns force_rms_N/moment_rms_Nm are pooled per-component RMS;\n'
        'the vector_rms columns contain 3D error-magnitude RMS.\n\n'
        'trials/<id>/ contains trial.json, full validation reports, monitor reports,\n'
        'and available timing/memory profiles. Each report includes per-episode metrics.\n'
        'The focused search places screening trials in screening/trials/<id>/.\n'
        'final/trials/<id>/ results, selections and test reports are included when available\n'
        'under the same relative paths used by the live search.\n\n'
        'This folder is a snapshot. Refresh from the repository root with\n'
        'python tools/export_search_results.py\n'
    )
    return snapshot


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('runs/focused_architecture_search'))
    parser.add_argument('--output', type=Path, default=Path('results/focused_architecture_search'))
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.output), indent=2))
