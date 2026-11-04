import json
from pathlib import Path
import subprocess
import sys
import os
import pytest
from helpers import make_attempt
from liquid_tcn.discovery import discover_sources
from liquid_tcn.metadata import assign_splits
from liquid_tcn.preprocess import prepare, verify_attempt


def attempt(root, platform='go1', round=0, retry=0, **kwargs):
    return make_attempt(root/platform/f'round-{round:05d}'/'collection'/f'attempt-{retry:04d}', **kwargs)


def test_live_tree_discovery_groups_retries_and_robots(tmp_path):
    root = tmp_path/'collection'
    a = attempt(root, samples=20)
    b = attempt(root, retry=1, samples=20)
    c = attempt(root, platform='go2', samples=20)
    job = json.loads((c/'job.json').read_text()); job['selected']['policy']['robot'] = 'go2'
    (c/'job.json').write_text(json.dumps(job))
    running = root/'go1'/'round-00001'/'collection'/'attempt-0000'; running.mkdir(parents=True)
    failed = root/'go1'/'round-00002'/'collection'/'attempt-0000'; failed.mkdir(parents=True)
    (failed/'result.json').write_text(json.dumps(dict(status='incomplete')))
    # Nested source snapshots and auxiliary streams are not independent attempts.
    nested = a/'source'/'attempt-0099'; nested.mkdir(parents=True)
    (nested/'result.json').write_text('bad json')
    sources, report = discover_sources(root, verify_attempt)
    assert {s['path'] for s in sources} == {str(p) for p in (a,b,c)}
    assert report['counts'] == dict(included=3, skipped=2, errors=0, groups=1)
    assign_splits(sources)
    assert len({s['split'] for s in sources}) == 1
    assert len({s['group'] for s in sources}) == 1
    # Adding another completed group does not change existing split assignments.
    old = {s['path']: (s['group'], s['split']) for s in sources}
    attempt(root, round=3, samples=20, shift=3.)
    more, _ = discover_sources(root, verify_attempt); assign_splits(more)
    assert all((s['group'],s['split']) == old[s['path']] for s in more if s['path'] in old)
    # Group identities are independent of which ancestor was passed as the root.
    single, _ = discover_sources(a, verify_attempt)
    assert single[0]['group'] == sources[0]['group']


def test_unfinished_auxiliary_and_snapshot_membership(tmp_path):
    root = tmp_path/'collection'
    a = attempt(root, samples=20)
    missing = attempt(root, round=1, samples=20, shift=1.)
    (missing/'events'/'result.json').unlink()
    late = root/'go1'/'round-00002'/'collection'/'attempt-0000'; late.mkdir(parents=True)
    def verify(path):
        # A worker completes while another attempt's hashes are being checked.
        if path == a:
            make_attempt(late, samples=20, shift=2.)
        return verify_attempt(path)
    sources, report = discover_sources(root, verify)
    assert [s['path'] for s in sources] == [str(a)]
    assert report['counts']['skipped'] == 2
    assert any(s['reason'] == 'incomplete_auxiliary_stream' for s in report['skipped'])
    later, _ = discover_sources(root, verify_attempt)
    assert {s['path'] for s in later} == {str(a), str(late)}


def test_corrupt_completed_attempt_fails_with_report(tmp_path):
    root = tmp_path/'collection'
    a = attempt(root, samples=20)
    with (a/'chunk-000000.npz').open('ab') as f:
        f.write(b'corrupt')
    output = tmp_path/'prepared'
    with pytest.raises(ValueError, match='invalid completed attempts'):
        prepare(dict(collection_root=str(root)), output)
    report = json.loads((output/'discovery.json').read_text())
    assert report['counts']['errors'] == 1
    assert not (output/'manifest.json').exists()


def test_empty_live_collection_and_ambiguous_config(tmp_path):
    root = tmp_path/'collection'; root.mkdir()
    (root/'attempt-0000').mkdir()
    with pytest.raises(ValueError, match='No completed attempts'):
        prepare(dict(collection_root=str(root)), tmp_path/'empty')
    assert json.loads((tmp_path/'empty'/'discovery.json').read_text())['counts']['skipped'] == 1
    with pytest.raises(ValueError, match='not both'):
        prepare(dict(collection_root=str(root), sources=[dict(path='unused')]), tmp_path/'mixed')


def test_prepare_cli_root_and_explicit_parity(tmp_path):
    root = tmp_path/'collection'
    a = attempt(root, samples=300)
    (root/'go2'/'attempt-0000').mkdir(parents=True)  # live worker has no result yet
    # Explicit fractions make this small fixture entirely training data.
    config = tmp_path/'prepare.json'; config.write_text(json.dumps(dict(fractions=[1.,0.,0.], shard_rows=71)))
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable,'-m','liquid_tcn','prepare','--root',str(root),
        '--config',str(config),'--output',str(tmp_path/'auto')],check=True,capture_output=True,text=True,
        env=dict(os.environ,PYTHONPATH=str(repo/'src')))
    summary = json.loads(result.stdout)
    assert summary['discovery']['included'] == 1 and summary['discovery']['skipped'] == 1
    automatic = json.loads((tmp_path/'auto'/'manifest.json').read_text())
    explicit = prepare(dict(sources=[dict(path=str(a),group='manual',split='train')], shard_rows=71),tmp_path/'explicit')
    assert automatic['windows'] == explicit['windows']
    assert automatic['normalization'] == explicit['normalization']
    assert automatic['artifacts'] == explicit['artifacts']
    assert len(automatic['discovery']['skipped']) == 1
