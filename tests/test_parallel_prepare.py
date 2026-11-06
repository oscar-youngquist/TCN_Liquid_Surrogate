import json
import numpy as np
import pytest
from helpers import make_attempt
from liquid_tcn.preprocess import prepare
from liquid_tcn.dataset import WindowDataset


def test_serial_parallel_artifacts_and_normalization_match(tmp_path, capsys):
    sources = []
    for i, split in enumerate(('train','validation','test')):
        path = make_attempt(tmp_path/f'raw-{i}', samples=70, shift=i, reset_at=30)
        sources.append(dict(path=str(path), group=str(i), split=split))
    config = dict(sources=sources, shard_rows=11, progress=False,
                  model=dict(channels=8, embedding_dim=4, kernel_size=3, dilations=[1,2], long_intervals=4))
    serial = prepare(dict(config, num_workers=0), tmp_path/'serial')
    parallel = prepare(dict(config, num_workers=2), tmp_path/'parallel')
    for key in ('artifacts','normalization','counts','series','sources','windows'):
        assert serial[key] == parallel[key], key
    assert not (tmp_path/'parallel'/'.prepare-work').exists()
    for split in ('train','validation','test'):
        a, b = WindowDataset(tmp_path/'serial', split), WindowDataset(tmp_path/'parallel', split)
        assert len(a) == len(b) > 0
        for key in ('motion','targets','episode_id','history_valid'):
            np.testing.assert_array_equal(a[-1+len(a)][key], b[-1+len(b)][key])
    assert 'Processing attempts' not in capsys.readouterr().err


@pytest.mark.parametrize('workers', [-1, 1.5, True])
def test_invalid_worker_configuration(tmp_path, workers):
    with pytest.raises(ValueError, match='num_workers'):
        prepare(dict(num_workers=workers), tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_progress_and_worker_failure_leave_no_manifest(tmp_path, capsys):
    path = make_attempt(tmp_path/'raw', samples=50)
    config = dict(sources=[dict(path=str(path), group='a', split='train')],
        model=dict(channels=8, embedding_dim=4, kernel_size=3, dilations=[1,2], long_intervals=4),
        progress=True)
    prepare(config, tmp_path/'success')
    assert 'Processing attempts' in capsys.readouterr().err
    # Checksums still match, but the row format cannot reconstruct causal input.
    from liquid_tcn.ported.storage import digest
    chunk = next(path.glob('chunk-*.npz'))
    with np.load(chunk) as f:
        arrays = {k:f[k] for k in f.files if k not in ('z','held_acceleration_W')}
    np.savez_compressed(chunk, **arrays)
    for name in ('chunks.json','result.json'):
        content = json.loads((path/name).read_text())
        entries = content if isinstance(content,list) else content['chunks']
        for entry in entries:
            if entry['file'] == chunk.name: entry['sha256'] = digest(chunk)
        (path/name).write_text(json.dumps(content))
    with pytest.raises(ValueError, match='Missing both'):
        prepare(dict(config, num_workers=2, progress=False), tmp_path/'failed')
    assert not (tmp_path/'failed'/'manifest.json').exists()
