import io
import pytest
from liquid_tcn.progress import TrainingProgress, metric_lines


@pytest.mark.parametrize('width', [20, 40, 80, 120])
def test_metric_wrapping_keeps_all_values(width):
    metrics = dict(phase='supervised', loss='1.234', sup='1.234', short='0.456',
                   long='0.789', lr='0.001', val='pending')
    lines = metric_lines(metrics, width)
    assert all(len(line) <= width for line in lines)
    for key, value in metrics.items():
        assert f'{key}={value}' in ' '.join(lines)


def test_resize_reflows_metrics_and_retires_rows(monkeypatch):
    width = [40]
    monkeypatch.setattr('liquid_tcn.progress.terminal_width', lambda stream: width[0])
    output = io.StringIO()
    progress = TrainingProgress(20000, initial=100, file=output)
    metrics = dict(phase='physics', loss='12.34', sup='10.12', short='0.123',
                   long='0.345', lr='0.0001', val='pending')
    try:
        progress.update()
        progress.set_postfix(refresh=True, **metrics)
        narrow_rows = len(progress.rows)
        assert narrow_rows > 1
        assert all(len(row.desc) <= 40 for row in progress.rows)
        width[0] = 160
        progress.update()
        progress.set_postfix(refresh=True, **metrics)
        assert len(progress.rows) == 1
        assert progress.bar.n == 102 and progress.bar.ncols == 160
        width[0] = 30
        progress.update()
        progress.set_postfix(refresh=True, **metrics)
        assert len(progress.rows) > narrow_rows
        assert 'rate_fmt' not in progress.bar.bar_format
    finally:
        progress.close()


def test_disabled_progress_is_silent():
    output = io.StringIO()
    progress = TrainingProgress(10, disable=True, file=output)
    progress.update()
    progress.set_postfix(refresh=True, loss='1')
    progress.close()
    assert output.getvalue() == ''
