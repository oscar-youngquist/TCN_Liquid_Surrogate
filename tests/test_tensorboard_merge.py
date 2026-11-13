"""A rolled-back session must not contribute stale or duplicate scalar points."""
import importlib.util
from pathlib import Path
from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader


def test_merge_resume_rollback(tmp_path):
    spec=importlib.util.spec_from_file_location('merge_tensorboard',Path(__file__).parents[1]/'tools/merge_tensorboard.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source=tmp_path/'original'
    def scalar(writer,step,tag,value):
        writer.add_event(event_pb2.Event(step=step,wall_time=100+step,summary=summary_pb2.Summary(
            value=[summary_pb2.Summary.Value(tag=tag,simple_value=value)])))
    writer=EventFileWriter(str(source))
    for step in (1,2,3):scalar(writer,step,'loss',step)
    scalar(writer,3,'stale',99)
    writer.close()
    writer=EventFileWriter(str(source))
    writer.add_event(event_pb2.Event(step=2,session_log=event_pb2.SessionLog(status=event_pb2.SessionLog.START)))
    scalar(writer,1,'loss',10)  # An overlapping tag/step also keeps the latest value.
    for step in (2,3,4):scalar(writer,step,'loss',step*10)
    writer.close()
    before={p.name:p.read_bytes() for p in source.iterdir()}
    result=module.merge(source,tmp_path/'merged')
    files=list((tmp_path/'merged').glob('events.out.tfevents.*'))
    assert len(files)==1 and result['purged_summary_events']==3
    found=[]
    for event in LegacyEventFileLoader(str(files[0])).Load():
        assert not event.HasField('session_log')
        for value in event.summary.value:
            found.append((event.step,value.tag,value.simple_value,event.wall_time))
    assert found==[(s,'loss',s*10,100+s) for s in (1,2,3,4)]
    assert {p.name:p.read_bytes() for p in source.iterdir()}==before
