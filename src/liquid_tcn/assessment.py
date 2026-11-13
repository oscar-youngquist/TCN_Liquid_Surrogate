"""Finish a single run's validation selection and assess its selected checkpoint."""
import argparse
import fcntl
from pathlib import Path
import signal
from threading import Event
import time
import torch
from .batch_logging import BatchLogger
from .checkpoint import load_checkpoint
from .dataset import WindowDataset, make_loader, close_loader
from .evaluate import evaluate
from .model import build_model, ModelConfig
from .normalization import Normalization
from .ported.storage import atomic_json
from .staged_validation import StageValidation
from .sweep import read_json


def select_checkpoint(run, data, subset, device, threads, cancelled_event, log):
    """Use the trainer's shortlist/full-validation policy without optimizer steps."""
    run=Path(run)
    selection_root = run/'selection' if (run/'monitoring.json').exists() else None
    selection=(selection_root or run.parent/'6')/'selection.json'
    if selection.exists():
        result=read_json(selection)
        log.emit('selection_reused',selected_update=result['selected_update'],validation_metric=result['normalized_mse'])
        return result
    saved=load_checkpoint(run/'latest.pt')
    config=dict(saved['config'],device=device,threads=threads,
                validation_batch_size=1024,validation_num_workers=0)
    model=build_model(ModelConfig(**saved['model_config'])).to(device)
    model.load_state_dict(saved['model'])
    norm=Normalization(saved['normalization'])
    dataset=WindowDataset(data,'validation')
    loader=make_loader(dataset,batch_size=1024,num_workers=0)
    policy=StageValidation('6',saved['effective_updates'],subset,selection_root=selection_root)
    try:
        # finish() only consumes the saved monitoring state and full loader.
        # It never calls the training save callback or a TensorBoard writer.
        policy.attach(model,norm,dataset,loader,config,run,saved,saved['update'],
                      None,cancelled_event,log,None)
        policy.finish(saved['update'])
        if cancelled_event.is_set():raise InterruptedError('Validation selection interrupted')
        if not selection.exists():
            raise ValueError('Training has not reached its maximum updates or monitoring patience stop; resume training before final selection')
        return read_json(selection)
    finally:
        policy.close();close_loader(loader)


def assess_checkpoint(data, checkpoint, destination, shared, config, device,
                      block_rows=2048, cancelled=lambda:False, log=None, window_only=False, inference_mode='full', compare_streaming=True):
    """Common selected-model assessment used by Unity workers and local runs."""
    from .offline import shared_events, assess
    from .benchmark import CancellableLoader
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    if not window_only:
        shared=Path(shared);shared.mkdir(parents=True,exist_ok=True)
        if not ((shared/'reference.json').exists() and (shared/'events.sqlite').exists()):
            with open(shared/'.reference.lock','a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                shared_events(data,shared,cancelled)
    report_path=destination/'window_test.json'
    if not report_path.exists():
        if log:log.emit('validation_start',scope='test',checkpoint=str(checkpoint))
        saved=load_checkpoint(checkpoint)
        model=build_model(ModelConfig(**saved['model_config'])).to(device)
        model.load_state_dict(saved['model'])
        loader=make_loader(WindowDataset(data,'test'),batch_size=1024,num_workers=0)
        class Cancellation:
            def cancelled(self):return cancelled()
        norm=Normalization(saved['normalization'])
        provider=None
        if inference_mode=='streaming':
            from .trajectories import cache_predictions
            from .streaming_assessment import CachedWindowPredictions
            cache_predictions(data,checkpoint,destination/'predictions',device,block_rows,cancelled,'streaming')
            provider=CachedWindowPredictions(destination/'predictions',norm,model.config.long_intervals)
        if str(device).startswith('cuda'):torch.cuda.synchronize(device)
        started=time.perf_counter()
        try:
            report=evaluate(model,CancellableLoader(loader,Cancellation()),norm,device,
                supervised=config.get('supervised','mse'),huber_delta=config.get('huber_delta',1.),
                lambda_short=config['lambda_short'],lambda_long=config['lambda_long'],physics_dtype=config['physics_dtype'],prediction_provider=provider)
            if str(device).startswith('cuda'):torch.cuda.synchronize(device)
            report.update(inference_mode=inference_mode,evaluation_seconds=time.perf_counter()-started,
                timing_scope='window loading and metric calculation; chronological prediction cache generation excluded')
            if provider is not None:
                report['prediction_cache_profile']=read_json(destination/'predictions/complete.json')
            if cancelled():raise InterruptedError('Test assessment interrupted')
            report['weighting']='overlapping eligible windows'
            atomic_json(report_path,report)
        finally:close_loader(loader)
        del model
        if log:log.emit('validation_end',scope='test',validation_metric=report['selection_metric'])
    if not window_only:
        assess(data,checkpoint,destination,shared,device,block_rows,cancelled,inference_mode)
    if cancelled():raise InterruptedError('Assessment interrupted')
    if compare_streaming and inference_mode=='full':
        if log:log.emit('streaming_assessment_start',checkpoint=str(checkpoint))
        assess_checkpoint(data,checkpoint,destination/'streaming',shared,config,device,block_rows,
                          cancelled,log,window_only,'streaming',False)
        from .streaming_assessment import compare_assessments
        compare_assessments(destination,window_only)
        if log:log.emit('streaming_assessment_complete')
    return read_json(report_path)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=Path('runs/full_local_40000/trials/full-s1/run'))
    parser.add_argument('--data',type=Path,help='Defaults to the saved launch_config.json data path')
    parser.add_argument('--subset',type=Path,help='Defaults to monitor_subset.json in the run ancestors')
    parser.add_argument('--output',type=Path,help='Defaults to the trial assessment directory')
    parser.add_argument('--device',default='cuda:1');parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--block-rows',type=int,default=2048)
    parser.add_argument('--window-only',action='store_true',help='Only existing full test-window metrics, without additional chronological analyses')
    args=parser.parse_args(argv)
    if min(args.threads,args.block_rows)<1:parser.error('Threads and block rows must be positive')
    run=args.run.resolve()
    data=args.data or Path(read_json(run/'launch_config.json')['data'])
    subset=args.subset or next((p/'monitor_subset.json' for p in (run,*run.parents) if (p/'monitor_subset.json').exists()),run/'monitor_subset.json')
    output=(args.output or (run if (run/'monitoring.json').exists() else run.parent)/'assessment').resolve()
    stopped=Event();previous={s:signal.signal(s,lambda *_:stopped.set()) for s in (signal.SIGTERM,signal.SIGINT)}
    log=BatchLogger('local-assessment',0);heartbeat=[time.monotonic()]
    def cancelled():
        if time.monotonic()-heartbeat[0]>=60:
            log.emit('assessment_heartbeat');heartbeat[0]=time.monotonic()
        return stopped.is_set()
    torch.set_num_threads(args.threads)
    try:
        with open(run/'.training.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            output.mkdir(parents=True,exist_ok=True)
            with open(output/'.assessment.lock','a') as assessment_lock:
                fcntl.flock(assessment_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                selected=select_checkpoint(run,data,subset,args.device,args.threads,stopped,log)
                checkpoint=Path(selected['best_checkpoint'])
                config=load_checkpoint(checkpoint)['config']
                log.emit('assessment_start',checkpoint=str(checkpoint),selected_update=selected['selected_update'])
                report=assess_checkpoint(data,checkpoint,output,output/'shared',config,args.device,
                    args.block_rows,cancelled,log,args.window_only)
                atomic_json(output/'selection.json',selected)
                text=(f"# Selected checkpoint assessment\n\nSelected update: {selected['selected_update']}\n\n"
                    f"Full-validation normalized MSE: {selected['normalized_mse']}\n\n"
                    f"Test normalized MSE (window weighted): {report['selection_metric']}\n\n"
                    "Full test metrics: [window_test.json](window_test.json).\n")
                text+='\nCached one-step comparison: [report](streaming_comparison.md).\n'
                if not args.window_only:text+='\nChronological metrics and offline analyses: [summary.json](summary.json).\n'
                temp=output/'report.md.tmp';temp.write_text(text);temp.replace(output/'report.md')
                log.emit('assessment_complete',output=str(output),selected_update=selected['selected_update'])
        return 0
    except (InterruptedError,KeyboardInterrupt):
        log.emit('interrupted');return 130
    finally:
        for s,h in previous.items():signal.signal(s,h)


if __name__=='__main__':raise SystemExit(main())
