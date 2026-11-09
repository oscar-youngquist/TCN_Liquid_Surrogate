"""Fixed monitoring windows and stage-specific checkpoint selection policy."""
import copy
import json
import math
import os
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from .dataset import WindowDataset, make_loader, close_loader, to_device
from .ported.storage import atomic_json, digest
from .checkpoint import load_checkpoint
from .evaluate import evaluate
from .objectives import objective

MONITOR_EVERY = 1000
RECOVERY_EVERY = 200


def monitoring_subset(data, path, seed=1729):
    path = Path(path)
    fingerprint = digest(Path(data)/'manifest.json')
    if path.exists():
        value = json.loads(path.read_text())
        return value
    ds = WindowDataset(data, 'validation')
    if not len(ds):
        raise ValueError('Empty validation split')
    ds._local()
    groups = {}
    for sid, series in ds.series.items():
        if series['split'] == 'validation':
            key = json.dumps([series.get(k) for k in ('group','robot','payload_id')])
            groups.setdefault(key, []).append(sid)
    counts = {key: int(np.isin(ds.index['series'], ids).sum()) for key,ids in groups.items()}
    size = max(1, int(math.floor(len(ds)*.05+.5)))
    quotas = {key: int(size*n/len(ds)) for key,n in counts.items()}
    remainder = sorted(groups, key=lambda k: (-(size*counts[k]/len(ds)-quotas[k]),k))
    for key in remainder[:size-sum(quotas.values())]:
        quotas[key] += 1
    rng = np.random.default_rng(seed)
    chosen = []
    for key in sorted(groups):
        positions = np.flatnonzero(np.isin(ds.index['series'],groups[key]))
        chosen.extend(rng.choice(positions, quotas[key], replace=False).tolist())
    value = dict(dataset_sha256=fingerprint, seed=seed, indices=sorted(chosen), windows=size,
                 total_windows=len(ds), fraction=size/len(ds), group_population=counts, group_samples=quotas)
    atomic_json(path,value)
    return value


@torch.no_grad()
def monitoring_mse(model, loader, normalization, device, cancelled=lambda:False, *,
                   lambda_short=0., lambda_long=0., physics_dtype='float64'):
    """Fixed MSE selection plus physics objectives; no lag or group diagnostics."""
    model.eval()
    sums = np.zeros(5); counts = np.zeros(5,dtype=np.int64); windows = 0
    physics = {h:dict(sums=np.zeros(3),count=0) for h in ('short','long')}
    for batch in loader:
        if cancelled():
            raise InterruptedError('Monitoring interrupted')
        batch = to_device(batch,device)
        p = model(normalization.normalize('motion',batch['motion']),normalization.normalize('properties',batch['properties']))
        result = objective(p,batch,normalization,lambda_short,lambda_long,
                           supervised='mse',physics_dtype=physics_dtype)
        sums += result['supervised_sums'].cpu().numpy()
        counts += result['supervised_counts'].cpu().numpy()
        for horizon in physics:
            terms = result['physics'][horizon]
            count = int(terms['count'])
            physics[horizon]['sums'] += terms['loss_by_channel'].cpu().numpy()*count
            physics[horizon]['count'] += count
        windows += len(p)
    if not (counts>0).all():
        raise ValueError('Empty monitoring supervision')
    losses = {}
    for horizon,acc in physics.items():
        if not acc['count']:
            raise ValueError(f'Empty monitoring {horizon} constraints')
        channels = acc['sums']/acc['count']
        losses[horizon] = dict(loss=float(channels.sum()),channels=dict(zip(('r','P','H'),channels.tolist())),valid_count=acc['count'])
    secondary = lambda_short*losses['short']['loss']+lambda_long*losses['long']['loss']
    mse = float(sums.sum()/(3*counts.sum()))
    return dict(scope='monitor_subset',selection_metric=mse,physics=losses,
                lambda_short=lambda_short,lambda_long=lambda_long,physics_objective=secondary,total_objective=mse+secondary,
                valid_counts=counts.tolist(),windows=windows)


def copy_atomic(source, target):
    target = Path(target)
    shutil.copyfile(source,str(target)+'.tmp')
    os.replace(str(target)+'.tmp',target)


class StageValidation:
    """Runtime policy; state is checkpointed separately from parameter identity.

    First entry into stage 6 starts empty patience/shortlist state at the resumed
    update. Earlier-stage monitoring never counts toward final-stage patience.
    """
    def __init__(self, stage, budget, subset_path, deadline=None):
        self.stage, self.budget = stage,budget
        self.subset_path = Path(subset_path)  # Legacy deadline argument is ignored.
        self.subset = json.loads(self.subset_path.read_text())
        self.subset_hash = digest(self.subset_path)
        self.state = None
        self.loader = None

    def attach(self, model, norm, dataset, full_loader, config, output, saved, update, save, stopped, log, writer):
        self.model,self.norm,self.dataset,self.full_loader = model,norm,dataset,full_loader
        self.config,self.output,self.save,self.stopped,self.log,self.writer = config,Path(output),save,stopped,log,writer
        prior = (saved or {}).get('stage_monitoring')
        self.state = copy.deepcopy(prior) if prior and prior['stage']==self.stage else dict(
            stage=self.stage,subset_sha256=self.subset_hash,entry_update=update,checks=[],shortlist=[],
            patience_best=None,stale=0,eligibility_update=config['pretrain_updates']+config['ramp_updates'])
        self.loader = make_loader(dataset,batch_size=config['validation_batch_size'] or config['batch_size'],
                                  num_workers=0,sampler=self.subset['indices'],pin_memory=config['pin_memory'])

    @property
    def patience_stop(self):
        return self.stage=='6' and self.state is not None and self.state['stale']>=3

    def cancelled(self):
        return self.stopped.is_set()

    def pending_monitor(self, update):
        return (update>self.state['entry_update'] and update % MONITOR_EVERY==0
                and (not self.state['checks'] or self.state['checks'][-1]['update']<update))

    def after_update(self, update):
        if self.cancelled():
            self.stopped.set()
            return
        if update % MONITOR_EVERY == 0:
            try:
                from .train import weights
                short,long = weights(self.config,update-1)
                report = monitoring_mse(self.model,self.loader,self.norm,self.config['device'],self.cancelled,
                    lambda_short=short,lambda_long=long,physics_dtype=self.config['physics_dtype'])
            except InterruptedError:
                self.stopped.set()
                return
            score = report['selection_metric']
            if not math.isfinite(score):
                raise FloatingPointError('Nonfinite monitoring score')
            report.update(update=update,subset_sha256=self.subset_hash,coverage=self.subset['fraction'])
            self.state['checks'].append(report)
            self.log.emit('validation_end',scope='monitor_subset',update=update,validation_metric=score,coverage=self.subset['fraction'],physics=report['physics'],physics_objective=report['physics_objective'])
            self.writer.add_scalar('monitor_subset/normalized_mse',score,update)
            for horizon,terms in report['physics'].items():
                for name,value in dict(terms['channels'],loss=terms['loss'],valid_count=terms['valid_count']).items():
                    self.writer.add_scalar(f'monitor_subset/{horizon}/{name}',value,update)
            for name in ('physics_objective','total_objective','lambda_short','lambda_long'):
                self.writer.add_scalar('monitor_subset/'+name,report[name],update)
            if self.stage=='6':
                if update>=self.state['eligibility_update']:
                    best = self.state['patience_best']
                    if best is None or score < best-self.config['min_delta']:
                        self.state.update(patience_best=score,stale=0)
                    else:
                        self.state['stale'] += 1
                row = dict(update=update,score=score,file=f'shortlist-{update}.pt')
                shortlist = sorted(self.state['shortlist']+[row],key=lambda r:(r['score'],r['update']))[:3]
                if row in shortlist:
                    # Save before adding its hash to latest.pt; avoid a self-hash.
                    self.save(row['file'])
                    row['sha256'] = digest(self.output/row['file'])
                    self.state['shortlist'] = shortlist
            # Commit monitoring state before removing superseded shortlist files.
            self.save('latest.pt')
            retained = {r['file'] for r in self.state['shortlist']}
            for path in self.output.glob('shortlist-*.pt'):
                if path.name not in retained:
                    path.unlink()
        elif update % RECOVERY_EVERY == 0:
            self.save('latest.pt')

    def finish(self, update):
        if self.cancelled():
            return
        reason = 'maximum_updates' if update==self.budget else 'patience' if self.patience_stop else None
        if reason is None:
            return
        folder = self.output.parent/self.stage
        folder.mkdir(exist_ok=True)
        latest = folder/'latest.pt'
        copy_atomic(self.output/'latest.pt',latest)
        choices = list(self.state['shortlist']) if self.stage=='6' else []
        choices = [r for r in choices if r['update']!=update]+[dict(update=update,file='latest.pt',sha256=digest(self.output/'latest.pt'))]
        evaluated = []
        from .benchmark import CancellableLoader
        for row in sorted(choices,key=lambda r:r['update']):
            if self.cancelled():
                raise InterruptedError('Selection interrupted; run remains incomplete')
            checkpoint = self.output/row['file']
            report_path = folder/f'validation-{row["update"]}.json'
            scope = 'shortlist_full' if self.stage=='6' else 'stage_endpoint'
            if report_path.exists():
                record = json.loads(report_path.read_text())
            else:
                saved = load_checkpoint(checkpoint)
                self.model.load_state_dict(saved['model'])
                self.log.emit('validation_start',scope=scope,update=row['update'])
                from .train import weights
                short,long = weights(self.config,row['update']-1)
                report = evaluate(self.model,CancellableLoader(self.full_loader,self),self.norm,self.config['device'],
                    supervised='mse',lambda_short=short,lambda_long=long,physics_dtype=self.config['physics_dtype'])
                if report['coverage']['fraction']!=1 or not math.isfinite(report['selection_metric']):
                    raise ValueError('Full finite validation required for selection')
                record = dict(scope=scope,checkpoint_sha256=row['sha256'],subset_sha256=self.subset_hash,report=report,update=row['update'])
                atomic_json(report_path,record)
                self.log.emit('validation_end',scope=scope,update=row['update'],validation_metric=report['selection_metric'])
            evaluated.append(dict(row,report_path=str(report_path),score=record['report']['selection_metric'],report=record['report']))
        if self.cancelled():
            raise InterruptedError('Checkpoint selection interrupted')
        selected = min(evaluated,key=lambda r:(r['score'],r['update']))
        copy_atomic(self.output/selected['file'],folder/'best.pt')
        copy_atomic(folder/'best.pt',self.output/'best.pt')
        report_path = self.output.parent/f'{self.stage}_best_validation.json'
        atomic_json(report_path,selected['report'])
        result = dict(update=update,stopping_reason=reason,selected_update=selected['update'],best_update=selected['update'],
            normalized_mse=selected['score'],selection_complete=True,scope='shortlist_full' if self.stage=='6' else 'stage_endpoint',
            best_checkpoint=str(folder/'best.pt'),latest_checkpoint=str(latest),resume_checkpoint=str(self.output/'latest.pt'),
            best_sha256=digest(folder/'best.pt'),latest_sha256=digest(latest),subset_sha256=self.subset_hash,
            subset_coverage=self.subset['fraction'],monitoring=copy.deepcopy(self.state),evaluations=[{k:v for k,v in r.items() if k!='report'} for r in evaluated],
            per_payload_report=str(report_path),**{k:selected['report'][k] for k in ('heads','peak_wrench','conservation','coverage')})
        atomic_json(folder/'selection.json',result)

    def close(self):
        if self.loader is not None:
            close_loader(self.loader)
