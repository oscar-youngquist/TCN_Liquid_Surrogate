"""Deterministic dependent search; only completed equal-budget results can rank."""
import copy
import hashlib
import itertools
import json
import math

BUDGETS = {str(i): n for i, n in enumerate((3000, 5000, 6000, 8000, 10000, 20000), 1)}
STAGES = [
    ('1', 'Supervised loss', 5), ('2', 'Physics learning rate', 3),
    ('3', 'Physics activation', 3), ('4', 'Physics weights', 4),
    ('5', 'Joint refinement', 8), ('6', 'Matched-seed confirmation and ablations', 12)]
SEEDS = (1, 2, 3)


def identity(config):
    # Paths and execution resources do not create new scientific candidates.
    value = copy.deepcopy(config)
    for key in ('output', 'resume', 'max_updates', 'data'):
        value.pop(key, None)
    value['training'].pop('device', None)
    value['training'].pop('threads', None)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def rank(ids, state, stage):
    results = []
    for trial in ids:
        row = state['trials'][trial]['stages'].get(stage, {})
        result = row.get('result') or {}
        endpoint = result.get('update') == BUDGETS[stage]
        patience = (stage=='6' and result.get('stopping_reason')=='patience'
                    and 0 < result.get('update',0) < BUDGETS[stage]
                    and result.get('monitoring',{}).get('stale',0)>=3
                    and result['update']>=result['monitoring']['eligibility_update'])
        if (row.get('status') != 'complete' or not (endpoint or patience)
                or not result.get('selection_complete')
                or result.get('scope') != ('shortlist_full' if stage=='6' else 'stage_endpoint')
                or (stage!='6' and result.get('best_update')!=BUDGETS[stage])
                or not math.isfinite(result.get('normalized_mse', float('nan')))):
            raise ValueError(f'Incomplete equal-budget comparison: {stage}/{trial}')
        results.append((result['normalized_mse'], trial))
    return [trial for _, trial in sorted(results)]


def nearby(value, tested, direction):
    """Geometric midpoint toward the next tested value, bounded by tested range."""
    choices = sorted(set(tested))
    eligible = [x for x in choices if x < value] if direction < 0 else [x for x in choices if x > value]
    if eligible:
        return math.sqrt(value * (max(eligible) if direction < 0 else min(eligible)))
    # At a boundary use the nearest in-range endpoint; dedup substitutes below.
    return value


def candidates(stage, baseline, history):
    """history contains ranked resolved configs, not test metrics or partial runs."""
    stage = str(stage)
    parent = baseline if stage == '1' else history[str(int(stage)-1)][0]
    def changed(**overrides):
        cfg = copy.deepcopy(parent)
        cfg['training'].update(overrides)
        return cfg
    if stage == '1':
        values = [changed(supervised='mse', huber_delta=1.)] + [changed(supervised='huber', huber_delta=d) for d in (.25, .5, 1., 2.)]
    elif stage == '2':
        values = [changed(physics_lr=v) for v in (.001, .0003, .0001)]
    elif stage == '3':
        values = [changed(pretrain_updates=p, ramp_updates=r) for p, r in ((100, 900), (500, 1500), (300, 1200))]
    elif stage == '4':
        values = [changed(lambda_short=s, lambda_long=l) for s, l in ((1, 1), (.3, 1), (1, 3), (.3, 3))]
    elif stage == '5':
        c = parent['training']
        runner_loss = history['1'][1]['training']
        loss = {k: runner_loss[k] for k in ('supervised', 'huber_delta')}
        runner_activation = history['3'][1]['training']
        activation = {k: runner_activation[k] for k in ('pretrain_updates', 'ramp_updates')}
        low = nearby(c['physics_lr'], (.0001, .0003, .001), -1)
        high = nearby(c['physics_lr'], (.0001, .0003, .001), 1)
        runner_weights = history['4'][1]['training']
        weights = {k: math.sqrt(c[k]*runner_weights[k]) for k in ('lambda_short', 'lambda_long')}
        values = [changed(), changed(**loss), changed(physics_lr=low), changed(physics_lr=high),
                  changed(**weights), changed(**loss, **weights), changed(**activation, **weights),
                  changed(physics_lr=low, **weights)]
        # Boundary midpoints or repeated winners may coincide. Stable, bounded
        # Cartesian substitutes guarantee eight distinct nearby interactions.
        lrs = sorted({low, high, c['physics_lr'], math.sqrt(.0001*.0003), math.sqrt(.0003*.001)})
        values += [changed(physics_lr=lr, **lo, **we, **ac) for lr, lo, we, ac in itertools.product(
            lrs, [{k:c[k] for k in loss}, loss], [{k:c[k] for k in weights}, weights],
            [{k:c[k] for k in activation}, activation])]
    elif stage == '6':
        c = parent['training']
        return [(f'{variant}_seed{seed}', changed(seed=seed, lambda_short=s, lambda_long=l))
                for variant, s, l in (('full',c['lambda_short'],c['lambda_long']), ('supervised',0,0),
                                     ('short',c['lambda_short'],0), ('long',0,c['lambda_long']))
                for seed in SEEDS]
    else:
        raise ValueError(stage)
    unique = {}
    for cfg in values:
        unique.setdefault(identity(cfg), cfg)
    result = list(unique.values())[:8] if stage == '5' else list(unique.values())
    if stage == '5' and len(result) != 8:
        raise ValueError('Could not generate eight distinct refinements')
    return [(f'candidate{i:02d}', cfg) for i, cfg in enumerate(result)]


def plan():
    return dict(stages=[dict(stage=s, name=n, candidates=c, absolute_updates=BUDGETS[s]) for s,n,c in STAGES],
                schedule_horizon=20000, seeds=list(SEEDS), ranking='stages 1-5: exact endpoint full-validation MSE; stage 6: shortlist full-validation MSE; ties by stable ID/update',
                recovery_every=200,monitor_every=1000,monitor_fraction=.05,
                monitoring_objectives=["normalized_supervised_mse","short_physics","long_physics"],
                final_patience=3,final_patience_eligibility='pretrain_updates + ramp_updates',
                final_shortlist='three lowest raw subset scores plus stopping update, deduplicated',
                refinement_rule='geometric midpoint to nearest tested LR; geometric midpoint to runner-up weights; stable Cartesian dedup substitution',
                dependent_generation='Only stage 1 resolved until predecessor completes', test_gate='all 12 final best checkpoints frozen')
