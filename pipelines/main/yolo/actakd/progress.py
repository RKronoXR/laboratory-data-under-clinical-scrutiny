"""Estimated work completion, separately for A/B/C/D and teacher T.

Each planned fit/evaluation is one work unit. An unfinished fit receives partial
credit from completed epochs. Shared modules have one ID: they contribute to all
strategies that depend on them but are counted only once in the overall total.
This is NOT an estimate of remaining time. No GPU/framework imports are needed.
"""
from __future__ import annotations
from pathlib import Path
from .config import is_hpo
from .io import atomic_json, atomic_bytes, read_json, write_csv, utc
from .search import candidates

ARMS = ('A','B','C','D')
LABELS = {'A':'Clinical only', 'B':'ACTA -> clinical',
          'C':'Clinical + distillation', 'D':'ACTA -> clinical + distillation',
          'T':'Original nnU-Net teacher'}


def _json(path):
    try:
        return read_json(path) if Path(path).is_file() else {}
    except (OSError, ValueError):
        return {}


def _fit_fraction(folder, epochs):
    folder = Path(folder)
    if (folder/'result.json').is_file():
        return 1.0
    current = _json(folder/'current.json')
    n = current.get('epoch', 0)
    # The native trainer writes "training" before the epoch, then
    # "epoch_completed" after the checkpoint. Student status is post-epoch.
    if current.get('state') == 'training':
        n = max(0, n-1)
    return min(.99, max(0., float(n)/max(1, int(epochs))))


def _prediction_fraction(folder, expected):
    if (folder/'result.json').is_file():
        return 1.0
    done = len(list((folder/'probabilities').glob('*.npy')))
    return min(.99, done/max(1, expected))


def calculate(c):
    out = Path(c['output_dir'])
    grid = candidates(c)
    nodes = {}
    selections = out/'selections'
    clinical_hpo = is_hpo(c,'clinical')
    distill_hpo = is_hpo(c,'distill')
    # With common clinical HPO, all four arms influence the chosen candidate,
    # hence all four depend on both shared modules for the selection sweep.
    teacher_owners = set(('T','C','D')) | (set(ARMS) if clinical_hpo else set())
    pretrain_owners = set(('B','D')) | (set(ARMS) if clinical_hpo else set())
    if distill_hpo:
        pretrain_owners.add('C')  # the common KD weight is scored jointly in C/D
    data = _json(out/'data/bundle.json')
    splits = {s['fold']:s for s in data.get('splits',[])}

    def add(key, owners, phase, fraction):
        fraction = min(1., max(0., float(fraction)))
        if key in nodes:
            nodes[key]['strategies'] = sorted(set(nodes[key]['strategies']) | set(owners))
            return
        nodes[key] = dict(task_id=key, phase=phase, strategies=sorted(owners),
                          fraction=fraction, complete=fraction==1.)

    add('shared/data_preparation', (*ARMS,'T'), 'Preparation',
        (out/'data/READY.json').is_file())
    for f in range(c['folds']):
        for seed in c['seeds']:
            cell = f'seed_{seed}_fold_{f}'
            base = out/'runs'/cell
            selected = selections/cell
            teacher_selected = (selected/'T.json').is_file()
            nteacher = len(grid['teacher'])
            for i in range(nteacher):
                folder = base/'T'/f'trial_{i:03d}'
                frac = 1. if teacher_selected else _fit_fraction(folder,c['training']['teacher_epochs'])
                # A teacher imported from a verified cache completes this module
                # without repeating any HPO fit represented by the module.
                add(f'{cell}/T/trial_{i:03d}',teacher_owners,'Teacher reuse/training',frac)
            ntrain = len(splits.get(f,{}).get('train_ids',[])) or c['expected_dcbr_bitewings']
            pred = out/'predictions'/cell/'T/distillation_train_cache'
            add(f'{cell}/T/targets',teacher_owners-{'T'},'Shared teacher targets',
                _prediction_fraction(pred,ntrain))
            for i,h in enumerate(grid['pretraining']):
                task = f'P/trial_{i:03d}' if is_hpo(c,'pretrain') else 'P/fixed'
                add(f'{cell}/{task}',pretrain_owners,'Shared ACTA pretraining',
                    _fit_fraction(base/task,h['epochs']))
                if is_hpo(c,'pretrain'):
                    task = f'P_probe/trial_{i:03d}'
                    add(f'{cell}/{task}',pretrain_owners,'ACTA transfer probe',
                        _fit_fraction(base/task,c['training']['max_epochs']))
            for i,h in enumerate(grid['clinical']):
                for arm in ARMS:
                    task=f'clinical/trial_{i:03d}/{arm}'
                    owners=ARMS if clinical_hpo else (arm,)
                    add(f'{cell}/{task}',owners,
                        'Shared clinical HPO' if clinical_hpo else 'Clinical training',
                        _fit_fraction(base/task,h['epochs']))
            for i,w in enumerate(grid['distillation_weights']):
                if float(w)==float(c['training']['distillation_weight']):
                    continue  # already included in the clinical C/D candidates
                for arm in ('C','D'):
                    task=f'distill/trial_{i:03d}/{arm}'
                    add(f'{cell}/{task}',('C','D'),'Shared distillation-weight HPO',
                        _fit_fraction(base/task,c['training']['max_epochs']))
            scopes=['development_validation']
            if c['reference25']['evaluate_after_all_development']:
                scopes.append('reference25')
            for arm in (*ARMS,'T'):
                for scope in scopes:
                    folder=out/'evaluations'/scope/cell/arm
                    if (folder/'DONE.json').is_file():
                        frac=1.
                    else:
                        n=(c['expected_reference_bitewings'] if scope=='reference25'
                           else len(splits.get(f,{}).get('validation_ids',[])))
                        pred=out/'predictions'/cell/arm/scope
                        frac=.95*_prediction_fraction(pred,max(1,n))
                    add(f'{cell}/{arm}/evaluate_{scope}',(arm,),
                        'Reference25 evaluation' if scope=='reference25' else 'Development validation',frac)
    add('shared/final_reports',(*ARMS,'T'),'Final statistics/export',
        (out/'COMPLETE.json').is_file())
    rows=[]
    cells=len(c['seeds'])*c['folds']
    for arm in (*ARMS,'T'):
        assigned=[r for r in nodes.values() if arm in r['strategies']]
        done=sum(r['fraction'] for r in assigned)
        pending=[r for r in assigned if not r['complete']]
        phases=[]
        for r in pending:
            if r['phase'] not in phases:
                phases.append(r['phase'])
        rows.append(dict(strategy=arm,label=LABELS[arm],
            percent=round(100*done/len(assigned),2),completed_work_units=round(done,3),
            total_work_units=len(assigned),fully_completed_units=sum(r['complete'] for r in assigned),
            development_models_evaluated=len(list((out/'evaluations/development_validation').glob(f'*/{arm}/DONE.json'))),
            reference25_models_evaluated=len(list((out/'evaluations/reference25').glob(f'*/{arm}/DONE.json'))),
            models_per_strategy=cells,pending_phases=phases))
    overall=100*sum(r['fraction'] for r in nodes.values())/max(1,len(nodes))
    return dict(updated_utc=utc(),profile=c.get('profile',''),run_directory=str(out),
        definition='Equal planned-work units with partial epoch/prediction credit. Not a time estimate.',
        shared_modules='One task ID, credited to each dependent strategy; counted once overall.',
        overall_percent=round(overall,2),unique_work_units=len(nodes),
        strategies=rows,tasks=list(nodes.values()))


def format_text(value):
    lines=['STRATEGY PROGRESS (%)',
           'Estimated completed work, not remaining time. Shared work is counted once overall.',
           'Strategy  Progress     Work units        DCBR models    Ref25 models']
    for r in value['strategies']:
        lines.append(f'{r["strategy"]:<9} {r["percent"]:6.1f}%    '
            f'{r["completed_work_units"]:7.2f}/{r["total_work_units"]:<6} '
            f'{r["development_models_evaluated"]:>3}/{r["models_per_strategy"]:<8} '
            f'{r["reference25_models_evaluated"]:>3}/{r["models_per_strategy"]}')
    lines.append(f'OVERALL: {value["overall_percent"]:.1f}% (shared modules counted once)')
    lines.append('DCBR = original development-validation folds. Ref25 is evaluation-only.')
    return '\n'.join(lines)+'\n'


def write_progress(c):
    value=calculate(c)
    if Path(c['output_dir']).is_dir():
        folder=Path(c['output_dir'])/'progress'
        atomic_json(folder/'STRATEGY_PROGRESS.json',value)
        write_csv(folder/'STRATEGY_PROGRESS.csv',value['strategies'])
        atomic_bytes(folder/'STRATEGY_PROGRESS.txt',format_text(value).encode())
    return value
