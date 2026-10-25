"""Moderate candidates and prespecified, arm-balanced shared HPO selection."""
from __future__ import annotations
import numpy as np
from scipy.stats import rankdata
from .config import is_hpo


def clinical_baseline(c):
    t=c['training']
    return dict(lr=t['baseline_lr'],weight_decay=t['baseline_weight_decay'],epochs=t['max_epochs'])


def teacher_baseline(c):
    t=c['training']
    return dict(lr=t['teacher_lr'],weight_decay=t['teacher_weight_decay'],foreground_sampling=t['teacher_foreground_sampling'])


def candidates(c):
    # Independent RNG streams: enabling one component does not change another's candidates.
    h=c['hpo'];t=c['training']
    def log(g,a):return float(np.exp(g.uniform(np.log(a[0]),np.log(a[1]))))
    g=np.random.default_rng(h['sampler_seed']+1)
    teacher=[teacher_baseline(c)]
    if is_hpo(c,'teacher'):
        teacher += [dict(lr=log(g,h['teacher_lr']),weight_decay=log(g,h['teacher_weight_decay']),
                         foreground_sampling=float(g.choice(h['teacher_foreground_sampling']))) for _ in range(h['teacher_trials']-1)]
    g=np.random.default_rng(h['sampler_seed']+2)
    pre=[dict(lr=t['baseline_lr'],weight_decay=t['baseline_weight_decay'],epochs=t['pretraining_epochs'])]
    if is_hpo(c,'pretrain'):
        pre += [dict(lr=log(g,h['student_lr']),weight_decay=log(g,h['student_weight_decay']),
                     epochs=int(g.choice(h['pretraining_epochs']))) for _ in range(h['pretraining_trials']-1)]
    g=np.random.default_rng(h['sampler_seed']+3)
    clinical=[clinical_baseline(c)]
    if is_hpo(c,'clinical'):
        clinical += [dict(lr=log(g,h['student_lr']),weight_decay=log(g,h['student_weight_decay']),epochs=t['max_epochs']) for _ in range(h['clinical_trials']-1)]
    weights=list(h['distillation_weights']) if is_hpo(c,'distill') else [t['distillation_weight']]
    return dict(teacher=teacher,pretraining=pre,clinical=clinical,distillation_weights=weights,
                clinical_kd_anchor=t['distillation_weight'])


def select_shared(results, arms):
    """results[candidate][arm] -> fit. Every arm contributes one equal rank.

    Primary ranking uses AUROC only. Ties: mean AUROC, then candidate index.
    AP and Dice are reported but do not silently alter this primary objective.
    """
    if not results or any(set(r) != set(arms) for r in results):
        raise ValueError('Shared HPO requires every candidate in every prespecified arm.')
    auc=np.array([[r[a]['metrics']['surface_auroc'] for a in arms] for r in results],dtype=float)
    if not np.isfinite(auc).all():
        raise ValueError('Undefined AUROC in shared HPO; inspect original development validation class coverage.')
    ranks=np.column_stack([rankdata(-auc[:,j],method='average') for j in range(len(arms))])
    mean=ranks.mean(axis=1);mean_auc=auc.mean(axis=1)
    winner=min(range(len(results)),key=lambda i:(mean[i],-mean_auc[i],i))
    audit=[dict(candidate=i,auroc={a:float(auc[i,j]) for j,a in enumerate(arms)},
                rank={a:float(ranks[i,j]) for j,a in enumerate(arms)},mean_rank=float(mean[i]),
                mean_auroc=float(mean_auc[i]),selected=i==winner) for i in range(len(results))]
    return winner,audit
