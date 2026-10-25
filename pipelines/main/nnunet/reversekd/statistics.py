"""Paired cluster bootstrap, conditional on the fitted models.

Seeds/folds are NOT independent patients and are never resampled as extra cases.
SD across seed means is reported separately by the reporting module.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
from .io import write_csv,atomic_json,event,check_pause
from .config import GROUPS

from itertools import combinations
# Keep all pairwise comparisons; orient F-E to match its planned KD contrast.
CONTRASTS=[(b,a) if (a,b)==('E','F') else (a,b) for a,b in combinations(GROUPS,2)]
MAIN_CONTRASTS=(('D','D0'),('C','C0'),('F','E'))

def contrast_interpretation(left,right):
    pair=frozenset((left,right))
    if pair in {frozenset(x) for x in MAIN_CONTRASTS}:
        return 'planned KD contrast: same starting weights, data and training budget; KD is the added treatment'
    if pair in {frozenset(('A','D')),frozenset(('B','C'))}:
        return 'KD plus additional clinical training; this comparison does not isolate KD'
    if 'T' in pair:
        return 'different input pipelines: T uses supplied anatomical proximal ROIs'
    return 'complete strategy comparison; interpret differences in training stages and order'
KEYS=['surface_auroc', 'surface_auprc_ap', 'surface_auprc_trapezoid', 'overlap_auroc', 'overlap_auprc_ap', 'overlap_auprc_trapezoid', 'sensitivity', 'specificity', 'mcc', 'precision', 'recall', 'ppv', 'npv', 'f1', 'accuracy', 'balanced_accuracy', 'youden_j', 'dice_positive_mean', 'dice_tp_mean', 'jaccard_positive_mean', 'jaccard_tp_mean', 'surface_sensitivity', 'surface_specificity', 'surface_mcc', 'surface_precision', 'surface_recall', 'surface_ppv', 'surface_npv', 'surface_f1', 'surface_accuracy', 'surface_balanced_accuracy', 'surface_youden_j', 'pixel_sensitivity', 'pixel_specificity', 'pixel_mcc', 'pixel_precision', 'pixel_recall', 'pixel_ppv', 'pixel_npv', 'pixel_f1', 'pixel_accuracy', 'pixel_balanced_accuracy', 'pixel_youden_j', 'pixel_dice_micro', 'pixel_jaccard_micro']


def _areas(y,s,w):
    # Tied scores are collapsed before the step AP / trapezoidal ROC integrations.
    order=np.argsort(-s,kind='stable');y=y[order];w=w[order];s=s[order]
    ends=np.r_[np.flatnonzero(np.diff(s)!=0),len(s)-1]
    tp=np.r_[0,np.cumsum(w*y)[ends]];fp=np.r_[0,np.cumsum(w*(1-y))[ends]]
    if tp[-1]<=0 or fp[-1]<=0:return np.nan,np.nan,np.nan
    tpr=tp/tp[-1];fpr=fp/fp[-1]
    precision=np.divide(tp,tp+fp,out=np.ones_like(tp),where=tp+fp>0)
    return float(np.sum(np.diff(fpr)*(tpr[1:]+tpr[:-1])/2)),float(np.sum(np.diff(tpr)*precision[1:])),float(np.sum(np.diff(tpr)*(precision[1:]+precision[:-1])/2))


def _binary(tp,fp,tn,fn):
    div=lambda a,b:a/b if b else np.nan
    sensitivity=div(tp,tp+fn);specificity=div(tn,tn+fp)
    den=(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    return dict(sensitivity=sensitivity,specificity=specificity,mcc=(tp*tn-fp*fn)/math.sqrt(den) if den else np.nan,
                accuracy=div(tp+tn,tp+fp+tn+fn),balanced_accuracy=(sensitivity+specificity)/2,
                ppv=div(tp,tp+fp),npv=div(tn,tn+fn),f1=div(2*tp,2*tp+fp+fn),precision=div(tp,tp+fp),recall=sensitivity,youden_j=sensitivity+specificity-1)


def calculate(a,w):
    y=a['y'];hit=a['hit'];alarm=a['alarm'];pos=w*y;neg=w*(1-y)
    value=_binary(float(np.dot(pos,hit)),float(np.dot(neg,alarm)),float(np.dot(neg,1-alarm)),float(np.dot(pos,1-hit)))
    surface=_binary(float(np.dot(pos,alarm)),float(np.dot(neg,alarm)),float(np.dot(neg,1-alarm)),float(np.dot(pos,1-alarm)))
    value.update({'surface_'+k:v for k,v in surface.items()})
    auc,ap,trap=_areas(y,a['score'],w);oauc,olap,otrap=_areas(y,np.where(y,a['overlap'],a['score']),w)
    value.update(surface_auroc=auc,surface_auprc_ap=ap,surface_auprc_trapezoid=trap,
        overlap_auroc=oauc,overlap_auprc_ap=olap,overlap_auprc_trapezoid=otrap,
        dice_positive_mean=float(np.dot(pos,a['dice'])/pos.sum()) if pos.sum() else np.nan,
        dice_tp_mean=float(np.dot(pos*hit,a['dice'])/(pos*hit).sum()) if (pos*hit).sum() else np.nan,
        jaccard_positive_mean=float(np.dot(pos,a['jaccard'])/pos.sum()) if pos.sum() else np.nan,
        jaccard_tp_mean=float(np.dot(pos*hit,a['jaccard'])/(pos*hit).sum()) if (pos*hit).sum() else np.nan)
    px={k:float(np.dot(w,a['pixel_'+k])) for k in ('tp','fp','tn','fn')}
    value.update({'pixel_'+k:v for k,v in _binary(**px).items()})
    den=2*px['tp']+px['fp']+px['fn'];union=px['tp']+px['fp']+px['fn']
    value['pixel_dice_micro']=2*px['tp']/den if den else np.nan
    value['pixel_jaccard_micro']=px['tp']/union if union else np.nan
    return np.array([value[k] for k in KEYS],dtype=float)


def _arrays(d,clusters):
    out=[]
    for (strategy,seed,fold,model),g in d.groupby(['strategy','seed','fold','model_id'],sort=True):
        a=dict(strategy=strategy,seed=seed,fold=fold,model=model,clusters=np.array([clusters[x] for x in g.group_id]),
               y=g.reference_positive.to_numpy(float),hit=(g.intersection_pixels.to_numpy(float)>0).astype(float),
               alarm=(g.predicted_pixels.to_numpy(float)>0).astype(float),score=g.score.to_numpy(float),
               overlap=g.overlap_score.fillna(0).to_numpy(float),dice=g.dice_positive.fillna(0).to_numpy(float),
               jaccard=g.jaccard_positive.fillna(0).to_numpy(float),
               **{'pixel_'+k:g['pixel_'+k].to_numpy(float) for k in ('tp','fp','tn','fn')})
        out.append(a)
    return out


def _average(values,arrays,strategies):
    """Same estimator as the tables: mean over folds within seed, then seeds."""
    answer=[]
    for strategy in strategies:
        seed_means=[]
        for seed in sorted({a['seed'] for a in arrays if a['strategy']==strategy}):
            ids=[i for i,a in enumerate(arrays) if a['strategy']==strategy and a['seed']==seed]
            sub=values[ids];n=np.isfinite(sub).sum(0)
            seed_means.append(np.divide(np.nansum(sub,0),n,out=np.full(len(KEYS),np.nan),where=n>0))
        sub=np.array(seed_means);n=np.isfinite(sub).sum(0)
        answer.append(np.divide(np.nansum(sub,0),n,out=np.full(len(KEYS),np.nan),where=n>0))
    return np.array(answer)


def cluster_bootstrap(frame,c,scope,setting,folder):
    d=frame.loc[frame.threshold_name==setting].copy()
    if d.empty:raise ValueError('Paired bootstrap requires evaluated surface rows.')
    if set(d.strategy)-set(GROUPS):raise ValueError('Unknown strategy label in surface results.')
    if 'scope' in d and set(d.scope)!={scope}:raise ValueError('Surface rows do not match the requested evaluation cohort.')
    strategies=[k for k in GROUPS if k in set(d.strategy)]
    required=['sample_id','seed','fold','strategy','group_id','reference_positive','reference_pixels']
    if d[required].isna().any().any():raise ValueError('Paired bootstrap keys, clusters and reference labels cannot be missing.')
    keys=['sample_id','seed','fold']
    if d.duplicated(['strategy',*keys]).any():
        raise ValueError('Duplicate case x seed x fold rows would count surfaces more than once.')
    if (d.groupby(['strategy','seed','fold']).model_id.nunique()!=1).any():
        raise ValueError('Each strategy x seed x fold must identify exactly one fitted model.')
    # Identical IDs alone are insufficient: copied source cases must retain the
    # same cluster and ground truth in every strategy and repetition.
    for field in ('group_id','bitewing_id','patient_id','reference_positive','reference_pixels'):
        if field in d and (d.groupby('sample_id')[field].nunique(dropna=False)>1).any():
            raise ValueError(f'Paired surface reference/cluster alignment differs for {field}.')
    cluster_ids=sorted(d.group_id.unique());clusters={x:i for i,x in enumerate(cluster_ids)}
    arrays=_arrays(d,clusters)
    # Every strategy must contain identical case x seed x fold evaluations.
    pairs={k:set(map(tuple,d.loc[d.strategy==k,['sample_id','seed','fold']].to_numpy())) for k in strategies}
    if len({frozenset(v) for v in pairs.values()})!=1:raise ValueError('Paired bootstrap requires the same evaluated surfaces and fitted-model repetitions in all groups.')
    point=_average(np.array([calculate(a,np.ones(len(a['y']))) for a in arrays]),arrays,strategies)
    reps=int(c['reporting']['bootstrap_replicates']);rng=np.random.default_rng(c['reporting']['bootstrap_seed'])
    draws=np.full((reps,len(strategies),len(KEYS)),np.nan)
    for b in range(reps):
        if b%100==0:
            check_pause(c['output_dir']);event(c['output_dir'],'bootstrap',scope=scope,threshold_name=setting,completed=b,total=reps)
        counts=rng.multinomial(len(cluster_ids),np.full(len(cluster_ids),1/len(cluster_ids)))
        values=np.array([calculate(a,counts[a['clusters']].astype(float)) for a in arrays])
        draws[b]=_average(values,arrays,strategies)
    alpha=1-c['reporting']['confidence_level'];q=[alpha/2,1-alpha/2];rows=[];differences=[]
    for i,strategy in enumerate(strategies):
        for j,key in enumerate(KEYS):
            values=draws[:,i,j];finite=values[np.isfinite(values)]
            ci=np.quantile(finite,q) if len(finite)>=max(100,int(.8*reps)) else [np.nan,np.nan]
            rows.append(dict(scope=scope,threshold_name=setting,strategy=strategy,metric=key,estimate=point[i,j],ci_low=ci[0],ci_high=ci[1],
                bootstrap_valid=len(finite),bootstrap_total=reps,clusters=len(clusters),method='paired cluster percentile bootstrap; fitted models fixed'))
    contrasts=[(left,right) for left,right in CONTRASTS if left in strategies and right in strategies]
    family_size=len(contrasts)
    for left,right in contrasts:
        i,k=strategies.index(left),strategies.index(right)
        for j,key in enumerate(KEYS):
            values=draws[:,i,j]-draws[:,k,j];finite=values[np.isfinite(values)]
            ci=np.quantile(finite,q) if len(finite)>=max(100,int(.8*reps)) else [np.nan,np.nan]
            differences.append(dict(scope=scope,threshold_name=setting,comparison=f'{left} minus {right}',metric=key,
                difference=point[i,j]-point[k,j],ci_low=ci[0],ci_high=ci[1],bootstrap_valid=len(finite),bootstrap_total=reps,
                ci_multiplicity=f'pointwise CI plus Bonferroni familywise CI across {family_size} pairwise comparisons per metric within cohort/threshold',
                comparison_family_size=family_size,
                planned_kd_contrast=frozenset((left,right)) in {frozenset(x) for x in MAIN_CONTRASTS},
                interpretation=contrast_interpretation(left,right),
                ci_familywise_low=np.quantile(finite,alpha/(2*family_size)) if len(finite)>=max(100,int(.8*reps)) else np.nan,
                ci_familywise_high=np.quantile(finite,1-alpha/(2*family_size)) if len(finite)>=max(100,int(.8*reps)) else np.nan,
                models_retrained=False))
    folder=Path(folder);write_csv(folder/f'{setting}_bootstrap_ci.csv',rows);write_csv(folder/f'{setting}_paired_differences.csv',differences)
    atomic_json(folder/f'{setting}_bootstrap_method.json',dict(replicates=reps,clusters=len(clusters),cluster_field='group_id',
        sd_distinct_from_ci=True,model_training_uncertainty_in_ci=False,folds_and_seeds_not_independent_patients=True,
        pairwise_comparisons=family_size,planned_kd_contrasts=[list(x) for x in MAIN_CONTRASTS],
        note='Conditional sampling uncertainty for these fitted models. Does not include rerunning model training or HPO. Dice-TP excludes missed lesions and uses model-specific subsets.'))
    return rows,differences
