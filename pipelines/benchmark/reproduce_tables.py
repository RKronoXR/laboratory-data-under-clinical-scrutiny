"""Reproduce manuscript benchmark tables from the supplied numerical records.

This module uses only the Python standard library. It does not train a model,
read an image, or infer predictions. The additional crop summaries are rebuilt
from regional prediction records, including their original partial coverage.
"""
from __future__ import annotations
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError('No numerical records available for '+str(path))
    with path.open('w', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def num(value):
    if value in (None,'','None','NA','nan','NaN'):
        return None
    x=float(value)
    return x if math.isfinite(x) else None


def boolean(x):
    return str(x).strip().lower() in ('true','1','1.0','yes')


def avg(values):
    v=[x for x in values if x is not None]
    return statistics.mean(v) if v else None


def sd(values):
    v=[x for x in values if x is not None]
    return statistics.stdev(v) if len(v)>1 else None


def ratio(x,y):return x/y if y else None


def summary(rows):
    tp=fp=tn=fn=0;dices=[];hits=[];events={}
    for r in rows:
        positive=boolean(r['reference_positive']);inter=int(float(r['intersection_pixels']));pred=int(float(r['predicted_pixels']))
        ref=int(float(r['reference_pixels']))
        if positive:
            tp+=int(inter>0);fn+=int(inter==0)
            d=2*inter/(ref+pred);dices.append(d)
            if inter:hits.append(d)
        else:
            fp+=int(pred>0);tn+=int(pred==0)
        score=float(r['overlap_score'] if positive else r['score'])
        if score not in events:events[score]=[0,0]
        events[score][0 if positive else 1]+=1
    positive=tp+fn;negative=tn+fp;ct=cf=0;auc=ap=0.
    for score in sorted(events,reverse=True):
        pr,pf=ct/positive,cf/negative;dp,dn=events[score];ct+=dp;cf+=dn
        recall,fpr=ct/positive,cf/negative
        auc+=(fpr-pf)*(recall+pr)/2
        ap+=(recall-pr)*ct/(ct+cf)
    sensitivity=ratio(tp,positive);specificity=ratio(tn,negative)
    denom=(tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)
    return dict(tp=tp,fp=fp,tn=tn,fn=fn,n_surfaces=len(rows),n_positive=positive,n_negative=negative,
                sensitivity=sensitivity,specificity=specificity,ppv=ratio(tp,tp+fp),npv=ratio(tn,tn+fn),
                f1=ratio(2*tp,2*tp+fp+fn),accuracy=ratio(tp+tn,len(rows)),
                balanced_accuracy=(sensitivity+specificity)/2,mcc=(tp*tn-fp*fn)/math.sqrt(denom) if denom else None,
                dice_positive_mean=avg(dices),dice_tp_mean=avg(hits),overlap_auroc=auc,overlap_auprc_ap=ap)


def run(repo, work):
    repo=Path(repo);work=Path(work);data=repo/'data/benchmark_reference';cfg=repo/'configs/benchmark';out=work/'reproduced_tables'
    provenance=json.loads((data/'provenance.json').read_text())
    for name,expected in provenance['files'].items():
        if hashlib.sha256((data/name).read_bytes()).hexdigest()!=expected:
            raise RuntimeError('Numerical input changed: '+name)
    native=read(data/'per_seed_native.csv');paired=read(data/'per_seed_paired.csv')
    groups=json.loads((cfg/'groups.json').read_text());friendly={g['source_group_id']:g['name'] for g in groups}
    all_long=[]
    for domain,rows in [('native',native),('paired',paired)]:
        expected=read(cfg/f'expected_{domain}_summary.csv')
        for e in expected:
            values=[num(r[e['metric']]) for r in rows if r['group_id']==e['group_id'] and r['threshold_name']==e['threshold_name']]
            a,b=avg(values),sd(values)
            for actual,gold,label in [(a,num(e['mean']),'mean'),(b,num(e['sd']),'SD')]:
                if actual is None or gold is None:
                    if actual!=gold:raise ValueError('Undefined summary differs: '+e['group_id']+' '+e['metric'])
                elif not math.isclose(actual,gold,rel_tol=1e-11,abs_tol=1e-12):
                    raise ValueError(f"Benchmark {label} differs: {e['group_id']} {e['metric']}")
            all_long.append(dict(group=friendly[e['group_id']],domain=domain,threshold=e['threshold_name'],metric=e['metric'],mean=a,sd=b,n_seeds=len([x for x in values if x is not None])))
    write(out/'benchmark_all_metrics.csv',all_long)
    table=[]
    metrics={'sensitivity':'Sensitivity','specificity':'Specificity','mcc':'MCC','dice_positive_mean':'Dice_Positive','overlap_auroc':'Localisation_ROC_area','overlap_auprc_ap':'Localisation_AP'}
    paper_order=['laboratory-yolo','laboratory-nnunet','laboratory-segresnet','clinical-bitewings-nnunet','clinical-crops-nnunet','clinical-bitewings-yolo','clinical-crops-yolo','clinical-bitewings-segresnet','clinical-crops-segresnet']
    for g in sorted(groups,key=lambda x:paper_order.index(x['name'])):
        domain='native' if g['dataset']=='acta_direct' else 'paired'
        row={'Pipeline':g['name']}
        for m,label in metrics.items():
            q=next(r for r in all_long if r['group']==g['name'] and r['domain']==domain and r['threshold']=='fixed' and r['metric']==m)
            row[label]=f"{q['mean']:.3f} ({q['sd']:.3f})"
        table.append(row)
    write(out/'Table_5_4.csv',table)
    # Retain the source snapshot's actual fold coverage for the appendix crop reports.
    regional=read(data/'additional_crop_regions.csv.gz');source_seed=read(data/'additional_crop_seed_metrics.csv')
    seed_summaries=[]
    for key in sorted({(r['architecture'],int(r['seed']),r['threshold_name']) for r in regional}):
        arch,seed,threshold=key
        rs=[r for r in regional if (r['architecture'],int(r['seed']),r['threshold_name'])==key]
        s=summary(rs);folds=sorted({int(r['fold']) for r in rs})
        expected=next(r for r in source_seed if r['architecture']==arch and int(r['seed'])==seed and r['threshold_name']==threshold)
        for k,val in s.items():
            if k not in expected:continue
            gold=num(expected[k])
            if val is None or gold is None:
                if val!=gold:raise ValueError('Additional crop undefined result differs: '+str(key)+' '+k)
            elif not math.isclose(val,gold,rel_tol=1e-10,abs_tol=1e-12):raise ValueError('Additional crop result differs: '+str(key)+' '+k)
        seed_summaries.append(dict(architecture=arch,seed=seed,threshold_name=threshold,n_folds=len(folds),**s))
    write(out/'additional_crop_per_seed_recomputed.csv',seed_summaries)
    display=[]
    for arch in ['yolo26_sem','monai_segresnet']:
        for threshold in ['fixed','youden']:
            ss=[r for r in seed_summaries if r['architecture']==arch and r['threshold_name']==threshold]
            # Three complete seeds are a mean+SD row; partial reports remain individual rows.
            chunks=[ss] if len(ss)==3 and all(r['n_folds']==5 for r in ss) else [[r] for r in ss]
            for chunk in chunks:
                row=dict(architecture=arch,threshold=threshold,seeds='/'.join(str(r['seed']) for r in chunk),fold_coverage='/'.join(str(r['n_folds']) for r in chunk))
                for metric in ['sensitivity','specificity','ppv','npv','accuracy','f1','mcc','dice_positive_mean','dice_tp_mean','overlap_auroc','overlap_auprc_ap']:
                    vals=[r[metric] for r in chunk];a,b=avg(vals),sd(vals)
                    row[metric]=f'{a:.3f} ({b:.3f})' if b is not None else (f'{a:.3f}' if a is not None else 'NA')
                row['pixel_micro_dice']='NR'
                display.append(row)
    write(out/'Tables_B21_B23.csv',display)
    detkeys=['architecture','seeds','fold_coverage','sensitivity','specificity','ppv','npv','accuracy','f1','mcc']
    write(out/'Table_B21.csv',[{k:r[k] for k in detkeys} for r in display if r['threshold']=='fixed'])
    write(out/'Table_B22.csv',[{k:r[k] for k in detkeys} for r in display if r['threshold']=='youden'])
    segkeys=['architecture','threshold','seeds','fold_coverage','dice_positive_mean','dice_tp_mean','pixel_micro_dice','overlap_auroc','overlap_auprc_ap']
    write(out/'Table_B23.csv',[{k:r[k] for k in segkeys} for r in display])
    threshold_table=[]
    for arch in ['yolo26_sem','monai_segresnet']:
        rs=[r for r in regional if r['architecture']==arch and r['threshold_name']=='youden']
        values={}
        for r in rs:
            key=(int(r['seed']),int(r['fold']));t=float(r['threshold'])
            if key in values and values[key]!=t:raise ValueError('One fitted model has inconsistent thresholds.')
            values[key]=t
        seeds=sorted({s for s,f in values});means=[statistics.mean(t for (s,f),t in values.items() if s==seed) for seed in seeds]
        threshold_table.append(dict(architecture=arch,n_models=len(values),n_seeds=len(seeds),mean=statistics.mean(means),sd=sd(means),minimum=min(values.values()),maximum=max(values.values())))
    write(out/'Table_B24.csv',threshold_table)
    validation=dict(status='PASS',benchmark_summary_checks=len(all_long),benchmark_table_rows=len(table),additional_seed_summaries=len(seed_summaries),
                    additional_regional_records=len(regional),table_files=['Table_5_4.csv','Table_B21.csv','Table_B22.csv','Table_B23.csv','Table_B24.csv'],
                    annotation_version='recorded-results',
                    scope='Numerical table reproduction; no retraining or new inference.')
    (out/'verification.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    print(json.dumps(validation,indent=2));print('Tables:',out)
    return validation
