"""Harmonized native-grid surface detection, spatial detection and segmentation."""
from __future__ import annotations
import io, math
from pathlib import Path
import numpy as np
from ._legacy_metrics import surface_record, count_metrics, summarize_records, curve_records
from .data import load_image, restore_probability
from .io import atomic_json, atomic_bytes, write_csv


def threshold_youden(rows,endpoint='overlap'):
    curve=curve_records(rows,endpoint)
    if curve is None: raise ValueError('Youden requires positive and negative original development validation surfaces. No silent fallback.')
    candidates=np.flatnonzero(curve['integer_j']==max(curve['integer_j']))
    i=min(candidates,key=lambda i:(abs(curve['threshold'][i]-.5),curve['threshold'][i]))
    return float(curve['threshold'][i])


def records_for_image(row,probability,surfaces,threshold=.5):
    native=restore_probability(probability,row['geometry'])
    result=[]
    for surface in surfaces:
        if surface['image_id']!=row['sample_id']:continue
        x1,y1,x2,y2=[int(surface[k]) for k in ('x1','y1','x2','y2')]
        ref=load_image(surface['reference_path'])>0;p=native[y1:y2,x1:x2]
        r=surface_record(ref,p,threshold)
        pred=p>=threshold
        r.update(surface)
        r.update(pixel_tp=int((pred&ref).sum()),pixel_fp=int((pred&~ref).sum()),
                 pixel_tn=int((~pred&~ref).sum()),pixel_fn=int((~pred&ref).sum()))
        r['surface_positive_prediction']=int(pred.any())
        r['correct_surface']=int(bool(pred.any())==bool(ref.any()))
        r['correct_overlap']=int(bool(r['detected']) if ref.any() else not pred.any())
        result.append(r)
    return result


def summarize(rows):
    rows=[{**r, "threshold_name":"fixed"} for r in rows]
    x=summarize_records(rows,eligible_only=False)
    y=np.array([bool(int(float(r['reference_positive']))) for r in rows]);p=np.array([int(r['predicted_pixels'])>0 for r in rows])
    sm=count_metrics(int((y&p).sum()),int((~y&p).sum()),int((~y&~p).sum()),int((y&~p).sum()))
    x.update({f'surface_{k}':v for k,v in sm.items()})
    pixel=[sum(int(r.get('pixel_'+k,0)) for r in rows) for k in ('tp','fp','tn','fn')]
    x.update({'pixel_'+k:v for k,v in count_metrics(*pixel).items()})
    tp,fp,tn,fn=pixel;x['pixel_dice_micro']=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
    return x


def objective(m):
    def number(k):
        v=m.get(k)
        if v is None or not math.isfinite(float(v)):return -math.inf
        return float(v)
    return tuple(number(k) for k in ('surface_auroc','surface_auprc_ap','dice_positive_mean'))


def evaluation_from_maps(images,surfaces,map_paths,thresholds,folder,model,scope):
    folder=Path(folder);rows=[];summary=[]
    for name,threshold in thresholds.items():
        part=[]
        for image in images:
            p=np.load(map_paths[image['sample_id']],allow_pickle=False)
            current=records_for_image(image,p,surfaces,threshold)
            for r in current:r.update(model_id=model['model_id'],strategy=model['strategy'],seed=model['seed'],fold=model['fold'],scope=scope,threshold_name='fixed')
            part.extend(current)
        s=summarize(part)
        for r in part:r['threshold_name']=name
        role='development_selection_and_reporting' if scope=='development_validation' else 'evaluation_only_previously_inspected'
        for r in part:r.update(evaluation_role=role,threshold_selection_dataset='original_dcbr_validation')
        summary.append(dict(evaluation_role=role,threshold_selection_dataset='original_dcbr_validation',model_id=model['model_id'],strategy=model['strategy'],seed=model['seed'],fold=model['fold'],scope=scope,threshold_name=name,threshold=threshold,**s))
        for mode in ('surface','overlap'):
            cur=curve_records(part,mode)
            if cur is not None:
                write_csv(folder/f'{name}_{mode}_curve.csv',[{k:float(cur[k][i]) for k in ('threshold','recall','precision','fpr','youden_j')} for i in range(len(cur['threshold']))])
        rows.extend(part)
    write_csv(folder/'surface_results.csv',rows);write_csv(folder/'metrics.csv',summary)
    atomic_json(folder/'metrics.json',summary)
    return summary
