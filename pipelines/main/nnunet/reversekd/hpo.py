# Historical reference only. Reverse V3 entry points do not execute this module.
"""Shared blockwise HPO on a grouped inner holdout, never the outer fold/test.

Every candidate is assessed by all six strategies. Prefix fits are cached by
training provenance. Search is coordinate-wise, not a joint exhaustive search.
"""
from __future__ import annotations
import copy,itertools,math
from pathlib import Path
import numpy as np
from .config import STAGES,HPO_BLOCKS
from .io import atomic_json,read_json,digest,file_hash,event,check_pause,write_csv

ORDER=('N','K','B_ACTA','C_ACTA','V','E_CLIN','F_CLIN')
ENDPOINTS=('A','B','C','D','E','F')

def candidates(c,block):
    h=c['hpo'][block]
    if block=='pretrain':
        return [dict(acta_pretrain_lr=float(lr),pretrain_weight_decay=float(wd),acta_pretrain_epochs=int(ep))
            for lr,wd,ep in itertools.product(h['lr'],h['weight_decay'],h['epochs'])]
    if block=='clinical':
        return [dict(clinical_lr=float(lr),clinical_weight_decay=float(wd))
            for lr,wd in itertools.product(h['lr'],h['weight_decay'])]
    if block=='finetune':
        return [dict(acta_finetune_lr=float(lr),finetune_weight_decay=float(wd))
            for lr,wd in itertools.product(h['lr'],h['weight_decay'])]
    if block=='distill':
        return [dict(distillation_weight=float(w),temperature=float(t)) for w,t in itertools.product(h['weight'],h['temperature'])]
    raise ValueError('Unknown HPO block: '+block)

def inner_partition(c,split,data):
    """Keep original fold intact. Split only its training IDs, by patient/BW."""
    from sklearn.model_selection import GroupShuffleSplit
    lookup={r['sample_id']:r for r in data['images']}
    ids=list(split['train_ids']);outer=set(split['validation_ids'])
    if set(ids)&outer:raise ValueError('Outer train/validation overlap.')
    group=lambda sid:lookup[sid].get('patient_id') or lookup[sid]['group_id']
    groups=np.array([group(x) for x in ids]);outer_groups={group(x) for x in outer}
    if set(groups)&outer_groups:raise ValueError('Related cases cross the original fold; repartitioning is forbidden.')
    positive={r['image_id']:set() for r in data['surfaces']}
    for r in data['surfaces']:positive.setdefault(r['image_id'],set()).add(int(r['reference_positive']))
    splitter=GroupShuffleSplit(n_splits=100,test_size=c['hpo']['inner_validation_fraction'],
        random_state=c['hpo']['split_seed']+split['fold'])
    for ti,vi in splitter.split(np.zeros(len(ids)),groups=groups):
        train=[ids[i] for i in ti];val=[ids[i] for i in vi]
        if {x for sid in train for x in positive.get(sid,set())}!={0,1}:continue
        if {x for sid in val for x in positive.get(sid,set())}!={0,1}:continue
        assert not set(train)&set(val) and not set(val)&outer
        return dict(fold=split['fold'],train_ids=train,validation_ids=val,
            outer_validation_ids=list(split['validation_ids']),grouping='patient_id else bitewing group_id',
            source='original_outer_training_only',reference25_used=False)
    raise ValueError('Could not form a grouped inner holdout with positive and sound surfaces in both parts.')

def parameter_config(c,values):
    result=copy.deepcopy(c);result['training'].update(values);return result

def graph(c,cell,split,targets,protocol,train_task,full_budget=False):
    """Return six endpoints and all seven stage checkpoints. Identical fits reuse."""
    out=Path(c['output_dir']);stages={};selected={};t=c['training']
    maps={key:targets[key] for key in split['train_ids']}
    for stage in ORDER:
        details=STAGES[stage];parent=stages.get(details['parent'])
        block='clinical' if details['domain']=='dcbr' else 'finetune' if parent else 'pretrain'
        if block=='clinical':
            epochs=t['clinical_epochs'] if full_budget else min(t['clinical_epochs'],c['hpo']['proxy_clinical_epochs'])
            lr=t['clinical_lr'];selection='clinical_validation'
        elif block=='finetune':
            epochs=t['acta_epochs'] if full_budget else min(t['acta_epochs'],c['hpo']['proxy_acta_finetune_epochs'])
            lr=t['acta_finetune_lr'];selection='clinical_validation'
        else:
            epochs=t.get('acta_pretrain_epochs',t['acta_epochs']);lr=t['acta_pretrain_lr'];selection='last'
        identity=dict(protocol=protocol,cell=cell,stage=stage,inner=split,lr=lr,epochs=epochs,
            weight_decay=t.get(block+'_weight_decay',t['weight_decay']),selection=selection,
            parent=parent['checkpoint_sha256'] if parent else None,
            distillation_weight=t['distillation_weight'] if details['kd'] else None,
            temperature=t['temperature'] if details['kd'] else None)
        key=digest(identity);folder=out/'hpo'/cell/'fits'/key
        # Non-KD signatures ignore KD parameters because those cannot affect the fit.
        cfg=copy.deepcopy(c)
        if not details['kd']:
            cfg['training']['distillation_weight']=0.;cfg['training']['temperature']=1.
        result=train_task(cfg,cell,stage,split,protocol,parent,maps if details['kd'] else None,
            overrides=dict(lr=lr,epochs=epochs,selection=selection),folder=folder,task_id=f'{cell}/hpo/{stage}/{key[:10]}')
        stages[stage]=result
        if details['final']:selected[details['final']]=result
    return selected,stages

def mean_ranks(trials):
    """Equal weight per strategy. AUROC, AP and Dice used lexicographically."""
    from scipy.stats import rankdata
    n=len(trials);rank=np.zeros((n,3))
    for index,key in enumerate(('surface_auroc','surface_auprc_ap','dice_positive_mean')):
        for label in ENDPOINTS:
            values=np.array([tr['scores'][label][key] for tr in trials],float)
            if not np.isfinite(values).all():raise ValueError('HPO objective contains undefined metrics.')
            rank[:,index]+=rankdata(-values,method='average')/len(ENDPOINTS)
    return rank

def write_progress(c):
    out=Path(c['output_dir']);rows=[];choices=[]
    for p in sorted((out/'hpo').glob('*/trials/*/*.json')):
        r=read_json(p)
        for label,m in r['scores'].items():rows.append(dict(cell=p.parents[2].name,block=r['block'],candidate=r['candidate'],strategy=label,
            parameters=str(r['parameters']),**m))
    for p in sorted((out/'hpo').glob('*/selected_*.json')):
        r=read_json(p);choices.append(dict(cell=p.parent.name,block=r['block'],candidate=r['candidate'],parameters=str(r['parameters']),mean_rank=str(r['mean_rank'])))
    write_csv(out/'reports/hpo_trials.csv',rows);write_csv(out/'reports/hpo_choices.csv',choices)

def tune(c,cell,split,data,targets,protocol,train_task):
    from .pipeline import verify_result
    out=Path(c['output_dir']);folder=out/'hpo'/cell;folder.mkdir(parents=True,exist_ok=True)
    signature=digest(dict(protocol=protocol,split=split,hpo=c['hpo'],enabled=c['hpo_enabled']))
    completed=folder/'COMPLETE.json'
    if completed.exists():
        r=read_json(completed)
        if r['signature']!=signature:raise RuntimeError('HPO provenance changed.')
        for stage,result in r['inner_models'].items():verify_result(result)
        return parameter_config(c,r['parameters']),r['final_overrides']
    inner=inner_partition(c,split,data);atomic_json(folder/'inner_partition.json',inner)
    values={};all_choices=[]
    for block in c['hpo_enabled']:
        check_pause(out);trials=[];space=candidates(c,block)
        for index,parameters in enumerate(space):
            path=folder/'trials'/block/f'{index:03d}.json'
            trial_signature=digest(dict(signature=signature,block=block,index=index,previous=values,parameters=parameters))
            if path.exists():
                trial=read_json(path)
                if trial['signature']!=trial_signature:raise RuntimeError('HPO candidate identity changed.')
                for r in trial['models'].values():verify_result(r)
            else:
                event(out,'hpo_candidate',task_id=f'{cell}/hpo/{block}/{index+1}',block=block,candidate=index+1,total=len(space))
                cfg=parameter_config(c,{**values,**parameters})
                models,_=graph(cfg,cell,inner,targets,protocol,train_task)
                scores={label:{key:float(models[label]['metrics'][key]) for key in ('surface_auroc','surface_auprc_ap','dice_positive_mean')} for label in ENDPOINTS}
                trial=dict(signature=trial_signature,block=block,candidate=index,parameters=parameters,scores=scores,models=models)
                atomic_json(path,trial)
            trials.append(trial);write_progress(c)
        ranks=mean_ranks(trials)
        best=min(range(len(trials)),key=lambda i:(*ranks[i],i))
        chosen=trials[best];values.update(chosen['parameters'])
        choice=dict(block=block,candidate=best,parameters=chosen['parameters'],mean_rank=ranks[best].tolist(),
            selection='equal strategy mean rank: AUROC then AP then Dice-positive',reference25_used=False,outer_validation_used=False)
        atomic_json(folder/f'selected_{block}.json',choice);all_choices.append(choice)
        event(out,'hpo_block_complete',task_id=f'{cell}/hpo/{block}',**choice)
    # After search, choose final training durations using ONLY inner validation.
    # Final models refit all original outer-training images for these fixed durations.
    cfg=parameter_config(c,values)
    models,stages=graph(cfg,cell,inner,targets,protocol,train_task,full_budget=True)
    overrides={stage:dict(epochs=int(r['best_epoch']),selection='last') for stage,r in stages.items()}
    value=dict(signature=signature,parameters=values,choices=all_choices,inner_split=inner,
        final_overrides=overrides,inner_models=models,
        outer_validation_used_for_hpo=False,reference25_used=False,
        teacher_retrained=False,mode='coordinate-wise shared grouped inner holdout; final epochs fixed before outer refit')
    atomic_json(completed,value);write_progress(c);event(out,'hpo_cell_complete',task_id=cell)
    return cfg,overrides
