"""Two tiny student checks using the real, already selected teacher targets.

No extra teacher is trained as a smoke test. These fits are excluded from study
results, budgets and HPO selection. They use only original training/validation data.
"""
from __future__ import annotations
import copy
from pathlib import Path
from .io import atomic_json,read_json,digest,event,check_pause


def smoke_student(c,protocol,data,split,maps):
    root=Path(c['output_dir'])/'preflight';receipt=root/'PASSED.json'
    if receipt.exists():
        if read_json(receipt)['protocol']!=protocol:raise RuntimeError('Preflight protocol changed.')
        return
    from .pipeline import _invoke
    cfg=copy.deepcopy(c);t=cfg['training']
    t.update(max_epochs=1,minimum_epochs=1,patience=1,validate_every=1,workers=0)
    train_ids=split['train_ids'][:min(8,len(split['train_ids']))]
    val_ids=split['validation_ids'];seed=int(c['seeds'][0])
    h=dict(lr=t['baseline_lr'],weight_decay=t['baseline_weight_decay'],epochs=1)
    def fit(label,init=None,teacher_maps=None,kd=0):
        s=dict(operation='train',task_id='preflight/'+label,directory=str(root/label),run_root=c['output_dir'],
            kind='student',strategy='PREFLIGHT',seed=seed,fold=split['fold'],hyperparameters=h,
            train_ids=train_ids,validation_ids=val_ids,initialization=init,teacher_maps=teacher_maps or {},
            distillation_weight=kd,protocol=protocol)
        s['signature']=digest(s);s['config']=cfg
        return _invoke(s,cfg)
    event(c['output_dir'],'preflight_student_start')
    baseline=fit('student_supervised');init={k:baseline[k] for k in ('checkpoint','checkpoint_sha256')}
    fit('student_distillation',init,{sid:maps[sid] for sid in train_ids},t['distillation_weight'])
    atomic_json(receipt,dict(protocol=protocol,extra_teacher_fits=0,counted_as_hpo=False,
        checks=['semantic_supervised_training','strict_weight_transfer','real_frozen_teacher_targets','aligned_pixel_distillation'],
        used_training_ids=train_ids,used_validation_ids=val_ids,used_new_internal_split=False,used_reference25=False))
    event(c['output_dir'],'preflight_student_passed')
