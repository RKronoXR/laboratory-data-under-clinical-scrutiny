"""Audited reuse of immutable artifacts, never inferred from model filenames."""
from __future__ import annotations
import json, time
from pathlib import Path
from .io import atomic_json, canonical, digest, file_hash, read_json, event


def log(c,kind,**fields):
    p=Path(c['output_dir'])/'audit/REUSE.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a') as f:f.write(canonical(dict(time=time.time(),kind=kind,**fields))+'\n')


def compatible_upgrade(c,old,value):
    """Preserve task identities for this specific, training-equivalent release."""
    if digest({k:v for k,v in old.items() if k!='signature'})!=old['signature']:
        raise RuntimeError('Original protocol receipt is corrupt.')
    for k in ('config','versions','python','inputs'):
        if old[k]!=value[k]:raise RuntimeError(f'Upgrade changed {k}. Use a NEW --run-name; existing work is protected.')
    base=read_json(Path(c['package_root'])/'docs/REUSE_BASELINE.json')
    mutable={'reversekd/config.py','reversekd/data.py','reversekd/pipeline.py','reversekd/monitor.py','reversekd/professor.py','reversekd/report.py','reversekd/worker.py'}
    if value['code'].get('reversekd/worker.py') not in (base['reversekd/worker.py'],'ef476a904f8ecddfce30efd9eb6e1869b962730c772be5b21e1027decbe562e5'):
        raise RuntimeError('Unrecognized worker code; migration refused.')
    if set(value['code'])!=set(base)|{'reversekd/cache.py'} or any(value['code'][k]!=v for k,v in base.items() if k not in mutable):
        raise RuntimeError('Numerical training code changed; compatible migration refused.')
    record=Path(c['output_dir'])/'audit/REUSE_UPGRADE.json'
    if record.exists():
        audit=read_json(record)
        if audit['original_signature']!=old['signature']:
            raise RuntimeError('Original protocol identity changed.')
        if audit['new_code']!=value['code']:
            previous=read_json(Path(c['package_root'])/'docs/SIGNATURE_FIX_PREVIOUS_CODE.json')
            changes={k for k in set(previous)|set(value['code']) if previous.get(k)!=value['code'].get(k)}
            if audit['new_code']!=previous or changes-{'reversekd/worker.py','reversekd/cache.py'}:
                raise RuntimeError('Code changed outside the reviewed serialization fix. Use a new run.')
            audit.setdefault('deployment_history',[]).append(dict(code=audit['new_code'],reason='Before signature serialization fix'))
            audit['new_code']=value['code'];audit['serialization_fix']='Duplicate signature removed; training and checkpoint identities preserved.'
            atomic_json(record,audit)
            log(c,'signature_serialization_fix',original_signature=old['signature'])
    else:
        allowed=read_json(Path(c['package_root'])/'docs/REUSE_GPU_GUARD_HASHES.json')
        previous=dict(old['code']);guard=previous.pop('reversekd/check.py',None)
        baseline=dict(base);baseline.pop('reversekd/check.py')
        if previous!=baseline or guard not in allowed:
            raise RuntimeError('Previous code is not a recognized V2 release. Automatic protocol migration refused.')
        atomic_json(record,dict(original_signature=old['signature'],old_code=old['code'],new_code=value['code'],
            original_counts=old['counts'],new_counts=value['counts'],
            reason='Only artifact caching, equivalent ACTA source sharing, progress accounting and strict protocol migration changed.',
            scientific_training_unchanged=True))
        log(c,'compatible_code_upgrade',original_signature=old['signature'])
    return old['signature']


def roots(c):
    """Only explicit neighboring reverse-KD outputs; original projects read only."""
    candidates=[Path(c['output_base']),Path(c['project_root'])/'acta_reverse_kd_v1/output']
    result=[]
    for base in candidates:
        if base.is_dir():
            for p in sorted(base.glob('*/protocol.json')):
                if p.parent not in result:result.append(p.parent)
    return result


def recipe(spec,config,ready):
    """Exact numerical recipe; ACTA-only V ignores unused clinical fold IDs."""
    source=spec['stage']=='V' and spec['domain']=='acta' and not spec['kd'] and not spec.get('initialization') and spec['selection']=='last'
    h=dict(spec['hyperparameters']);t=config['training']
    h.setdefault('weight_decay',t['weight_decay'])
    h.setdefault('distillation_weight',t['distillation_weight'])
    h.setdefault('temperature',t['temperature'])
    # These fields do not affect the computation when KD is disabled.
    if not spec['kd']:
        h.pop('distillation_weight',None);h.pop('temperature',None)
    training={k:v for k,v in t.items() if k not in (
        'clinical_lr','acta_pretrain_lr','acta_finetune_lr','clinical_epochs','acta_epochs',
        'clinical_weight_decay','finetune_weight_decay','pretrain_weight_decay',
        'distillation_weight','temperature','weight_decay','acta_final_selection','save_last_checkpoint')}
    parent=(spec.get('initialization') or {}).get('checkpoint_sha256')
    maps={k:(v['probability_sha256'],v['roi_sha256']) for k,v in spec.get('teacher_maps',{}).items()}
    return digest(dict(stage=spec['stage'],domain=spec['domain'],kd=spec['kd'],seed=spec['seed'],
        train_ids=None if source else sorted(spec['train_ids']),validation_ids=None if source else sorted(spec['validation_ids']),
        selection=spec['selection'],hyperparameters=h,training=training,parent=parent,maps=maps,
        geometry=[config[k] for k in ('full_image_size','crop_image_size','teacher_image_size')],
        architecture=config.get('architecture_signature'),normalization=config.get('normalization_scheme'),
        normalization_mask=config.get('normalization_use_mask'),batch_dice=config.get('batch_dice'),
        # READY is signed and unchanged, includes original split and all dataset hashes.
        data=ready))


def trusted(root,c,source_only=False):
    p=read_json(root/'protocol.json')
    if digest({k:v for k,v in p.items() if k!='signature'})!=p['signature']:return False
    baseline=read_json(Path(c['package_root'])/'docs/REUSE_BASELINE.json')
    # The reviewed V1 difference is parameter lookup with identical defaults.
    v1=read_json(Path(c['package_root'])/'docs/REUSE_V1_SOURCE_HASHES.json') if source_only else {}
    keys=('reversekd/student.py','reversekd/data.py','reversekd/_source.py') if source_only else ('reversekd/student.py','reversekd/data.py','reversekd/_source.py','reversekd/metrics.py','reversekd/_legacy_metrics.py')
    for key in keys:
        actual=file_hash(Path(c['package_root'])/key)
        if p['code'].get(key) not in (baseline.get(key),actual,v1.get(key)):return False
    current=read_json(Path(c['output_dir'])/'protocol.json')
    return p['versions']==current['versions'] and p['inputs']==current['inputs'] and p['python']==current['python']


def reuse_training(c,spec):
    """Import a completed identical fit; a partial local fit always resumes locally."""
    folder=Path(spec['directory'])
    if (folder/'last.pth').is_file():return None
    ready=Path(c['output_dir'])/'data/READY.json'
    if not ready.is_file() or not (Path(c['output_dir'])/'protocol.json').is_file():return None
    ready_hash=file_hash(ready);wanted=recipe(spec,c,ready_hash)
    from .pipeline import verify_result
    for root in roots(c):
        try:
            if not trusted(root,c,source_only=spec['stage']=='V'):
                log(c,'candidate_run_rejected',source=str(root),reason='Training implementation/environment/plans not identical');continue
            if file_hash(root/'data/READY.json')!=ready_hash:continue
        except (OSError,KeyError,ValueError):continue
        # Includes final stages and HPO trial caches. Alias receipts are skipped.
        files=sorted((root/'runs').glob('*/'+spec['stage']+'/result.json'))
        files+=sorted((root/'hpo').glob('**/result.json'))
        for path in files:
            if path==folder/'result.json':continue
            try:
                r=read_json(path);s=r.get('spec',{})
                if r.get('reused') or s.get('operation')!='train' or s.get('stage')!=spec['stage']:continue
                cfg=s.get('config')
                if not cfg or recipe(s,cfg,ready_hash)!=wanted:continue
                recorded=digest({k:v for k,v in s.items() if k not in ('signature','config')})
                if s.get('signature')!=recorded or r['signature']!=recorded:continue
                verify_result(r)
                if spec['stage']!='V':
                    calibration=Path(s['directory'])/'validation_surfaces.csv'
                    if not calibration.is_file():continue
                    r={**r,'reuse_calibration':str(calibration),'reuse_calibration_sha256':file_hash(calibration)}
                # Retain the original spec directory for the original calibration CSV.
                reused={**r,'signature':spec['signature'],'reused':True,'requested_spec':spec,
                        'reuse_source_result':str(path),'reuse_source_signature':r['signature'],'reuse_recipe':wanted}
                log(c,'student_checkpoint_reused',task=spec['task_id'],source=str(path),checkpoint_sha256=r['checkpoint_sha256'],
                    reason='Exact architecture, data, seed, initialization, loss, hyperparameters and validation recipe match')
                event(c['output_dir'],'student_checkpoint_reused',task_id=spec['task_id'],source=str(path))
                return reused
            except (OSError,KeyError,ValueError,RuntimeError) as exc:
                log(c,'candidate_checkpoint_rejected',source=str(path),reason=str(exc))
    return None


_target_index={}

def find_target(c,signature):
    """Identical teacher/image/geometry/ROI content only; no label-derived keys."""
    cache_key=c['output_dir']
    if cache_key not in _target_index:
        index={}
        for root in roots(c):
            for path in sorted((root/'targets').glob('**/maps/*.json')):
                try:
                    record=read_json(path);index.setdefault(record.get('signature'),[]).append((path,record))
                except (OSError,ValueError):continue
        _target_index[cache_key]=index
    for receipt,r in _target_index[cache_key].get(signature,[]):
        try:
            if r.get('signature')!=signature or r.get('ground_truth_masks_used') is not False:continue
            if not all(Path(p).is_file() and file_hash(p)==h for p,h in r['artifacts'].items()):continue
            probability=next(p for p in r['artifacts'] if p.endswith('.npy'))
            roi=next(p for p in r['artifacts'] if p.endswith('_roi.png'))
            log(c,'teacher_target_reused',source=str(receipt),signature=signature)
            return probability,roi,r
        except (OSError,KeyError,ValueError,StopIteration):continue
    return None


def reuse_data(c,target,reference):
    from .data import _verify,_duplicates
    for root in roots(c):
        source=root/('reference_data' if reference else 'data')
        if source==target or not (source/'READY.json').is_file():continue
        try:
            bundle=_verify(source,c,reference)
            if reference:
                _duplicates(read_json(Path(c['output_dir'])/'data/pixels.json')+read_json(source/'pixels.json'))
            # The immutable bundle keeps its original absolute paths.
            target.symlink_to(source.resolve(),target_is_directory=True)
            log(c,'prepared_data_reused',source=str(source),destination=str(target),reference=reference)
            return bundle
        except (OSError,KeyError,ValueError,RuntimeError) as exc:
            log(c,'candidate_data_rejected',source=str(source),reason=str(exc))
    return None
