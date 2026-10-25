"""Read-only reuse of original full-bitewing teachers on their original folds.

Checkpoint selection on the original validation fold is legitimate development
selection, not a reason for retraining. Reference25 is never consulted here.
"""
from __future__ import annotations
import hashlib
import json
import re
import shutil
from pathlib import Path
from .io import (atomic_json, atomic_bytes, read_json, file_hash, digest,
                 write_csv, utc, event)
from .config import discover_plans, is_hpo
from .search import candidates, teacher_baseline


def canonical_id(value):
    match = re.fullmatch(r'(?:bw|dcbr_)?0*(\d+)', str(value))
    if not match:
        raise ValueError(f'Unrecognized full-bitewing identifier: {value}')
    return f'dcbr_{int(match.group(1)):03d}'


def legacy_root(c):
    p = Path(c['teacher_reuse']['legacy_root'])
    return p if p.is_absolute() else Path(c['project_root'])/p


def legacy_split_path(c):
    p = Path(c['teacher_reuse']['legacy_splits'])
    return p if p.is_absolute() else Path(c['project_root'])/p


def legacy_splits(c):
    p = legacy_split_path(c)
    if not p.is_file():
        return None
    splits = read_json(p)
    if len(splits) != c['folds']:
        raise ValueError('Original full-bitewing fold count differs from the requested protocol.')
    result = []
    for f, s in enumerate(splits):
        train = [canonical_id(v) for v in s['train']]
        val = [canonical_id(v) for v in s['val']]
        if len(set(train)) != len(train) or len(set(val)) != len(val) or set(train)&set(val):
            raise ValueError(f'Invalid original split {f}: overlapping/duplicated identifiers.')
        result.append(dict(fold=f, train_ids=train, validation_ids=val))
    return result


def identity(c, data, split, seed):
    """Teacher-only identity, independent of student resolution and student HPO."""
    from .data import teacher_images
    images = {r['sample_id']: r for r in teacher_images(data['images'])}
    files = {sid: dict(image=file_hash(images[sid]['image_path']),
                      mask=file_hash(images[sid]['mask_path']))
             for sid in sorted(split['train_ids'] + split['validation_ids'])}
    plan = read_json(discover_plans(c))
    training = {k:v for k,v in c['training'].items() if k.startswith('teacher_')}
    training.pop('teacher_workers', None)
    return dict(schema=3, partition_policy=c['partition_policy'], seed=int(seed),
        fold=split['fold'], train_ids=sorted(split['train_ids']),
        validation_ids=sorted(split['validation_ids']),
        teacher_image_size=c['teacher_image_size'], input_preprocessing='original_letterbox',
        files=files, plan_configuration=plan['configurations']['2d'],
        labels={'background':0,'caries':1}, framework=c['nnunet_version'],
        training=training, search_grid=candidates(c)['teacher'],
        selection='surface_auroc' if is_hpo(c,'teacher') else 'fixed_hyperparameters',
        code={k:file_hash(Path(c['package_root'])/'actakd'/k)
              for k in ('teacher.py','native_trainer.py','data.py','_source.py','reuse.py')})


def cache_path(c, ident):
    return Path(c['output_base'])/'_shared'/'teachers'/digest(ident)


def read_cache(c, ident):
    folder = cache_path(c, ident)
    manifest = folder/'manifest.json'
    if not manifest.is_file():
        return None
    m = read_json(manifest)
    if m.get('identity') != ident:
        return None
    for rel, h in m['files'].items():
        if not (folder/rel).is_file() or file_hash(folder/rel) != h:
            return dict(compatible=False, reason='SHARED_CACHE_CORRUPTED', path=str(manifest))
    return dict(compatible=True, source='verified_v3_shared_cache', manifest=str(manifest),
        checkpoint=str(folder/'checkpoint_best.pth'),
        checkpoint_sha256=m['files']['checkpoint_best.pth'],
        plans_file=str(folder/'plans.json'), dataset_file=str(folder/'dataset.json'),
        selected_hyperparameters=m['selected_hyperparameters'],
        source_selection=m.get('selection_audit',{}), teacher_image_size=c['teacher_image_size'])


def legacy_checkpoint(c, seed, fold):
    base = legacy_root(c)/'output/runs'/f'seed_{seed}'/f'fold_{fold}'/'full/training'
    paths = sorted(base.glob(
        f'native_results/Dataset*_DCBRFullBW/*__nnUNetPlans__2d/fold_{fold}/checkpoint_best.pth'))
    return (paths[0] if len(paths)==1 else None), base


def _old_digest(value):
    # Exact JSON serialization used by the original dcbr_compare.io module.
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _load_checkpoint_metadata(path):
    """Trusted user-owned checkpoint, CPU only, with memory mapping when supported."""
    import torch
    try:
        ck = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    except (TypeError, RuntimeError):
        ck = torch.load(path, map_location='cpu', weights_only=False)
    init = ck.get('init_args', {})
    result = dict(protocol_seed=ck.get('protocol_seed'),
        protocol_signature=ck.get('protocol_signature'), trainer_name=ck.get('trainer_name'),
        fold=init.get('fold'), configuration=init.get('configuration'),
        has_network_weights=isinstance(ck.get('network_weights'),dict))
    del ck
    return result


def _legacy_data_provenance(c, data):
    """Verify frozen metadata and original input pixels, not student-grid equality."""
    import numpy as np
    from .data import load_image, teacher_images
    root = legacy_root(c)
    protocol_path = root/'output/protocol.json'
    config_path = root/'output/effective_config.json'
    bundle_path = root/'output/data_bundle.json'
    missing = [str(p) for p in (protocol_path, config_path, bundle_path) if not p.is_file()]
    if missing:
        raise ValueError('ORIGINAL_METADATA_MISSING: ' + ', '.join(missing))
    protocol = read_json(protocol_path)
    signature = protocol.get('signature')
    if signature != _old_digest({k:v for k,v in protocol.items() if k!='signature'}):
        raise ValueError('ORIGINAL_PROTOCOL_SIGNATURE_CHANGED')
    cfg = read_json(config_path)
    if protocol.get('configuration') != cfg:
        raise ValueError('ORIGINAL_EFFECTIVE_CONFIGURATION_CHANGED')
    if int(cfg.get('full_size', -1)) != int(c['teacher_image_size']):
        raise ValueError('ORIGINAL_TEACHER_INPUT_SIZE_DIFFERS: '
                         f'original={cfg.get("full_size")}, configured={c["teacher_image_size"]}. '
                         'Restore teacher_image_size to the recorded original size.')
    bundle = read_json(bundle_path)
    original = legacy_splits(c)
    for s in original:
        f = str(s['fold'])
        for key, expected in (('train_ids_by_fold', s['train_ids']),
                              ('validation_ids_by_fold', s['validation_ids'])):
            got = [canonical_id(v) for v in bundle.get(key,{}).get(f,[])]
            if set(got) != set(expected):
                raise ValueError(f'ORIGINAL_NATIVE_SPLIT_PROVENANCE_CHANGED: fold {f}, {key}')
    # The earlier preparation froze the shared crop/full-bitewing assignments.
    manifest = root.parent/'output/data/bitewings.csv'
    expected = protocol.get('frozen_manifest_hashes',{}).get('bitewings.csv')
    if not expected or not manifest.is_file() or file_hash(manifest)!=expected:
        raise ValueError('ORIGINAL_SHARED_BITEWING_MANIFEST_MISSING_OR_CHANGED')
    inventory = protocol.get('full_input_inventory', [])
    proof = {(canonical_id(r['bitewing_id']),r['kind']):r['sha256'] for r in inventory}
    current = teacher_images(data['images'])
    if len(proof)!=len(current)*2 or len(inventory)!=len(proof):
        raise ValueError('ORIGINAL_FULL_INPUT_INVENTORY_INCOMPLETE')
    raw = root/'work/raw'/bundle['dataset_name']
    if not raw.is_dir():
        raw = Path(bundle['raw_directory'])
    for image in current:
        sid = image['sample_id']
        number = int(sid.rsplit('_',1)[1])
        for kind,key,sub,suffix in [('image','image_path','imagesTr','_0000.png'),
                                    ('mask','mask_path','labelsTr','.png')]:
            old = raw/sub/(f'bw{number:03d}'+suffix)
            if not old.is_file() or file_hash(old)!=proof.get((sid,kind)):
                raise ValueError(f'ORIGINAL_INPUT_MISSING_OR_CHANGED: {sid} {kind}')
            if not np.array_equal(load_image(old), load_image(image[key])):
                raise ValueError(f'CURRENT_DATA_DIFFER_FROM_ORIGINAL_TEACHER_PIXELS: {sid} {kind}')
    return dict(signature=signature, configuration=cfg, inventory_verified=True,
        protocol_path=str(protocol_path), protocol_sha256=file_hash(protocol_path),
        original_split_sha256=file_hash(legacy_split_path(c)))


def audit_one(c, data, split, seed):
    ident = identity(c, data, split, seed)
    common = dict(cell=f'seed_{seed}_fold_{split["fold"]}', seed=seed,
                  fold=split['fold'], identity=ident)
    cached = read_cache(c,ident) if c['teacher_reuse']['shared_cache'] else None
    if cached and cached['compatible']:
        return dict(**common, compatible=True, reasons=[], candidate=cached, action='reuse')
    ck, folder = legacy_checkpoint(c,seed,split['fold'])
    if is_hpo(c,'teacher'):
        # A fixed legacy teacher does not bypass an explicitly requested HPO.
        return dict(**common, compatible=False, reasons=[],
                    candidate_checkpoint=str(ck) if ck else str(folder), action='requested_teacher_hpo')
    reasons = []
    # A damaged copy must not invalidate an intact original teacher.
    warnings = [cached['reason']] if cached and not cached['compatible'] else []
    original = legacy_splits(c)
    if ck is None:
        reasons.append('CHECKPOINT_MISSING_OR_AMBIGUOUS')
    if original is None:
        reasons.append('ORIGINAL_SPLIT_FILE_MISSING')
    else:
        old = original[split['fold']]
        if set(old['train_ids']) != set(split['train_ids']):
            reasons.append('TRAINING_CASES_DIFFER_FROM_ORIGINAL_PARTITION')
        if set(old['validation_ids']) != set(split['validation_ids']):
            reasons.append('VALIDATION_CASES_DIFFER_FROM_ORIGINAL_PARTITION')
    if ck is not None and not reasons:
        try:
            proof = _legacy_data_provenance(c, data)
            result_path, spec_path = folder/'result.json', folder/'spec.json'
            if not result_path.is_file() or not spec_path.is_file():
                raise ValueError('ORIGINAL_TRAINING_RESULT_OR_SPEC_MISSING')
            result, spec = read_json(result_path), read_json(spec_path)
            expected_sig = _old_digest(dict(protocol=proof['signature'],seed=seed,fold=split['fold']))
            if result.get('signature') != expected_sig or spec.get('signature') != expected_sig:
                raise ValueError('ORIGINAL_TRAINING_SIGNATURE_MISMATCH')
            if any(int(r.get('seed',-1))!=seed or int(r.get('fold',-1))!=split['fold'] for r in (result,spec)):
                raise ValueError('ORIGINAL_SEED_OR_FOLD_MISMATCH')
            ck_hash = file_hash(ck)
            if result.get('checkpoint_sha256') != ck_hash:
                raise ValueError('ORIGINAL_CHECKPOINT_HASH_MISMATCH')
            metadata = _load_checkpoint_metadata(ck)
            if metadata['protocol_seed']!=seed or metadata['protocol_signature']!=expected_sig:
                raise ValueError('CHECKPOINT_EMBEDDED_SEED_OR_PROTOCOL_MISMATCH')
            if int(metadata.get('fold',-1))!=split['fold'] or metadata.get('configuration')!='2d':
                raise ValueError('CHECKPOINT_EMBEDDED_FOLD_OR_CONFIGURATION_MISMATCH')
            if metadata['trainer_name']!='nnUNetTrainerDCBRFullBW' or not metadata['has_network_weights']:
                raise ValueError('ORIGINAL_TEACHER_ARCHITECTURE_MISMATCH')
            model = ck.parent.parent
            plans_path, dataset_path = model/'plans.json', model/'dataset.json'
            if not plans_path.is_file() or not dataset_path.is_file():
                raise ValueError('ORIGINAL_MODEL_PLANS_OR_DATASET_MISSING')
            plan, dataset = read_json(plans_path), read_json(dataset_path)
            expected_plan=read_json(legacy_root(c)/'output/protocol.json')['copied_plans']
            expected_dataset=read_json(legacy_root(c)/'output/protocol.json')['copied_dataset_json']
            if plan != expected_plan or dataset != expected_dataset:
                raise ValueError('ORIGINAL_MODEL_PLANS_OR_LABELS_CHANGED')
            if dataset.get('labels')!={'background':0,'caries':1} or plan.get('image_reader_writer')!='NaturalImage2DIO':
                raise ValueError('ORIGINAL_TEACHER_INPUT_OR_LABEL_ENCODING_MISMATCH')
            oldcfg=proof['configuration']
            h=dict(lr=float(oldcfg['nnunet_initial_lr']),
                   weight_decay=float(oldcfg['nnunet_weight_decay']),
                   foreground_sampling=float(oldcfg['nnunet_oversample_foreground']))
            source=dict(compatible=True, source='verified_original_full_bitewing_teacher',
                checkpoint=str(ck), checkpoint_sha256=ck_hash,
                plans_file=str(plans_path), dataset_file=str(dataset_path),
                selected_hyperparameters=h, teacher_image_size=c['teacher_image_size'],
                provenance=proof, original_checkpoint_selection='EMA_foreground_Dice_on_original_validation_fold')
            return dict(**common, compatible=True, reasons=[], warnings=warnings, candidate=source, action='reuse')
        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
            reasons.append(str(exc))
    return dict(**common, compatible=False, reasons=reasons, warnings=warnings,
                candidate_checkpoint=str(ck) if ck else str(folder), action='retrain_incompatible')


def alert(c,cell,status,reasons=None,**fields):
    """Write the visible text file BEFORE a teacher retraining can start."""
    out=Path(c['output_dir']);directory=out/'alerts';directory.mkdir(parents=True,exist_ok=True)
    record=dict(time_utc=utc(),cell=cell,status=status,reasons=reasons or [],**fields)
    prior=directory/'teachers'/f'{cell}.json'
    previous=read_json(prior) if prior.exists() else {}
    record={**previous,**record}
    if reasons is None:record['reasons']=previous.get('reasons',[])
    atomic_json(prior,record)
    with (directory/'teacher_history.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush()
    refresh_alert_text(c)
    event(out,'teacher_'+status,cell=cell,reasons=record['reasons'],trigger=record.get('trigger'))


def refresh_alert_text(c):
    out=Path(c['output_dir']);files=sorted((out/'alerts/teachers').glob('*.json'))
    records=[read_json(p) for p in files]
    lines=['TEACHER REUSE / RETRAINING ALERTS',f'Run: {out.name}',
           'This file is updated BEFORE any replacement teacher training starts.',
           'INCOMPATIBILITY retraining is distinct from explicitly requested teacher HPO.',
           'Los modelos originales nunca se modifican ni se borran.','']
    if not records:lines.append('AUDIT PENDING. No conclusion about teacher compatibility yet.')
    for r in records:
        trigger=r.get('trigger','')
        title='!!! REENTRENAMIENTO POR INCOMPATIBILIDAD !!!' if trigger=='incompatibility' else 'TEACHER STATUS'
        lines += [f'{title} | {r["cell"]} | {r["status"].upper()}',f'Updated UTC: {r["time_utc"]}']
        for key in ('candidate_checkpoint','new_checkpoint','worker_log','trigger','cache_manifest'):
            if r.get(key):lines.append(f'{key}: {r[key]}')
        lines += ['Reason: '+x for x in r.get('reasons',[])]
        lines.append('')
    text='\n'.join(lines)+'\n'
    atomic_bytes(out/'TEACHER_RETRAINING_ALERTS.txt',text.encode())
    atomic_bytes(out/'alerts/TEACHER_RETRAINING_ALERTS.txt',text.encode())
    write_csv(out/'alerts/TEACHER_STATUS.csv',[{**r,'reasons':' | '.join(r.get('reasons',[]))} for r in records])


def audit_all(c,data):
    output=[]
    for split in data['splits']:
        for seed in c['seeds']:
            row=audit_one(c,data,split,seed);output.append(row)
            cell=row['cell'];p=Path(c['output_dir'])/'alerts/teachers'/f'{cell}.json'
            # Do not erase a previously started/completed warning on resume.
            if not p.exists():
                status='reuse_verified' if row['compatible'] else 'hpo_planned' if row['action']=='requested_teacher_hpo' else 'retraining_planned'
                alert(c,cell,status,row['reasons'],trigger='explicit_hpo' if is_hpo(c,'teacher') else 'reuse' if row['compatible'] else 'incompatibility',
                      candidate_checkpoint=row.get('candidate_checkpoint',row.get('candidate',{}).get('checkpoint')))
    atomic_json(Path(c['output_dir'])/'alerts/TEACHER_COMPATIBILITY_AUDIT.json',output)
    return output


def publish(c,ident,result,selection_audit=None):
    if not c['teacher_reuse']['shared_cache']:return None
    folder=cache_path(c,ident);folder.mkdir(parents=True,exist_ok=True)
    existing=read_cache(c,ident)
    if existing and existing['compatible']:
        if existing['checkpoint_sha256']!=result['checkpoint_sha256']:
            raise RuntimeError('A verified teacher cache already contains a different checkpoint for the same protocol identity. Refusing overwrite.')
        return existing
    if (folder/'manifest.json').exists():
        import time
        quarantine=folder.with_name(folder.name+'.quarantined_'+str(time.time_ns()))
        folder.rename(quarantine);folder.mkdir(parents=True,exist_ok=True)
        event(c['output_dir'],'teacher_corrupt_cache_preserved',quarantine=str(quarantine))
    inputs={'checkpoint_best.pth':result['checkpoint'], 'plans.json':result['native_bundle']['plans_file'],
            'dataset.json':result['native_bundle']['dataset_file']}
    for name,path in inputs.items():
        p=folder/name
        if p.exists() and file_hash(p)!=file_hash(path):
            raise RuntimeError(f'Incomplete cache contains mismatching artifact: {p}')
        if not p.exists():
            tmp=p.with_suffix(p.suffix+'.building');shutil.copy2(path,tmp);tmp.replace(p)
    atomic_json(folder/'manifest.json',dict(identity=ident,files={x:file_hash(folder/x) for x in inputs},
        selected_hyperparameters=result['spec']['hyperparameters'],selection_audit=selection_audit or {},
        created_utc=utc(),source_checkpoint=result['checkpoint'],source_run=c['output_dir']))
    return read_cache(c,ident)
