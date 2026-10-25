"""Exact, read-only imports of selected V3 baselines and their committed results.

No training or inference fallback exists here. A is the selected V3 T, including
its original input pixels, native plans, thresholds and numeric evaluations.
"""
from __future__ import annotations
import copy
import csv
import io
from pathlib import Path
import numpy as np
from .io import atomic_bytes, atomic_json, digest, file_hash, read_json
from .data import load_image, teacher_images

SUPPORTED_TRAINERS = {
    'nnUNetTrainerACTAKD': ('v3_native', 'd948aed2fb7adb312d6fa43a036f897c57f987026bb16c14203f739913d0726e'),
    'nnUNetTrainerDCBRFullBW': ('legacy_dcbr', '8831365afddc8c067ff9d762cd4fc182e8c255576ecee64791b6546c574658a6'),
}
REQUIRED_EVALUATION = ('metrics.csv', 'metrics.json', 'surface_results.csv')
TEACHER_RESULT_CODE = {
    'reversekd/metrics.py':'05b3b793f84bbfdc783709cf06b29a8308c90919f2a7d26da02751626511b07b',
    'reversekd/_legacy_metrics.py':'f7d02fee8dddc04b906071b90de2a837f35bf9bd9783f959998dfbef7b7fbe0e',
    'reversekd/_source.py':'acad9ebc5773d449c362720742f97db8417961c4aa7c9c349218e75ace94bcb2',
    'reversekd/professor.py':'6ba1c203a7c90f13b10b870e1ebff2238cf667ee32315e6d9ff43e4456b95284',
}
# Reviewed V1 professor differs only by adding content-verified target-cache
# lookup. Crop inference, stitching and numeric results are byte-identical code.
APPROVED_PROFESSOR_CODE = {
    TEACHER_RESULT_CODE['reversekd/professor.py'],
    '3d65db10a94461d9fb19da08fb4fd553eeaf04b3d21b2746572b01f16db7eaf9',
}
# Captured from the user's installed engine in ACTA_CODE_CONTEXT_20261008.
# A same-version locally modified training engine must not pass the audit.
NATIVE_RUNTIME_SOURCES = {
    'trainer':'7096efb2040135eb60df3c8bf39cdbcc373e299671fcee365b3aff2f4da4dfbd',
    'loader':'03e07ba75daa520f14943364a04e6e359f60dc317310f64a9904010f499f6445',
}


def _check_file(path, expected=None):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f'Required source baseline artifact missing: {p}. No retraining or inference fallback.')
    if expected is not None and file_hash(p) != expected:
        raise RuntimeError(f'Source baseline artifact changed: {p}')
    return p


def _copy_exact(source, destination, expected=None):
    src = _check_file(source, expected)
    dst = Path(destination)
    if dst.resolve() == src.resolve():
        raise RuntimeError('Baseline destination must not be the source artifact.')
    if dst.exists():
        _check_file(dst, file_hash(src))
    else:
        atomic_bytes(dst, src.read_bytes())
    return str(dst)


def _receipt(c, cell, label, **fields):
    p = Path(c['output_dir']) / 'reuse' / cell / f'{label}.json'
    old = read_json(p) if p.exists() else {}
    value = {**old, 'schema': 1, 'cell': cell, 'group': label,
             'training_reused': True, 'development_copied': False,
             'reference_copied': False, **old, **fields}
    atomic_json(p, value)
    return value


def _load_checkpoint(path):
    import torch
    try:
        return torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    except (TypeError, RuntimeError):
        return torch.load(path, map_location='cpu', weights_only=False)


def _trainer_source(c, name):
    if name not in SUPPORTED_TRAINERS:
        raise RuntimeError(f'Unverified native teacher trainer: {name}')
    kind, expected = SUPPORTED_TRAINERS[name]
    if kind == 'v3_native':
        path = Path(c['source_v3']) / 'actakd' / 'native_trainer.py'
    else:
        path = Path(c.get('legacy_native_trainer_path') or
                    (Path(c['project_root']) / 'dcbr_full_vs_crops' / 'dcbr_nnunet' / 'native_trainer.py'))
    _check_file(path, expected)
    return kind, str(path), expected


def _source_reuse_candidate(source):
    spec = source.get('spec', {})
    if not isinstance(spec, dict):
        raise RuntimeError('Source teacher spec must be an object.')
    candidate = spec.get('reuse_candidate')
    recorded = source.get('reuse_source')
    if candidate is not None:
        if not isinstance(candidate, dict):
            raise RuntimeError('Source reuse_candidate must be an object.')
        if isinstance(recorded, str) and candidate.get('source') not in (None, recorded):
            raise RuntimeError('Source reuse label differs from its recorded candidate.')
        return candidate
    if isinstance(recorded, dict):
        return recorded
    raise RuntimeError('Legacy source provenance is missing from spec.reuse_candidate.')


def _legacy_runtime(source, configuration, values, conf):
    """Recover inherited runtime choices from the actual historical trainer log."""
    checkpoint=Path(source.get('original_checkpoint') or _source_reuse_candidate(source).get('checkpoint') or source['checkpoint'])
    path=checkpoint.parent/'debug.json'
    if not path.exists() and {'nnunet_n_proc_da','nnunet_compile'}<=set(configuration):
        return int(configuration['nnunet_n_proc_da']),bool(configuration['nnunet_compile']),{}
    debug=read_json(_check_file(path))
    numeric={'epochs':'num_epochs','patience':'_pc_patience','min_delta':'_pc_min_delta',
             'initial_lr':'initial_lr','weight_decay':'weight_decay','train_iterations':'num_iterations_per_epoch',
             'validation_iterations':'num_val_iterations_per_epoch','foreground_sampling':'oversample_foreground_percent'}
    for field, key in numeric.items():
        if key not in debug or float(debug[key])!=float(values[field]):
            raise RuntimeError(f'Historical actual native debug/configuration differs: {key}')
    spec=source['spec']
    if int(debug.get('_pc_seed',-1))!=int(spec['seed']) or int(debug.get('fold',-1))!=int(spec['fold']):
        raise RuntimeError('Historical actual native debug belongs to another seed/fold.')
    if 'dataloader_train.num_processes' in debug:
        workers=int(debug['dataloader_train.num_processes'])
    elif 'SingleThreadedAugmenter' in debug.get('dataloader_train',''):
        workers=0
    else:
        raise RuntimeError('Historical augmentation worker count cannot be verified.')
    if 'nnunet_n_proc_da' in configuration and int(configuration['nnunet_n_proc_da'])!=workers:
        raise RuntimeError('Historical requested/actual augmentation worker count differs.')
    network=debug.get('network','')
    ordinary=conf.get('architecture',{}).get('network_class_name',conf.get('network_arch_class_name','')).rsplit('.',1)[-1]
    if network=='OptimizedModule':compiled=True
    elif ordinary and network==ordinary:compiled=False
    else:raise RuntimeError('Historical network compilation state cannot be verified.')
    if 'nnunet_compile' in configuration and bool(configuration['nnunet_compile'])!=compiled:
        raise RuntimeError('Historical configured/actual compilation differs.')
    return workers,compiled,dict(debug_file=str(path),debug_sha256=file_hash(path),actual_runtime_verified=True)


def derive_native_recipe(c, source, payload):
    """Use the *selected* checkpoint's actual recorded regimen, never defaults."""
    spec = source['spec']; cfg = spec['config']
    name = payload.get('trainer_name')
    kind, path, trainer_hash = _trainer_source(c, name)
    bundle = source['native_bundle']
    plans = read_json(_check_file(bundle['plans_file']))
    dataset = read_json(_check_file(bundle['dataset_file']))
    conf = plans['configurations']['2d']
    if plans.get('image_reader_writer') != 'NaturalImage2DIO':
        raise RuntimeError('Selected T uses a different native image reader.')
    if dataset.get('labels') != {'background': 0, 'caries': 1} or len(dataset.get('channel_names', {})) != 1:
        raise RuntimeError('Selected T label/channel encoding differs.')
    init = payload.get('init_args', {})
    if init.get('plans', {}).get('configurations', {}).get('2d') != conf:
        raise RuntimeError('Selected T embedded/source network plans differ.')
    if init.get('dataset_json') != dataset:
        raise RuntimeError('Selected T embedded/source dataset encoding differs.')
    if conf.get('normalization_schemes') not in (['ZScoreNormalization'], ['NoNormalization']):
        raise RuntimeError('Selected T has unverifiable/global fitted normalization.')
    if int(source.get('teacher_image_size', cfg.get('teacher_image_size', -1))) != 384:
        raise RuntimeError('Selected V3 T is not the original 384-pixel clinical input.')
    if kind == 'legacy_dcbr':
        proof = _source_reuse_candidate(source).get('provenance')
        if not proof or not proof.get('inventory_verified'):
            raise RuntimeError('Legacy T lacks verified original data/configuration provenance.')
        old = proof['configuration']
        if int(old.get('full_size', -1)) != 384:
            raise RuntimeError('Legacy T original full_size differs from 384.')
        keys = {'initial_lr':'nnunet_initial_lr', 'weight_decay':'nnunet_weight_decay',
                'epochs':'nnunet_max_epochs', 'patience':'nnunet_patience',
                'min_delta':'nnunet_min_delta', 'train_iterations':'nnunet_train_iterations',
                'validation_iterations':'nnunet_validation_iterations',
                'foreground_sampling':'nnunet_oversample_foreground'}
        values = {k: old[v] for k, v in keys.items()}
        workers,compile_enabled,runtime_proof=_legacy_runtime(source,old,values,conf)
    else:
        t = cfg['training']; h = spec['hyperparameters']
        values = dict(initial_lr=h['lr'], weight_decay=h['weight_decay'],
                      foreground_sampling=h['foreground_sampling'], epochs=t['teacher_epochs'],
                      patience=t['teacher_patience'], min_delta=t['teacher_min_delta'],
                      train_iterations=t['teacher_train_iterations'],
                      validation_iterations=t['teacher_validation_iterations'])
        workers, compile_enabled = t['teacher_workers'], t['teacher_compile']
        runtime_proof={}
    axes = payload.get('inference_allowed_mirroring_axes')
    if axes is None or not all(int(a) in (0, 1) for a in axes):
        raise RuntimeError('Selected T lacks valid recorded inference mirroring axes.')
    t = cfg['training']
    if 'teacher_mirroring' not in t:
        raise RuntimeError('V3 teacher inference mirroring setting is not recorded.')
    recipe = dict(schema=1, framework='nnunetv2', framework_version=cfg['nnunet_version'],
                  trainer_name=name, trainer_kind=kind, trainer_source=path,
                  trainer_sha256=trainer_hash, input_size=384, configuration='2d',
                  plan_configuration=conf, labels=dataset['labels'],
                  image_reader_writer=plans['image_reader_writer'],
                  normalization_schemes=conf['normalization_schemes'],
                  use_mask_for_norm=conf['use_mask_for_norm'],
                  optimizer='SGD', momentum=.99, nesterov=True, gradient_clip=12.,
                  deep_supervision=True, workers=int(workers), compile=bool(compile_enabled),
                  scheduler={'name':'PolyLRScheduler', 'exponent':.9},
                  selection='EMA_foreground_Dice', augmentation='native_nnUNet_2.8.1',
                  inference={'tile_step_size':.5, 'use_gaussian':True,
                             'use_mirroring':bool(t['teacher_mirroring']),
                             'mirroring_axes':list(axes),'compile':False},
                  native_runtime_sources=dict(NATIVE_RUNTIME_SOURCES),
                  plans_sha256=file_hash(bundle['plans_file']),
                  dataset_sha256=file_hash(bundle['dataset_file']), **values)
    recipe['provenance']=runtime_proof
    if recipe['framework_version'] != c['nnunet_version']:
        raise RuntimeError('Source T nnU-Net version differs from the controlled run.')
    for key in ('epochs', 'patience', 'train_iterations', 'validation_iterations'):
        recipe[key] = int(recipe[key])
        if recipe[key] < 1: raise RuntimeError(f'Invalid native recipe {key}.')
    for key in ('initial_lr', 'weight_decay', 'min_delta', 'foreground_sampling'):
        recipe[key] = float(recipe[key])
        if not np.isfinite(recipe[key]): raise RuntimeError(f'Invalid native recipe {key}.')
    # Both audited subclasses inherit the same native procedures. Their metadata
    # names and equivalent status wrappers do not alter the controlled regimen.
    excluded = {'trainer_name', 'trainer_kind', 'trainer_source', 'trainer_sha256',
                'plans_sha256', 'dataset_sha256','provenance'}
    recipe['recipe_signature'] = digest({k:v for k,v in recipe.items() if k not in excluded})
    return recipe


def _audit_source_data(c, data):
    base = Path(c['source_output'])
    ready = read_json(_check_file(base/'data/READY.json'))
    source = read_json(_check_file(base/'data/bundle.json'))
    if ready.get('source_zip_sha256') != file_hash(c['datasets_zip']):
        raise RuntimeError('V3 and corrected runs do not use the same frozen Datasets.zip.')
    from .reuse import legacy_split_path
    if ready.get('original_splits_sha256') != file_hash(legacy_split_path(c)):
        raise RuntimeError('Selected V3 source split provenance differs.')
    if ready.get('teacher_image_size') != 384 or c['full_image_size'] != 384:
        raise RuntimeError('A and all corrected clinical inputs must preserve source T 384 pixels.')
    if source['splits'] != data['splits']:
        raise RuntimeError('Source/current complete split records differ.')
    old = {r['sample_id']:r for r in teacher_images(source['images'])}
    current = {r['sample_id']:r for r in data['images']}
    if set(old) != set(current): raise RuntimeError('Clinical source/current image IDs differ.')
    for sid, row in current.items():
        prior = old[sid]
        if prior['geometry'] != row['geometry']: raise RuntimeError(f'Clinical geometry differs: {sid}')
        for key in ('image_path','mask_path'):
            p = _check_file(prior[key])
            rel = str(p.relative_to(base/'data'))
            _check_file(p, ready['artifacts'][rel])
            if not np.array_equal(load_image(p), load_image(row[key])):
                raise RuntimeError(f'Clinical source/current decoded pixels differ: {sid}, {key}')
    # Native evaluation reference ROIs and labels must also remain identical.
    def references(rows):
        return {r['sample_id']:{k:v for k,v in r.items() if k != 'reference_path'} for r in rows}
    if references(source['surfaces']) != references(data['surfaces']):
        raise RuntimeError('Source/current surface ROI/label metadata differ.')
    prev = {r['sample_id']:r for r in source['surfaces']}
    for row in data['surfaces']:
        p = _check_file(prev[row['sample_id']]['reference_path'])
        _check_file(p, ready['artifacts'][str(p.relative_to(base/'data'))])
        if not np.array_equal(load_image(p), load_image(row['reference_path'])):
            raise RuntimeError('Source/current native surface reference pixels differ.')
    # Required source Reference25 results must use the same frozen dataset file.
    ref_ready = read_json(_check_file(base/'reference_data/READY.json'))
    if ref_ready.get('source_zip_sha256') != ready['source_zip_sha256']:
        raise RuntimeError('Source Reference25 dataset provenance differs.')
    for rel, expected in ref_ready.get('artifacts', {}).items():
        p = base/'reference_data'/rel
        if not p.resolve().is_relative_to((base/'reference_data').resolve()):
            raise RuntimeError('Source reference artifact escapes its data directory.')
        _check_file(p, expected)
    return source


def _audit_evaluation(base, cell, source_label, scope, checkpoint_hash):
    folder = Path(base)/'evaluations'/scope/cell/source_label
    done = read_json(_check_file(folder/'DONE.json'))
    if done.get('checkpoint_sha256') != checkpoint_hash:
        raise RuntimeError('Source committed evaluation belongs to another checkpoint.')
    for name in REQUIRED_EVALUATION:
        if name not in done.get('artifacts', {}): raise RuntimeError(f'Source evaluation did not commit {name}.')
    for name, expected in done['artifacts'].items():
        p = folder/name
        if not p.resolve().is_relative_to(folder.resolve()): raise RuntimeError('Source artifact path escapes evaluation.')
        _check_file(p, expected)
    threshold = _check_file(Path(base)/'thresholds'/cell/f'{source_label}.json')
    if read_json(threshold) != done['threshold']:
        raise RuntimeError('Source evaluation/frozen threshold disagree.')
    prediction = Path(base)/'predictions'/cell/source_label/scope/'result.json'
    if not prediction.exists() and source_label=='T':
        prediction=Path(base)/'targets'/cell/(scope+'_T')/'result.json'
    maps = read_json(_check_file(prediction))
    task=read_json(_check_file(prediction.parent/'task.json'))
    if task.get('signature')!=maps.get('signature'):
        raise RuntimeError('Source prediction task/result signatures differ.')
    task_hash=task.get('checkpoint_sha256') or task.get('teacher',{}).get('checkpoint_sha256')
    if task_hash!=checkpoint_hash:
        raise RuntimeError('Source prediction task used another checkpoint.')
    expected_images={r['sample_id'] for r in task.get('images',[])}
    if not expected_images or expected_images!=set(maps.get('maps',{})):
        raise RuntimeError('Source prediction image coverage differs from committed task.')
    if not maps.get('maps'):raise RuntimeError('Source prediction map receipt is incomplete.')
    for sid, path in maps['maps'].items():
        if isinstance(path,dict):
            for path_key,hash_key in (('probability','probability_sha256'),('roi_mask','roi_sha256')):
                _check_file(path[path_key],path[hash_key])
        else:
            if sid not in maps.get('map_hashes',{}):raise RuntimeError('Source prediction hash receipt is incomplete.')
            _check_file(path,maps['map_hashes'][sid])
    return dict(folder=str(folder), receipt=str(folder/'DONE.json'),
                threshold=str(threshold), predictions=str(prediction))


def _teacher_results_base(c):
    value=c.get('teacher_results_source')
    if not value:raise FileNotFoundError('teacher_results_source is required for exact professor result import.')
    p=Path(value)
    return p if p.is_absolute() else Path(c['project_root'])/p


def _audit_teacher_data(c, current, source_base, scope):
    folder='data' if scope=='development_validation' else 'reference_data'
    base=Path(source_base)/folder
    ready=read_json(_check_file(base/'READY.json'))
    if ready.get('source_zip_sha256')!=file_hash(c['datasets_zip']):
        raise RuntimeError('Professor source dataset ZIP differs.')
    old=read_json(_check_file(base/'bundle.json'))
    if scope=='development_validation' and old['splits']!=current['splits']:
        raise RuntimeError('Professor source original seed/fold split records differ.')
    originals={r['sample_id']:r for r in old['images']}
    images={r['sample_id']:r for r in current['images']}
    if set(originals)!=set(images):raise RuntimeError('Professor native source image identifiers differ.')
    for sid, row in images.items():
        prior=originals[sid]
        for key in ('native_image_path','native_mask_path'):
            if key not in row or key not in prior:raise RuntimeError('Professor native input provenance missing.')
            p=_check_file(prior[key]);_check_file(p,ready['artifacts'][str(p.relative_to(base))])
            if not np.array_equal(load_image(p),load_image(row[key])):
                raise RuntimeError(f'Professor native source pixels differ: {sid}, {key}')
    def comparable(rows):
        return {r['sample_id']:{k:v for k,v in r.items() if k!='reference_path'} for r in rows}
    if comparable(old['surfaces'])!=comparable(current['surfaces']):
        raise RuntimeError('Professor anatomical ROI/label metadata differ.')
    source_refs={r['sample_id']:r for r in old['surfaces']}
    for row in current['surfaces']:
        p=_check_file(source_refs[row['sample_id']]['reference_path'])
        _check_file(p,ready['artifacts'][str(p.relative_to(base))])
        if not np.array_equal(load_image(p),load_image(row['reference_path'])):
            raise RuntimeError('Professor native reference mask pixels differ.')
    return ready


def verify_professor_result_import(c, data, professors):
    """Audit all old T result sets, preserving their exact projection protocol.

    The old standalone professor was evaluated after its crop predictions were
    stitched to a 768-pixel grid. That original result pipeline is retained for
    exact reuse; the corrected pupil/KD training grid remains 384 pixels.
    """
    expected_cells={(s,f) for s in c['seeds'] for f in range(c['folds'])}
    if set(professors)!=expected_cells:raise RuntimeError('All selected source professor cells must be audited before training.')
    base=_teacher_results_base(c);protocol=read_json(_check_file(base/'protocol.json'))
    if protocol.get('signature')!=digest({k:v for k,v in protocol.items() if k!='signature'}):
        raise RuntimeError('Professor source frozen protocol signature changed.')
    cfg=protocol['config']
    for key in ('teacher_model','ultralytics_version','teacher_image_size','partition_policy'):
        if cfg.get(key)!=c.get(key):raise RuntimeError(f'Professor source {key} differs.')
    if cfg.get('teacher_image_size')!=384 or cfg.get('threshold')!=c.get('threshold'):
        raise RuntimeError('Professor source crop resolution/threshold rule differs.')
    for name, expected in TEACHER_RESULT_CODE.items():
        actual=protocol.get('code',{}).get(name)
        if (actual not in APPROVED_PROFESSOR_CODE if name=='reversekd/professor.py' else actual!=expected):
            raise RuntimeError(f'Professor source unverified prediction/metric code: {name}')
    _audit_teacher_data(c,data,base,'development_validation')
    # Reference25 is already a previously inspected secondary dataset in V3.
    # Comparing provenance reads labels without using them for new selection.
    reference=read_json(_check_file(Path(c['source_output'])/'reference_data/bundle.json'))
    _audit_teacher_data(c,reference,base,'reference25')
    output={};errors=[]
    for (seed,fold), model in professors.items():
        cell=f'seed_{seed}_fold_{fold}'
        try:
            prior=read_json(_check_file(base/'selections'/cell/'T.json'))
            if prior.get('checkpoint_sha256')!=model['checkpoint_sha256']:
                raise RuntimeError('Old standalone professor is not the same selected V3 P checkpoint.')
            if (prior.get('seed'),prior.get('fold'),prior.get('source'))!=(seed,fold,'ACTA_only_V3_P'):
                raise RuntimeError('Old professor source seed/fold/role differs.')
            _check_file(prior['checkpoint'],prior['checkpoint_sha256'])
            evaluations={scope:_audit_evaluation(base,cell,'T',scope,model['checkpoint_sha256'])
                         for scope in ('development_validation','reference25')}
            model.update(source_result_import=True,source_group='T',source_results_output=str(base),
                         source_evaluations=evaluations,source_projection_grid=int(cfg['full_image_size']),
                         source_results_protocol_sha256=file_hash(base/'protocol.json'))
            output[(seed,fold)]=model
            _receipt(c,cell,'T',status='audited',checkpoint_sha256=model['checkpoint_sha256'],
                     source_project=str(base),source_group='T',weights_source_project=c['source_output'],
                     weights_source_group='P',source_projection_grid=int(cfg['full_image_size']))
        except (KeyError,ValueError,RuntimeError,FileNotFoundError) as exc:
            errors.append(f'{cell}: {exc}')
            _receipt(c,cell,'T',status='missing' if isinstance(exc,FileNotFoundError) else 'failed',
                     checkpoint_sha256=model['checkpoint_sha256'],source_project=str(base),source_group='T',error=str(exc))
    if errors:raise RuntimeError('Professor source result import blocked before training:\n'+'\n'.join(errors))
    return output


def verify_baseline_import(c, data):
    """Audit every selected T and both source evaluations before any new fit."""
    _audit_source_data(c, data)
    records = {}; errors = []
    for seed in c['seeds']:
        for fold in range(c['folds']):
            cell = f'seed_{seed}_fold_{fold}'
            selection = Path(c['source_output'])/'selections'/cell/'T.json'
            try:
                r = read_json(_check_file(selection)); s = r['spec']; cfg = s['config']
                if (s['seed'], s['fold'], s['kind'], s['strategy']) != (seed, fold, 'teacher', 'T'):
                    raise RuntimeError('Selected T source seed/fold/role differs.')
                split = next(x for x in data['splits'] if x['fold']==fold)
                for key in ('train_ids','validation_ids'):
                    if s[key] != split[key]: raise RuntimeError(f'Selected T original {key} differ.')
                ck = _check_file(r['checkpoint'], r['checkpoint_sha256']); payload = _load_checkpoint(ck)
                if payload.get('protocol_seed') != seed or payload.get('init_args',{}).get('fold') != fold:
                    raise RuntimeError('Selected T embedded seed/fold differs.')
                if payload.get('init_args',{}).get('configuration') != '2d':
                    raise RuntimeError('Selected T embedded native configuration differs.')
                # A legacy task has a cell-specific signature, validated by V3's
                # source audit and retained in its source candidate metadata.
                if r.get('reused'):
                    candidate = s.get('reuse_candidate', {})
                    if candidate.get('checkpoint_sha256') != r['checkpoint_sha256']:
                        raise RuntimeError('Imported V3 T candidate/selected weights differ.')
                    if payload.get('trainer_name') == 'nnUNetTrainerACTAKD':
                        if not payload.get('protocol_signature'): raise RuntimeError('Source teacher checkpoint signature missing.')
                    elif payload.get('trainer_name') == 'nnUNetTrainerDCBRFullBW':
                        proof = candidate.get('provenance', {})
                        if not proof or not proof.get('inventory_verified'): raise RuntimeError('Legacy source candidate proof missing.')
                        # Exact historical JSON serialization uses digest from
                        # the legacy module, identical to this module's digest.
                        if payload.get('protocol_signature') != digest(dict(protocol=proof['signature'],seed=seed,fold=fold)):
                            raise RuntimeError('Legacy checkpoint embedded protocol signature differs.')
                        proof_path = _check_file(proof['protocol_path'],proof['protocol_sha256'])
                        original_protocol=read_json(proof_path)
                        if original_protocol.get('signature') != proof['signature']:
                            raise RuntimeError('Legacy original protocol provenance differs.')
                        if digest({k:v for k,v in original_protocol.items() if k!='signature'}) != proof['signature']:
                            raise RuntimeError('Legacy frozen protocol signature changed.')
                        if original_protocol.get('configuration') != proof['configuration']:
                            raise RuntimeError('Legacy original training configuration differs.')
                elif payload.get('protocol_signature') != s.get('signature'):
                    raise RuntimeError('V3 teacher checkpoint embedded protocol signature differs.')
                weights = payload.get('network_weights')
                if not isinstance(weights, dict) or not weights:
                    raise RuntimeError('Selected T has no native network weights.')
                for value in weights.values():
                    if hasattr(value,'is_floating_point') and value.is_floating_point():
                        import torch
                        if not torch.isfinite(value).all(): raise RuntimeError('Source T contains nonfinite weights.')
                recipe = derive_native_recipe(c, r, payload); del payload
                evaluations = {scope:_audit_evaluation(c['source_output'],cell,'T',scope,r['checkpoint_sha256'])
                               for scope in ('development_validation','reference25')}
                records[(seed,fold)] = dict(source=r, source_selection=str(selection),
                                            native_recipe=recipe, source_evaluations=evaluations)
                _receipt(c,cell,'A',status='audited',checkpoint_sha256=r['checkpoint_sha256'],
                         source_project=c['source_output'],source_group='T',source_selection=str(selection))
            except (KeyError, ValueError, RuntimeError, FileNotFoundError) as exc:
                errors.append(f'{cell}: {exc}')
                _receipt(c,cell,'A',status='missing' if isinstance(exc,FileNotFoundError) else 'failed',
                         source_project=c['source_output'],source_group='T',error=str(exc))
    if errors: raise RuntimeError('Baseline import blocked before training:\n'+'\n'.join(errors))
    signatures = {r['native_recipe']['recipe_signature'] for r in records.values()}
    if len(signatures) != 1:
        raise RuntimeError('Selected V3 T models have different native training regimens. No silent harmonization.')
    atomic_json(Path(c['output_dir'])/'audit/NATIVE_BASELINE_AUDIT.json',
                {'schema':1,'all_cells_verified':True,'training_fallback':False,
                 'recipe_signature':next(iter(signatures)),
                 'cells':{f'seed_{s}_fold_{f}':r['native_recipe'] for (s,f),r in records.items()}})
    return records


def import_native_baselines(c, data):
    audited = verify_baseline_import(c, data); imported = {}
    for (seed,fold), row in audited.items():
        cell = f'seed_{seed}_fold_{fold}'; src = row['source']
        dst = Path(c['output_dir'])/'baselines'/cell/'A'; bundle={}
        checkpoint = _copy_exact(src['checkpoint'],dst/'checkpoint_best.pth',src['checkpoint_sha256'])
        for key, name in (('plans_file','plans.json'),('dataset_file','dataset.json')):
            bundle[key] = _copy_exact(src['native_bundle'][key],dst/name)
        calibration = Path(src['spec']['directory'])/'validation_surfaces.csv'
        copied_calibration = _copy_exact(calibration,dst/'validation_surfaces.csv')
        _copy_exact(row['source_selection'],dst/'original_selection.json')
        model = {**copy.deepcopy(src),'checkpoint':checkpoint,'native_bundle':bundle,
                 'native_recipe':row['native_recipe'],'source_evaluations':row['source_evaluations'],
                 'source_selection':row['source_selection'],'source_group':'T',
                 'source_output':c['source_output'],'baseline_import':True,'reused':True,
                 'training_reused':True,'new_training':False,'epochs_completed':0,
                 'original_checkpoint':src['checkpoint'],'reuse_calibration':copied_calibration,
                 'reuse_calibration_sha256':file_hash(copied_calibration)}
        # Preserve original role in archived original_selection; downstream A
        # remains tied to the source seed/fold and native source recipe.
        model['spec'] = {**src['spec'],'directory':str(dst),'strategy':'A','stage':'SOURCE_A',
                         'domain':'dcbr','kd':False,'operation':'import_native_baseline'}
        atomic_json(dst/'result.json',model); imported[(seed,fold)] = model
        _receipt(c,cell,'A',status='imported',checkpoint_sha256=model['checkpoint_sha256'],
                 native_recipe_signature=model['native_recipe']['recipe_signature'],source_project=c['source_output'],source_group='T')
    return imported


def _rename_csv(content, cell, label):
    # CSV string values are preserved exactly; only identity columns change.
    stream=io.StringIO(content.decode());reader=csv.DictReader(stream)
    out=io.StringIO();writer=csv.DictWriter(out,fieldnames=reader.fieldnames);writer.writeheader()
    for row in reader:
        if 'strategy' in row: row['strategy']=label
        if 'model_id' in row: row['model_id']=f'{label}_{cell}'
        writer.writerow(row)
    return out.getvalue().encode()


def copy_source_evaluation(c, cell, label, model, scope, protocol):
    """Copy immutable numeric A results. Reference receipt waits for the lock."""
    is_professor=label=='T' and model.get('source_result_import')
    if not model.get('baseline_import') and not is_professor:
        return False
    if not is_professor and (label!='A' or model.get('source_group')!='T'):
        raise RuntimeError('An imported native source baseline may only become A.')
    source_base=model['source_results_output'] if is_professor else model['source_output']
    source = _audit_evaluation(source_base,cell,'T',scope,model['checkpoint_sha256'])
    src=Path(source['folder']);dst=Path(c['output_dir'])/'evaluations'/scope/cell/label
    archive=Path(c['output_dir'])/'baselines'/cell/label/'original_evaluations'/scope
    done=read_json(source['receipt']);artifacts={}
    if scope=='reference25' and not (Path(c['output_dir'])/'DEVELOPMENT_FROZEN.json').is_file():
        # Stage under the archived baseline, never among reported evaluation
        # folders. The source is verified now, but no test result is committed.
        for name,expected in done['artifacts'].items():_copy_exact(src/name,archive/name,expected)
        _copy_exact(source['receipt'],archive/'DONE.json')
        _receipt(c,cell,label,status='imported',reference_staged=True)
        return True
    for name, expected in done['artifacts'].items():
        _copy_exact(src/name,archive/name,expected)
        value=(src/name).read_bytes()
        if name.endswith('.csv') and name in REQUIRED_EVALUATION:
            value=_rename_csv(value,cell,label)
        elif name=='metrics.json':
            rows=read_json(src/name)
            rows=[{**r,'strategy':label,'model_id':f'{label}_{cell}'} for r in rows]
            from .io import canonical
            value=(canonical(rows)+'\n').encode()
        target=dst/name
        if target.exists() and target.read_bytes()!=value: raise RuntimeError('Imported evaluation changed after commitment.')
        if not target.exists():atomic_bytes(target,value)
        artifacts[name]=file_hash(target)
    _copy_exact(source['receipt'],archive/'DONE.json')
    _copy_exact(source['threshold'],Path(c['output_dir'])/'thresholds'/cell/f'{label}.json')
    maps=read_json(source['predictions']);pred=Path(c['output_dir'])/'predictions'/cell/label/scope
    paths={};hashes={}
    for sid,path in maps['maps'].items():
        if isinstance(path,dict):
            # The previous professor worker commits anatomical ROI descriptors.
            copied={**path}
            for path_key,hash_key in (('probability','probability_sha256'),('roi_mask','roi_sha256')):
                copied[path_key]=_copy_exact(path[path_key],pred/'probabilities'/Path(path[path_key]).name,path[hash_key])
            paths[sid]=copied
        else:
            paths[sid]=_copy_exact(path,pred/'probabilities'/Path(path).name,maps['map_hashes'][sid])
            hashes[sid]=maps['map_hashes'][sid]
    atomic_json(pred/'result.json',dict(maps=paths,map_hashes=hashes,source_receipt=source['predictions'],
                                      checkpoint_sha256=model['checkpoint_sha256'],inference_reused=True,
                                      source_projection_grid=model.get('source_projection_grid',384)))
    signature=digest(dict(checkpoint=model['checkpoint_sha256'],scope=scope,protocol=protocol,source_receipt=file_hash(source['receipt'])))
    receipt=dict(signature=signature,checkpoint_sha256=model['checkpoint_sha256'],threshold=done['threshold'],artifacts=artifacts,
                 source_receipt=source['receipt'],source_receipt_sha256=file_hash(source['receipt']),
                 source_artifacts=done['artifacts'],numeric_results_copied_exactly=True,inference_reused=True,new_training=False)
    target=dst/'DONE.json'
    if target.exists() and read_json(target)!=receipt: raise RuntimeError('Imported evaluation receipt changed.')
    atomic_json(target,receipt)
    prior_path=Path(c['output_dir'])/'reuse'/cell/f'{label}.json'
    prior=read_json(prior_path) if prior_path.exists() else {}
    scope_hashes={**prior.get('source_result_sha256_by_scope',{}),scope:file_hash(source['receipt'])}
    _receipt(c,cell,label,status='complete',source_result_sha256=file_hash(source['receipt']),
             source_result_sha256_by_scope=scope_hashes,evaluation_status='copied',
             source_projection_grid=model.get('source_projection_grid',384),
             **{('development_copied' if scope=='development_validation' else 'reference_copied'):True})
    return True


def record_professor_reuse(c, cell, model):
    """Record exact selected P weight reuse independently of evaluation reuse."""
    _check_file(model['checkpoint'],model['checkpoint_sha256'])
    if model.get('source') != 'ACTA_only_V3_P' or model.get('clinical_finetuned'):
        raise RuntimeError('Reverse professor must be the exact selected V3 ACTA-only P.')
    return _receipt(c,cell,'T',status='complete',checkpoint_sha256=model['checkpoint_sha256'],
                    source_project=c['source_output'],source_group='P',
                    source_selection=model['source_selection'],new_training=False,
                    evaluation_status='pending',
                    result_reuse_note='V3 P has no committed clinical comparator results; its exact weights are reused.')
