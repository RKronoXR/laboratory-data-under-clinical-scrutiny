"""Seven shared fits per cell; six student endpoints; all development frozen first."""
from __future__ import annotations
import os,subprocess,sys,time
from pathlib import Path
from .io import atomic_json,read_json,digest,file_hash,event,check_pause,PauseRequested,read_csv
from .config import STAGES,cells,GROUPS
from .data import prepare
from .metrics import threshold_youden,evaluation_from_maps

def verify_result(result):
    if result.get('reuse_calibration') and file_hash(result['reuse_calibration'])!=result['reuse_calibration_sha256']:
        raise RuntimeError('Imported calibration artifact changed.')
    if 'checkpoint' in result:
        if not Path(result['checkpoint']).is_file() or file_hash(result['checkpoint'])!=result['checkpoint_sha256']:
            raise RuntimeError('A selected checkpoint changed or disappeared.')
    for key,value in result.get('maps',{}).items():
        if isinstance(value,dict):
            for path_key,hash_key in [('probability','probability_sha256'),('roi_mask','roi_sha256')]:
                if not Path(value[path_key]).is_file() or file_hash(value[path_key])!=value[hash_key]:
                    raise RuntimeError(f'Target artifact changed: {key}')
        elif not Path(value).is_file() or file_hash(value)!=result['map_hashes'][key]:
            raise RuntimeError(f'Prediction artifact changed: {key}')

def invoke(c,spec):
    folder=Path(spec['directory']);folder.mkdir(parents=True,exist_ok=True)
    spec['signature']=digest({k:v for k,v in spec.items() if k not in ('signature','config')});spec['config']=c
    p=folder/'result.json'
    if p.exists():
        result=read_json(p)
        if result['signature']!=spec['signature']:raise RuntimeError(f'Task signature changed: {spec["task_id"]}')
        verify_result(result);return result
    check_pause(c['output_dir'])
    # Old full-frame fits are never accepted as native controlled-stage fits.
    # Completed corrected fits resume only by their exact local signed receipt.
    atomic_json(folder/'task.json',spec)
    log=folder/'worker.log';env=dict(os.environ,PYTHONUNBUFFERED='1')
    with log.open('ab',buffering=0) as f:
        proc=subprocess.Popen([sys.executable,'-u',str(Path(c['package_root'])/'run.py'),'_worker',str(folder/'task.json')],
                               stdout=f,stderr=subprocess.STDOUT,env=env)
        from .processes import record
        atomic_json(Path(c['output_dir'])/'worker.json',dict(**record(proc.pid),task_id=spec['task_id'],directory=str(folder),log=str(log),started=time.time()))
        last_change=time.monotonic();last_signature=None;timeout=c['runtime']['worker_no_progress_timeout_seconds']
        while proc.poll() is None:
            try:
                current=Path(c['output_dir'])/'current.json'
                signature=(current.stat().st_mtime_ns,log.stat().st_size) if current.exists() else (0,log.stat().st_size)
                if signature!=last_signature:last_change=time.monotonic();last_signature=signature
                if time.monotonic()-last_change>timeout:
                    proc.terminate()
                    try:proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=15)
                    raise RuntimeError(f'Task made no log/progress update for {timeout}s: {spec["task_id"]}. See {log}. It was stopped; use RESUME.sh after reviewing the error.')
                time.sleep(2)
            except BaseException:
                if proc.poll() is None:
                    proc.terminate()
                    try:proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=15)
                raise
    if proc.returncode==75:raise PauseRequested('Task paused at a checkpoint boundary.')
    if proc.returncode:raise RuntimeError(f'Task failed: {spec["task_id"]}. See {log}')
    if not p.is_file():raise RuntimeError(f'Worker exited without a committed result: {spec["task_id"]}')
    result=read_json(p);verify_result(result);return result

def target_task(c,cell,teacher,images,surfaces,scope,protocol):
    folder=Path(c['output_dir'])/'targets'/cell/scope
    spec=dict(operation='targets',task_id=f'{cell}/targets/{scope}',directory=str(folder),
              teacher=teacher,images=images,surfaces=[s for s in surfaces if s['image_id'] in {i['sample_id'] for i in images}],
              protocol=protocol)
    return invoke(c,spec)['maps']

def train_task(c,cell,stage,split,protocol,parent=None,maps=None,overrides=None,folder=None,task_id=None):
    details=STAGES[stage];t=c['training'];domain=details['domain']
    if domain=='dcbr':lr=c['native_recipe']['initial_lr'];epochs=c['native_recipe']['epochs'];selection='native_ema_foreground_dice'
    else:
        lr=c['native_recipe']['initial_lr'];epochs=t['acta_epochs']
        selection='native_ema_foreground_dice' if parent else 'last'
    overrides=overrides or {}
    epochs=overrides.get('epochs',epochs);lr=overrides.get('lr',lr);selection=overrides.get('selection',selection)
    block='clinical' if domain=='dcbr' else 'finetune' if parent else 'pretrain'
    initialization={k:parent[k] for k in ('checkpoint','checkpoint_sha256','native_bundle','native_recipe') if k in parent} if parent else None
    spec=dict(operation='train',task_id=task_id or f'{cell}/{stage}',directory=str(folder or Path(c['output_dir'])/'runs'/cell/stage),
              stage=stage,kind='student',domain=domain,strategy=details['final'] or 'V',kd=details['kd'],
              seed=int(cell.split('_')[1]),fold=split['fold'],train_ids=split['train_ids'],validation_ids=split['validation_ids'],
              hyperparameters=dict(lr=lr,epochs=epochs,weight_decay=c['native_recipe']['weight_decay'],distillation_weight=t['distillation_weight'],temperature=t['temperature']),initialization=initialization,selection=selection,
              teacher_maps=maps or {},protocol=protocol,native_recipe=c['native_recipe'],
              source_native_bundle=c['native_bundle'],optimizer_reset_at_stage_start=True,
              supervised_continuation_control=('D0' if stage=='K' else 'C0' if stage=='C_ACTA' else 'E' if stage=='F_CLIN' else None))
    return invoke(c,spec)

def freeze_threshold(c,cell,label,model,rows_path):
    path=Path(c['output_dir'])/'thresholds'/cell/f'{label}.json'
    signature=digest(dict(checkpoint=model['checkpoint_sha256'],rows=file_hash(rows_path),threshold=c['threshold']))
    if path.exists():
        old=read_json(path)
        if old['signature']!=signature:raise RuntimeError('Frozen operating threshold provenance changed.')
        return old
    rows=read_csv(rows_path)
    for r in rows:
        for k in ('reference_positive','predicted_pixels','reference_pixels','intersection_pixels'):r[k]=int(float(r[k]))
        for k in ('score','overlap_score','dice_positive'):r[k]=float(r[k]) if r.get(k) else None
    threshold=threshold_youden(rows,c['threshold']['youden_endpoint'])
    value=dict(signature=signature,fixed=c['threshold']['fixed'],youden=threshold,endpoint=c['threshold']['youden_endpoint'],
               checkpoint_sha256=model['checkpoint_sha256'],calibration_file=str(rows_path),calibration_sha256=file_hash(rows_path),
               source='inner_dcbr_validation' if c['hpo_enabled'] else 'original_dcbr_development_validation',reference25_used=False)
    atomic_json(path,value);return value

def evaluate(c,cell,label,model,images,surfaces,scope,protocol):
    folder=Path(c['output_dir'])/'evaluations'/scope/cell/label;receipt=folder/'DONE.json'
    threshold=read_json(Path(c['output_dir'])/'thresholds'/cell/f'{label}.json') if scope=='reference25' else None
    signature=digest(dict(checkpoint=model['checkpoint_sha256'],scope=scope,protocol=protocol,
                          images=[i['sample_id'] for i in images]))
    if receipt.exists():
        old=read_json(receipt)
        if old['signature']!=signature:raise RuntimeError('Committed evaluation identity changed.')
        if any(not (folder/k).is_file() or file_hash(folder/k)!=v for k,v in old['artifacts'].items()):
            raise RuntimeError('Committed evaluation artifact changed.')
        return
    if label=='T':
        descriptors=target_task(c,cell,model,images,surfaces,scope+'_T',protocol)
        maps={key:value['probability'] for key,value in descriptors.items()}
    else:
        spec=dict(operation='predict',task_id=f'{cell}/{label}/predict_{scope}',
            directory=str(Path(c['output_dir'])/'predictions'/cell/label/scope),model=model,images=images,protocol=protocol)
        maps=invoke(c,spec)['maps']
    if scope=='development_validation':
        if label=='T':
            # Score the teacher on its clinical validation crops to calibrate its
            # own threshold; no clinical weights are fitted to that fold.
            from .metrics import records_for_image
            from .io import write_csv
            import numpy as np
            rows=[]
            for image in images:rows.extend(records_for_image(image,np.load(maps[image['sample_id']],allow_pickle=False),surfaces))
            calibration=Path(c['output_dir'])/'teachers'/cell/'validation_surfaces.csv';write_csv(calibration,rows)
        else:calibration=Path(model['spec']['directory'])/'validation_surfaces.csv'
        if c['hpo_enabled']:
            hpo=read_json(Path(c['output_dir'])/'hpo'/cell/'COMPLETE.json')
            if label=='T':
                import numpy as np
                from .metrics import records_for_image
                from .io import write_csv
                inner_ids=set(hpo['inner_split']['validation_ids'])
                bundle=read_json(Path(c['output_dir'])/'data/bundle.json')
                inner_images=[i for i in bundle['images'] if i['sample_id'] in inner_ids]
                desc=target_task(c,cell,model,inner_images,bundle['surfaces'],'inner_calibration_T',protocol)
                inner_rows=[]
                for im in inner_images:inner_rows.extend(records_for_image(im,np.load(desc[im['sample_id']]['probability'],allow_pickle=False),bundle['surfaces']))
                calibration=Path(c['output_dir'])/'hpo'/cell/'teacher_inner_validation_surfaces.csv';write_csv(calibration,inner_rows)
            else:calibration=Path(hpo['inner_models'][label]['spec']['directory'])/'validation_surfaces.csv'
        threshold=freeze_threshold(c,cell,label,model,calibration)
    descriptor=dict(model_id=f'{label}_{cell}',strategy=label,seed=int(cell.split('_')[1]),fold=int(cell.split('_')[3]))
    evaluation_from_maps(images,surfaces,maps,{k:threshold[k] for k in ('fixed','youden')},folder,descriptor,scope,threshold_source='inner_dcbr_validation' if c['hpo_enabled'] else 'original_dcbr_validation')
    atomic_json(receipt,dict(signature=signature,checkpoint_sha256=model['checkpoint_sha256'],threshold=threshold,
        assisted_anatomical_roi_input=(label=='T'),artifacts={p.name:file_hash(p) for p in folder.iterdir() if p.is_file() and p.name!='DONE.json'}))
    from .report import refresh
    refresh(c,plots=False)

def run(c,protocol):
    from .professor import audit_all,snapshot
    from .source_baselines import import_native_baselines,copy_source_evaluation,verify_professor_result_import
    from .report import refresh,finish,export_results
    out=Path(c['output_dir']);event(out,'prepare_data')
    data=prepare(c);event(out,'audit_source_baselines')
    # Audit ALL source cells before any fit. No automatic source retraining.
    baselines=import_native_baselines(c,data)
    audits=audit_all(c,data);teachers={(r['seed'],r['fold']):r for r in audits}
    data_signature=file_hash(out/'data/READY.json');protocol=digest(dict(protocol=protocol,data=data_signature))
    # Publish all already-completed source results before any new experiment.
    # A missing late-cell source copy cannot waste earlier GPU fits.
    professors={(seed,fold):snapshot(c,teachers[(seed,fold)]) for seed,fold in cells(c)}
    professors=verify_professor_result_import(c,data,professors)
    for seed,fold in cells(c):
        cell=f'seed_{seed}_fold_{fold}'
        baseline=baselines[(seed,fold)]
        teacher=professors[(seed,fold)]
        atomic_json(out/'selections'/cell/'A.json',baseline)
        atomic_json(out/'selections'/cell/'T.json',teacher)
        for label,model in [('A',baseline),('T',teacher)]:
            if not copy_source_evaluation(c,cell,label,model,'development_validation',protocol):
                raise RuntimeError(f'Required existing {label} development results cannot be copied. No new fit or recalculation was started.')
    refresh(c,plots=True);export_results(c)
    for seed,fold in cells(c):
        check_pause(out);cell=f'seed_{seed}_fold_{fold}'
        baseline=baselines[(seed,fold)]
        teacher=professors[(seed,fold)];split=next(s for s in data['splits'] if s['fold']==fold)
        preflight_folder=out/'preflight'/cell
        fit_c={**c,'native_recipe':baseline['native_recipe'],'native_bundle':baseline['native_bundle'],'source_native_bundle':baseline['native_bundle'],
               'native_gate_path':str(preflight_folder/'NATIVE_EQUIVALENCE.json')}
        train=[i for i in data['images'] if i['sample_id'] in split['train_ids']]
        validation=[i for i in data['images'] if i['sample_id'] in split['validation_ids']]
        targets=target_task(c,cell,teacher,train,data['surfaces'],'training_only',protocol)
        if not (preflight_folder/'result.json').is_file():
            invoke(fit_c,dict(operation='preflight',task_id=f'{cell}/native_preflight',directory=str(preflight_folder),
                teacher=teacher,teacher_maps={train[0]['sample_id']:targets[train[0]['sample_id']]},
                baseline_model=baseline,native_recipe=baseline['native_recipe'],protocol=protocol))
        selected={'A':baseline,'T':teacher};stages={'A':baseline}
        # D/D0 share source T; C/C0 inherit their matching continuation.
        # E/F share the exact ACTA source checkpoint. All stage optimizers reset.
        for stage in ('N0','K','B_ACTA','C0_ACTA','C_ACTA','V','E_CLIN','F_CLIN'):
            details=STAGES[stage]
            parent=stages[details['parent']] if details['parent'] else None
            result=train_task(fit_c,cell,stage,split,protocol,parent,targets if details['kd'] else None)
            stages[stage]=result
            if details['final']:
                label=details['final'];selected[label]=result
                atomic_json(out/'selections'/cell/f'{label}.json',result)
                evaluate(fit_c,cell,label,result,validation,data['surfaces'],'development_validation',protocol)
        atomic_json(out/'cells'/f'{cell}.json',dict(seed=seed,fold=fold,
            selected_checkpoints={k:r['checkpoint_sha256'] for k,r in selected.items()},
            copied_groups=['A','T'],controlled_pairs=[['D','D0'],['C','C0'],['F','E']]))
        refresh(c,plots=True);export_results(c)
        event(out,'cell_complete',seed=seed,fold=fold)
    frozen=dict(protocol=protocol,models={},thresholds={})
    for seed,fold in cells(c):
        cell=f'seed_{seed}_fold_{fold}'
        for label in GROUPS:
            model=read_json(out/'selections'/cell/f'{label}.json');verify_result(model)
            frozen['models'][cell+'/'+label]=model['checkpoint_sha256']
            frozen['thresholds'][cell+'/'+label]=file_hash(out/'thresholds'/cell/f'{label}.json')
    p=out/'DEVELOPMENT_FROZEN.json'
    if p.exists() and read_json(p)!=frozen:raise RuntimeError('Development models/thresholds changed after freeze.')
    atomic_json(p,frozen);event(out,'prepare_reference25')
    reference=prepare(c,reference=True)
    for seed,fold in cells(c):
        cell=f'seed_{seed}_fold_{fold}';baseline=baselines[(seed,fold)]
        fit_c={**c,'native_recipe':baseline['native_recipe'],'native_bundle':baseline['native_bundle'],'source_native_bundle':baseline['native_bundle']}
        for label in GROUPS:
            model=read_json(out/'selections'/cell/f'{label}.json')
            if label in ('A','T'):
                if not copy_source_evaluation(c,cell,label,model,'reference25',protocol):
                    raise RuntimeError(f'Required source Reference25 results for {label} are missing. They were not recomputed.')
            else:evaluate(fit_c,cell,label,model,reference['images'],reference['surfaces'],'reference25',protocol)
        refresh(c,plots=True);export_results(c)
    atomic_json(out/'TRAINING_AND_EVALUATION_COMPLETE.json',dict(protocol=protocol))
    finish(c);export_results(c)
    atomic_json(out/'COMPLETE.json',dict(protocol=protocol,training_and_evaluation=True,
        statistics_status=str(out/'reports/statistics_status.json')))
    event(out,'complete')
