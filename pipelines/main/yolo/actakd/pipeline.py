"""Original grouped validation folds, shared HPO modules, and locked Reference25 assessment."""
from __future__ import annotations
import os,sys,time,subprocess,shutil
from pathlib import Path
from .io import *
from .config import freeze,counts,is_hpo
from .data import prepare, teacher_images
from .search import candidates,clinical_baseline,select_shared
from .metrics import objective,threshold_youden,evaluation_from_maps


def _verify_model(r):
    if not Path(r['checkpoint']).is_file() or file_hash(r['checkpoint'])!=r['checkpoint_sha256']:
        raise RuntimeError('Selected checkpoint missing/changed. It cannot be silently retrained under an existing selection.')


def _invoke(spec,c):
    folder=Path(spec['directory']);folder.mkdir(parents=True,exist_ok=True)
    target=folder/'result.json'
    if target.exists():
        r=read_json(target)
        if r['signature']!=spec['signature']:raise RuntimeError(f'Task signature changed: {spec["task_id"]}')
        return r
    check_pause(c['output_dir']);atomic_json(folder/'task.json',spec)
    env=dict(os.environ,PYTHONUNBUFFERED='1',OMP_NUM_THREADS=str(c['training']['cpu_threads']),MKL_NUM_THREADS=str(c['training']['cpu_threads']))
    log=folder/'worker.log'
    with log.open('ab',buffering=0) as f:
        p=subprocess.Popen([sys.executable,str(Path(c['package_root'])/'run.py'),'_worker',str(folder/'task.json')],stdout=f,stderr=subprocess.STDOUT,env=env)
        atomic_json(Path(c['output_dir'])/'worker.json',dict(pid=p.pid,task_id=spec['task_id'],log=str(log),directory=str(folder)))
        code=p.wait()
    if code==75:raise PauseRequested('Worker paused at a safe boundary.')
    if code!=0:raise RuntimeError(f'Task failed: {spec["task_id"]}. See {log}')
    return read_json(target)


def _spec(c,protocol,cell,task,kind,strategy,h,split,initialization=None,teacher_maps=None,kd=0.):
    folder=Path(c['output_dir'])/'runs'/cell/task
    value=dict(operation='train',task_id=f'{cell}/{task}',directory=str(folder),run_root=c['output_dir'],
        kind=kind,strategy=strategy,seed=int(cell.split('_')[1]),fold=split['fold'],hyperparameters=h,
        train_ids=split['train_ids'],validation_ids=split['validation_ids'],initialization=initialization,
        teacher_maps=teacher_maps or {},distillation_weight=kd,protocol=protocol)
    # Fingerprint identities/config/teacher checkpoint, not only filenames.
    value['signature']=digest(value);value['config']=c
    return value


def _train(c,protocol,cell,task,kind,strategy,h,split,initialization=None,teacher_maps=None,kd=0.):
    s=_spec(c,protocol,cell,task,kind,strategy,h,split,initialization,teacher_maps,kd)
    return _invoke(s,c)


def _select(results):
    if not results:raise ValueError('Empty hyperparameter search.')
    return max(enumerate(results),key=lambda ir:(objective(ir[1]['metrics']),-ir[0]))[1]


def _prune(results,keep,c):
    if not c['reporting']['prune_unselected_checkpoints']:return
    kept={str(Path(r['checkpoint']).resolve()) for r in keep}
    for r in results:
        folder=Path(r['spec']['directory'])
        if not folder.resolve().is_relative_to(Path(c['output_dir']).resolve()):
            continue  # Never prune original teachers or another run's shared assets.
        for pattern in ('*.pt','**/checkpoint*.pth'):
            for p in folder.glob(pattern):
                if str(p.resolve()) not in kept:p.unlink(missing_ok=True)
        atomic_json(folder/'retention.json',dict(selected_checkpoint_retained=str(Path(r['checkpoint']).resolve()) in kept,
            note='Unselected/optimizer checkpoints pruned after selection; metrics, hyperparameters and checkpoint hashes retained.'))


def _selection_path(c,cell,phase):return Path(c['output_dir'])/'selections'/cell/f'{phase}.json'


def _get_selected(c,cell,phase):
    p=_selection_path(c,cell,phase)
    if not p.exists():return None
    r=read_json(p);_verify_model(r);return r


def _save_selected(c,cell,phase,r):atomic_json(_selection_path(c,cell,phase),r)


def _report(c,plots=False):
    from .report import refresh
    refresh(c,plots=plots)
    from .progress import write_progress
    write_progress(c)


def _search_phase(c,protocol,cell,phase,kind,grid,split,init=None,maps=None,weights=None):
    old=_get_selected(c,cell,phase)
    if old:return old
    results=[]
    for i,h in enumerate(grid):
        r=_train(c,protocol,cell,f'{phase}/trial_{i:03}',kind,phase,h,split,init,maps,weights[i] if weights else 0.)
        results.append(r);_report(c)
    chosen=_select(results);_save_selected(c,cell,phase,chosen);_prune(results,[chosen],c)
    return chosen


def _pretrain_search(c,protocol,cell,grid,split):
    old=_get_selected(c,cell,'P')
    if old:return old
    if not is_hpo(c,'pretrain'):
        chosen=_train(c,protocol,cell,'P/fixed','pretrain','P',grid[0],split)
        _save_selected(c,cell,'P',chosen)
        return chosen
    pre=[];probe=[]
    for i,h in enumerate(grid):
        r=_train(c,protocol,cell,f'P/trial_{i:03}','pretrain','P',h,split)
        init={k:r[k] for k in ('checkpoint','checkpoint_sha256')}
        baseline=dict(lr=c['training']['baseline_lr'],weight_decay=c['training']['baseline_weight_decay'],epochs=c['training']['max_epochs'])
        q=_train(c,protocol,cell,f'P_probe/trial_{i:03}','student','P_probe',baseline,split,init)
        pre.append(r);probe.append(q);_report(c)
    best=_select(probe);i=next(i for i,q in enumerate(probe) if q['checkpoint']==best['checkpoint'])
    chosen={**pre[i],'transfer_probe_metrics':best['metrics'],'transfer_probe_task':best['spec']['task_id']}
    _save_selected(c,cell,'P',chosen);_prune(pre,[chosen],c);_prune(probe,[],c)
    return chosen


def _predict(c,cell,label,model,images,scope):
    directory=Path(c['output_dir'])/'predictions'/cell/label/scope
    source=Path(model['spec']['directory'])/'result.json'
    spec=dict(operation='predict',task_id=f'{cell}/{label}/predict_{scope}',directory=str(directory),source_result=str(source),
        checkpoint_sha256=model['checkpoint_sha256'],images=images,kind=model['spec']['kind'],strategy=label,seed=model['spec']['seed'],fold=model['spec']['fold'])
    spec['signature']=digest(spec);spec['config']=c
    r=_invoke(spec,c)
    for k,p in r['maps'].items():
        if not Path(p).is_file() or file_hash(p)!=r['map_hashes'][k]:raise RuntimeError(f'Prediction artifact changed: {p}')
    return r['maps']


def _threshold(c,cell,label,model):
    path=Path(c['output_dir'])/'thresholds'/cell/f'{label}.json'
    source=Path(model['spec']['directory'])/'validation_surfaces.csv'
    if path.exists():
        value=read_json(path)
        if value['checkpoint_sha256']!=model['checkpoint_sha256'] or value['calibration_sha256']!=file_hash(source):raise RuntimeError('Frozen threshold provenance changed.')
        return value
    rows=read_csv(source)
    for r in rows:
        for key in ('reference_positive','predicted_pixels','intersection_pixels','reference_pixels'):r[key]=int(float(r[key]))
        for key in ('score','overlap_score','dice_positive'):
            r[key]=float(r[key]) if r.get(key) else None
    threshold=threshold_youden(rows,c['threshold']['youden_endpoint'])
    value=dict(fixed=c['threshold']['fixed'],youden=threshold,youden_endpoint=c['threshold']['youden_endpoint'],
        source='original_development_validation',calibration_file=str(source),calibration_sha256=file_hash(source),checkpoint_sha256=model['checkpoint_sha256'],
        n_surfaces=len(rows),n_positive=sum(r['reference_positive'] for r in rows),
        tie_rule='Maximum integer Youden J; closest to 0.5; then smaller threshold.',reference25_used=False,independent_test_used=False,
        development_validation_reused_for_reporting=True)
    atomic_json(path,value);return value


def _evaluate(c,cell,label,model,images,surfaces,scope):
    if label=='T':
        images=teacher_images(images)
    threshold=_threshold(c,cell,label,model);folder=Path(c['output_dir'])/'evaluations'/scope/cell/label
    descriptor=dict(model_id=f'{label}_{cell}',strategy=label,seed=model['spec']['seed'],fold=model['spec']['fold'])
    signature=digest(dict(model=model['checkpoint_sha256'],threshold=threshold,image_ids=[i['sample_id'] for i in images]))
    receipt=folder/'DONE.json'
    if receipt.exists():
        old=read_json(receipt)
        if old['signature']!=signature:raise RuntimeError('Evaluation changed after freeze.')
        if any(not (folder/name).is_file() or file_hash(folder/name)!=h for name,h in old['artifacts'].items()):raise RuntimeError('Committed evaluation artifact was changed or removed.')
        return
    maps=_predict(c,cell,label,model,images,scope)
    evaluation_from_maps(images,surfaces,maps,{k:threshold[k] for k in ('fixed','youden')},folder,descriptor,scope)
    artifacts={str(p.relative_to(folder)):file_hash(p) for p in folder.glob('*') if p.is_file() and p.name!='DONE.json'}
    atomic_json(receipt,dict(signature=signature,checkpoint_sha256=model['checkpoint_sha256'],threshold=threshold,artifacts=artifacts))
    _report(c,plots=True)


def _teacher_phase(c,protocol,cell,split,data,grid):
    from . import reuse
    old=_get_selected(c,cell,'T')
    if old:return old
    seed=int(cell.split('_')[1]);audit=reuse.audit_one(c,data,split,seed)
    if audit['compatible']:
        source=audit['candidate']
        spec=_spec(c,protocol,cell,'T/reused','teacher','T',source['selected_hyperparameters'],split)
        spec.update(operation='import_teacher',reuse_candidate=source)
        spec['signature']=digest({k:v for k,v in spec.items() if k not in ('signature','config')})
        selected=_invoke(spec,c)
        _save_selected(c,cell,'T',selected)
        reuse.alert(c,cell,'reused',[],trigger='reuse',new_checkpoint=selected['checkpoint'],
                    cache_manifest=source.get('manifest',''))
        return selected
    if not is_hpo(c,'teacher') and c['teacher_reuse']['on_incompatible']!='retrain':
        raise RuntimeError('Teacher reuse failed: '+ '; '.join(audit['reasons']))
    trigger='explicit_hpo' if is_hpo(c,'teacher') else 'incompatibility'
    status='hpo_running' if trigger=='explicit_hpo' else 'retraining_running'
    reuse.alert(c,cell,status,audit['reasons'],trigger=trigger,
                candidate_checkpoint=audit.get('candidate_checkpoint'),
                worker_log=str(Path(c['output_dir'])/'runs'/cell/'T/trial_000/worker.log'))
    try:
        results=[]
        for i,h in enumerate(grid):
            r=_train(c,protocol,cell,f'T/trial_{i:03}','teacher','T',h,split)
            results.append(r);_report(c)
        selected=_select(results)
        # Publish to the shared cache before pruning. C/D and later compatible
        # NO_HPO/HPO-student runs can reuse this exact trained teacher.
        choice=dict(selected_hyperparameters=selected['spec']['hyperparameters'],
                    trials=[dict(hyperparameters=r['spec']['hyperparameters'],metrics=r['metrics']) for r in results])
        shared=reuse.publish(c,audit['identity'],selected,choice)
        _save_selected(c,cell,'T',selected);_prune(results,[selected],c)
        reuse.alert(c,cell,'hpo_completed' if trigger=='explicit_hpo' else 'retraining_completed',
                    trigger=trigger,new_checkpoint=selected['checkpoint'],cache_manifest=(shared or {}).get('manifest',''))
        return selected
    except PauseRequested:
        reuse.alert(c,cell,'paused',trigger=trigger);raise
    except Exception as exc:
        reuse.alert(c,cell,'failed',trigger=trigger,error=str(exc));raise


def _shared_clinical(c,protocol,cell,grid,split,init,maps):
    receipt=Path(c['output_dir'])/'hpo_selections'/cell/'clinical.json'
    if receipt.exists():
        value=read_json(receipt)
        for r in value['selected_models'].values():_verify_model(r)
        return value['selected_models']
    results=[];anchor=float(c['training']['distillation_weight'])
    for i,h in enumerate(grid):
        quartet={}
        for arm in ('A','B','C','D'):
            quartet[arm]=_train(c,protocol,cell,f'clinical/trial_{i:03}/{arm}','student',arm,h,split,
                init if arm in ('B','D') else None,maps if arm in ('C','D') else None,
                anchor if arm in ('C','D') else 0.)
            _report(c)
        results.append(quartet)
    winner,audit=select_shared(results,('A','B','C','D'))
    selected=results[winner]
    atomic_json(receipt,dict(objective='surface_auroc',selection_rule='minimum mean within-arm AUROC rank; tie: mean AUROC, then candidate index',
        candidate_audit=audit,selected_hyperparameters=grid[winner],kd_anchor=anchor,selected_models=selected,
        selection_data='original_development_validation',reference25_used=False))
    _prune([r for q in results for r in q.values()],list(selected.values()),c)
    return selected


def _shared_distillation(c,protocol,cell,weights,split,init,maps,clinical):
    receipt=Path(c['output_dir'])/'hpo_selections'/cell/'distill.json'
    if receipt.exists():
        value=read_json(receipt)
        for r in value['selected_models'].values():_verify_model(r)
        return value['selected_models']
    anchor=float(c['training']['distillation_weight']);h=clinical['A']['spec']['hyperparameters'];results=[]
    for i,weight in enumerate(weights):
        if float(weight)==anchor:
            pair={arm:clinical[arm] for arm in ('C','D')}
        else:
            pair={}
            for arm in ('C','D'):
                pair[arm]=_train(c,protocol,cell,f'distill/trial_{i:03}/{arm}','student',arm,h,split,
                    init if arm=='D' else None,maps,float(weight))
                _report(c)
        results.append(pair)
    winner,audit=select_shared(results,('C','D'));selected=results[winner]
    atomic_json(receipt,dict(objective='surface_auroc',candidate_audit=audit,
        selected_weight=float(weights[winner]),selected_models=selected,
        clinical_hyperparameters=h,selection_data='original_development_validation',reference25_used=False,
        note='Anchor C/D checkpoints are reused exactly; not retrained. This is a staged search, not the full Cartesian clinical-by-weight grid.'))
    # Also retain the clinical phase winners to make partial-run recovery safe.
    _prune([r for pair in results for r in pair.values()],list(selected.values())+list(clinical.values()),c)
    return selected


def run(c):
    out=Path(c['output_dir']);out.mkdir(parents=True,exist_ok=True)
    with exclusive_lock(out/'.run.lock'):
        protocol=freeze(c);event(out,'preparing',counts=counts(c));data=prepare(c)
        from .teacher import setup_bundle
        from . import reuse
        reuse.audit_all(c,data)
        # Native training data are prepared lazily only if a teacher actually needs training.
        grid=candidates(c);atomic_json(out/'candidate_configurations.json',grid)
        # Fold-major order provides one complete T/A/B/C/D cell at a time.
        for split in data['splits']:
            for seed in c['seeds']:
                check_pause(out);cell=f'seed_{seed}_fold_{split["fold"]}'
                event(out,'cell_start',cell=cell)
                from .progress import write_progress
                write_progress(c)
                T=_teacher_phase(c,protocol,cell,split,data,grid['teacher'])
                train=[r for r in data['images'] if r['sample_id'] in split['train_ids']]
                maps=_predict(c,cell,'T',T,train,'distillation_train_cache')
                from .preflight import smoke_student
                smoke_student(c,protocol,data,split,maps)
                P=_pretrain_search(c,protocol,cell,grid['pretraining'],split)
                init={k:P[k] for k in ('checkpoint','checkpoint_sha256')}
                clinical=_shared_clinical(c,protocol,cell,grid['clinical'],split,init,maps)
                distill=_shared_distillation(c,protocol,cell,grid['distillation_weights'],split,init,maps,clinical)
                final={**clinical,**distill}
                for label,r in final.items():_save_selected(c,cell,label,r)
                # This is the ORIGINAL validation fold, also used for selection.
                # Its reported scores are descriptive development results, not unbiased test scores.
                heldout=[r for r in data['images'] if r['sample_id'] in split['validation_ids']]
                for label,model in {'T':T,**final}.items():
                    _evaluate(c,cell,label,model,heldout,data['surfaces'],'development_validation')
                cell_receipt=out/'cells'/f'{cell}.json'
                if not cell_receipt.exists():
                    event(out,'cell_complete',cell=cell)
                    from .export import result_archive
                    result_archive(c)
                    atomic_json(cell_receipt,dict(complete=True,protocol=protocol))
                    write_progress(c)
        locked={str(p.relative_to(out)):file_hash(p) for root in ('selections','thresholds','hpo_selections') for p in (out/root).glob('**/*.json')}
        existing=out/'DEVELOPMENT_FROZEN.json'
        payload=dict(protocol_signature=protocol,artifacts=locked,reference25_used_for_hpo=False)
        if existing.exists() and read_json(existing)!=payload:raise RuntimeError('Development freeze changed.')
        atomic_json(existing,payload)
        if c['reference25']['evaluate_after_all_development']:
            ref=prepare(c,reference=True)
            for split in data['splits']:
                for seed in c['seeds']:
                    cell=f'seed_{seed}_fold_{split["fold"]}'
                    for label in ('T','A','B','C','D'):
                        check_pause(out);r=_get_selected(c,cell,label)
                        _evaluate(c,cell,label,r,ref['images'],ref['surfaces'],'reference25')
        event(out,'final_reporting')
        from .report import finish
        finish(c)
        from .export import model_archive,result_archive
        if c['reporting']['export_model_zip']:model_archive(c)
        atomic_json(out/'COMPLETE.json',dict(time_utc=utc(),counts=counts(c),protocol_signature=protocol))
        result_archive(c)
        from .compare_runs import compare_available
        compare_available(c)
        write_progress(c)
        event(out,'complete',results=str(out/'exports/RESULTS.zip'),models=str(out/'exports/MODELS.zip'))
