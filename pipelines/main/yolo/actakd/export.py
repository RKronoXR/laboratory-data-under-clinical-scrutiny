"""Portable selected-model bundle and separate results-only ZIP."""
from __future__ import annotations
import io,os,shutil,tempfile,zipfile
from pathlib import Path
from .io import atomic_json,atomic_bytes,read_json,file_hash,write_csv,utc


def _zip_tree(source,destination):
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    tmp=destination.with_name(destination.name+'.building')
    try:
        with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
            for p in sorted(Path(source).rglob('*')):
                if p.is_file():z.write(p,str(p.relative_to(source)))
        os.replace(tmp,destination)
    finally:tmp.unlink(missing_ok=True)


def result_archive(c):
    out=Path(c['output_dir']);exports=out/'exports';exports.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.results_',dir=exports) as t:
        build=Path(t)
        for name in ('reports','thresholds','selections','hpo_selections','alerts','progress'):
            if (out/name).is_dir():shutil.copytree(out/name,build/name)
        for name in ('protocol.json','candidate_configurations.json','DEVELOPMENT_FROZEN.json','COMPLETE.json','TEACHER_RETRAINING_ALERTS.txt'):
            if (out/name).exists():shutil.copy2(out/name,build/name)
        for name in ('split_counts.csv','surfaces.csv','resolution_audit.csv','READY.json','ORIGINAL_PARTITION_AUDIT.json','original_splits_final.json'):
            p=out/'data'/name
            if p.exists():(build/'data_audit').mkdir(exist_ok=True);shutil.copy2(p,build/'data_audit'/name)
        shutil.copytree(Path(c['package_root'])/'docs',build/'documentation')
        shutil.copy2(Path(c['package_root'])/'README.md',build/'README.md')
        atomic_json(build/'EXPORT.json',dict(created_utc=utc(),complete=(out/'COMPLETE.json').exists(),
             includes='Results, graphs, model selections, threshold provenance and protocol; no images or model weights.'))
        _zip_tree(build,exports/'RESULTS.zip')


def model_archive(c):
    import torch
    out=Path(c['output_dir']);exports=out/'exports';exports.mkdir(exist_ok=True)
    selections=sorted((out/'selections').glob('*/*.json'))
    with tempfile.TemporaryDirectory(prefix='.models_',dir=exports) as t:
        build=Path(t);index=[]
        for source in selections:
            cell=source.parent.name;label=source.stem;r=read_json(source)
            ck=Path(r['checkpoint'])
            if not ck.is_file() or file_hash(ck)!=r['checkpoint_sha256']:raise RuntimeError(f'Selected model missing/changed: {source}')
            model_id=f'{label}_{cell}';folder=build/'models'/model_id;folder.mkdir(parents=True)
            payload=torch.load(ck,map_location='cpu',weights_only=False)
            teacher=label=='T'
            weights={'state_dict':payload['network_weights'] if teacher else payload['state_dict']}
            b=io.BytesIO();torch.save(weights,b);atomic_bytes(folder/'weights.pt',b.getvalue())
            descriptor=dict(model_id=model_id,strategy=label,seed=r['spec']['seed'],fold=r['spec']['fold'],
                architecture='nnUNetv2_2d' if teacher else 'yolo26_sem',student_model=c['student_model'],
                input_size=c['teacher_image_size'] if teacher else c['crop_image_size'] if label=='P' else c['full_image_size'],weights='weights.pt',
                weights_sha256=file_hash(folder/'weights.pt'),source_checkpoint_sha256=r['checkpoint_sha256'],
                hyperparameters=r['spec']['hyperparameters'],distillation_weight=r['spec'].get('distillation_weight',0),
                role='shared_pretraining_module' if label=='P' else 'teacher_comparator' if teacher else 'final_student',
                expected_ultralytics=c['ultralytics_version'],expected_nnunet=c['nnunet_version'])
            if teacher:
                shutil.copy2(r['native_bundle']['plans_file'],folder/'plans.json');shutil.copy2(r['native_bundle']['dataset_file'],folder/'dataset.json')
                descriptor.update(plans='plans.json',dataset='dataset.json',mirroring=c['training']['teacher_mirroring'],
                    inference_allowed_mirroring_axes=payload.get('inference_allowed_mirroring_axes'))
            threshold=out/'thresholds'/cell/f'{label}.json'
            if threshold.exists():
                th=read_json(threshold);descriptor['thresholds']={k:th[k] for k in ('fixed','youden','youden_endpoint')}
            atomic_json(folder/'model.json',descriptor);index.append({k:v for k,v in descriptor.items() if k not in ('hyperparameters','thresholds')})
        shutil.copytree(Path(c['package_root'])/'actakd',build/'actakd',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copy2(Path(c['package_root'])/'PREDICT.py',build/'PREDICT.py')
        write_csv(build/'MODEL_INDEX.csv',index)
        atomic_json(build/'software.json',dict(ultralytics=c['ultralytics_version'],nnunetv2=c['nnunet_version']))
        text='''SELECTED MODELS\n\nT and A/B/C/D are separately fitted models, not an ensemble. P denotes the ACTA pretraining module reused by B/D.\nEach model.json identifies the architecture, seed, fold, input size, weights hash and original-development-validation thresholds.\nThe bundle does not require teachers to run a student. No data or optimizer states are included.\n\nExample (from this extracted folder, using the Spark project Python):\n  python PREDICT.py --bundle . --model A_seed_17_fold_0 --image /path/bitewing.png --output predictions --threshold fixed\n\nUse --threshold youden to apply this model's frozen original-development-validation threshold.\nP modules have no clinical threshold and are not final clinical models.\nWeights are research outputs, not independently certified clinical software.\n'''
        atomic_bytes(build/'README_MODELS.txt',text.encode());_zip_tree(build,exports/'MODELS.zip')
