"""Fixed settings, immutable run identity and a seven-stage dependency graph."""
from __future__ import annotations
import importlib.metadata
import re
import sys
from pathlib import Path
import yaml
from .io import atomic_json, digest, file_hash, read_json

LABELS = {'A':'DCBR full', 'B':'DCBR full -> ACTA',
          'C':'DCBR full + KD -> ACTA', 'D':'DCBR full + KD',
          'E':'ACTA -> DCBR full', 'F':'ACTA -> DCBR full + KD', 'T':'YOLO ACTA, evaluated on proximal crops'}
GROUPS = tuple(LABELS)
STAGES = {
    'N': {'domain':'dcbr', 'parent':None, 'kd':False, 'final':'A'},
    'K': {'domain':'dcbr', 'parent':None, 'kd':True, 'final':'D'},
    'V': {'domain':'acta', 'parent':None, 'kd':False, 'final':None},
    'B_ACTA': {'domain':'acta', 'parent':'N', 'kd':False, 'final':'B'},
    'C_ACTA': {'domain':'acta', 'parent':'K', 'kd':False, 'final':'C'},
    'E_CLIN': {'domain':'dcbr', 'parent':'V', 'kd':False, 'final':'E'},
    'F_CLIN': {'domain':'dcbr', 'parent':'V', 'kd':True, 'final':'F'},
}
DEPENDENCIES = {'A':['N'], 'B':['N','B_ACTA'], 'C':['K','C_ACTA'],
                'D':['K'], 'E':['V','E_CLIN'], 'F':['V','F_CLIN']}

def load(path, run_name=None, use_last=False, hpo=None):
    path = Path(path).resolve()
    c = yaml.safe_load(path.read_text())
    package = Path(__file__).resolve().parents[1]
    root = Path(c['project_root']).expanduser()
    root = (root if root.is_absolute() else package/root).resolve()
    base = Path(c.get('output_base','output'))
    base = (base if base.is_absolute() else package/base).resolve()
    if use_last and run_name is None and (base/'LAST_RUN.json').is_file():
        run_name = read_json(base/'LAST_RUN.json').get('run_name')
    if run_name and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}',run_name):
        raise ValueError('Invalid --run-name; use letters, numbers, dot, underscore or hyphen.')
    enabled = parse_hpo(hpo if hpo is not None else c.get('hpo',{}).get('enabled',[]))
    last_path=base/'LAST_RUN.json'
    if use_last and hpo is None and last_path.is_file():
        last=read_json(last_path)
        if last.get('run_name')==run_name: enabled=parse_hpo(last.get('hpo_enabled',enabled))
    profile='HPO_'+'_'.join(enabled).upper() if enabled else 'NO_HPO'
    namespace = profile+('__'+run_name if run_name else '')
    c.update(package_root=str(package), project_root=str(root), config_path=str(path),
             output_base=str(base), output_dir=str(base/namespace), run_name=run_name,
             profile=profile, hpo_enabled=enabled)
    for key in ('datasets_zip','source_v3','student_plans'):
        p=Path(c[key]).expanduser();c[key]=str((p if p.is_absolute() else root/p).resolve())
    c['source_output'] = str(Path(c['source_v3'])/'output'/c['source_profile'])
    if c['folds']!=5 or c['seeds']!=[17,42,2026]:
        raise ValueError('This package preserves the original three seeds and five folds.')
    if c['partition_policy']!='original_train_validation_folds':
        raise ValueError('Original folds are mandatory.')
    if c['training']['acta_final_selection'] not in ('clinical_validation','last'):
        raise ValueError('acta_final_selection must be clinical_validation or last.')
    for name in ('full_image_size','crop_image_size','teacher_image_size'):
        if int(c[name])<32 or c[name]%32:
            raise ValueError(f'{name} must be a positive multiple of 32.')
    if c['crop_image_size']!=c['teacher_image_size']:
        raise ValueError('ACTA student/teacher crops must preserve the same 384-pixel geometry.')
    if c['training']['optimizer']!='SGD':
        raise ValueError('This controlled nnU-Net protocol uses SGD for every student stage.')
    if c['hpo_enabled']:
        h=c['hpo']
        if not 0<h['inner_validation_fraction']<.5:raise ValueError('HPO holdout fraction must be between 0 and .5.')
        from .hpo import candidates
        for block in c['hpo_enabled']:
            space=candidates(c,block)
            if not space or any(not all(float(v)>0 for v in row.values()) for row in space):raise ValueError('HPO values must be positive.')
        if any(int(h[k])<1 for k in ('proxy_clinical_epochs','proxy_acta_finetune_epochs')):raise ValueError('HPO epoch budgets must be positive.')
    return c

HPO_BLOCKS=('pretrain','clinical','finetune','distill')

def parse_hpo(value):
    if isinstance(value,str):
        value=[] if value.lower() in ('off','none','') else list(HPO_BLOCKS) if value.lower()=='all' else value.split(',')
    value=[str(x).strip().lower() for x in value]
    if set(value)-set(HPO_BLOCKS):raise ValueError('HPO choices: off, all, pretrain,clinical,finetune,distill')
    return [x for x in HPO_BLOCKS if x in value]

def cells(c):
    # Finish a complete set of six strategies before moving to the next cell.
    return [(s,f) for f in range(c['folds']) for s in c['seeds']]

def counts(c):
    n=len(c['seeds'])*c['folds']
    return dict(seed_fold_cells=n, teacher_checkpoints_reused=n, new_teacher_fits=0,
                final_nnunet_models=6*n, final_teacher_comparators=n,
                shared_clinical_stages=2*n, acta_source_pretraining_stages=n if c['hpo_enabled'] else len(c['seeds']),
                acta_finetuning_stages=2*n, clinical_after_acta_stages=2*n,
                training_stages_total=len(STAGES)*n if c['hpo_enabled'] else 6*n+len(c['seeds']),
                logical_training_stages=len(STAGES)*n,
                development_evaluations=len(GROUPS)*n,
                reference25_evaluations=len(GROUPS)*n, hpo=bool(c['hpo_enabled']),
                hpo_blocks=c['hpo_enabled'], hpo_candidates_per_cell=sum(len(__import__('reversekd.hpo',fromlist=['candidates']).candidates(c,b)) for b in c['hpo_enabled']),
                hpo_max_stage_calls_per_cell=(sum(len(__import__('reversekd.hpo',fromlist=['candidates']).candidates(c,b)) for b in c['hpo_enabled'])+1)*len(STAGES) if c['hpo_enabled'] else 0,
                additional_fit_for_naming=0)

def freeze(c):
    out=Path(c['output_dir']);out.mkdir(parents=True,exist_ok=True)
    package=Path(c['package_root'])
    files=[*sorted((package/'reversekd').glob('*.py')), package/'run.py',package/'analysis/glmm.R']
    code={str(p.relative_to(package)):file_hash(p) for p in files}
    versions={}
    for name in ('torch','nnunetv2','ultralytics','numpy','Pillow','scikit-learn'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]='not installed'
    inputs={'plans':file_hash(c['student_plans']),
            'splits':file_hash(Path(c['project_root'])/c['teacher_reuse']['legacy_splits'])}
    value=dict(config={k:v for k,v in c.items() if k!='config_path'},code=code,
               versions=versions,python=sys.version,inputs=inputs,counts=counts(c))
    signature=digest(value);p=out/'protocol.json'
    if p.exists() and read_json(p)['signature']!=signature:
        from .cache import compatible_upgrade
        return compatible_upgrade(c,read_json(p),value)
    atomic_json(p,dict(signature=signature,**value))
    return signature
