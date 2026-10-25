"""Frozen configuration, explicit per-component HPO and isolated run namespaces."""
from __future__ import annotations
import importlib.metadata
import re
import sys
from pathlib import Path
import yaml
from .io import digest, file_hash, atomic_json, read_json, utc

COMPONENTS = ('teacher', 'pretrain', 'clinical', 'distill')
RECOMMENDED = ('pretrain', 'clinical', 'distill')


def parse_hpo(value=None):
    if value is None or value == 'none':
        return []
    if value in ('recommended', 'moderate'):
        return list(RECOMMENDED)
    if value == 'all':
        return list(COMPONENTS)
    requested = str(value).lower().split(',')
    if any(not x or x not in COMPONENTS for x in requested) or len(set(requested)) != len(requested):
        raise ValueError('Use --hpo none, recommended, all, or a comma-separated subset of teacher,pretrain,clinical,distill.')
    return [x for x in COMPONENTS if x in requested]


def load(path, hpo=None, run_name=None, use_last=False):
    path = Path(path).resolve()
    c = yaml.safe_load(path.read_text())
    package = Path(__file__).resolve().parents[1]
    root = Path(c['project_root']).expanduser()
    if not root.is_absolute():
        root = (package / root).resolve()
    base = Path(c.get('output_base', 'output'))
    base = (base if base.is_absolute() else package / base).resolve()
    if use_last and hpo is None and run_name is None and (base/'LAST_RUN.json').exists():
        last = read_json(base/'LAST_RUN.json')
        hpo, run_name = last['hpo'], last.get('run_name')
    enabled = parse_hpo(hpo)
    if run_name and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', run_name):
        raise ValueError('--run-name must be 1-64 letters, numbers, dot, underscore or hyphen, without path separators.')
    profile = 'HPO_' + '_'.join(x.upper() for x in enabled) if enabled else 'NO_HPO'
    namespace = profile + ('__'+run_name if run_name else '')
    zp = Path(c['datasets_zip']).expanduser()
    c.update(project_root=str(root), output_dir=str(base/namespace), output_base=str(base),
             datasets_zip=str((zp if zp.is_absolute() else root/zp).resolve()),
             package_root=str(package), config_path=str(path), hpo_enabled=enabled,
             profile=profile, run_name=run_name, run_id=namespace)
    if c['folds'] < 3 or len(set(c['seeds'])) != len(c['seeds']):
        raise ValueError('Need at least 3 folds and unique seeds.')
    for k in ('full_image_size', 'crop_image_size', 'teacher_image_size'):
        if c[k] < 32 or c[k] % 32:
            raise ValueError(f'{k} must be a positive multiple of 32.')
    if c.get('partition_policy') != 'original_train_validation_folds':
        raise ValueError('V3 requires the original training/validation folds; no new internal split is supported.')
    if not c['teacher_reuse'].get('preserve_original_splits', False):
        raise ValueError('Preserving the original DCBR splits is mandatory.')
    if c['threshold']['youden_endpoint'] not in ('surface', 'overlap'):
        raise ValueError('Unknown Youden endpoint.')
    for key in ('teacher_trials','pretraining_trials','clinical_trials'):
        if int(c['hpo'][key]) < 1:
            raise ValueError(f'{key} must be at least 1.')
    if c['hpo']['objective'] != 'surface_auroc':
        raise ValueError('This prespecified protocol selects HPO with surface AUROC.')
    weights = c['hpo']['distillation_weights']
    if len(set(weights)) != len(weights) or any(x <= 0 for x in weights):
        raise ValueError('KD candidate weights must be unique and strictly positive.')
    if c['training']['distillation_weight'] not in weights:
        raise ValueError('The fixed KD anchor must be one of the KD search candidates, to permit exact reuse.')
    return c


def is_hpo(c, component):
    return component in c.get('hpo_enabled', [])


def counts(c):
    """Count actual scheduled fits, not checkpoint files. Selected fits are reused."""
    cells = len(c['seeds']) * c['folds']
    p = c['hpo']['pretraining_trials'] if is_hpo(c,'pretrain') else 1
    probes = p if is_hpo(c,'pretrain') else 0
    clinical = c['hpo']['clinical_trials'] if is_hpo(c,'clinical') else 1
    weights = len(c['hpo']['distillation_weights']) if is_hpo(c,'distill') else 1
    # C/D at the fixed KD anchor have already been fitted in the clinical stage.
    additional_kd = 2 * (weights - 1)
    yolos = cells * (p + probes + 4 * clinical + additional_kd)
    th = cells * c['hpo']['teacher_trials'] if is_hpo(c,'teacher') else 0
    fallback = 0 if is_hpo(c,'teacher') else cells
    return dict(profile=c.get('profile','test'), hpo_enabled=c.get('hpo_enabled',[]), seed_fold_cells=cells,
        teacher_configurations_per_cell=c['hpo']['teacher_trials'] if is_hpo(c,'teacher') else 1,
        teacher_search_fits_max=th, teacher_fallback_fits_max=fallback,
        teacher_fits_if_all_compatible=th, teacher_fits_if_none_compatible=th+fallback,
        pretraining_fits=p*cells, transfer_probe_fits=probes*cells,
        clinical_fits=4*clinical*cells, additional_distillation_fits=additional_kd*cells,
        student_training_stages_total=yolos, training_stages_total=yolos+th+fallback,
        training_stages_if_all_teachers_reusable=yolos+th,
        final_teacher_models=cells, final_student_models=4*cells, shared_pretraining_modules=cells,
        additional_final_refits=0, preflight_student_fits_max=2,
        estimated_hours_reuse=yolos*c['runtime_estimates']['yolo_minutes']/60 + th*c['runtime_estimates']['nnunet_minutes']/60,
        estimated_hours_no_reuse=yolos*c['runtime_estimates']['yolo_minutes']/60 + (th+fallback)*c['runtime_estimates']['nnunet_minutes']/60,
        note='Upper bounds before compatible caches. KD anchor fits and selected HPO checkpoints are reused, never refitted just for naming. Times are assumptions, not measured promises.')


def discover_plans(c):
    v = c['teacher_plans']; root = Path(c['project_root'])
    if v != 'AUTO':
        p = Path(v); candidates = [p if p.is_absolute() else root/p]
    else:
        preferred = root/'dcbr_full_vs_crops/nnunet/work/preprocessed/Dataset702_DCBRFullBW/nnUNetPlans.json'
        candidates = [preferred] if preferred.is_file() else sorted((root/'dcbr_full_vs_crops/nnunet').glob('**/plans.json'))
        if not candidates:
            candidates = sorted((root/'work/nnunet/preprocessed').glob('Dataset*_DCBR/nnUNetPlans.json'))
    valid = [(p,read_json(p)) for p in candidates if p.is_file() and '2d' in read_json(p).get('configurations',{})]
    if not valid:
        raise FileNotFoundError('No DCBR 2D nnU-Net plan found. Set teacher_plans to its full path.')
    if len({digest(j['configurations']['2d']) for _,j in valid}) != 1:
        raise ValueError('Multiple different 2D plans found. Set an explicit teacher_plans path.')
    return valid[0][0]


# Operational GPU-gate patch. Scientific code, dataset splits, weights,
# metrics and hyperparameters are untouched. This exact allowlist lets an
# in-flight V3 study retain its frozen protocol identity and checkpoint IDs.
_ORIGINAL_RUNTIME_HASHES = {
    'actakd/check.py':'bcba1d7f4076b03fdc5607a88ea9d4dc9eefe6efc2cac940664bb37fe635c4fc',
    'actakd/config.py':'bf2e6577091b5e979a14f7544348d1d60e20adc7f64793d1e4816129819d09d9',
}
_APPROVED_GATE_SHA256 = '33efdf0134a149457eb1c27b827731b0e28f287c9d157b901115494708cad9f8'


def _allowed_runtime_gate_patch(old, current):
    # Preserve all scientific settings and package hashes except the two
    # files implementing this audited runtime-only GPU policy.
    if set(old)-{'signature'} != set(current): return False
    if any(old.get(k) != current[k] for k in current if k!='code'):return False
    former=old.get('code',{}); latter=current.get('code',{})
    if set(former)!=set(latter):return False
    diffs={name for name in latter if former[name]!=latter[name]}
    if diffs != set(_ORIGINAL_RUNTIME_HASHES):return False
    if any(former.get(name)!=sha for name,sha in _ORIGINAL_RUNTIME_HASHES.items()):return False
    if latter['actakd/check.py']!=_APPROVED_GATE_SHA256:return False
    return True


def freeze(c):
    out = Path(c['output_dir']); out.mkdir(parents=True,exist_ok=True)
    code = {str(p.relative_to(c['package_root'])):file_hash(p) for p in sorted(Path(c['package_root']).glob('actakd/*.py'))}
    for extra in ('run.py','PREDICT.py','analysis/glmm.R'):
        code[extra] = file_hash(Path(c['package_root'])/extra)
    versions = {}
    for name in ('torch','ultralytics','nnunetv2','numpy','scipy','scikit-learn','Pillow'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = 'not installed'
    payload = dict(config={k:v for k,v in c.items() if k!='config_path'}, code=code, counts=counts(c), python=sys.version, versions=versions)
    sig = digest(payload); target = out/'protocol.json'
    if target.exists():
        old=read_json(target)
        if old['signature']==sig:
            return sig
        if _allowed_runtime_gate_patch(old,payload):
            audit=out/'GPU_GATE_RUNTIME_PATCH.json'
            if not audit.exists():
                atomic_json(audit,dict(original_protocol_signature=old['signature'],
                    recorded_at_utc=utc(),change='GPU admission gate only: ignore verified Linux GUI/display processes; continue to block real/unknown GPU processes',
                    old_hashes={k:old['code'][k] for k in _ORIGINAL_RUNTIME_HASHES},
                    new_hashes={k:code[k] for k in _ORIGINAL_RUNTIME_HASHES},
                    scientific_settings_changed=False,output_task_signatures_changed=False,
                    note='The original frozen protocol.json is deliberately preserved. Do not use this exception for any further code modifications.'))
            print('AUDITED RUNTIME PATCH: original scientific protocol identity preserved; GPU policy audit: '+str(audit),flush=True)
            return old['signature']
        raise RuntimeError('Frozen configuration/code/environment differs. Existing results are protected. Restore it or choose a NEW --run-name.')
    atomic_json(target, dict(signature=sig, **payload))
    return sig
