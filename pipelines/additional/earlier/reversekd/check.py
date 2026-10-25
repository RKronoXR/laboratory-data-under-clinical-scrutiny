"""Non-interactive preflight. Fail visibly; never wait for a busy GPU or install."""
from __future__ import annotations
import importlib.util,importlib.metadata,os,shutil,subprocess
from pathlib import Path
from .io import atomic_json,read_json,file_hash,digest

def check_files(c):
    from .reuse import legacy_split_path
    for p in (Path(c['datasets_zip']),Path(c['student_plans']),legacy_split_path(c),
              Path(c['source_output'])/'data/bundle.json',Path(c['source_output'])/'data/READY.json'):
        if not p.is_file():raise FileNotFoundError(f'Required input missing: {p}')
    from .professor import descriptor_path
    for seed in c['seeds']:
        for fold in range(c['folds']):
            p=descriptor_path(c,seed,fold)
            if not p.is_file():raise FileNotFoundError(f'No ACTA-only YOLO selection for seed {seed}, fold {fold}: {p}')

def environment_check(c,cuda=True):
    check_files(c)
    for name,expected in (('ultralytics',c['ultralytics_version']),('nnunetv2',c['nnunet_version'])):
        got=importlib.metadata.version(name)
        if got!=expected:raise RuntimeError(f'{name} {got} installed; expected {expected}. No frameworks are upgraded automatically.')
    for name in ('torch','numpy','pandas','PIL','yaml','matplotlib','sklearn','scipy','dynamic_network_architectures'):
        if importlib.util.find_spec(name) is None:raise RuntimeError(f'Missing dependency: {name}. Use the project .venv.')
    if cuda and c['training']['device']!='cpu':
        import torch
        if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable; no automatic CPU fallback.')
    parent=Path(c['output_dir'])
    while not parent.exists():parent=parent.parent
    free=shutil.disk_usage(parent).free/1024**3
    if free<c['reporting']['min_free_gb']:raise RuntimeError(f'Insufficient free disk: {free:.1f} GiB.')
    return dict(free_gb=free,frameworks_verified=True)

def gpu_processes(c):
    import csv
    if c['training']['device']=='cpu':return []
    if not shutil.which('nvidia-smi'):raise RuntimeError('nvidia-smi unavailable; cannot verify the GPU is free.')
    command=['nvidia-smi','-i',str(c['training']['device']),'--query-compute-apps=pid,process_name','--format=csv,noheader,nounits']
    r=subprocess.run(command,capture_output=True,text=True,timeout=15)
    if r.returncode:raise RuntimeError('GPU process check failed: '+r.stderr.strip())
    pids=[]
    for row in csv.reader(r.stdout.splitlines()):
        if not row or not row[0].strip().isdigit():continue
        pid=int(row[0].strip())
        name=row[1].strip() if len(row)>1 else ''
        # This identified desktop service is not an experiment queue.
        if pid==os.getpid() or name=='/usr/libexec/gnome-remote-desktop-daemon':continue
        pids.append(pid)
    return pids

def require_idle_gpu(c):
    pids=gpu_processes(c)
    if pids:raise RuntimeError(f'GPU busy with compute PID(s) {pids}. The experiment was NOT queued or left waiting. Pause that job, then run START.sh again.')

def freeze_plans(c):
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    source=Path(c['student_plans']);plans=read_json(source);pm=PlansManager(plans)
    conf=pm.get_configuration('2d')
    if plans.get('image_reader_writer')!='NaturalImage2DIO':raise ValueError('A NaturalImage2DIO plan is required.')
    schemes=conf.normalization_schemes
    if len(schemes)!=1 or schemes[0] not in ('ZScoreNormalization','NoNormalization'):
        raise ValueError('Only the previous single-channel plan with per-image normalization is supported.')
    from functools import reduce
    strides=conf.network_arch_init_kwargs.get('strides')
    if not strides or any(len(x)!=2 for x in strides):raise ValueError('A 2D nnU-Net network topology is required.')
    divisor=[reduce(lambda a,b:a*b,(int(s[i]) for s in strides),1) for i in (0,1)]
    for name in ('full_image_size','crop_image_size'):
        if any(c[name]%x for x in divisor):raise ValueError(f'{name} is not divisible by inherited network pooling {divisor}.')
    dst=Path(c['output_dir'])/'audit/student_plans.json';dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists() and read_json(dst)!=plans:raise RuntimeError('Frozen nnU-Net plan changed.')
    atomic_json(dst,plans)
    architecture=dict(class_name=conf.network_arch_class_name,kwargs=conf.network_arch_init_kwargs,
        required_imports=conf.network_arch_init_kwargs_req_import,normalization=schemes,
        normalization_use_mask=list(conf.use_mask_for_norm),deep_supervision=c['training']['deep_supervision'],
        labels={'background':0,'caries':1},channels=1)
    c.update(frozen_plans=str(dst),normalization_scheme=schemes[0],
             normalization_use_mask=bool(conf.use_mask_for_norm[0]),batch_dice=bool(conf.batch_dice),
             architecture_signature=digest(architecture),pooling_divisor=divisor)
    atomic_json(dst.parent/'STUDENT_ARCHITECTURE.json',architecture)
    return c
