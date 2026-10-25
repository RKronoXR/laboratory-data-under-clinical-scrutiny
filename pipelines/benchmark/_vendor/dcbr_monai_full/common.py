from pathlib import Path
import time, os
from dcbr_compare.io import atomic_json, atomic_bytes, read_json, file_hash, digest, write_csv
SEEDS=(17,42,2026)

def out(project): return Path(project)/'monai_full/output'
def read(path, default=None):
    return read_json(path) if Path(path).is_file() else ({} if default is None else default)
def state(project, **values):
    atomic_json(out(project)/'session.json',dict(pid=os.getpid(),updated_at=time.time(),**values))
def activity(project, **values):
    atomic_json(out(project)/'current.json',dict(updated_at=time.time(),**values))
def check_stop(project):
    if (out(project)/'STOP').exists(): raise KeyboardInterrupt('Pausa MONAI solicitada')
def folder(project,seed,fold): return out(project)/'runs'/f'seed_{seed}'/f'fold_{fold}'/'full'
def frames(project):
    import pandas as pd
    base=Path(project)/'output/data'
    kwargs=dict(dtype={'bitewing_id':str,'sample_id':str,'group_id':str,'patient_id':str},keep_default_na=False)
    return pd.read_csv(base/'bitewings.csv',**kwargs),pd.read_csv(base/'surfaces.csv',**kwargs)
def effective(project): return read_json(out(project)/'effective_config.json')
def mask_path(row):
    for key in ('reference_mask_path','crop_mask_path','mask_path'):
        if row.get(key): return Path(row[key])
    raise ValueError('Falta máscara proximal existente')
def frozen(project):
    rec=read_json(out(project)/'protocol.json')
    for path,expected in rec['source_hashes'].items():
        if file_hash(path)!=expected: raise RuntimeError('Cambió una entrada o código congelado: '+path)
    return rec
