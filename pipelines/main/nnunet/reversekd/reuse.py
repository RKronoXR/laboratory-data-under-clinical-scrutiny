"""Load the unchanged original DCBR partitions. Never make replacement folds."""
from pathlib import Path
import re
from .io import read_json

def canonical_id(value):
    m = re.fullmatch(r'(?:bw|dcbr_)?0*(\d+)', str(value))
    if not m:
        raise ValueError(f'Unrecognized original bitewing identifier: {value}')
    return f'dcbr_{int(m[1]):03d}'

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
        raise ValueError('The original number of folds differs from the protocol.')
    result = []
    for f, s in enumerate(splits):
        tr = [canonical_id(x) for x in s['train']]
        va = [canonical_id(x) for x in s['val']]
        if len(tr)!=len(set(tr)) or len(va)!=len(set(va)) or set(tr)&set(va):
            raise ValueError(f'Duplicated or overlapping original fold {f}.')
        result.append(dict(fold=f, train_ids=tr, validation_ids=va))
    return result
