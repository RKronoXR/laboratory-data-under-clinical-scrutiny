"""Atomic files, fingerprints, and safe process coordination."""
from __future__ import annotations
import contextlib, csv, hashlib, io, json, math, os, time
from pathlib import Path

class PauseRequested(Exception):
    pass

def clean(v):
    if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (tuple,list)): return [clean(x) for x in v]
    if isinstance(v, Path): return str(v)
    if hasattr(v, 'item'): return clean(v.item())
    if isinstance(v, float) and not math.isfinite(v): return None
    return v

def canonical(v): return json.dumps(clean(v), sort_keys=True, ensure_ascii=False, allow_nan=False)
def digest(v): return hashlib.sha256(canonical(v).encode()).hexdigest()
def file_hash(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def atomic_bytes(p, value):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+f'.{os.getpid()}.tmp')
    try:
        with tmp.open('wb') as f: f.write(value); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,p)
    finally: tmp.unlink(missing_ok=True)

def atomic_json(p,v): atomic_bytes(p,(canonical(v)+'\n').encode())
def read_json(p): return json.loads(Path(p).read_text())
def write_csv(p,rows):
    rows=[clean(r) for r in rows]
    if not rows: atomic_bytes(p,b''); return
    cols=list(dict.fromkeys(k for row in rows for k in row))
    s=io.StringIO(); w=csv.DictWriter(s,fieldnames=cols); w.writeheader(); w.writerows(rows)
    atomic_bytes(p,s.getvalue().encode())
def read_csv(p):
    with Path(p).open(newline='') as f: return list(csv.DictReader(f))
def utc(): return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def event(root,name,**fields):
    p=Path(root); p.mkdir(parents=True,exist_ok=True)
    v=dict(time_utc=utc(),event=name,**fields)
    with (p/'events.jsonl').open('a') as f: f.write(canonical(v)+'\n'); f.flush()
    atomic_json(p/'current.json',v)
    print(canonical(v),flush=True)
def check_pause(root):
    if (Path(root)/'PAUSE').exists(): raise PauseRequested('Pause requested; last completed epoch is preserved.')
@contextlib.contextmanager
def exclusive_lock(p):
    import fcntl
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a+') as f:
        try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e: raise RuntimeError(f'Another process holds {p}') from e
        f.seek(0); f.truncate(); f.write(str(os.getpid())); f.flush()
        try: yield
        finally: fcntl.flock(f,fcntl.LOCK_UN)
