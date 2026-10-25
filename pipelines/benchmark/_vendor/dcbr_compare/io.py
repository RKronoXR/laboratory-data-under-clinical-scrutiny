from __future__ import annotations
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import time
from pathlib import Path


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, Path): return str(value)
    if hasattr(value, 'item'): return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def canonical(value):
    return json.dumps(clean(value), sort_keys=True, ensure_ascii=False, allow_nan=False)


def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def atomic_bytes(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        with tmp.open('wb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_json(path, value): atomic_bytes(path, (canonical(value) + '\n').encode())
def read_json(path): return json.loads(Path(path).read_text())


def write_csv(path, rows):
    rows = [clean(r) for r in rows]
    if not rows:
        atomic_bytes(path, b''); return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    buf = io.StringIO(); w = csv.DictWriter(buf, fieldnames=keys)
    w.writeheader(); w.writerows(rows); atomic_bytes(path, buf.getvalue().encode())


def event(root, event_name, **fields):
    payload = {'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'event': event_name, **fields}
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    with (root / 'events.jsonl').open('a') as f:
        f.write(canonical(payload) + '\n'); f.flush()
    atomic_json(root / 'current.json', payload)
    print(canonical(payload), flush=True)


@contextlib.contextmanager
def exclusive_lock(path):
    import fcntl
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as f:
        try: fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Ya hay otra ejecución activa en esta carpeta.')
        f.seek(0); f.truncate(); f.write(str(os.getpid())); f.flush()
        try: yield
        finally: fcntl.flock(f, fcntl.LOCK_UN)
