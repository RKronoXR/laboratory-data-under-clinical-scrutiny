"""Identify live Linux processes without trusting a stale/reused PID."""
import os,signal,time
from pathlib import Path

def identity(pid):
    try:
        text=(Path('/proc')/str(pid)/'stat').read_text()
        fields=text[text.rfind(')')+2:].split()
        if fields[0]=='Z':return None
        return fields[19]
    except (FileNotFoundError,ProcessLookupError,PermissionError,IndexError):return None

def record(pid):return dict(pid=int(pid),process_start=identity(pid),recorded=time.time())

def alive(value):
    if not value or not value.get('pid'):return False
    actual=identity(value['pid'])
    return actual is not None and actual==value.get('process_start')

def terminate_group(proc):
    if proc.poll() is not None:return
    try:os.killpg(proc.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:proc.wait(timeout=15)
    except Exception:
        try:os.killpg(proc.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        proc.wait(timeout=15)
