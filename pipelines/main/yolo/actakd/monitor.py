from __future__ import annotations
import os,subprocess,time
from pathlib import Path
from .io import read_json

def gpu():
    try:
        r=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw','--format=csv'],capture_output=True,text=True,timeout=8)
        return r.stdout.strip() or r.stderr.strip()
    except (FileNotFoundError,subprocess.TimeoutExpired):return 'GPU telemetry unavailable.'

def show(c):
    out=Path(c['output_dir']);lines=['ACTA KD STUDY v3: '+str(out),'HPO: '+(','.join(c.get('hpo_enabled',[])) or 'OFF'),'',gpu(),'']
    import shutil
    host=out if out.exists() else Path(c['package_root'])
    lines.append(f'Disk free: {shutil.disk_usage(host).free/1024**3:.1f} GiB')
    if hasattr(os,'getloadavg'):lines.append(f'CPU load (1/5/15 min): {os.getloadavg()}')
    mem=Path('/proc/meminfo')
    if mem.exists():
        vals={line.split(':')[0]:int(line.split()[1]) for line in mem.read_text().splitlines() if len(line.split())>=2 and line.split()[1].isdigit()}
        total=vals.get('MemTotal',0)/1024**2;avail=vals.get('MemAvailable',0)/1024**2
        lines.append(f'RAM: {total-avail:.1f}/{total:.1f} GiB')
    decisions=[read_json(p) for p in (out/'alerts/teachers').glob('*.json')]
    affected=[r for r in decisions if r.get('trigger')=='incompatibility']
    if affected:
        lines.append(f'!!! TEACHER INCOMPATIBILITY: {len(affected)} cell(s). See {out}/TEACHER_RETRAINING_ALERTS.txt')
        for r in affected:
            if r['status'] in ('retraining_running','paused','failed'):lines.append(f'  {r["cell"]}: {r["status"]} | '+ '; '.join(r.get('reasons',[])[:2]))
    if (out/'current.json').exists():
        v=read_json(out/'current.json');lines.append('State: '+str(v.get('event'))+' | '+str(v.get('time_utc')))
    if (out/'PAUSE').exists():lines.append('PAUSE REQUESTED. A running worker stops after a safe checkpoint boundary.')
    if (out/'worker.json').exists():
        w=read_json(out/'worker.json');lines.append('Current/latest worker: '+w['task_id'])
        p=Path(w['directory'])/'current.json'
        if p.exists():
            s=read_json(p);lines.append(f'Epoch {s.get("epoch","?")}/{s.get("max_epochs","?")} | best epoch {s.get("best_epoch","?")}')
        lines.append('Log: '+w['log'])
    from .progress import write_progress,format_text
    lines.extend(['',format_text(write_progress(c))])
    p=out/'reports/MONITOR.txt'
    lines.extend(['',p.read_text() if p.exists() else 'Preparing or waiting for the first completed training stage.'])
    return '\n'.join(lines)

def watch(c,repeat=False,interval=15):
    while True:
        if repeat and os.isatty(1):print('\033[2J\033[H',end='')
        print(show(c),flush=True)
        if not repeat:return
        try:time.sleep(interval)
        except KeyboardInterrupt:return
