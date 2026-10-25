"""Read-only process/heartbeat/progress monitor. Never imports a GPU framework."""
from __future__ import annotations
import json,os,subprocess,time
from pathlib import Path
from .io import read_json
from .config import STAGES,DEPENDENCIES,cells,GROUPS,counts
from .processes import alive

def read_if(path,default=None):
    try:return read_json(path)
    except (FileNotFoundError,json.JSONDecodeError):return default

def fraction(c,cell,stage):
    folder=Path(c['output_dir'])/'runs'/cell/stage
    if (folder/'result.json').is_file():return 1.
    cur=read_if(folder/'current.json',{})
    if not cur:return 0.
    epoch=max(0,int(cur.get('epoch',0)))
    if 'batch' in cur:epoch=max(0,epoch-1)+cur['batch']/max(1,cur['batches'])
    return min(.99,epoch/max(1,cur.get('max_epochs',1)))

def snapshot(c):
    out=Path(c['output_dir']);supervisor=read_if(out/'supervisor.json',{})
    heartbeat=read_if(out/'heartbeat.json',{})
    controller=read_if(out/'controller.json',{})
    worker=read_if(out/'worker.json',{})
    current=read_if(out/'current.json',{})
    failure=read_if(out/'FAILED.json');paused=(out/'PAUSE').exists()
    supervisor_alive=alive(supervisor);controller_alive=alive(controller)
    worker_alive=False
    if worker:
        worker_alive=alive(worker)
    heartbeat_age=time.time()-heartbeat.get('time',0) if heartbeat else None
    current_path=out/'current.json'
    progress_age=time.time()-current_path.stat().st_mtime if current_path.is_file() else None
    if (out/'COMPLETE.json').is_file():status='COMPLETE'
    elif failure:status='FAILED'
    elif paused and not controller_alive:status='PAUSED'
    elif paused:status='PAUSE_REQUESTED'
    elif supervisor_alive and controller_alive:
        status='RUNNING' if progress_age is None or progress_age<c['runtime']['supervisor_no_progress_timeout_seconds'] else 'ACTIVE_WITHOUT_RECENT_PROGRESS'
    elif supervisor_alive:status='STARTING' if heartbeat_age is not None and heartbeat_age<15 else 'SUPERVISOR_WITHOUT_CONTROLLER'
    elif (out/'protocol.json').exists():status='STOPPED'
    else:status='NOT_STARTED'
    values=[];group=[]
    for label,deps in DEPENDENCIES.items():
        fractions=[];final=dev=ref=0
        for seed,fold in cells(c):
            cell=f'seed_{seed}_fold_{fold}'
            fractions += [fraction(c,cell,s) for s in deps]
            final+=int((out/'selections'/cell/f'{label}.json').is_file())
            dev+=int((out/'evaluations/development_validation'/cell/label/'DONE.json').is_file())
            ref+=int((out/'evaluations/reference25'/cell/label/'DONE.json').is_file())
        group.append(dict(strategy=label,fraction=sum(fractions)/len(fractions),final=final,development=dev,reference=ref))
    for seed,fold in cells(c):
        values += [fraction(c,f'seed_{seed}_fold_{fold}',s) for s in STAGES if c['hpo_enabled'] or s!='V']
    if not c['hpo_enabled']:
        values += [max(fraction(c,f'seed_{seed}_fold_{fold}','V') for fold in range(c['folds'])) for seed in c['seeds']]
    return dict(status=status,supervisor_alive=supervisor_alive,controller_alive=controller_alive,worker_alive=worker_alive,
        heartbeat_age_seconds=heartbeat_age,progress_age_seconds=progress_age,current=current,worker=worker,
        groups=group,training_fraction=sum(values)/len(values),failure=failure,
        teachers_verified=len(read_if(out/'audit/TEACHERS.json',[])),
        teacher_train_targets=sum((out/'targets'/f'seed_{s}_fold_{f}'/'training_only/result.json').is_file() for s,f in cells(c)))

def bar(fraction):
    n=round(12*fraction);return '█'*n+'░'*(12-n)+f' {100*fraction:5.1f}%'

def format_text(c,state,view='all'):
    out=Path(c['output_dir']);n=len(cells(c))
    lines=['ACTA | YOLO PROFESOR → nnU-Net ALUMNO',
        f'Estado: {state["status"]} | HPO: {c["hpo_enabled"] or "desactivado"}',
        f'Supervisor vivo: {state["supervisor_alive"]} | Controlador vivo: {state["controller_alive"]} | Worker vivo: {state["worker_alive"]}',
        'Último latido: '+(f'{state["heartbeat_age_seconds"]:.0f}s' if state['heartbeat_age_seconds'] is not None else 'sin registro')+
        ' | Último progreso: '+(f'{state["progress_age_seconds"]:.0f}s' if state['progress_age_seconds'] is not None else 'sin registro'),
        'Etapa actual: '+str(state['current'].get('event','sin registro'))+
        ' | '+str(state['current'].get('task_id',''))]
    if state['failure']:lines+=['ERROR: '+str(state['failure'].get('error','ver FAILED.json'))]
    if view in ('all','training','assessment'):
        lines+=['',f'Profesores ACTA verificados: {state["teachers_verified"]}/{n} | Targets clínicos: {state["teacher_train_targets"]}/{n}',
            'Trabajo de entrenamiento: '+bar(state['training_fraction']),
            'Reutilización auditada: '+str(out/'audit/REUSE.jsonl'),
            'Etapas previstas sin duplicación: '+str(counts(c)['training_stages_total']),
            'Grupo   Progreso de etapas     Finales   Eval DCBR   Eval Reference25']
        for row in state['groups']:
            lines.append(f'{row["strategy"]:<7} {bar(row["fraction"])}    {row["final"]:2}/{n}     {row["development"]:2}/{n}        {row["reference"]:2}/{n}')
        td=sum((out/'evaluations/development_validation'/f'seed_{s}_fold_{f}'/'T/DONE.json').is_file() for s,f in cells(c))
        tr=sum((out/'evaluations/reference25'/f'seed_{s}_fold_{f}'/'T/DONE.json').is_file() for s,f in cells(c))
        lines+= [f'T YOLO: profesor reutilizado; eval DCBR {td}/{n}; Reference25 {tr}/{n}',
                  'Los porcentajes representan trabajo planificado, no tiempo restante. Etapas compartidas contadas una vez.']
    if view in ('all','hpo'):
        total_candidates=sum(len(__import__('reversekd.hpo',fromlist=['candidates']).candidates(c,b)) for b in c['hpo_enabled'])*n
        completed=len(list((out/'hpo').glob('*/trials/*/*.json')))
        final_hpo=len(list((out/'hpo').glob('*/COMPLETE.json')))
        lines+=['',f'HPO: candidatos {completed}/{total_candidates}; particiones terminadas {final_hpo}/{n if c["hpo_enabled"] else 0}',
            'HPO está separado del progreso de los ajustes finales; fits idénticos se reutilizan.']
        choices=out/'reports/hpo_choices.csv'
        if choices.is_file() and choices.stat().st_size:lines+=['Selecciones HPO: '+str(choices)]
    if view in ('all','metrics'):
        report=out/'reports/MONITOR.txt'
        if report.is_file():
            text=report.read_text();start=text.find('DEVELOPMENT_VALIDATION | FIXED')
            lines+=['',text[start:] if start>=0 else text]
        else:lines+=['','Todavía no hay modelos evaluados.']
    if view=='all' and c['training']['device']!='cpu':
        try:
            r=subprocess.run(['nvidia-smi','-i',str(c['training']['device']),'--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=5)
            if r.returncode==0:lines+=['GPU: utilización, memoria usada/total, temperatura: '+r.stdout.strip()]
        except (OSError,subprocess.TimeoutExpired):pass
    if state['worker'].get('log'):lines+=['Log activo/último: '+state['worker']['log']]
    lines+=['Resultados: '+str(out/'RESULTS_TO_SHARE.zip'),
        'Monitor de solo lectura. Ctrl+C cierra el monitor y no detiene los experimentos.']
    return '\n'.join(lines)

def watch(c,continuous=False,interval=15,view='all'):
    try:
        while True:
            if continuous and os.isatty(1):print('\033[2J\033[H',end='')
            print(format_text(c,snapshot(c),view),flush=True)
            if not continuous:return
            time.sleep(interval)
    except KeyboardInterrupt:return
