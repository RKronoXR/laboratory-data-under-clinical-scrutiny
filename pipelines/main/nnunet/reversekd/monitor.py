"""Read-only process/heartbeat/progress monitor. Never imports a GPU framework."""
from __future__ import annotations
import json,os,subprocess,time
from pathlib import Path
from .io import read_json
from .config import STAGES,DEPENDENCIES,cells,GROUPS,counts
from .processes import alive

REUSED_GROUPS=('A','T')
REUSE_COMPLETE='REUTILIZADO · COMPLETADO'

def read_if(path,default=None):
    try:return read_json(path)
    except (FileNotFoundError,json.JSONDecodeError):return default

def source_receipt(c,cell,label):
    """Read the explicit audit receipt, never infer a completed copy from a path."""
    value=read_if(Path(c['output_dir'])/'reuse'/cell/(label+'.json'),{})
    return value if isinstance(value,dict) else {}

def source_checkpoint_ready(receipt):
    return (receipt.get('training_reused') is True and bool(receipt.get('checkpoint_sha256'))
            and receipt.get('status')!='failed')

def source_scope_ready(c,cell,label,scope):
    receipt=source_receipt(c,cell,label)
    flag='reference_copied' if scope=='reference25' else 'development_copied'
    path=Path(c['output_dir'])/'evaluations'/scope/cell/label/'DONE.json'
    source_hash=receipt.get('source_result_sha256_by_scope',{}).get(scope) or receipt.get('source_result_sha256')
    done=read_if(path,{})
    consistent=not done.get('source_receipt_sha256') or done['source_receipt_sha256']==source_hash
    return receipt.get('status')!='failed' and receipt.get(flag) is True and bool(source_hash) and path.is_file() and consistent

def source_progress(c,label):
    """Training and copied evaluation have separate completion denominators."""
    rows=[(f'seed_{s}_fold_{f}',source_receipt(c,f'seed_{s}_fold_{f}',label)) for s,f in cells(c)]
    total=len(rows)
    checkpoints=sum(source_checkpoint_ready(r) for _,r in rows)
    development=sum(source_scope_ready(c,cell,label,'development_validation') for cell,_ in rows)
    reference=sum(source_scope_ready(c,cell,label,'reference25') for cell,_ in rows)
    failed=sum(r.get('status')=='failed' for _,r in rows)
    missing=sum(r.get('status')=='missing' or not r for _,r in rows)
    # A usable checkpoint alone cannot mark copied comparator results complete.
    status=REUSE_COMPLETE if total and development==total and not failed else ('BLOQUEADO · FUENTE FALLIDA' if failed else 'BLOQUEADO · RESULTADOS FUENTE FALTANTES')
    if checkpoints<total and not failed:status='BLOQUEADO · CHECKPOINT FUENTE FALTANTE'
    sources=sorted({(r.get('source_project','?'),r.get('source_group','?')) for _,r in rows if r})
    def short_project(project):
        path=Path(project)
        return (path.parent.parent.name+'/'+path.name if path.parent.name=='output' else path.name) if path.is_absolute() else project
    return dict(strategy=label,total=total,checkpoints=checkpoints,development=development,
                reference=reference,failed=failed,missing=missing,status=status,
                source='; '.join(short_project(project)+' / '+group for project,group in sources),new_epochs=0)

def reused_stage(stage):
    spec=STAGES.get(stage,{})
    return stage.startswith('REUSED_') or bool(spec.get('reused') or spec.get('reuse') or spec.get('source_reused'))

def training_stages():
    return [stage for stage in STAGES if not reused_stage(stage)]

def fraction(c,cell,stage):
    if reused_stage(stage):
        label=STAGES.get(stage,{}).get('final') or stage.removeprefix('REUSED_') or 'A'
        return float(source_checkpoint_ready(source_receipt(c,cell,label)))
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
    worker_alive=alive(worker) if worker else False
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
    for label in GROUPS:
        deps=DEPENDENCIES.get(label,())
        fractions=[];final=dev=ref=0
        for seed,fold in cells(c):
            cell=f'seed_{seed}_fold_{fold}'
            fractions += [fraction(c,cell,s) for s in deps if not reused_stage(s)]
            final+=int((out/'selections'/cell/f'{label}.json').is_file())
            if label in REUSED_GROUPS:
                dev+=int(source_scope_ready(c,cell,label,'development_validation'))
                ref+=int(source_scope_ready(c,cell,label,'reference25'))
            else:
                dev+=int((out/'evaluations/development_validation'/cell/label/'DONE.json').is_file())
                ref+=int((out/'evaluations/reference25'/cell/label/'DONE.json').is_file())
        reuse=source_progress(c,label) if label in REUSED_GROUPS else None
        average=(reuse['checkpoints']/max(1,reuse['total']) if reuse else sum(fractions)/max(1,len(fractions)))
        group.append(dict(strategy=label,fraction=average,final=final,development=dev,reference=ref,reuse=reuse))
    stages=training_stages()
    for seed,fold in cells(c):
        values += [fraction(c,f'seed_{seed}_fold_{fold}',s) for s in stages]
    return dict(status=status,supervisor_alive=supervisor_alive,controller_alive=controller_alive,worker_alive=worker_alive,
        heartbeat_age_seconds=heartbeat_age,progress_age_seconds=progress_age,current=current,worker=worker,
        groups=group,training_fraction=sum(values)/max(1,len(values)),failure=failure,
        namespace=Path(c['output_dir']).name,
        reuse={g:source_progress(c,g) for g in REUSED_GROUPS if g in GROUPS},
        teachers_verified=len(read_if(out/'audit/TEACHERS.json',[])),
        teacher_train_targets=sum((out/'targets'/f'seed_{s}_fold_{f}'/'training_only/result.json').is_file() for s,f in cells(c)))

def bar(fraction):
    fraction=min(1.,max(0.,fraction));n=round(12*fraction)
    return '█'*n+'░'*(12-n)+f' {100*fraction:5.1f}%'

def reuse_lines(c,state=None):
    state=snapshot(c) if state is None else state;n=len(cells(c))
    lines=['YA HECHO EN EL REPOSITORIO | reutilización auditada, 0 épocas nuevas y 0 entrenamiento GPU']
    for label in REUSED_GROUPS:
        r=state.get('reuse',{}).get(label) or source_progress(c,label)
        source=r['source'] or ('acta_kd_study_v3 / T' if label=='A' else 'acta_kd_study_v3 / P (YOLO ACTA)')
        lines.append(f'{label}: {r["status"]} | {source}')
        lines.append(f'   Pesos auditados {r["checkpoints"]}/{n} | resultados DCBR copiados {r["development"]}/{n} | RefStd copiado {r["reference"]}/{n}')
    locked=not (Path(c['output_dir'])/'DEVELOPMENT_FROZEN.json').is_file()
    lines.append('RefStd: BLOQUEADO hasta cerrar desarrollo.' if locked else 'RefStd: habilitado; las copias pendientes y evaluaciones se muestran por separado.')
    return lines

def format_text(c,state,view='all'):
    out=Path(c['output_dir']);n=len(cells(c))
    lines=['ACTA | YOLO PROFESOR → nnU-Net ALUMNO',
        'Corrida controlada: '+state.get('namespace',out.name),
        f'Estado: {state["status"]} | HPO: {c["hpo_enabled"] or "desactivado"}',
        f'Supervisor vivo: {state["supervisor_alive"]} | Controlador vivo: {state["controller_alive"]} | Worker vivo: {state["worker_alive"]}',
        'Último latido: '+(f'{state["heartbeat_age_seconds"]:.0f}s' if state['heartbeat_age_seconds'] is not None else 'sin registro')+
        ' | Último progreso: '+(f'{state["progress_age_seconds"]:.0f}s' if state['progress_age_seconds'] is not None else 'sin registro'),
        'Etapa actual: '+str(state['current'].get('event','sin registro'))+
        ' | '+str(state['current'].get('task_id',''))]
    if state['failure']:lines+=['ERROR: '+str(state['failure'].get('error','ver FAILED.json'))]
    lines+=['',*reuse_lines(c,state)]
    if view in ('all','training','assessment'):
        lines+=['',f'Profesores ACTA verificados: {state["teachers_verified"]}/{n} | Targets clínicos: {state["teacher_train_targets"]}/{n}',
            'ENTRENAMIENTO NUEVO: '+bar(state['training_fraction'])+' | no incluye copiar A/T ni evaluación del estudio',
            'Reutilización auditada: '+str(out/'reuse'),
            'Etapas nuevas previstas sin duplicación: '+str(counts(c)['training_stages_total']),
            'Grupo   Progreso de etapas     Finales   Eval DCBR   Eval Reference25']
        for r in state['groups']:
            tag=(' | '+r['reuse']['status']+'; 0 épocas nuevas') if r.get('reuse') else ''
            lines.append(f'{r["strategy"]:<7} {bar(r["fraction"])}    {r["final"]:2}/{n}     {r["development"]:2}/{n}        {r["reference"]:2}/{n}'+tag)
        lines+=['Los porcentajes representan trabajo planificado, no tiempo restante. Etapas compartidas contadas una vez.']
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
