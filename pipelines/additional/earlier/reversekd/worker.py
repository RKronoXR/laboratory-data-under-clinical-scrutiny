"""A fresh GPU process for each task, with explicit completion/error receipts."""
import os,traceback
from pathlib import Path
from .io import read_json,atomic_json,file_hash,event,PauseRequested,check_pause

def execute(path):
    spec=read_json(path);c=spec['config'];folder=Path(spec['directory'])
    os.environ.update(CUDA_VISIBLE_DEVICES='' if c['training']['device']=='cpu' else str(c['training']['device']),
        YOLO_CONFIG_DIR=str(Path(c['output_dir'])/'runtime/ultralytics'),YOLO_AUTOINSTALL='false',
        nnUNet_compile='false',WANDB_MODE='disabled',WANDB_DISABLED='true',YOLO_OFFLINE='true',
        OMP_NUM_THREADS=str(c['training']['cpu_threads']),MKL_NUM_THREADS=str(c['training']['cpu_threads']))
    try:
        check_pause(c['output_dir']);event(c['output_dir'],'task_start',task_id=spec['task_id'],operation=spec['operation'])
        data=read_json(Path(c['output_dir'])/'data/bundle.json')
        if spec['operation']=='train':
            from .student import train
            result=train(spec,c,data)
        elif spec['operation']=='targets':
            from .professor import targets
            maps=targets(c,spec['teacher'],spec['images'],spec['surfaces'],folder/'maps')
            result=dict(maps=maps)
        elif spec['operation']=='predict':
            from .student import predict
            maps=predict(c,spec['model'],spec['images'],folder/'maps',spec['task_id'])
            result=dict(maps=maps,map_hashes={k:file_hash(p) for k,p in maps.items()})
        elif spec['operation']=='preflight':
            from .preflight import preflight
            result=preflight(c,data,spec['teacher'],spec['teacher_maps'],folder)
        else:raise ValueError('Unknown task operation.')
        if result.get('signature',spec['signature'])!=spec['signature']:
            raise RuntimeError('Worker result signature differs from the requested task.')
        atomic_json(folder/'result.json',{**result,'signature':spec['signature']})
        (folder/'ERROR.json').unlink(missing_ok=True)
        event(c['output_dir'],'task_complete',task_id=spec['task_id']);return 0
    except (PauseRequested,KeyboardInterrupt):
        event(c['output_dir'],'paused',task_id=spec['task_id']);return 75
    except Exception as exc:
        atomic_json(folder/'ERROR.json',dict(error=str(exc),traceback=traceback.format_exc()))
        event(c['output_dir'],'task_failed',task_id=spec['task_id'],error=str(exc));traceback.print_exc();return 1
