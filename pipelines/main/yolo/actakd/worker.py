"""One GPU task per isolated subprocess. No framework is imported by the monitor."""
from pathlib import Path
import os,traceback
from .io import read_json,atomic_json,file_hash,event,PauseRequested,check_pause

def execute(path):
    spec=read_json(path);c=spec['config'];out=Path(c['output_dir']);folder=Path(spec['directory'])
    os.environ['CUDA_VISIBLE_DEVICES']='' if c['training']['device']=='cpu' else str(c['training']['device'])
    os.environ['YOLO_CONFIG_DIR']=str(out/'runtime/ultralytics');os.environ['YOLO_AUTOINSTALL']='false'
    os.environ['nnUNet_compile']='false'
    os.environ.update(WANDB_MODE='disabled',WANDB_DISABLED='true',YOLO_OFFLINE='true')
    try:
        check_pause(out)
        from . import student,teacher
        data=read_json(out/'data/bundle.json')
        if spec['operation']=='train':
            event(out,'task_start',task_id=spec['task_id'],strategy=spec['strategy'],kind=spec['kind'],seed=spec['seed'],fold=spec['fold'])
            result=(teacher if spec['kind']=='teacher' else student).train(spec,c,data)
        elif spec['operation']=='import_teacher':
            result=teacher.import_teacher(spec,c,data)
        elif spec['operation']=='predict':
            model=read_json(spec['source_result']);images=spec['images']
            result=dict(maps=(teacher if model['spec']['kind']=='teacher' else student).predict(model,images,folder/'probabilities',c))
            result['map_hashes']={k:file_hash(p) for k,p in result['maps'].items()}
        else:raise ValueError('Unknown worker operation.')
        atomic_json(folder/'result.json',dict(signature=spec['signature'],**result))
        event(out,'task_complete',task_id=spec['task_id'],strategy=spec.get('strategy'),kind=spec.get('kind'),seed=spec.get('seed'),fold=spec.get('fold'))
        return 0
    except (PauseRequested,KeyboardInterrupt):
        event(out,'paused',task_id=spec['task_id']);return 75
    except Exception as exc:
        atomic_json(folder/'ERROR.json',dict(error=str(exc),traceback=traceback.format_exc()))
        event(out,'task_failed',task_id=spec['task_id'],error=str(exc));traceback.print_exc();return 1
