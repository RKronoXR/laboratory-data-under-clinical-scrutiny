"""Native nnU-Net v2 training and inference, isolated from previous experiments."""
from __future__ import annotations
import copy,io,os,shutil,time
from pathlib import Path
import numpy as np
from .io import atomic_json,atomic_bytes,read_json,file_hash,digest,event,check_pause
from .config import discover_plans
from .data import teacher_images, probability_to_grid
from .metrics import records_for_image,summarize


def setup_bundle(c,data):
    out=Path(c['output_dir']);base=out/'nnunet';receipt=base/'bundle.json'
    if receipt.exists():
        bundle=read_json(receipt)
        for name,expected in bundle['frozen_file_hashes'].items():
            if not Path(name).is_file() or file_hash(name)!=expected:raise RuntimeError(f'Frozen teacher plan/split metadata changed: {name}')
        return bundle
    path=discover_plans(c);plans=copy.deepcopy(read_json(path))
    conf=plans['configurations']['2d']
    if any(x not in ('ZScoreNormalization','NoNormalization') for x in conf['normalization_schemes']):
        raise ValueError('Teacher plan uses fitted/global normalization. A fold-specific normalization implementation is required; refusing leakage.')
    if plans.get('image_reader_writer')!='NaturalImage2DIO':raise ValueError('A NaturalImage2DIO DCBR plan is required.')
    if len(conf['normalization_schemes'])!=1:raise ValueError('Expected the existing single-channel DCBR teacher plan.')
    name='Dataset813_ACTAKDDCBR';plans['dataset_name']=name
    raw=base/'raw'/name;pre=base/'preprocessed'/name
    raw.mkdir(parents=True,exist_ok=True);pre.mkdir(parents=True,exist_ok=True)
    dataset={'channel_names':{'0':'gray'},'labels':{'background':0,'caries':1},'numTraining':len(data['images']),'file_ending':'.png'}
    for image in teacher_images(data['images']):
        sid=image['sample_id']
        for fld,key,suffix in [('imagesTr','image_path','_0000.png'),('labelsTr','mask_path','.png')]:
            p=raw/fld/(sid+suffix);p.parent.mkdir(parents=True,exist_ok=True)
            if not p.exists():shutil.copy2(image[key],p)
            if file_hash(p)!=file_hash(image[key]):raise RuntimeError('Native nnU-Net source copy mismatch.')
    atomic_json(raw/'dataset.json',dataset);atomic_json(pre/'dataset.json',dataset)
    atomic_json(pre/'nnUNetPlans.json',plans)
    atomic_json(pre/'splits_final.json',[{'train':s['train_ids'],'val':s['validation_ids']} for s in data['splits']])
    atomic_json(pre/'dataset_fingerprint.json',{'note':'Fixed previously specified 2D plan. No data-dependent replanning. Per-image normalization only.',
        'original_plan_sha256':file_hash(path)})
    b=dict(dataset_name=name,raw_root=str(raw.parent),preprocessed_root=str(pre.parent),raw_directory=str(raw),pre_directory=str(pre),
        plans_file=str(pre/'nnUNetPlans.json'),dataset_file=str(pre/'dataset.json'),original_plan=str(path),original_plan_sha256=file_hash(path),
        inherited_patch_size=conf['patch_size'],inherited_batch_size=conf['batch_size'])
    b['frozen_file_hashes']={str(q):file_hash(q) for q in (path,pre/'nnUNetPlans.json',pre/'dataset.json',pre/'splits_final.json')}
    atomic_json(receipt,b);return b


def environment(bundle,directory,c):
    os.environ.update(nnUNet_raw=bundle['raw_root'],nnUNet_preprocessed=bundle['preprocessed_root'],
        nnUNet_results=str(Path(directory)/'native_results'),nnUNet_n_proc_DA=str(c['training']['teacher_workers']),
        nnUNet_compile='true' if c['training']['teacher_compile'] else 'false')


def preprocess(bundle,c):
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    pm=PlansManager(bundle['plans_file']);conf=pm.get_configuration('2d');pp=conf.preprocessor_class(verbose=False)
    dataset=read_json(bundle['dataset_file']);raw=Path(bundle['raw_directory']);pre=Path(bundle['pre_directory'])/conf.data_identifier
    pre.mkdir(parents=True,exist_ok=True)
    gt=Path(bundle['pre_directory'])/'gt_segmentations';gt.mkdir(exist_ok=True)
    for image in sorted((raw/'imagesTr').glob('*_0000.png')):
        check_pause(c['output_dir']);sid=image.name[:-9];mask=raw/'labelsTr'/f'{sid}.png'
        receipt=pre/(sid+'.receipt.json');prefix=pre/sid
        source_sig=digest({'image':file_hash(image),'mask':file_hash(mask),'plans':file_hash(bundle['plans_file'])})
        if receipt.exists():
            rec=read_json(receipt)
            if rec['signature']!=source_sig or not all(Path(p).is_file() and file_hash(p)==h for p,h in rec['files'].items()):
                raise RuntimeError(f'Corrupted native preprocessing: {sid}')
        else:
            tmp=pre/(sid+'_building');pp.run_case_save(str(tmp),[str(image)],str(mask),pm,conf,dataset)
            generated=[]
            for suffix in ('.b2nd','_seg.b2nd','.pkl'):
                p=Path(str(prefix)+suffix);os.replace(str(tmp)+suffix,p);generated.append(p)
            atomic_json(receipt,dict(signature=source_sig,files={str(p):file_hash(p) for p in generated}))
        if not (gt/mask.name).exists():shutil.copy2(mask,gt/mask.name)


def effective_cfg(c,h):
    t=c['training']
    return dict(nnunet_max_epochs=t['teacher_epochs'],nnunet_patience=t['teacher_patience'],nnunet_min_delta=t['teacher_min_delta'],
        nnunet_initial_lr=h['lr'],nnunet_weight_decay=h['weight_decay'],nnunet_train_iterations=t['teacher_train_iterations'],
        nnunet_validation_iterations=t['teacher_validation_iterations'],nnunet_oversample_foreground=h['foreground_sampling'])


def trainer_for(spec,c,bundle):
    import torch
    from . import native_trainer
    native_trainer.configure({'spec':spec,'cfg':effective_cfg(c,spec['hyperparameters']),'stop_file':str(Path(c['output_dir'])/'PAUSE')})
    plans=copy.deepcopy(read_json(bundle['plans_file']));plans['continue_training']=True
    expected=Path(spec['directory'])/'native_results'/bundle['dataset_name']/'nnUNetTrainerACTAKD__nnUNetPlans__2d'/f'fold_{spec["fold"]}'
    plans['continue_training']=(expected/'checkpoint_latest.pth').exists() or (expected/'checkpoint_best.pth').exists()
    device=torch.device('cpu' if c['training']['device']=='cpu' else 'cuda')
    trainer=native_trainer.nnUNetTrainerACTAKD(plans,'2d',int(spec['fold']),read_json(bundle['dataset_file']),device)
    if Path(trainer.output_folder).resolve()!=expected.resolve():raise RuntimeError('nnU-Net output directory escaped isolation.')
    return trainer


def _predict(trainer,images,folder,c,save=True):
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.imageio.natural_image_reader_writer import NaturalImage2DIO
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    trainer.set_deep_supervision_enabled(False);trainer.network.eval()
    for p in trainer.network.parameters():p.requires_grad_(False)
    predictor=nnUNetPredictor(tile_step_size=.5,use_gaussian=True,use_mirroring=bool(c['training']['teacher_mirroring']),
        perform_everything_on_device=trainer.device.type=='cuda',device=trainer.device,verbose=False,verbose_preprocessing=False,allow_tqdm=False)
    # Exactly one network, not an ensemble. Parameter list is needed by the single-array inference API.
    predictor.manual_initialization(trainer.network,trainer.plans_manager,trainer.configuration_manager,
        [trainer.network.state_dict()],trainer.dataset_json,trainer.__class__.__name__,trainer.inference_allowed_mirroring_axes)
    reader=NaturalImage2DIO()
    for image in images:
        check_pause(c['output_dir']);p=folder/(image['sample_id']+'.npy')
        if p.exists():prob=np.load(p,allow_pickle=False)
        else:
            arr,props=reader.read_images([image['image_path']])
            with torch.inference_mode(): _,pr=predictor.predict_single_npy_array(arr,props,save_or_return_probabilities=True)
            pr=np.asarray(pr)
            if pr.ndim==4 and pr.shape[1]==1:pr=pr[:,0]
            if pr.ndim!=3 or pr.shape[0]!=2:raise RuntimeError(f'Expected two-class nnU-Net probabilities: {pr.shape}')
            prob=pr[1].astype(np.float32)
            if save:
                b=io.BytesIO();np.save(b,prob,allow_pickle=False);atomic_bytes(p,b.getvalue())
        if prob.shape!=(image['geometry']['size'],image['geometry']['size']) or not np.isfinite(prob).all() or prob.min()<0 or prob.max()>1:
            raise RuntimeError('Invalid nnU-Net probability grid or values.')
        yield image,prob,str(p)


def train(spec,c,data):
    import torch
    bundle=setup_bundle(c,data);environment(bundle,spec['directory'],c);preprocess(bundle,c)
    trainer=trainer_for(spec,c,bundle)
    modeldir=Path(trainer.output_folder);last=modeldir/'checkpoint_latest.pth';best=modeldir/'checkpoint_best.pth'
    done=Path(spec['directory'])/'TRAINED.json'
    if not done.exists():
        if last.exists():trainer.load_checkpoint(str(last))
        trainer.run_training()
        if not best.is_file():raise RuntimeError('Native teacher produced no best checkpoint.')
        atomic_json(done,{'best_epoch':trainer._pc_best_epoch,'epochs_completed':trainer.current_epoch})
    trainer.load_checkpoint(str(best));rows=[]
    val=teacher_images([i for i in data['images'] if i['sample_id'] in spec['validation_ids']])
    for image,prob,_ in _predict(trainer,val,Path(spec['directory'])/'validation_probabilities',c,save=False):
        rows.extend(records_for_image(image,prob,data['surfaces']))
    from .io import write_csv
    write_csv(Path(spec['directory'])/'validation_surfaces.csv',rows)
    result=dict(checkpoint=str(best),checkpoint_sha256=file_hash(best),metrics=summarize(rows),native_bundle=bundle,
        teacher_image_size=c['teacher_image_size'],
        native_model_dir=str(modeldir),spec=spec,**read_json(done))
    del trainer
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return result


def predict(result,images,folder,c):
    """Read-only single-checkpoint inference; never construct a training controller.

    Works for verified shared teachers as well as new native fits. No writes to
    the original model folder, no optimizer loading, and no teacher gradients.
    """
    import torch
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.imageio.natural_image_reader_writer import NaturalImage2DIO
    bundle=result['native_bundle']
    if file_hash(result['checkpoint'])!=result['checkpoint_sha256']:
        raise RuntimeError('Teacher checkpoint changed.')
    pm=PlansManager(read_json(bundle['plans_file']));conf=pm.get_configuration('2d')
    dataset=read_json(bundle['dataset_file']);device=torch.device('cpu' if c['training']['device']=='cpu' else 'cuda:0')
    payload=torch.load(result['checkpoint'],map_location='cpu',weights_only=False)
    state=payload['network_weights']
    state={k.removeprefix('_orig_mod.'):v for k,v in state.items()}
    network=get_network_from_plans(conf.network_arch_class_name,conf.network_arch_init_kwargs,
        conf.network_arch_init_kwargs_req_import,determine_num_input_channels(pm,conf,dataset),
        pm.get_label_manager(dataset).num_segmentation_heads,allow_init=False,deep_supervision=False)
    network.load_state_dict(state,strict=True);network.to(device).eval()
    for parameter in network.parameters():parameter.requires_grad_(False)
    predictor=nnUNetPredictor(tile_step_size=.5,use_gaussian=True,
        use_mirroring=bool(c['training']['teacher_mirroring']),perform_everything_on_device=device.type=='cuda',
        device=device,verbose=False,verbose_preprocessing=False,allow_tqdm=False)
    predictor.manual_initialization(network,pm,conf,[state],dataset,payload.get('trainer_name','nnUNetTrainerACTAKD'),
        payload.get('inference_allowed_mirroring_axes'))
    reader=NaturalImage2DIO();folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);paths={}
    for image in images:
        check_pause(c['output_dir']);p=folder/(image['sample_id']+'.npy')
        if p.exists():prob=np.load(p,allow_pickle=False)
        else:
            source_image=teacher_images([image])[0]
            arr,props=reader.read_images([source_image['image_path']])
            with torch.inference_mode():_,pr=predictor.predict_single_npy_array(arr,props,save_or_return_probabilities=True)
            pr=np.asarray(pr)
            if pr.ndim==4 and pr.shape[1]==1:pr=pr[:,0]
            if pr.ndim!=3 or pr.shape[0]!=2:raise RuntimeError(f'Invalid teacher probability shape: {pr.shape}')
            prob=pr[1].astype(np.float32)
            prob=probability_to_grid(prob,source_image['geometry'],image['geometry'])
            b=io.BytesIO();np.save(b,prob,allow_pickle=False);atomic_bytes(p,b.getvalue())
        size=image['geometry']['size']
        if prob.shape!=(size,size) or not np.isfinite(prob).all() or prob.min()<0 or prob.max()>1:
            raise RuntimeError('Invalid teacher probability values/grid. No silent axis or resolution change.')
        paths[image['sample_id']]=str(p)
    del predictor,network,payload,state
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return paths


def import_teacher(spec,c,data):
    """Evaluate a verified immutable checkpoint on the original validation fold."""
    from .io import write_csv
    source=spec['reuse_candidate']
    if file_hash(source['checkpoint'])!=source['checkpoint_sha256']:
        raise RuntimeError('Teacher selected for reuse changed after its audit.')
    # Keep a local immutable snapshot so old run results do not depend on a
    # future change to a shared cache or an external teacher directory.
    folder=Path(spec['directory']);folder.mkdir(parents=True,exist_ok=True)
    local={}
    for key,name in [('checkpoint','checkpoint_best.pth'),('plans_file','plans.json'),('dataset_file','dataset.json')]:
        src=Path(source[key]);dst=folder/name
        if dst.exists() and file_hash(dst)!=file_hash(src):
            raise RuntimeError('Existing imported teacher snapshot differs. No overwrite is permitted.')
        if not dst.exists():
            tmp=dst.with_suffix(dst.suffix+'.building');shutil.copy2(src,tmp);os.replace(tmp,dst)
        local[key]=str(dst)
    result=dict(checkpoint=local['checkpoint'],checkpoint_sha256=source['checkpoint_sha256'],
        native_bundle={'plans_file':local['plans_file'],'dataset_file':local['dataset_file']},
        spec=spec,teacher_image_size=source['teacher_image_size'],reused=True,reuse_source=source.get('source'),original_checkpoint=source['checkpoint'],best_epoch=None,epochs_completed=0)
    val=teacher_images([r for r in data['images'] if r['sample_id'] in spec['validation_ids']])
    maps=predict(result,val,Path(spec['directory'])/'validation_probabilities',c);rows=[]
    for image in val:rows.extend(records_for_image(image,np.load(maps[image['sample_id']],allow_pickle=False),data['surfaces']))
    write_csv(Path(spec['directory'])/'validation_surfaces.csv',rows)
    result['metrics']=summarize(rows)
    return result
