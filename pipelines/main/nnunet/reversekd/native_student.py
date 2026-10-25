"""Source-T native nnU-Net recipe for controlled Reverse continuations.

No custom full-image supervised training/inference loop is used. Reused A is
evaluated separately by source_baselines; this module trains later stages only.
Clinical recipe fields come from audited selected V3 T artifacts, never from
nominal new YAML defaults. Exact equality is gated on the target installation.
"""
from __future__ import annotations
import copy
import importlib.metadata
import io
import os
import random
from pathlib import Path
import numpy as np
from .io import atomic_bytes, atomic_json, check_pause, digest, event, file_hash, read_json, write_csv, PauseRequested
from .native_data import native_images, setup_bundle, preprocess, prepare_targets


def validate_recipe(recipe):
    required=('schema','framework','framework_version','input_size','configuration','plan_configuration',
              'initial_lr','weight_decay','epochs','patience','min_delta','train_iterations',
              'validation_iterations','foreground_sampling','workers','compile','optimizer',
              'momentum','nesterov','scheduler','selection','augmentation','inference')
    missing=[k for k in required if k not in recipe]
    if missing: raise RuntimeError('Incomplete audited source recipe: '+','.join(missing))
    if recipe['framework']!='nnunetv2' or recipe['framework_version']!='2.8.1':
        raise RuntimeError('Unsupported source framework/version.')
    if recipe['input_size']!=384 or recipe['configuration']!='2d':
        raise RuntimeError('Unsupported source native input/configuration.')
    if (recipe['optimizer'],float(recipe['momentum']),bool(recipe['nesterov']))!=('SGD',.99,True):
        raise RuntimeError('Original native optimizer differs; no substitute is allowed.')
    if recipe['scheduler']!={'name':'PolyLRScheduler','exponent':.9}:
        raise RuntimeError('Original native scheduler differs.')
    if recipe['selection']!='EMA_foreground_Dice' or recipe['augmentation']!='native_nnUNet_2.8.1':
        raise RuntimeError('Unsupported source loss/augmentation/checkpoint procedure.')
    inference=recipe['inference']
    if float(inference['tile_step_size'])!=.5 or not inference['use_gaussian']:
        raise RuntimeError('Original source inference differs.')
    if inference.get('compile',False):
        raise RuntimeError('V3 source-T evaluation uses the uncompiled read-only network.')
    if int(recipe['epochs'])!=250 or float(recipe['initial_lr'])!=.01:
        raise RuntimeError('This controlled protocol requires audited source epochs250 / lr0.01.')
    if not (0<float(recipe['foreground_sampling'])<1) or int(recipe['workers'])<0:
        raise RuntimeError('Invalid source native sampling/worker recipe.')
    return recipe


def recipe_signature(recipe):
    """Equivalent audited class names do not change effective native procedure."""
    ignored={'trainer_name','trainer_kind','trainer_sha256','trainer_source','provenance','plans_sha256','dataset_sha256',
             'source','recipe_signature','procedure_signature'}
    return digest({k:v for k,v in recipe.items() if k not in ignored})


def environment_identity():
    import inspect
    import sys
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
    from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
    packages=('torch','nnunetv2','batchgeneratorsv2','batchgenerators','dynamic-network-architectures',
              'numpy','scipy','scikit-image','blosc2')
    return dict(python=sys.version,versions={p:importlib.metadata.version(p) for p in packages},
        native_sources={name:file_hash(inspect.getfile(cls)) for name,cls in
                       [('trainer',nnUNetTrainer),('loader',nnUNetDataLoader),('spatial',SpatialTransform)]},
        backend_sources={name:file_hash(Path(__file__).with_name(name)) for name in
                         ('native_student.py','native_data.py','native_kd_trainer.py')})


def effective_cfg(spec,recipe):
    validate_recipe(recipe);h=spec['hyperparameters']
    clinical=spec['domain']=='dcbr'
    if clinical and (float(h['lr'])!=float(recipe['initial_lr']) or int(h['epochs'])!=int(recipe['epochs'])
            or float(h['weight_decay'])!=float(recipe['weight_decay'])):
        raise RuntimeError('Clinical stage changed the audited source-T supervised recipe.')
    if not clinical and int(h['epochs'])!=100:
        raise RuntimeError('The prespecified ACTA context uses 100 native epochs.')
    return dict(nnunet_max_epochs=int(h['epochs']),nnunet_patience=int(recipe['patience']),
        nnunet_min_delta=float(recipe['min_delta']),nnunet_initial_lr=float(h['lr']),
        nnunet_weight_decay=float(h['weight_decay']),nnunet_train_iterations=int(recipe['train_iterations']),
        nnunet_validation_iterations=int(recipe['validation_iterations']),
        nnunet_oversample_foreground=float(recipe['foreground_sampling']))


def _environment(bundle,spec,recipe):
    os.environ.update(nnUNet_raw=bundle['raw_root'],nnUNet_preprocessed=bundle['preprocessed_root'],
        nnUNet_results=str(Path(spec['directory'])/'native_results'),
        nnUNet_n_proc_DA=str(recipe['workers']),nnUNet_compile='true' if recipe['compile'] else 'false')


def _seed(seed):
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def load_initialization(trainer,initialization,recipe):
    """Transfer network weights only. New optimizer/scheduler/EMA/epoch remain0."""
    import torch
    if file_hash(initialization['checkpoint'])!=initialization['checkpoint_sha256']:
        raise RuntimeError('Transferred selected checkpoint changed.')
    previous=initialization.get('native_recipe')
    if previous is not None and recipe_signature(previous)!=recipe_signature(recipe):
        raise RuntimeError('Transferred checkpoint native recipe differs from the selected source T.')
    if 'native_bundle' not in initialization:
        raise RuntimeError('Transfer requires the immutable source native plans/dataset snapshot.')
    from .native_data import validate_native_plan
    previous_bundle=initialization['native_bundle']
    validate_native_plan(read_json(previous_bundle['plans_file']),read_json(previous_bundle['dataset_file']),recipe)
    checkpoint=torch.load(initialization['checkpoint'],map_location='cpu',weights_only=False)
    if 'network_weights' not in checkpoint:raise RuntimeError('Expected a native nnU-Net checkpoint.')
    state={k.removeprefix('_orig_mod.'):v for k,v in checkpoint['network_weights'].items()}
    trainer.initialize()
    network=getattr(trainer.network,'_orig_mod',trainer.network)
    network.load_state_dict(state,strict=True)
    if trainer.current_epoch!=0 or trainer.optimizer.state or trainer._pc_best_epoch!=0 or trainer._pc_stale!=0:
        raise RuntimeError('Transfer must reset optimizer, schedule, EMA and epoch state.')


def trainer_for(spec,c,bundle,recipe,targets=None):
    import torch
    _environment(bundle,spec,recipe)
    from . import native_kd_trainer
    if importlib.metadata.version('nnunetv2')!='2.8.1':raise RuntimeError('nnUNet 2.8.1 is mandatory.')
    if str(c['training']['device'])!='cpu' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no silent CPU training.')
    spec=copy.deepcopy(spec);spec.setdefault('run_root',c['output_dir'])
    native_kd_trainer.configure(dict(spec=spec,cfg=effective_cfg(spec,recipe),recipe_signature=recipe_signature(recipe),
        stop_file=str(Path(c['output_dir'])/'PAUSE'),kd_targets=targets or {},
        selection='last' if spec['selection']=='last' else 'EMA_foreground_Dice'))
    _environment(bundle,spec,recipe)
    plans=copy.deepcopy(read_json(bundle['plans_file']))
    expected=Path(spec['directory'])/'native_results'/bundle['dataset_name']/'nnUNetTrainerReverseControlled__nnUNetPlans__2d'/f'fold_{spec["fold"]}'
    plans['continue_training']=(expected/'checkpoint_latest.pth').exists() or (expected/'checkpoint_best.pth').exists()
    device=torch.device('cpu' if c['training']['device']=='cpu' else 'cuda')
    trainer=native_kd_trainer.nnUNetTrainerReverseControlled(plans,'2d',int(spec['fold']),read_json(bundle['dataset_file']),device)
    if Path(trainer.output_folder).resolve()!=expected.resolve():raise RuntimeError('Native output escaped stage isolation.')
    return trainer


def _native_spec(spec,data):
    result=copy.deepcopy(spec)
    result['native_train_ids']=list(spec['train_ids']) if spec['domain']=='dcbr' else [i['sample_id'] for i in data['acta']]
    result['native_validation_ids']=list(spec['validation_ids'])
    return result


def _source_bundle(spec,c):
    source=spec.get('source_native_bundle') or c.get('source_native_bundle')
    if source is None and spec.get('initialization'):source=spec['initialization'].get('native_bundle')
    if source is None:raise RuntimeError('Native source plans/dataset snapshot is required; no fallback plan discovery.')
    return source


def _require_gate(c,recipe,spec=None):
    spec=spec or {}
    path=Path(spec.get('native_gate_path') or c.get('native_gate_path') or Path(c['output_dir'])/'preflight'/'NATIVE_EQUIVALENCE.json')
    if not path.is_file():raise RuntimeError('Target native equivalence preflight is mandatory before fitting.')
    gate=read_json(path)
    if gate.get('recipe_signature')!=recipe_signature(recipe) or not gate.get('passed'):
        raise RuntimeError('Native equivalence gate does not cover this source recipe.')
    if gate.get('environment')!=environment_identity():
        raise RuntimeError('Native equivalence installation changed.')
    verify_archived_runtime(recipe,gate['environment'])


def verify_archived_runtime(recipe,identity):
    for name,expected in recipe.get('native_runtime_sources',{}).items():
        if identity['native_sources'].get(name)!=expected:
            raise RuntimeError('Installed native '+name+' source differs from the archived Spark implementation.')


def train(spec,c,data):
    import torch
    recipe=validate_recipe(spec.get('native_recipe') or c.get('native_recipe') or {})
    _require_gate(c,recipe,spec)
    effective_cfg(spec,recipe)
    if spec['kd'] and spec['domain']!='dcbr':raise RuntimeError('KD is prespecified only on clinical training images.')
    if not spec['kd'] and spec.get('teacher_maps'):raise RuntimeError('Non-KD stage received professor targets.')
    folder=Path(spec['directory']);folder.mkdir(parents=True,exist_ok=True)
    spec=_native_spec(spec,data);spec.setdefault('run_root',c['output_dir'])
    bundle=setup_bundle(c,data,recipe,_source_bundle(spec,c))
    _environment(bundle,spec,recipe);preprocess(bundle,c)
    targets=prepare_targets(spec,c,bundle) if spec['kd'] else {}
    _seed(int(spec['seed']))
    trainer=trainer_for(spec,c,bundle,recipe,targets)
    modeldir=Path(trainer.output_folder);latest=modeldir/'checkpoint_latest.pth';best=modeldir/'checkpoint_best.pth'
    marker=folder/'TRAINED.json'
    if marker.exists() and read_json(marker)['signature']!=spec['signature']:
        raise RuntimeError('Native stage completion identity changed.')
    if not marker.exists():
        if latest.exists():trainer.load_checkpoint(str(latest))
        elif spec.get('initialization'):load_initialization(trainer,spec['initialization'],recipe)
        try:
            trainer.run_training()
        except KeyboardInterrupt as exc:
            raise PauseRequested('Native stage paused at the last completed epoch.') from exc
        chosen=modeldir/'checkpoint_final.pth' if spec['selection']=='last' else best
        if not chosen.is_file():raise RuntimeError('Native stage did not produce the prespecified selected checkpoint.')
        atomic_json(marker,dict(signature=spec['signature'],best_epoch=trainer.current_epoch if spec['selection']=='last' else trainer._pc_best_epoch,
            epochs_completed=trainer.current_epoch,selection='last' if spec['selection']=='last' else 'EMA_foreground_Dice'))
    done=read_json(marker)
    chosen=modeldir/'checkpoint_final.pth' if spec['selection']=='last' else best
    if not latest.is_file():raise RuntimeError('Native stage must retain the resumable last checkpoint.')
    result=dict(signature=spec['signature'],checkpoint=str(chosen),checkpoint_sha256=file_hash(chosen),
        last_checkpoint=str(latest),last_checkpoint_sha256=file_hash(latest),native_bundle=bundle,native_recipe=recipe,
        spec=spec,architecture_signature=digest(recipe['plan_configuration']),optimizer_reset_at_stage_start=True,
        optimizer_steps=None,metrics=None,**{k:v for k,v in done.items() if k!='signature'})
    del trainer
    if torch.cuda.is_available():torch.cuda.empty_cache()
    if spec['stage']!='V':
        from .metrics import records_for_image,summarize
        validation=[i for i in data['images'] if i['sample_id'] in spec['validation_ids']]
        maps=predict(c,result,validation,folder/'validation_probabilities',spec['task_id'])
        rows=[]
        for image in native_images(validation):rows.extend(records_for_image(image,np.load(maps[image['sample_id']],allow_pickle=False),data['surfaces']))
        write_csv(folder/'validation_surfaces.csv',rows);result['metrics']=summarize(rows)
    return result


def predict(c,model,images,folder,task_id):
    """Read-only native sliding windows, Gaussian weighting and source mirrors."""
    import torch
    # The V3 worker explicitly disabled compilation for source-T read-only
    # inference. This is separate from the original teacher's training setting.
    os.environ['nnUNet_compile']='false'
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.imageio.natural_image_reader_writer import NaturalImage2DIO
    recipe=validate_recipe(model.get('native_recipe') or c.get('native_recipe') or {})
    if file_hash(model['checkpoint'])!=model['checkpoint_sha256']:raise RuntimeError('Selected native checkpoint changed.')
    bundle=model['native_bundle'];plans=read_json(bundle['plans_file']);dataset=read_json(bundle['dataset_file'])
    from .native_data import validate_native_plan
    validate_native_plan(plans,dataset,recipe)
    pm=PlansManager(plans);conf=pm.get_configuration('2d')
    payload=torch.load(model['checkpoint'],map_location='cpu',weights_only=False)
    state={k.removeprefix('_orig_mod.'):v for k,v in payload['network_weights'].items()}
    network=get_network_from_plans(conf.network_arch_class_name,conf.network_arch_init_kwargs,
        conf.network_arch_init_kwargs_req_import,determine_num_input_channels(pm,conf,dataset),
        pm.get_label_manager(dataset).num_segmentation_heads,allow_init=False,deep_supervision=False)
    device=torch.device('cpu' if c['training']['device']=='cpu' else 'cuda')
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA native inference unavailable.')
    network.load_state_dict(state,strict=True);network.to(device).eval()
    for p in network.parameters():p.requires_grad_(False)
    inference=recipe['inference']
    if payload.get('inference_allowed_mirroring_axes')!=inference.get('mirroring_axes'):
        # JSON stores tuples as lists, so compare normalized representations.
        if tuple(payload.get('inference_allowed_mirroring_axes') or ())!=tuple(inference.get('mirroring_axes') or ()):
            raise RuntimeError('Checkpoint mirroring axes differ from the source recipe.')
    predictor=nnUNetPredictor(tile_step_size=float(inference['tile_step_size']),use_gaussian=bool(inference['use_gaussian']),
        use_mirroring=bool(inference['use_mirroring']),perform_everything_on_device=device.type=='cuda',device=device,
        verbose=False,verbose_preprocessing=False,allow_tqdm=False)
    predictor.manual_initialization(network,pm,conf,[state],dataset,payload.get('trainer_name','nnUNetTrainerReverseControlled'),
        payload.get('inference_allowed_mirroring_axes'))
    reader=NaturalImage2DIO();folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);paths={}
    for index,image in enumerate(native_images(images),1):
        check_pause(c['output_dir']);path=folder/(image['sample_id']+'.npy')
        identity=digest(dict(checkpoint=model['checkpoint_sha256'],image=file_hash(image['image_path']),
            plans=file_hash(bundle['plans_file']),dataset=file_hash(bundle['dataset_file']),inference=inference))
        marker=path.with_suffix('.receipt.json')
        if marker.exists():
            receipt=read_json(marker)
            if receipt['signature']!=identity or file_hash(path)!=receipt['sha256']:raise RuntimeError('Native prediction cache changed.')
            probability=np.load(path,allow_pickle=False)
        else:
            if path.exists():raise RuntimeError('Uncommitted native probability file; no silent reuse.')
            arr,props=reader.read_images([image['image_path']])
            with torch.inference_mode():_,probabilities=predictor.predict_single_npy_array(arr,props,save_or_return_probabilities=True)
            probabilities=np.asarray(probabilities)
            if probabilities.ndim==4 and probabilities.shape[1]==1:probabilities=probabilities[:,0]
            if probabilities.shape!=(2,384,384):raise RuntimeError('Invalid native restored probability grid.')
            probability=probabilities[1].astype(np.float32)
            buffer=io.BytesIO();np.save(buffer,probability,allow_pickle=False);atomic_bytes(path,buffer.getvalue())
            atomic_json(marker,dict(signature=identity,sha256=file_hash(path)))
        if probability.shape!=(384,384) or not np.isfinite(probability).all() or np.any((probability<0)|(probability>1)):
            raise RuntimeError('Invalid native probabilities.')
        paths[image['sample_id']]=str(path)
        event(c['output_dir'],'student_prediction_image',task_id=task_id,completed=index,total=len(images))
    del predictor,network,payload,state
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return paths


def _native_state_equal(a,b):
    """Exact comparison of network, SGD buffers and AMP scaler state."""
    import torch
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and a.shape==b.shape and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,dict):
        return isinstance(b,dict) and a.keys()==b.keys() and all(_native_state_equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):
        return type(a) is type(b) and len(a)==len(b) and all(_native_state_equal(x,y) for x,y in zip(a,b))
    return type(a) is type(b) and a==b


def _native_step_state(trainer):
    """Verification snapshot only; no study checkpoint is written."""
    network=getattr(trainer.network,'_orig_mod',trainer.network)
    return dict(weights={k:v.detach().cpu().clone() for k,v in network.state_dict().items()},
        optimizer=copy.deepcopy(trainer.optimizer.state_dict()),
        scaler=copy.deepcopy(trainer.grad_scaler.state_dict()) if trainer.grad_scaler else None)


def _restore_native_step_state(trainer,state):
    getattr(trainer.network,'_orig_mod',trainer.network).load_state_dict(state['weights'],strict=True)
    trainer.optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
    if trainer.grad_scaler:trainer.grad_scaler.load_state_dict(copy.deepcopy(state['scaler']))


def _alpha0_loss_check(native_losses,candidate_loss):
    """Use PyTorch 2.14's float32 comparison for the logged scalar only.

    Native CUDA reductions need not reproduce a scalar bit-for-bit. ULP count
    is recorded, never used as an inferred limit from one earlier batch.
    The fixed dtype tolerances are those published for torch.testing.assert_close;
    network, SGD and AMP updates must still be bit-for-bit identical separately.
    This helper never changes the loss tensor used for backward or training.
    """
    if len(native_losses)!=3:raise RuntimeError('Alpha0 gate requires a native step and two native replays.')
    values=[np.asarray(value) for value in native_losses];candidate=np.asarray(candidate_loss);first=values[0]
    if (first.dtype!=np.dtype('float32') or first.size!=1 or
            any(v.shape!=first.shape or v.dtype!=first.dtype or not np.isfinite(v).all() for v in [*values,candidate])):
        raise RuntimeError('Alpha0 loss must be a finite scalar with the same native dtype.')
    # PyTorch 2.14 torch.testing.assert_close default dtype tolerances.
    # https://docs.pytorch.org/docs/2.14/testing.html
    rtol,atol=1.3e-6,1e-5
    ulp=max(float(np.abs(np.spacing(v)).max()) for v in values)
    span=max(float(np.abs(v.astype(np.float64)-first.astype(np.float64)).max()) for v in values)
    difference=float(np.abs(candidate.astype(np.float64)-first.astype(np.float64)).max())
    close=lambda a,b:bool(np.all(np.abs(a.astype(np.float64)-b.astype(np.float64))<=atol+rtol*np.abs(b.astype(np.float64))))
    if any(not close(v,first) for v in values):
        raise RuntimeError(f'Native supervised loss is inconsistent within float32 precision: native={[v.tolist() for v in values]}.')
    if not close(candidate,first):
        raise RuntimeError(f'Alpha0 loss differs beyond float32 precision: native={[v.tolist() for v in values]}, candidate={candidate.tolist()}.')
    return dict(native_losses=[v.tolist() for v in values],candidate_loss=candidate.tolist(),
        native_dtype=str(first.dtype),native_repeat_span=span,candidate_abs_difference=difference,
        native_ulp=ulp,native_drift_ulps=span/ulp if ulp else 0.,
        comparison='pytorch_2.14_float32_assert_close',rtol=rtol,atol=atol,
        candidate_within_float32_tolerance=True,
        candidate_matches_native_replay=any(np.array_equal(v,candidate) for v in values),
        loss_bit_exact_to_first=bool(np.array_equal(first,candidate)))


def preflight(c,data,teacher,teacher_maps,folder,baseline_model):
    """Mandatory target-installation parity gate. It performs no study fit.

    Four actual clinical patches use source-T weights and the original native
    loader/transforms. Images, DS targets, RNG streams, weights, optimizer and
    AMP updates must be bit-identical. Alpha0 logged scalar loss uses the
    standard PyTorch float32 comparison. Positive-alpha KL gradients
    are confined to transported ROIs. No study training flag is changed.
    """
    import torch
    from .native_kd_trainer import roi_kl
    recipe=validate_recipe(baseline_model.get('native_recipe') or c.get('native_recipe') or {})
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    identity=environment_identity();signature=digest(dict(recipe=recipe_signature(recipe),environment=identity,
        baseline=baseline_model['checkpoint_sha256'],teacher=teacher['checkpoint_sha256']))
    verify_archived_runtime(recipe,identity)
    receipt=folder/'NATIVE_EQUIVALENCE.json'
    if receipt.exists():
        previous=read_json(receipt)
        if previous.get('signature')!=signature or not previous.get('passed'):
            raise RuntimeError('Native equivalence preflight identity changed.')
        return {**{k:v for k,v in previous.items() if k!='signature'},'gate_signature':previous['signature']}
    if c['training']['device']!='cpu' and not torch.cuda.is_available():
        raise RuntimeError('The mandatory native equivalence gate requires the requested CUDA device.')
    split=next(s for s in data['splits'] if s['fold']==baseline_model['spec']['fold'])
    available=[sid for sid in split['train_ids'] if sid in teacher_maps]
    if not available:raise RuntimeError('Native preflight needs a real training-only ROI target.')
    sid=available[0];validation_id=split['validation_ids'][0]
    mini=dict(images=[i for i in data['images'] if i['sample_id'] in (sid,validation_id)],acta=[],
              splits=[dict(train_ids=[sid],validation_ids=[validation_id])])
    gate_c=dict(c,output_dir=str(folder/'scratch'))
    bundle=setup_bundle(gate_c,mini,recipe,baseline_model['native_bundle'])
    seed=int(baseline_model['spec']['seed']);fold=int(baseline_model['spec']['fold'])
    base=dict(directory=str(folder/'scratch'/'controllers'),task_id='native_equivalence',run_root=str(folder),
        signature=signature,seed=seed,fold=fold,stage='GATE',domain='dcbr',kd=False,selection='clinical_validation',
        train_ids=[sid],validation_ids=[validation_id],native_train_ids=[sid],native_validation_ids=[validation_id],
        initialization=baseline_model,hyperparameters=dict(lr=recipe['initial_lr'],epochs=recipe['epochs'],
            weight_decay=recipe['weight_decay'],distillation_weight=0.,temperature=1.))
    _environment(bundle,base,recipe);preprocess(bundle,gate_c)
    candidate_spec=dict(base,kd=True,teacher_maps={sid:teacher_maps[sid]})
    targets=prepare_targets(candidate_spec,gate_c,bundle)
    _seed(seed);original=trainer_for(base,gate_c,bundle,recipe);load_initialization(original,baseline_model,recipe)
    _seed(seed);candidate=trainer_for(candidate_spec,gate_c,bundle,recipe,targets);load_initialization(candidate,baseline_model,recipe)
    # Zero workers is a verification context only. The study uses the exact
    # audited source worker count. It permits direct parent RNG comparison.
    prior_workers=os.environ.get('nnUNet_n_proc_DA');os.environ['nnUNet_n_proc_DA']='0'
    original_loader=candidate_loader=None
    def rng():
        return (random.getstate(),np.random.get_state(),torch.get_rng_state(),
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    def same_rng(a,b):
        return (a[0]==b[0] and a[1][0]==b[1][0] and np.array_equal(a[1][1],b[1][1]) and a[1][2:]==b[1][2:]
                and torch.equal(a[2],b[2]) and (a[3] is None and b[3] is None or
                    a[3] is not None and b[3] is not None and len(a[3])==len(b[3]) and all(torch.equal(x,y) for x,y in zip(a[3],b[3]))))
    try:
        _seed(seed);original_loader,original_val=original.get_dataloaders()
        _seed(seed);candidate_loader,candidate_val=candidate.get_dataloaders()
        alpha0_checks=[]
        for index in range(4):
            _seed(seed+index);source_batch=next(original_loader);source_rng=rng()
            _seed(seed+index);candidate_batch=next(candidate_loader);candidate_rng=rng()
            if (list(source_batch['keys'])!=list(candidate_batch['keys']) or
                    not torch.equal(source_batch['data'],candidate_batch['data'][:,:1]) or not same_rng(source_rng,candidate_rng)):
                raise RuntimeError('KD transport changed native sampled images, keys or random streams.')
            a=source_batch['target'];b=candidate_batch['target'];a=a if isinstance(a,list) else [a];b=b if isinstance(b,list) else [b]
            if len(a)!=len(b) or any(not torch.equal(x,y) for x,y in zip(a,b)):
                raise RuntimeError('KD transport changed native deep-supervision labels.')
            before=_native_step_state(original)
            if not _native_state_equal(before,_native_step_state(candidate)):
                raise RuntimeError('Alpha0 gate starts from different network, optimizer or AMP state.')
            _seed(seed+100+index);original_result=original.train_step(source_batch)
            after=_native_step_state(original)
            native_losses=[np.asarray(original_result['loss']).copy()]
            # The unmodified native scalar can differ by float32 rounding even
            # when its SGD/AMP update is identical. Measure this batch locally,
            # and restore the first update before proceeding to the next batch.
            try:
                for _ in range(2):
                    _restore_native_step_state(original,before)
                    _seed(seed+100+index);repeated=original.train_step(source_batch)
                    if not _native_state_equal(after,_native_step_state(original)):
                        raise RuntimeError('Native replay changed the SGD/AMP update; exact equivalence cannot be verified.')
                    native_losses.append(np.asarray(repeated['loss']).copy())
            finally:
                _restore_native_step_state(original,after)
            _seed(seed+100+index);candidate_result=candidate.train_step(candidate_batch)
            if not _native_state_equal(after,_native_step_state(candidate)):
                raise RuntimeError('Alpha0 changed the native network, SGD buffers or AMP update.')
            checked=_alpha0_loss_check(native_losses,candidate_result['loss'])
            alpha0_checks.append(dict(batch=index+1,updates='bit_exact',**checked))
            atomic_json(folder/'ALPHA0_CHECKS.json',dict(completed=len(alpha0_checks),total=4,checks=alpha0_checks))
            event(c['output_dir'],'native_alpha0_check',batch=index+1,total=4,updates='bit_exact',**checked)
        device=original.device
        _seed(seed);logits=torch.randn(1,2,17,19,device=device,requires_grad=True)
        probability=torch.full((1,17,19),.8,device=device,requires_grad=True)
        support=torch.zeros_like(probability);support[:,4:12,3:11]=1
        kd=roi_kl(logits,probability,support,1.);kd.backward()
        outside=(support==0)[:,None].expand_as(logits)
        if torch.count_nonzero(logits.grad[outside]) or not torch.count_nonzero(logits.grad[~outside]) or probability.grad is not None:
            raise RuntimeError('ROI KL gradients escaped the prescribed anatomical support.')
        empty=roi_kl(logits.detach(),probability.detach(),torch.zeros_like(support),1.)
        if float(empty)!=0:raise RuntimeError('Empty native ROI patch must have exactly zero KD loss.')
        # Match the legacy source's native preprocessed sliding-window + export
        # path against the raw-array API used for all Reverse predictions.
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
        checkpoint=torch.load(baseline_model['checkpoint'],map_location='cpu',weights_only=False)
        state={k.removeprefix('_orig_mod.'):v for k,v in checkpoint['network_weights'].items()}
        getattr(original.network,'_orig_mod',original.network).load_state_dict(state,strict=True)
        original.set_deep_supervision_enabled(False);original.network.eval()
        axes=checkpoint.get('inference_allowed_mirroring_axes')
        if tuple(axes or ())!=tuple(recipe['inference'].get('mirroring_axes') or ()):
            raise RuntimeError('Source checkpoint mirroring axes changed.')
        infer=recipe['inference']
        os.environ['nnUNet_compile']='false'
        source_predictor=nnUNetPredictor(tile_step_size=infer['tile_step_size'],use_gaussian=infer['use_gaussian'],
            use_mirroring=infer['use_mirroring'],perform_everything_on_device=device.type=='cuda',device=device,
            verbose=False,verbose_preprocessing=False,allow_tqdm=False)
        source_predictor.manual_initialization(getattr(original.network,'_orig_mod',original.network),original.plans_manager,original.configuration_manager,
            None,original.dataset_json,checkpoint.get('trainer_name'),axes)
        native_dataset=original.dataset_class(original.preprocessed_dataset_folder,[sid])
        native_data,_,_,properties=native_dataset.load_case(sid)
        with torch.inference_mode():
            native_logits=source_predictor.predict_sliding_window_return_logits(torch.from_numpy(np.array(native_data[:],copy=True))).cpu()
        _,native_probabilities=convert_predicted_logits_to_segmentation_with_correct_shape(native_logits,
            original.plans_manager,original.configuration_manager,original.label_manager,properties,return_probabilities=True)
        native_probabilities=np.asarray(native_probabilities)
        if native_probabilities.ndim==4 and native_probabilities.shape[1]==1:native_probabilities=native_probabilities[:,0]
        selected_image=[i for i in mini['images'] if i['sample_id']==sid]
        raw_maps=predict(c,baseline_model,selected_image,folder/'raw_inference','native_inference_equivalence')
        raw_probability=np.load(raw_maps[sid],allow_pickle=False)
        if not np.array_equal(native_probabilities[1].astype(np.float32),raw_probability):
            raise RuntimeError('Native raw inference changed source preprocessing, sliding windows or export.')
        result=dict(signature=signature,recipe_signature=recipe_signature(recipe),environment=identity,passed=True,
            device=str(device),baseline_checkpoint_sha256=baseline_model['checkpoint_sha256'],
            teacher_checkpoint_sha256=teacher['checkpoint_sha256'],actual_clinical_training_image=sid,
            native_batch_checks=4,image_labels_rng='bit_exact',
            supervised_loss_sgd_amp_alpha0='native_replay_loss_and_bit_exact_updates',
            supervised_loss_alpha0='pytorch_2.14_float32_assert_close',
            network_optimizer_scaler_alpha0='bit_exact',alpha0_checks=alpha0_checks,
            native_preprocessed_vs_raw_inference='bit_exact',
            outside_roi_gradient=0,teacher_gradient=None,empty_roi_loss=0,
            source_weights_transferred=True,optimizer_scheduler_ema_reset=True,
            verification_workers=0,study_workers=recipe['workers'],
            cross_process_native_augmentation_continuation_bit_exact=False)
        atomic_json(receipt,result)
        return {**{k:v for k,v in result.items() if k!='signature'},'gate_signature':result['signature']}
    finally:
        if prior_workers is None:os.environ.pop('nnUNet_n_proc_DA',None)
        else:os.environ['nnUNet_n_proc_DA']=prior_workers
        for loader in (original_loader,candidate_loader,locals().get('original_val'),locals().get('candidate_val')):
            if hasattr(loader,'_finish'):loader._finish()
        del original,candidate
        if torch.cuda.is_available():torch.cuda.empty_cache()
