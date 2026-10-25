"""Native preprocessing plus KD transport without changing image/label RNG.

Native nnU-Net 2.8.1 supplies patch sampling and all original image/label
transforms. Auxiliary teacher targets are never normalized as image channels,
never used to sample patches, and never added to segmentation targets.
"""
from __future__ import annotations
import copy
import io
import os
import pickle
import shutil
from pathlib import Path
import numpy as np
from .io import atomic_bytes, atomic_json, check_pause, digest, file_hash, read_json


def native_images(images):
    """Clinical input is the unchanged source T grid. ACTA is its crop grid."""
    result = []
    for image in images:
        if image.get('source') == 'acta':
            row = dict(image)
        else:
            row = dict(image, image_path=image.get('teacher_image_path', image['image_path']),
                       mask_path=image.get('teacher_mask_path', image['mask_path']),
                       geometry=image.get('teacher_geometry', image['geometry']))
        if int(row['geometry']['size']) != 384:
            raise RuntimeError('Source native recipe requires the unchanged 384-pixel grid.')
        result.append(row)
    return result


def validate_native_plan(plans, dataset, recipe):
    if recipe.get('framework_version') != '2.8.1' or recipe.get('input_size') != 384:
        raise ValueError('Only the audited native 2.8.1 / 384 source recipe is supported.')
    if recipe.get('configuration') != '2d' or plans.get('image_reader_writer') != 'NaturalImage2DIO':
        raise ValueError('An audited NaturalImage2DIO 2d plan is required.')
    conf = plans['configurations']['2d']
    if recipe['plan_configuration'] != conf:
        raise RuntimeError('Native source configuration differs from the frozen plan.')
    if len(conf['normalization_schemes']) != 1 or any(
            name not in ('ZScoreNormalization', 'NoNormalization') for name in conf['normalization_schemes']):
        raise ValueError('Only audited per-image normalization is supported; no fitted/global normalization.')
    if sorted(int(v) for v in dataset['labels'].values()) != [0, 1]:
        raise ValueError('Native target transport requires the original binary, non-region labels.')
    if len(dataset['channel_names']) != 1 or dataset.get('file_ending') != '.png':
        raise ValueError('The original grayscale PNG dataset recipe is required.')


def setup_bundle(c, data, recipe, source_bundle):
    from .native_student import recipe_signature
    plans = read_json(source_bundle['plans_file'])
    dataset = read_json(source_bundle['dataset_file'])
    validate_native_plan(plans, dataset, recipe)
    images = native_images([*data['images'], *data['acta']])
    if len({r['sample_id'] for r in images}) != len(images):
        raise RuntimeError('Native dataset identifiers overlap between domains.')
    signature = digest(dict(plans=file_hash(source_bundle['plans_file']),
        dataset=file_hash(source_bundle['dataset_file']), recipe=recipe_signature(recipe),
        inputs=[dict(id=r['sample_id'], image=file_hash(r['image_path']), mask=file_hash(r['mask_path'])) for r in images]))
    base = Path(c['output_dir'])/'native'/signature[:20]
    receipt = base/'BUNDLE.json'
    if receipt.exists():
        result = read_json(receipt)
        if result['signature'] != signature:
            raise RuntimeError('Native preprocessing source identity changed.')
        for p, expected in result['frozen_file_hashes'].items():
            if file_hash(p) != expected:
                raise RuntimeError('Frozen native data/plan metadata changed.')
        return result
    name = 'Dataset814_ACTAReverseControlled'
    raw = base/'raw'/name
    pre = base/'preprocessed'/name
    raw.mkdir(parents=True, exist_ok=True); pre.mkdir(parents=True, exist_ok=True)
    plans = copy.deepcopy(plans); plans['dataset_name'] = name
    dataset = copy.deepcopy(dataset); dataset['numTraining'] = len(images)
    paths = []
    for image in images:
        check_pause(c['output_dir'])
        for subfolder, key, suffix in [('imagesTr','image_path','_0000.png'), ('labelsTr','mask_path','.png')]:
            dst = raw/subfolder/(image['sample_id']+suffix)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists(): shutil.copy2(image[key], dst)
            if file_hash(dst) != file_hash(image[key]):
                raise RuntimeError('Native source PNG copy is not byte-identical.')
            paths.append(dst)
    for p, value in [(raw/'dataset.json', dataset), (pre/'dataset.json', dataset),
                     (pre/'nnUNetPlans.json', plans), (pre/'splits_final.json',
                      [{'train':s['train_ids'], 'val':s['validation_ids']} for s in data['splits']])]:
        atomic_json(p, value); paths.append(p)
    atomic_json(pre/'dataset_fingerprint.json', dict(note='Frozen original plan; no replanning; per-image normalization only.'))
    result = dict(signature=signature, dataset_name=name, raw_root=str(raw.parent), preprocessed_root=str(pre.parent),
        raw_directory=str(raw), pre_directory=str(pre), plans_file=str(pre/'nnUNetPlans.json'),
        dataset_file=str(pre/'dataset.json'), source_plans_file=source_bundle['plans_file'],
        source_dataset_file=source_bundle['dataset_file'], recipe=recipe,
        frozen_file_hashes={str(p):file_hash(p) for p in paths})
    atomic_json(receipt, result)
    return result


def preprocess(bundle, c):
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    pm = PlansManager(bundle['plans_file']); conf = pm.get_configuration('2d')
    preprocessor = conf.preprocessor_class(verbose=False)
    dataset = read_json(bundle['dataset_file']); raw = Path(bundle['raw_directory'])
    pre = Path(bundle['pre_directory'])/conf.data_identifier
    pre.mkdir(parents=True, exist_ok=True)
    gt = Path(bundle['pre_directory'])/'gt_segmentations'; gt.mkdir(exist_ok=True)
    for image in sorted((raw/'imagesTr').glob('*_0000.png')):
        check_pause(c['output_dir']); sid=image.name[:-9]; mask=raw/'labelsTr'/(sid+'.png')
        signature = digest(dict(image=file_hash(image), mask=file_hash(mask), plans=file_hash(bundle['plans_file'])))
        marker = Path(bundle['pre_directory'])/'case_receipts'/(sid+'.json'); prefix=pre/sid
        if marker.exists():
            old = read_json(marker)
            if old['signature'] != signature or any(file_hash(p) != h for p,h in old['files'].items()):
                raise RuntimeError('Changed native preprocessed case: '+sid)
        else:
            temp=pre/(sid+'_building')
            preprocessor.run_case_save(str(temp), [str(image)], str(mask), pm, conf, dataset)
            generated=[]
            for suffix in ('.b2nd','_seg.b2nd','.pkl'):
                p=Path(str(prefix)+suffix); os.replace(str(temp)+suffix, p); generated.append(p)
            atomic_json(marker, dict(signature=signature, files={str(p):file_hash(p) for p in generated}))
        destination=gt/mask.name
        if destination.exists() and file_hash(destination) != file_hash(mask):
            raise RuntimeError('Native validation reference changed.')
        if not destination.exists(): shutil.copy2(mask, destination)


def targets_to_native(probability, support, properties, native_shape, pm, conf):
    """Apply the saved image-only transpose/crop/resample to ROI targets."""
    probability=np.asarray(probability, dtype=np.float32); support=np.asarray(support, dtype=np.float32)
    if probability.shape != support.shape or probability.ndim != 2 or probability.shape != (384,384):
        raise ValueError('KD target geometry must be the unchanged clinical source grid.')
    if not np.isfinite(probability).all() or np.any((probability<0)|(probability>1)) or not np.isin(support,[0,1]).all():
        raise ValueError('Invalid KD target probability/support.')
    forward=list(pm.transpose_forward)
    p=probability[None,None].transpose([0,*[a+1 for a in forward]])
    roi=support[None,None].transpose([0,*[a+1 for a in forward]])
    bbox=properties['bbox_used_for_cropping']
    slices=(slice(None),*[slice(int(lo),int(hi)) for lo,hi in bbox])
    p=p[slices];roi=roi[slices]
    if tuple(p.shape[1:]) != tuple(properties['shape_after_cropping_and_before_resampling']):
        raise RuntimeError('KD crop metadata disagrees with native image preprocessing.')
    spacing=[properties['spacing'][a] for a in forward]
    target=list(conf.spacing)
    if len(target)<len(native_shape):target=[spacing[0],*target]
    mass=conf.resampling_fn_data(roi, native_shape, spacing, target)
    numerator=conf.resampling_fn_data(p*roi, native_shape, spacing, target)
    binary=conf.resampling_fn_seg(roi, native_shape, spacing, target)>0.5
    q=np.divide(numerator,mass,out=np.zeros_like(numerator),where=mass>1e-6)
    return np.concatenate((np.clip(q,0,1),binary.astype(np.float32)),axis=0).astype(np.float32)


def prepare_targets(spec, c, bundle):
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
    from .data import load_image
    maps=spec.get('teacher_maps',{})
    if set(maps) != set(spec['native_train_ids']):
        raise RuntimeError('KD maps must cover exactly the clinical training fold, without validation.')
    pm=PlansManager(bundle['plans_file']);conf=pm.get_configuration('2d')
    pre=Path(bundle['pre_directory'])/conf.data_identifier
    dataset=infer_dataset_class(str(pre))(str(pre),spec['native_train_ids'])
    result={};folder=Path(spec['directory'])/'native_teacher_targets';folder.mkdir(parents=True,exist_ok=True)
    for sid in spec['native_train_ids']:
        check_pause(c['output_dir']);entry=maps[sid]
        for path, expected in [('probability','probability_sha256'),('roi_mask','roi_sha256')]:
            if file_hash(entry[path]) != entry[expected]: raise RuntimeError('Teacher target changed: '+sid)
        data,_,_,props=dataset.load_case(sid)
        signature=digest(dict(target=entry,preprocessing=file_hash(Path(bundle['pre_directory'])/'case_receipts'/(sid+'.json'))))
        path=folder/(sid+'.npy');receipt=folder/(sid+'.json')
        if receipt.exists():
            rec=read_json(receipt)
            if rec['signature'] != signature or file_hash(path) != rec['sha256']:
                raise RuntimeError('Native teacher transport cache changed.')
        else:
            p=np.load(entry['probability'],allow_pickle=False);roi=load_image(entry['roi_mask'])
            values=targets_to_native(p,roi,props,data.shape[1:],pm,conf)
            buffer=io.BytesIO();np.save(buffer,values,allow_pickle=False);atomic_bytes(path,buffer.getvalue())
            atomic_json(receipt,dict(signature=signature,sha256=file_hash(path)))
        result[sid]=dict(path=str(path),sha256=file_hash(path))
    return result


class TeacherTargetDataset:
    """Native sampling metadata unchanged; append only continuous auxiliaries."""
    def __init__(self, base, target_paths):
        self.base=base;self.targets=target_paths;self.identifiers=base.identifiers
        if set(self.identifiers)!=set(self.targets):raise ValueError('Training target IDs differ.')
    def __getattr__(self,name):return getattr(self.base,name)
    def __len__(self):return len(self.base)
    def load_case(self, identifier):
        data,seg,previous,properties=self.base.load_case(identifier)
        item=self.targets[identifier]
        if file_hash(item['path'])!=item['sha256']:raise RuntimeError('Teacher transport changed.')
        auxiliary=np.load(item['path'],allow_pickle=False)
        if tuple(auxiliary.shape)!=(2,*data.shape[1:]):raise RuntimeError('Native image/KD geometry mismatch.')
        return np.concatenate((np.asarray(data[:]),auxiliary),axis=0),seg,previous,properties


class TransportTeacherTargets:
    """Original transform objects, original random calls, original labels.

    Auxiliaries bypass intensity and DS transforms. Spatial parameters are drawn
    once by the original object and replayed deterministically onto p and ROI.
    No teacher target is transported through a segmentation-label interpolator.
    """
    def __init__(self, original, original_channels=1):
        if not hasattr(original,'transforms'):raise RuntimeError('Unsupported native transform composition.')
        self.original=original;self.channels=original_channels
        for transform in original.transforms:
            name=type(transform).__name__
            if name in ('Convert3DTo2DTransform','Convert2DTo3DTransform'):
                raise RuntimeError('KD transport supports the audited pure 2d recipe only.')
        if sum(type(t).__name__=='SpatialTransform' for t in original.transforms)!=1:
            raise RuntimeError('Native KD requires exactly one audited SpatialTransform.')

    def __call__(self, **state):
        import torch
        import torch.nn.functional as F
        image=state['image']
        if image.shape[0]!=self.channels+2:raise RuntimeError('Missing native target transport channels.')
        p=image[-2:-1].clone();support=image[-1:].clone()
        state=dict(state,image=image[:self.channels])
        for transform in self.original.transforms:
            name=type(transform).__name__
            if name=='SpatialTransform':
                parameters=transform.get_parameters(**state)
                state=transform.apply(state,**parameters)
                # Transport numerator and support with the same native image
                # resampler, then normalize. ROI edges must not dilute p.
                numerator=transform._apply_to_image(p*support,**parameters)
                mass=transform._apply_to_image(support,**parameters)
                if parameters['grid'] is None:
                    binary=mass>0.5
                else:
                    binary=F.grid_sample(support[None],parameters['grid'][None],mode='nearest',
                                         padding_mode='zeros',align_corners=transform.align_corners)[0]>0.5
                p=torch.where(mass>1e-6,numerator/mass.clamp_min(1e-6),torch.zeros_like(numerator)).clamp(0,1)
                support=binary.to(p.dtype)
            elif name=='MirrorTransform':
                parameters=transform.get_parameters(**state)
                state=transform.apply(state,**parameters)
                axes=[int(a)+1 for a in parameters['axes']]
                if axes:p=torch.flip(p,axes);support=torch.flip(support,axes)
            else:
                state=transform(**state)
        if state['image'].shape[1:]!=p.shape[1:]:raise RuntimeError('Unrecognized spatial transform in native recipe.')
        state['image']=torch.cat((state['image'],p,support),dim=0)
        return state
