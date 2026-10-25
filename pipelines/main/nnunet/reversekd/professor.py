"""Reuse the verified V3 ACTA-only YOLO checkpoint and align anatomy ROI targets."""
from __future__ import annotations
import io
import shutil
from pathlib import Path
import numpy as np
from .data import load_image, letterbox, restore_probability
from .io import atomic_bytes,atomic_json,read_json,file_hash,digest,event,check_pause,write_csv

def descriptor_path(c,seed,fold):
    return Path(c['source_output'])/'selections'/f'seed_{seed}_fold_{fold}'/'P.json'

def audit_source_data(c,data):
    base=Path(c['source_output']);ready=read_json(base/'data/READY.json')
    source=read_json(base/'data/bundle.json')
    old={r['sample_id']:r for r in source['acta']}
    current={r['sample_id']:r for r in data['acta']}
    if set(old)!=set(current):raise RuntimeError('The trained YOLO ACTA source pool differs from the current ACTA pool.')
    expected=file_hash(Path(c['project_root'])/c['teacher_reuse']['legacy_splits'])
    if ready.get('original_splits_sha256')!=expected:
        raise RuntimeError('The V3 checkpoint metadata do not belong to the unchanged original DCBR partitions.')
    for index,sid in enumerate(sorted(current),1):
        a,b=old[sid],current[sid]
        if a['group_id']!=b['group_id'] or a['geometry']!=b['geometry']:
            raise RuntimeError(f'The ACTA teacher input geometry differs: {sid}.')
        for key in ('image_path','mask_path'):
            if not np.array_equal(load_image(a[key]),load_image(b[key])):
                raise RuntimeError(f'The ACTA teacher source pixels differ: {sid}, {key}.')
        if index%100==0:
            event(c['output_dir'],'audit_acta_source',completed=index,total=len(current))
    value=dict(acta_ids_sha256=digest(sorted(current)),original_splits_sha256=expected,
               n_acta_images=len(current),source_output=str(base),decoded_pixels_identical=True)
    atomic_json(Path(c['output_dir'])/'audit/ACTA_SOURCE_AUDIT.json',value)
    return value

def audit_one(c,seed,fold):
    """Check provenance and tensor signature, not filename alone. CPU-only."""
    import torch
    p=descriptor_path(c,seed,fold)
    if not p.is_file():raise FileNotFoundError(f'ACTA-only teacher selection missing: {p}. No new teacher is trained automatically.')
    r=read_json(p);s=r['spec'];cfg=s.get('config',{})
    if s.get('kind')!='pretrain' or s.get('strategy')!='P':
        raise RuntimeError('The requested professor is not an ACTA-only V3 pretraining checkpoint.')
    if (int(s['seed']),int(s['fold']))!=(seed,fold):
        raise RuntimeError('Teacher seed/fold metadata mismatch.')
    if cfg.get('student_model')!=c['teacher_model'] or cfg.get('crop_image_size')!=c['teacher_image_size']:
        raise RuntimeError('Teacher architecture/crop size differ from the requested protocol.')
    if cfg.get('ultralytics_version')!=c['ultralytics_version']:
        raise RuntimeError('Teacher framework version differs from the requested protocol.')
    if s.get('initialization') or s.get('teacher_maps') or float(s.get('distillation_weight',0))!=0:
        raise RuntimeError('The ACTA professor has an unexpected training initialization or clinical distillation source.')
    ck=Path(r['checkpoint'])
    if not ck.is_file() or file_hash(ck)!=r['checkpoint_sha256']:
        raise RuntimeError(f'Teacher checkpoint missing or changed: {ck}')
    payload=torch.load(ck,map_location='cpu',weights_only=False)
    if payload.get('signature')!=s.get('signature') or payload.get('student_model')!=c['teacher_model']:
        raise RuntimeError('Teacher embedded checkpoint signature does not match its training record.')
    if 'state_dict' not in payload or not payload['state_dict']:
        raise RuntimeError('No semantic teacher weights in the selected checkpoint.')
    if any(v.is_floating_point() and not torch.isfinite(v).all() for v in payload['state_dict'].values()):
        raise RuntimeError('Nonfinite tensor in teacher checkpoint.')
    del payload
    return dict(seed=seed,fold=fold,source_selection=str(p),checkpoint=str(ck),
                checkpoint_sha256=r['checkpoint_sha256'],signature=s['signature'],
                teacher_model=c['teacher_model'],crop_image_size=c['teacher_image_size'],
                source='ACTA_only_V3_P',reused=True,clinical_finetuned=False)

def audit_all(c,data):
    audit_source_data(c,data);rows=[]
    for fold in range(c['folds']):
        for seed in c['seeds']:
            rows.append(audit_one(c,seed,fold))
            event(c['output_dir'],'audit_teacher',seed=seed,fold=fold,completed=len(rows),total=15)
    folder=Path(c['output_dir'])/'audit'
    atomic_json(folder/'TEACHERS.json',rows);write_csv(folder/'TEACHERS.csv',rows)
    return rows

def snapshot(c,teacher):
    folder=Path(c['output_dir'])/'teachers'/f'seed_{teacher["seed"]}_fold_{teacher["fold"]}'
    dst=folder/'teacher.pt';folder.mkdir(parents=True,exist_ok=True)
    if dst.exists() and file_hash(dst)!=teacher['checkpoint_sha256']:
        raise RuntimeError('Immutable local teacher snapshot changed.')
    if not dst.exists():
        tmp=folder/'teacher.pt.building';shutil.copy2(teacher['checkpoint'],tmp);tmp.replace(dst)
    result={**teacher,'original_checkpoint':teacher['checkpoint'],'checkpoint':str(dst)}
    atomic_json(folder/'source.json',result)
    return result

def build_model(c,teacher,device):
    import torch
    from ultralytics.nn.tasks import SemanticSegmentationModel
    from types import SimpleNamespace
    if file_hash(teacher['checkpoint'])!=teacher['checkpoint_sha256']:
        raise RuntimeError('Frozen professor checkpoint changed.')
    model=SemanticSegmentationModel(teacher['teacher_model'],ch=3,nc=2,verbose=False)
    model.args=SimpleNamespace(data='acta_binary_semantic')
    if model.model[-1].__class__.__name__!='SemanticSegment' or model.model[-1].nc!=2:
        raise RuntimeError('Expected a semantic probability map. Detection/instance YOLO is not supported.')
    payload=torch.load(teacher['checkpoint'],map_location='cpu',weights_only=False)
    model.load_state_dict(payload['state_dict'],strict=True)
    model.to(device).eval()
    for p in model.parameters():p.requires_grad_(False)
    return model

def stitch_roi_predictions(shape,regions,predictions,size):
    """Regions are anatomical rectangles, never lesion-dependent. Average overlaps."""
    total=np.zeros(shape,np.float32);coverage=np.zeros(shape,np.uint16)
    if len(regions)!=len(predictions):raise ValueError('ROI/prediction count mismatch.')
    for region,p in zip(regions,predictions):
        x1,y1,x2,y2=[int(region[k]) for k in ('x1','y1','x2','y2')]
        if not (0<=x1<x2<=shape[1] and 0<=y1<y2<=shape[0]):raise ValueError('ROI outside the radiograph.')
        p=np.asarray(p,np.float32)
        if p.shape!=(size,size) or not np.isfinite(p).all() or p.min()<0 or p.max()>1:
            raise ValueError('Invalid teacher ROI probabilities.')
        _,geometry=letterbox(np.zeros((y2-y1,x2-x1),np.uint8),size)
        native=restore_probability(p,geometry)
        total[y1:y2,x1:x2]+=native;coverage[y1:y2,x1:x2]+=1
    probability=np.divide(total,coverage,out=np.zeros_like(total),where=coverage>0)
    return probability,(coverage>0).astype(np.uint8)

def target_signature(teacher,image,regions,c):
    return digest(dict(teacher=teacher['checkpoint_sha256'],image=file_hash(image['native_image_path']),
        geometry=image['geometry'],crop_size=c['teacher_image_size'],
        rois=[{k:r[k] for k in ('sample_id','x1','y1','x2','y2')} for r in regions]))

def targets(c,teacher,images,surfaces,folder):
    """Predict on actual clinical ROI crops, then put soft outputs on the full grid."""
    import torch
    import torch.nn.functional as F
    from PIL import Image
    device=torch.device('cpu' if c['training']['device']=='cpu' else 'cuda:0')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    model=None;paths={}
    for index,image in enumerate(images,1):
        check_pause(c['output_dir']);sid=image['sample_id']
        # Reference labels and lesion masks do not enter this construction.
        regions=[s for s in surfaces if s['image_id']==sid]
        signature=target_signature(teacher,image,regions,c)
        pp=folder/f'{sid}.npy';mp=folder/f'{sid}_roi.png';receipt=folder/f'{sid}.json'
        valid=False
        if receipt.exists():
            old=read_json(receipt)
            if old['signature']!=signature:raise RuntimeError('Teacher target geometry/source changed.')
            valid=all(Path(p).is_file() and file_hash(p)==h for p,h in old['artifacts'].items())
            if not valid:raise RuntimeError('A committed teacher target changed or disappeared.')
        if not valid:
            from .cache import find_target
            found=find_target(c,signature)
            if found is not None:
                source_p,source_roi,source_receipt=found
                # Copy immutable maps, keeping the destination receipt self-contained.
                atomic_bytes(pp,Path(source_p).read_bytes());atomic_bytes(mp,Path(source_roi).read_bytes())
                atomic_json(receipt,{**source_receipt,'artifacts':{str(q):file_hash(q) for q in (pp,mp)},'reused_from':source_p})
                valid=True
        if not valid:
            if model is None:model=build_model(c,teacher,device)
            native=load_image(image['native_image_path']);arrays=[]
            for r in regions:
                x1,y1,x2,y2=[int(r[k]) for k in ('x1','y1','x2','y2')]
                crop,_=letterbox(native[y1:y2,x1:x2],c['teacher_image_size'])
                arrays.append(np.repeat(crop[None],3,axis=0).astype(np.float32)/255)
            predictions=[];batch=c['training']['teacher_batch_size']
            for start in range(0,len(arrays),batch):
                x=torch.from_numpy(np.stack(arrays[start:start+batch])).to(device)
                with torch.inference_mode():
                    logits=model(x)
                    if isinstance(logits,(list,tuple)):logits=logits[0]
                    if logits.ndim!=4 or logits.shape[1]!=2:raise RuntimeError('Teacher did not return binary semantic logits.')
                    prob=F.interpolate(logits.float(),size=x.shape[-2:],mode='bilinear',align_corners=False).softmax(1)[:,1]
                predictions.extend(prob.cpu().numpy())
                event(c['output_dir'],'teacher_roi_batch',task_id=f'{teacher["seed"]}/{teacher["fold"]}/{sid}',
                      image_id=sid,completed=min(start+batch,len(arrays)),total=len(arrays))
            native_prob,native_roi=stitch_roi_predictions(native.shape,regions,predictions,c['teacher_image_size'])
            p,geometry=letterbox(native_prob,image['geometry']['size'])
            roi,_=letterbox(native_roi,image['geometry']['size'],is_mask=True)
            if geometry!=image['geometry']:raise RuntimeError('Professor/student full-image alignment mismatch.')
            p[roi==0]=0
            buf=io.BytesIO();np.save(buf,p.astype(np.float32),allow_pickle=False);atomic_bytes(pp,buf.getvalue())
            buf=io.BytesIO();Image.fromarray(roi).save(buf,format='PNG');atomic_bytes(mp,buf.getvalue())
            atomic_json(receipt,dict(signature=signature,artifacts={str(q):file_hash(q) for q in (pp,mp)},
                anatomical_rois_only=True,ground_truth_masks_used=False,n_rois=len(regions)))
        paths[sid]=dict(probability=str(pp),roi_mask=str(mp),
                       probability_sha256=file_hash(pp),roi_sha256=file_hash(mp))
        event(c['output_dir'],'teacher_target_image',image_id=sid,completed=index,total=len(images))
    if model is not None:del model
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return paths
