"""Train one new crop fit; load continuous validation probabilities only."""
import copy
import io
import os
from pathlib import Path
import numpy as np
from PIL import Image
from .io import atomic_bytes,atomic_json,digest,file_hash,read_json,write_csv,event
from .data import run_dir,read_frame


def prepare_nnunet(project,df,eff):
    root=Path(project)/'work/nnunet';name='Dataset702_DCBRCrops100'
    raw=root/'raw'/name;pre=root/'preprocessed'/name
    raw.mkdir(parents=True,exist_ok=True);pre.mkdir(parents=True,exist_ok=True)
    for r in df.itertuples():
        for src,dst in [(Path(r.image_path),raw/'imagesTr'/f'{r.sample_id}_0000.png'),(Path(r.mask_path),raw/'labelsTr'/f'{r.sample_id}.png')]:
            dst.parent.mkdir(parents=True,exist_ok=True)
            if dst.is_symlink():
                if dst.resolve()!=src.resolve():raise RuntimeError('A native dataset link changed.')
            elif dst.exists():raise RuntimeError('Refusing to replace an existing native dataset file.')
            else:dst.symlink_to(src.resolve())
    dataset={'channel_names':{'0':'grayscale'},'labels':{'background':0,'caries':1},'numTraining':len(df),'file_ending':'.png'}
    atomic_json(raw/'dataset.json',dataset);atomic_json(pre/'dataset.json',dataset)
    atomic_json(pre/'nnUNetPlans.json',eff['nnunet_plans'])
    splits=[{'train':df.loc[df.fold.astype(int)!=f,'sample_id'].tolist(),'val':df.loc[df.fold.astype(int)==f,'sample_id'].tolist()} for f in range(5)]
    atomic_json(pre/'splits_final.json',splits)
    return dict(raw_root=str(root/'raw'),preprocessed_root=str(root/'preprocessed'),raw_directory=str(raw),preprocessed_directory=str(pre),
                plans_file=str(pre/'nnUNetPlans.json'),dataset_json_file=str(raw/'dataset.json'),manifest_path=str(Path(project)/'output/data/manifest.csv'),
                configuration='2d',plans_name='nnUNetPlans',dataset_name=name,validation_ids_by_fold={str(f):splits[f]['val'] for f in range(5)})


def infer_yolo(spec,result,val):
    import torch
    import ultralytics
    ck=torch.load(result['checkpoint'],map_location='cpu',weights_only=False)
    net=ck.get('ema')
    if net is None:net=ck.get('model')
    if net is None:raise RuntimeError('Missing semantic YOLO network.')
    device=torch.device('cuda:0');net=net.float().to(device).eval();paths={}
    for i,r in enumerate(val.itertuples(),1):
        if Path(spec['stop_file']).exists():raise KeyboardInterrupt('Pause requested between validation predictions.')
        path=Path(spec['directory'])/'probabilities'/f'{r.sample_id}.npy'
        if not path.is_file():
            with Image.open(r.image_path) as im:a=np.asarray(im.convert('RGB'),dtype=np.float32)/255.
            x=torch.from_numpy(a.transpose(2,0,1).copy()).unsqueeze(0).to(device)
            with torch.inference_mode():
                logits=net(x)
                if isinstance(logits,(tuple,list)):logits=logits[0]
                if not torch.is_tensor(logits) or logits.ndim!=4 or logits.shape[1] not in (1,2):raise RuntimeError('YOLO must expose logits; hard-mask fallback cannot support ROC/PR.')
                logits=torch.nn.functional.interpolate(logits.float(),size=a.shape[:2],mode='bilinear',align_corners=False)
                p=logits.sigmoid()[:,0] if logits.shape[1]==1 else logits.softmax(dim=1)[:,1]
                p=p[0].cpu().numpy().astype(np.float32)
            buf=io.BytesIO();np.save(buf,p,allow_pickle=False);atomic_bytes(path,buf.getvalue())
        paths[r.sample_id]=str(path)
        if i==1 or i%10==0 or i==len(val):event(Path(spec['run_root']),'validation_inference',state='validation_inference',architecture='yolo26_sem',seed=spec['seed'],fold=spec['fold'],processed=i,total=len(val))
    return paths


def probability_path(result,sid):
    if sid in result.get('probability_paths',{}):return result['probability_paths'][sid]
    return result.get('native_npz_paths',{})[sid]


def load_probability(path):
    path=Path(path)
    if path.suffix=='.npz':
        with np.load(path,allow_pickle=False) as z:p=z['probabilities']
        if p.ndim==4 and p.shape[1]==1:p=p[:,0]
        if p.ndim!=3 or p.shape[0]!=2:raise ValueError('Expected native two-class probabilities.')
        p=p[1]
    else:p=np.load(path,allow_pickle=False)
    p=np.asarray(p,dtype=np.float32)
    if p.shape!=(384,384) or not np.isfinite(p).all() or p.min()<0 or p.max()>1:raise ValueError(f'Invalid crop probabilities: {path}')
    return p


def train_fit(project,item,protocol):
    project=Path(project);eff=read_json(project/'output/effective_config.json');df=read_frame(protocol['manifest'])
    arch,seed,fold=item['architecture'],item['seed'],item['fold'];folder=run_dir(project,arch,seed,fold)
    signature=digest({'protocol':protocol['signature'],'architecture':arch,'seed':seed,'fold':fold})
    spec={**item,'signature':signature,'directory':str(folder),'run_root':str(project/'output/activity'),'stop_file':str(project/'output/STOP'),'condition':'crops'}
    val=df[df.fold.astype(int)==fold];train=df[df.fold.astype(int)!=fold]
    if arch=='yolo26_sem':
        from .yolo import train_candidate
        result=train_candidate(spec,train,val,val,eff['yolo']);paths=infer_yolo(spec,result,val)
        result={**result,'architecture':arch,'probability_paths':paths}
        atomic_json(folder/'result.json',result)
    elif arch=='monai_segresnet':
        from .monai import train
        result=train(spec,df,eff['historical_config'])
    elif arch=='nnunetv2':
        from .nnunet import train_and_predict
        result=train_and_predict(spec,prepare_nnunet(project,df,eff),eff['nnunet'])
    else:raise ValueError(arch)
    from .metrics import surface_record,summarize_records
    rows=[];hashes={result['checkpoint']:file_hash(result['checkpoint'])}
    for r in val.itertuples():
        if Path(spec['stop_file']).exists():raise KeyboardInterrupt('Pause requested during held-out metric calculation.')
        path=probability_path(result,r.sample_id);p=load_probability(path)
        with Image.open(r.mask_path) as im:ref=np.asarray(im)>0
        rec=surface_record(ref,p,.5);rec.update(architecture=arch,seed=seed,fold=fold,sample_id=r.sample_id,bitewing_id=r.bitewing_id,group_id=r.group_id,threshold_name='fixed',eligible=True)
        rows.append(rec);hashes[path]=file_hash(path)
    write_csv(folder/'per_surface_fixed.csv',rows);hashes[str(folder/'per_surface_fixed.csv')]=file_hash(folder/'per_surface_fixed.csv')
    result.update(seed=seed,fold=fold,signature=signature)
    atomic_json(folder/'result.json',result);hashes[str(folder/'result.json')]=file_hash(folder/'result.json')
    atomic_json(folder/'complete.json',dict(protocol_signature=protocol['signature'],signature=signature,architecture=arch,seed=seed,fold=fold,
                trained=result,summary=summarize_records(rows),n_surfaces=len(rows),artifact_hashes=hashes))
    return result
