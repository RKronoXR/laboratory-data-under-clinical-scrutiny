"""Historical SegResNet crop protocol with resumable optimizer and RNG state."""
from __future__ import annotations
import io
import random
import time
from pathlib import Path
import numpy as np
from .io import atomic_bytes,atomic_json,file_hash,read_json,write_csv,event


def save_torch(path,value):
    import torch
    buf=io.BytesIO();torch.save(value,buf);atomic_bytes(path,buf.getvalue())


def rng_nodes(transform):
    seen=set();result=[]
    def visit(obj):
        if id(obj) in seen:return
        seen.add(id(obj))
        if hasattr(obj,'R') and hasattr(obj.R,'get_state'):result.append(obj)
        if hasattr(obj,'__dict__'):
            for val in vars(obj).values():
                for x in (val if isinstance(val,(tuple,list)) else (val,)):
                    if hasattr(x,'set_random_state') or hasattr(x,'transforms'):visit(x)
    visit(transform);return result


def transforms(train):
    import torch
    from monai.data import PILReader
    from monai.transforms import Compose,LoadImaged,EnsureChannelFirstd,ScaleIntensityd,RandFlipd,RandAffined,RandScaleIntensityd,RandShiftIntensityd,EnsureTyped
    items=[LoadImaged(keys=['image','label'],reader=PILReader(reverse_indexing=True),image_only=True),
           EnsureChannelFirstd(keys=['image','label']),ScaleIntensityd(keys=['image'],minv=0.,maxv=1.)]
    if train:
        items += [RandFlipd(keys=['image','label'],prob=.5,spatial_axis=1),
                  RandAffined(keys=['image','label'],prob=.30,rotate_range=(np.deg2rad(5.),),translate_range=(8,8),scale_range=(.05,.05),mode=('bilinear','nearest'),padding_mode='zeros'),
                  RandScaleIntensityd(keys=['image'],factors=.10,prob=.30),RandShiftIntensityd(keys=['image'],offsets=.10,prob=.30)]
    return Compose(items+[EnsureTyped(keys=['image','label'],dtype=torch.float32)])


def build(cfg,device):
    from monai.networks.nets import SegResNet
    m=cfg['monai']
    return SegResNet(spatial_dims=2,in_channels=1,out_channels=1,init_filters=int(m['init_filters']),blocks_down=tuple(m['blocks_down']),blocks_up=tuple(m['blocks_up']),norm='BATCH').to(device)


def train(spec,df,cfg):
    import torch
    from monai.data import DataLoader,Dataset
    from monai.losses import DiceFocalLoss
    from monai.utils import set_determinism
    folder=Path(spec['directory']);folder.mkdir(parents=True,exist_ok=True)
    done=folder/'result.json'
    if done.is_file():
        result=read_json(done)
        if result['signature']!=spec['signature'] or file_hash(result['checkpoint'])!=result['checkpoint_sha256']:raise RuntimeError('Completed MONAI checkpoint changed.')
        return result
    stop=Path(spec['stop_file'])
    def check():
        if stop.exists():raise KeyboardInterrupt('Pause requested; the last complete epoch is saved.')
    check()
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required; this queue never silently switches training to CPU.')
    device=torch.device('cuda:0');tc=cfg['training'];set_determinism(seed=spec['seed'])
    train=df[df.fold.astype(int)!=spec['fold']];val=df[df.fold.astype(int)==spec['fold']]
    def records(frame):return [{'image':r.image_path,'label':r.mask_path,'sample_id':r.sample_id} for r in frame.itertuples()]
    tr=transforms(True);ev=transforms(False)
    loader=DataLoader(Dataset(records(train),tr),batch_size=int(tc['batch_size']),shuffle=True,num_workers=int(tc['num_workers']),pin_memory=True)
    validation=DataLoader(Dataset(records(val),ev),batch_size=int(tc['batch_size']),shuffle=False,num_workers=int(tc['num_workers']),pin_memory=True)
    net=build(cfg,device);opt=torch.optim.AdamW(net.parameters(),lr=float(tc['learning_rate']),weight_decay=float(tc['weight_decay']))
    scaler=torch.amp.GradScaler('cuda',enabled=True);lossfn=DiceFocalLoss(sigmoid=True,lambda_dice=1.,lambda_focal=1.)
    best=-1.;best_epoch=0;stale=0;steps=0;start_epoch=1;elapsed=0.;history=[]
    last=folder/'last.pt';bestfile=folder/'best.pt';completed=folder/'training_done.json'
    if last.is_file():
        ck=torch.load(last,map_location='cpu',weights_only=False)
        if ck['signature']!=spec['signature'] or ck['seed']!=spec['seed'] or ck['fold']!=spec['fold']:raise RuntimeError('Incompatible MONAI resume checkpoint.')
        net.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler'])
        best,best_epoch,stale,steps,start_epoch,elapsed=ck['best'],ck['best_epoch'],ck['stale'],ck['steps'],ck['epoch']+1,ck['elapsed']
        history=ck['history'];random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
        nodes=rng_nodes(tr)
        if len(nodes)!=len(ck['transform_rng']):raise RuntimeError('The MONAI transform graph changed.')
        for node,rng in zip(nodes,ck['transform_rng']):node.R.set_state(rng)
        if not bestfile.exists() or file_hash(bestfile)!=ck['best_sha256']:raise RuntimeError('Selected MONAI weights changed.')
        del ck
    started=time.monotonic()
    def status(state,**fields):
        row=dict(state=state,architecture='monai_segresnet',condition='crops',seed=spec['seed'],fold=spec['fold'],max_epochs=int(tc['max_epochs']),
                 best_epoch=best_epoch,best_metric=best,stale=stale,patience=int(tc['patience']),seconds=elapsed+time.monotonic()-started,**fields)
        atomic_json(folder/'current.json',row);event(Path(spec['run_root']),state,**row)
    if not completed.is_file():
        for epoch in range(start_epoch,int(tc['max_epochs'])+1):
            if history and stale>=int(tc['patience']):break
            check();net.train();losses=[]
            for i,batch in enumerate(loader,1):
                x=batch['image'].to(device);y=(batch['label']>0).float().to(device);opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda',enabled=True):loss=lossfn(net(x),y)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite MONAI loss.')
                scaler.scale(loss).backward();scaler.step(opt);scaler.update();steps+=1;losses.append(float(loss.detach()))
                if i==1 or i%5==0 or i==len(loader):status('training',epoch=epoch,batch=i,batches=len(loader),optimizer_steps=steps)
            net.eval();values=[]
            with torch.inference_mode():
                for i,batch in enumerate(validation,1):
                    y=batch['label'].to(device)>0;pred=net(batch['image'].to(device)).sigmoid()>=.5
                    for p,t in zip(pred,y):
                        if t.any():values.append(2.*float((p&t).sum())/float(p.sum()+t.sum()))
                    if i==1 or i%5==0 or i==len(validation):status('validation',epoch=epoch,batch=i,batches=len(validation),optimizer_steps=steps)
            if not values:raise RuntimeError('No reference-positive validation surfaces.')
            metric=float(np.mean(values))
            if metric>best+float(tc['min_delta']):
                best,best_epoch,stale=metric,epoch,0
                save_torch(bestfile,dict(model=net.state_dict(),epoch=epoch,metric=metric,signature=spec['signature'],seed=spec['seed'],fold=spec['fold']))
            else:stale+=1
            row=dict(epoch=epoch,train_loss=float(np.mean(losses)),positive_dice=metric,best=best,best_epoch=best_epoch,stale=stale,optimizer_steps=steps,seconds=elapsed+time.monotonic()-started)
            history.append(row)
            payload=dict(model=net.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),signature=spec['signature'],seed=spec['seed'],fold=spec['fold'],epoch=epoch,
                         best=best,best_epoch=best_epoch,stale=stale,steps=steps,elapsed=row['seconds'],history=history,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                         torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),transform_rng=[n.R.get_state() for n in rng_nodes(tr)],best_sha256=file_hash(bestfile))
            save_torch(last,payload);del payload
            write_csv(folder/'history.csv',history);atomic_json(folder/'epochs'/f'{epoch:04d}.json',row)
            status('epoch_saved',epoch=epoch,optimizer_steps=steps);check()
            if stale>=int(tc['patience']):break
        if not history:raise RuntimeError('No MONAI epoch was completed.')
        atomic_json(completed,dict(signature=spec['signature'],best_epoch=best_epoch,epochs_completed=history[-1]['epoch'],seconds=elapsed+time.monotonic()-started,optimizer_steps=steps))
    ck=torch.load(bestfile,map_location='cpu',weights_only=False);net.load_state_dict(ck['model']);net.eval()
    probs=folder/'probabilities';probs.mkdir(exist_ok=True)
    with torch.inference_mode():
        for i,batch in enumerate(validation,1):
            check();p=net(batch['image'].to(device)).sigmoid().cpu().numpy()[:,0]
            for sid,value in zip(batch['sample_id'],p):
                # PILReader's historical reverse indexing is restored to PNG H,W.
                buf=io.BytesIO();np.save(buf,value.T.astype(np.float32),allow_pickle=False);atomic_bytes(probs/f'{sid}.npy',buf.getvalue())
            status('validation_inference',epoch=history[-1]['epoch'],batch=i,batches=len(validation))
    result={**read_json(completed),'architecture':'monai_segresnet','checkpoint':str(bestfile),'checkpoint_sha256':file_hash(bestfile),
            'selection_metric':'validation_positive_surface_Dice','probability_paths':{sid:str(probs/f'{sid}.npy') for sid in val.sample_id}}
    atomic_json(done,result);return result
