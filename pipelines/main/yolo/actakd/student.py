"""YOLO26 semantic segmentation with native supervised loss plus pixel KL.

A study-specific PyTorch loop is used for exact teacher-target alignment,
atomic resumable checkpoints, and harmonized validation. This is not the stock
Ultralytics training command and never loads an object-detection head.
"""
from __future__ import annotations
import io,math,random,time,copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .io import atomic_json,atomic_bytes,read_json,write_csv,event,check_pause,file_hash
from .data import load_image
from .metrics import records_for_image,summarize,objective


def seed_everything(seed):
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


def build_model(c):
    from ultralytics.nn.tasks import SemanticSegmentationModel
    model=SemanticSegmentationModel(c['student_model'],ch=3,nc=2,verbose=False)
    model.args=SimpleNamespace(data='acta_kd_binary_semantic')
    if model.model[-1].__class__.__name__!='SemanticSegment' or model.model[-1].nc!=2:
        raise RuntimeError('Expected the YOLO26 two-class semantic segmentation head, not boxes or instance masks.')
    return model


class ImageDataset:
    def __init__(self,rows,teacher_maps=None):self.rows=rows;self.teacher_maps=teacher_maps or {}
    def __len__(self):return len(self.rows)
    def __getitem__(self,idx):
        import torch
        r=self.rows[idx];im=load_image(r['image_path']).astype(np.float32)/255.;mask=load_image(r['mask_path']).astype(np.int64)
        g=r['geometry'];valid=np.zeros(mask.shape,np.float32)
        y,x,h,w=[int(g[k]) for k in ('top','left','resized_height','resized_width')];valid[y:y+h,x:x+w]=1
        if r['sample_id'] in self.teacher_maps:
            p=np.load(self.teacher_maps[r['sample_id']],allow_pickle=False).astype(np.float32)
            if p.shape!=im.shape or not np.isfinite(p).all() or p.min()<0 or p.max()>1:raise ValueError('Invalid teacher target grid.')
        else:p=np.zeros(im.shape,np.float32)
        return torch.from_numpy(np.repeat(im[None],3,axis=0)),torch.from_numpy(mask),torch.from_numpy(valid),torch.from_numpy(p)


def augment(x,y,valid,p,c):
    import torch
    import torch.nn.functional as F
    b=x.shape[0];angle=(torch.rand(b,device=x.device)*2-1)*float(c['training']['rotate_degrees'])*math.pi/180
    theta=torch.zeros((b,2,3),device=x.device,dtype=x.dtype)
    theta[:,0,0]=angle.cos();theta[:,0,1]=-angle.sin();theta[:,1,0]=angle.sin();theta[:,1,1]=angle.cos()
    grid=F.affine_grid(theta,x.shape,align_corners=False)
    x=F.grid_sample(x,grid,mode='bilinear',padding_mode='zeros',align_corners=False)
    y=F.grid_sample(y[:,None].float(),grid,mode='nearest',align_corners=False)[:,0].long()
    valid=F.grid_sample(valid[:,None],grid,mode='nearest',align_corners=False)[:,0]
    p=F.grid_sample(p[:,None],grid,mode='bilinear',align_corners=False)[:,0]
    a=float(c['training']['intensity_fraction'])
    brightness=1+(torch.rand((b,1,1,1),device=x.device)*2-1)*a
    x=(x*brightness).clamp(0,1);y=y.masked_fill(valid<.5,255)
    return x,y,valid,p


def kl_loss(logits,teacher_probability,valid,temperature):
    import torch
    import torch.nn.functional as F
    if temperature<=0:raise ValueError('Temperature must be positive.')
    logits=F.interpolate(logits.float(),size=teacher_probability.shape[-2:],mode='bilinear',align_corners=False)
    p=teacher_probability.detach().float().clamp(1e-6,1-1e-6)
    q=torch.stack([1-p,p],1).log().div(temperature).softmax(1)
    point=F.kl_div((logits/temperature).log_softmax(1),q,reduction='none').sum(1)*temperature**2
    return (point*valid).sum()/valid.sum().clamp_min(1)


def probabilities(net,images,c):
    import torch
    import torch.nn.functional as F
    net.eval();device=next(net.parameters()).device;batch=int(c['training']['inference_batch'])
    for start in range(0,len(images),batch):
        current=images[start:start+batch];arrays=[np.repeat(load_image(i['image_path'])[None],3,axis=0).astype(np.float32)/255 for i in current]
        x=torch.from_numpy(np.stack(arrays)).to(device)
        with torch.inference_mode():
            raw=net(x)
            if isinstance(raw,(list,tuple)):raw=raw[0]
            if raw.ndim!=4 or raw.shape[1]!=2:raise RuntimeError('Student did not return Bx2xHxW logits.')
            p=F.interpolate(raw.float(),size=x.shape[-2:],mode='bilinear',align_corners=False).softmax(1)[:,1].cpu().numpy()
        for image,prob in zip(current,p):yield image,prob.astype(np.float32)


def evaluate(net,images,surfaces,c):
    rows=[]
    for image,p in probabilities(net,images,c):rows.extend(records_for_image(image,p,surfaces))
    return summarize(rows),rows


def save_torch(p,payload):
    import torch
    b=io.BytesIO();torch.save(payload,b);atomic_bytes(p,b.getvalue())


def train(spec,c,data):
    import torch
    from torch.utils.data import DataLoader
    from ultralytics.utils.loss import SemanticSegmentationLoss
    t=c['training'];folder=Path(spec['directory']);folder.mkdir(parents=True,exist_ok=True)
    device=torch.device('cpu' if t['device']=='cpu' else 'cuda:0');seed=int(spec['seed'])
    seed_everything(seed);torch.set_num_threads(t['cpu_threads'])
    net=build_model(c).to(device)
    pretrain=spec['kind']=='pretrain';images=data['acta'] if pretrain else [r for r in data['images'] if r['sample_id'] in spec['train_ids']]
    val=[] if pretrain else [r for r in data['images'] if r['sample_id'] in spec['validation_ids']]
    init=spec.get('initialization')
    if init:
        if file_hash(init['checkpoint'])!=init['checkpoint_sha256']:raise RuntimeError('Shared pretraining checkpoint changed.')
        state=torch.load(init['checkpoint'],map_location='cpu',weights_only=False)
        net.load_state_dict(state['state_dict'],strict=True)
    for p in net.parameters():p.requires_grad_(True)
    h=spec['hyperparameters'];optimizer=torch.optim.AdamW(net.parameters(),lr=h['lr'],weight_decay=h['weight_decay'])
    epochs=int(h['epochs']);scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda e:((1+math.cos(math.pi*min(e,epochs)/epochs))/2)*(1-t['lrf'])+t['lrf'])
    scaler=torch.amp.GradScaler('cuda',enabled=t['amp'] and device.type=='cuda')
    criterion=SemanticSegmentationLoss(net)
    last=folder/'last.pt';best=folder/'best.pt';start=0;best_pair=(-math.inf,)*3;best_epoch=0;steps=0
    if last.exists():
        ck=torch.load(last,map_location='cpu',weights_only=False)
        if ck['signature']!=spec['signature']:raise RuntimeError('Student checkpoint signature mismatch.')
        net.load_state_dict(ck['state_dict'],strict=True);optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler']);scaler.load_state_dict(ck['scaler'])
        start=ck['epoch'];best_pair=tuple(ck['best_pair']);best_epoch=ck['best_epoch'];steps=ck['steps']
        if best_epoch==start and (not best.exists() or read_best_epoch(best)!=best_epoch):
            save_torch(best,{k:ck[k] for k in ('state_dict','epoch','signature','student_model')})
        if ck.get('training_finished'):start=epochs
    teacher_maps=spec.get('teacher_maps',{})
    if spec.get('distillation_weight',0)>0 and set(teacher_maps)!=set(spec['train_ids']):raise RuntimeError('Teacher targets must cover exactly the student TRAIN partition.')
    ds=ImageDataset(images,teacher_maps);timer=time.monotonic();latest_metrics={};training_finished=False
    for epoch in range(start,epochs):
        check_pause(c['output_dir']);seed_everything(seed+1009*epoch)
        gen=torch.Generator().manual_seed(seed+1009*epoch)
        loader=DataLoader(ds,batch_size=t['batch_size'],shuffle=True,num_workers=t['workers'],pin_memory=device.type=='cuda',generator=gen,persistent_workers=False)
        net.train();losses=[]
        for x,y,v,p in loader:
            x,y,v,p=[a.to(device,non_blocking=True) for a in (x,y,v,p)];x,y,v,p=augment(x,y,v,p,c)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=t['amp'] and device.type=='cuda'):
                raw=net(x);base,_=criterion(raw,{'semantic_mask':y})
                supervised=base/x.shape[0]
                kd=kl_loss(raw[0] if isinstance(raw,tuple) else raw,p,v,t['temperature']) if teacher_maps else supervised.new_zeros(())
                loss=supervised+float(spec.get('distillation_weight',0))*kd
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite student loss. No automatic learning-rate/batch-size change.')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(),t['gradient_clip'])
            before=scaler.get_scale();scaler.step(optimizer);scaler.update()
            steps+=int(scaler.get_scale()>=before)
            losses.append([float(loss.detach()),float(supervised.detach()),float(kd.detach())])
        scheduler.step();e=epoch+1;improved=False
        if pretrain:
            improved=True;best_epoch=e;best_pair=(float(e),0.,0.)
        elif e==1 or e%t['validate_every']==0 or e==epochs:
            latest_metrics,rows=evaluate(net,val,data['surfaces'],c);pair=objective(latest_metrics)
            if not math.isfinite(pair[0]):raise RuntimeError('Development-validation surface AUROC is undefined.')
            if pair>best_pair:best_pair=pair;best_epoch=e;improved=True
            write_csv(folder/'latest_validation_surfaces.csv',rows)
        training_finished=(e==epochs or (not pretrain and e>=t['minimum_epochs'] and e-best_epoch>=t['patience']))
        state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        if any(v.is_floating_point() and not torch.isfinite(v).all() for v in state.values()):raise FloatingPointError('Nonfinite student parameters; checkpoint rejected.')
        ck=dict(state_dict=state,epoch=e,signature=spec['signature'],student_model=c['student_model'],optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict(),best_pair=best_pair,best_epoch=best_epoch,steps=steps,training_finished=training_finished)
        save_torch(last,ck)
        if improved:save_torch(best,{k:ck[k] for k in ('state_dict','epoch','signature','student_model')})
        means=np.mean(losses,axis=0)
        status=dict(task_id=spec['task_id'],strategy=spec['strategy'],kind=spec['kind'],seed=seed,fold=spec['fold'],epoch=e,max_epochs=epochs,
            best_epoch=best_epoch,seconds=time.monotonic()-timer,optimizer_steps=steps,loss=float(means[0]),supervised_loss=float(means[1]),
            distillation_loss=float(means[2]),learning_rate=optimizer.param_groups[0]['lr'],metrics=latest_metrics)
        atomic_json(folder/'current.json',status);atomic_json(folder/'epochs'/f'{e:04}.json',status)
        event(c['output_dir'],'student_epoch',**status);check_pause(c['output_dir'])
        if training_finished:break
    if not best.is_file():raise RuntimeError('No student best checkpoint exists.')
    ck=torch.load(best,map_location='cpu',weights_only=False);net.load_state_dict(ck['state_dict'],strict=True)
    metrics=None
    if not pretrain:
        metrics,rows=evaluate(net,val,data['surfaces'],c);write_csv(folder/'validation_surfaces.csv',rows)
    result=dict(checkpoint=str(best),checkpoint_sha256=file_hash(best),best_epoch=ck['epoch'],epochs_completed=read_best_epoch(last),metrics=metrics,spec=spec)
    del net
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return result


def read_best_epoch(path):
    import torch
    return int(torch.load(path,map_location='cpu',weights_only=False)['epoch'])


def predict(result,images,folder,c):
    import torch
    if file_hash(result['checkpoint'])!=result['checkpoint_sha256']:raise RuntimeError('Student checkpoint changed.')
    seed_everything(result['spec']['seed']);net=build_model(c)
    ck=torch.load(result['checkpoint'],map_location='cpu',weights_only=False);net.load_state_dict(ck['state_dict'],strict=True)
    net.to('cpu' if c['training']['device']=='cpu' else 'cuda:0');folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    paths={};todo=[]
    for i in images:
        p=folder/(i['sample_id']+'.npy');paths[i['sample_id']]=str(p)
        if not p.exists():todo.append(i)
    for image,p in probabilities(net,todo,c):
        b=io.BytesIO();np.save(b,p,allow_pickle=False);atomic_bytes(paths[image['sample_id']],b.getvalue())
    del net
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return paths
