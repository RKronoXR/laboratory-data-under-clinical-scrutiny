"""Plan-defined nnU-Net 2D, full-frame inputs and an aligned Dice+CE+ROI-KL loop.

This is a study-specific training loop, not the stock nnUNetv2_train command.
All six arms use the same network, normalization, augmentation and inference.
The full clinical image is retained; no lesion-centred patch sampling is used.
"""
from __future__ import annotations
import io,math,random,time
from pathlib import Path
import numpy as np
from .data import load_image
from .io import atomic_bytes,atomic_json,read_json,file_hash,event,check_pause,write_csv,digest
from .metrics import summarize,records_for_image,objective

def seed_everything(seed):
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True

def build_model(c):
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    pm=PlansManager(read_json(c['frozen_plans']));conf=pm.get_configuration('2d')
    return get_network_from_plans(conf.network_arch_class_name,conf.network_arch_init_kwargs,
        conf.network_arch_init_kwargs_req_import,1,2,allow_init=True,
        deep_supervision=bool(c['training']['deep_supervision']))

def valid_mask(geometry):
    s=int(geometry['size']);valid=np.zeros((s,s),np.float32)
    y,x,h,w=[int(geometry[k]) for k in ('top','left','resized_height','resized_width')]
    valid[y:y+h,x:x+w]=1
    return valid

class ImageDataset:
    def __init__(self,images,teacher_maps=None):self.images=images;self.maps=teacher_maps or {}
    def __len__(self):return len(self.images)
    def __getitem__(self,index):
        import torch
        image=self.images[index];im=load_image(image['image_path']).astype(np.float32)
        mask=load_image(image['mask_path']).astype(np.int64);valid=valid_mask(image['geometry'])
        mask[valid<.5]=255
        p=np.zeros(im.shape,np.float32);roi=np.zeros(im.shape,np.float32)
        if image['sample_id'] in self.maps:
            source=self.maps[image['sample_id']]
            p=np.load(source['probability'],allow_pickle=False).astype(np.float32)
            roi=load_image(source['roi_mask']).astype(np.float32)
            if p.shape!=im.shape or roi.shape!=im.shape or not np.isfinite(p).all() or p.min()<0 or p.max()>1:
                raise ValueError('Teacher/student target shape or probability mismatch.')
            if not np.isin(roi,[0,1]).all():raise ValueError('ROI mask is not binary.')
            if np.any((roi>.5)&(valid<.5)):raise ValueError('Teacher KD support enters image padding.')
        return tuple(torch.from_numpy(a) for a in (im[None],mask,valid,p,roi))

def normalize(x,valid,c):
    import torch
    name=c['normalization_scheme']
    if name=='NoNormalization':return x
    if name!='ZScoreNormalization':raise ValueError('Unsupported inherited normalization.')
    if c['normalization_use_mask']:
        v=valid[:,None];n=v.sum((2,3),keepdim=True).clamp_min(1)
        mean=(x*v).sum((2,3),keepdim=True)/n
        std=((x-mean).square()*v).sum((2,3),keepdim=True).div(n).sqrt().clamp_min(1e-8)
        return (x-mean).div(std)*v
    mean=x.mean((2,3),keepdim=True)
    std=x.std((2,3),keepdim=True,correction=0).clamp_min(1e-8)
    return (x-mean)/std

def augment(x,y,valid,p,roi,c):
    import torch
    import torch.nn.functional as F
    b=x.shape[0];a=(torch.rand(b,device=x.device)*2-1)*c['training']['rotate_degrees']*math.pi/180
    theta=torch.zeros((b,2,3),device=x.device,dtype=x.dtype)
    theta[:,0,0]=a.cos();theta[:,0,1]=-a.sin();theta[:,1,0]=a.sin();theta[:,1,1]=a.cos()
    grid=F.affine_grid(theta,x.shape,align_corners=False)
    x=F.grid_sample(x,grid,mode='bilinear',align_corners=False)
    y=F.grid_sample(y[:,None].float(),grid,mode='nearest',align_corners=False)[:,0].long()
    valid=F.grid_sample(valid[:,None],grid,mode='nearest',align_corners=False)[:,0]
    p=F.grid_sample(p[:,None],grid,mode='bilinear',align_corners=False)[:,0]
    roi=F.grid_sample(roi[:,None],grid,mode='nearest',align_corners=False)[:,0]*valid
    frac=c['training']['intensity_fraction']
    x=(x*(1+(torch.rand((b,1,1,1),device=x.device)*2-1)*frac)).clamp(0,255)
    y=y.masked_fill(valid<.5,255)
    return normalize(x,valid,c),y,valid,p,roi

def kl_loss(logits,teacher_probability,roi,temperature):
    import torch
    import torch.nn.functional as F
    if temperature<=0:raise ValueError('Temperature must be positive.')
    logits=F.interpolate(logits.float(),size=teacher_probability.shape[-2:],mode='bilinear',align_corners=False)
    p=teacher_probability.detach().float().clamp(1e-6,1-1e-6)
    q=torch.stack((1-p,p),1).log().div(temperature).softmax(1)
    point=F.kl_div((logits/temperature).log_softmax(1),q,reduction='none').sum(1)*temperature**2
    if not torch.any(roi>.5):raise RuntimeError('KD batch contains no anatomical ROI support.')
    return (point*roi).sum()/roi.sum().clamp_min(1)

def build_loss(c):
    from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
    from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
    return DC_and_CE_loss(dict(batch_dice=c['batch_dice'],smooth=1e-5,do_bg=False,ddp=False),{},
                          weight_ce=1,weight_dice=1,ignore_label=255,dice_class=MemoryEfficientSoftDiceLoss)

def supervised_loss(raw,y,criterion):
    import torch.nn.functional as F
    values=list(raw) if isinstance(raw,(list,tuple)) else [raw]
    weights=np.array([1/(2**i) for i in range(len(values))],np.float32)
    if len(values)>1:weights[-1]=0
    weights/=weights.sum();loss=values[0].sum()*0
    for value,weight in zip(values,weights):
        if weight==0:continue
        target=F.interpolate(y[:,None].float(),size=value.shape[-2:],mode='nearest').long()
        loss=loss+float(weight)*criterion(value.float(),target)
    return loss

def save_torch(path,payload):
    import torch
    buf=io.BytesIO();torch.save(payload,buf);atomic_bytes(path,buf.getvalue())

def probabilities(net,images,c):
    import torch
    import torch.nn.functional as F
    net.eval();device=next(net.parameters()).device
    for start in range(0,len(images),c['training']['inference_batch']):
        part=images[start:start+c['training']['inference_batch']]
        x=torch.from_numpy(np.stack([load_image(i['image_path'])[None].astype(np.float32) for i in part])).to(device)
        v=torch.from_numpy(np.stack([valid_mask(i['geometry']) for i in part])).to(device)
        with torch.inference_mode():
            raw=net(normalize(x,v,c));logits=raw[0] if isinstance(raw,(list,tuple)) else raw
            if logits.ndim!=4 or logits.shape[1]!=2:raise RuntimeError('Expected Bx2xHxW nnU-Net semantic logits.')
            p=F.interpolate(logits.float(),size=x.shape[-2:],mode='bilinear',align_corners=False).softmax(1)[:,1].cpu().numpy()
        for image,prob in zip(part,p):
            if not np.isfinite(prob).all():raise FloatingPointError('Nonfinite student probability.')
            yield image,prob.astype(np.float32)

def evaluate(net,images,surfaces,c,task_id=None,honor_pause=True):
    rows=[]
    for index,(image,p) in enumerate(probabilities(net,images,c),1):
        rows.extend(records_for_image(image,p,surfaces))
        if task_id:event(c['output_dir'],'student_validation_image',task_id=task_id,completed=index,total=len(images))
        if honor_pause:check_pause(c['output_dir'])
    return summarize(rows),rows

def load_weights(net,initialization,c):
    import torch
    if file_hash(initialization['checkpoint'])!=initialization['checkpoint_sha256']:
        raise RuntimeError('Student initialization checkpoint changed.')
    ck=torch.load(initialization['checkpoint'],map_location='cpu',weights_only=False)
    if ck.get('architecture_signature')!=c['architecture_signature']:
        raise RuntimeError('nnU-Net stage architectures/normalization differ.')
    net.load_state_dict(ck['network_weights'],strict=True)

def train(spec,c,data):
    import torch
    from torch.utils.data import DataLoader
    t=c['training'];folder=Path(spec['directory']);folder.mkdir(parents=True,exist_ok=True)
    device=torch.device('cpu' if t['device']=='cpu' else 'cuda:0')
    seed_everything(spec['seed']);torch.set_num_threads(t['cpu_threads'])
    net=build_model(c).to(device)
    if spec.get('initialization'):load_weights(net,spec['initialization'],c)
    for p in net.parameters():p.requires_grad_(True)
    images=data['acta'] if spec['domain']=='acta' else [r for r in data['images'] if r['sample_id'] in spec['train_ids']]
    validation=[r for r in data['images'] if r['sample_id'] in spec['validation_ids']]
    maps=spec.get('teacher_maps',{})
    if spec['kd'] and set(maps)!=set(spec['train_ids']):
        raise RuntimeError('KD maps must cover exactly the original training fold.')
    if not spec['kd'] and maps:raise RuntimeError('Non-KD stage received professor targets.')
    for v in maps.values():
        if file_hash(v['probability'])!=v['probability_sha256'] or file_hash(v['roi_mask'])!=v['roi_sha256']:
            raise RuntimeError('Professor target changed before training.')
    ds=ImageDataset(images,maps);h=spec['hyperparameters'];epochs=h['epochs']
    optimizer=torch.optim.SGD(net.parameters(),lr=h['lr'],weight_decay=h.get('weight_decay',t['weight_decay']),
                             momentum=t['momentum'],nesterov=t['nesterov'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda e:max(1e-4,(1-min(e,epochs)/epochs)**.9))
    scaler=torch.amp.GradScaler('cuda',enabled=t['amp'] and device.type=='cuda')
    criterion=build_loss(c);last=folder/'last.pth';best=folder/'best.pth'
    start=0;best_pair=(-math.inf,)*3;best_epoch=0;steps=0;finished=False;latest={}
    if last.is_file():
        ck=torch.load(last,map_location='cpu',weights_only=False)
        if ck['signature']!=spec['signature']:raise RuntimeError('Resumed task signature changed.')
        net.load_state_dict(ck['network_weights'],strict=True);optimizer.load_state_dict(ck['optimizer'])
        scheduler.load_state_dict(ck['scheduler']);scaler.load_state_dict(ck['scaler'])
        start=ck['epoch'];best_pair=tuple(ck['best_pair']);best_epoch=ck['best_epoch'];steps=ck['optimizer_steps']
        finished=ck['training_finished']
        if best_epoch==start and (not best.exists() or torch.load(best,map_location='cpu',weights_only=False)['epoch']!=best_epoch):
            save_torch(best,{k:ck[k] for k in ('network_weights','epoch','signature','architecture_signature')})
        del ck
    selecting_clinical=spec['selection']=='clinical_validation'
    batch=t['acta_batch_size'] if spec['domain']=='acta' else t['batch_size']
    timer=time.monotonic()
    for epoch in range(start,epochs if not finished else start):
        check_pause(c['output_dir']);seed_everything(spec['seed']+1009*epoch)
        loader=DataLoader(ds,batch_size=batch,shuffle=True,num_workers=t['workers'],
                         pin_memory=device.type=='cuda',generator=torch.Generator().manual_seed(spec['seed']+1009*epoch))
        net.train();total=np.zeros(3,np.float64);n_batches=0
        event(c['output_dir'],'epoch_start',task_id=spec['task_id'],epoch=epoch+1,max_epochs=epochs,batches=len(loader))
        for index,(x,y,v,p,roi) in enumerate(loader,1):
            x,y,v,p,roi=[a.to(device,non_blocking=True) for a in (x,y,v,p,roi)]
            x,y,v,p,roi=augment(x,y,v,p,roi,c);optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=t['amp'] and device.type=='cuda'):
                raw=net(x);sup=supervised_loss(raw,y,criterion)
                logits=raw[0] if isinstance(raw,(list,tuple)) else raw
                kd=kl_loss(logits,p,roi,h.get('temperature',t['temperature'])) if spec['kd'] else sup.new_zeros(())
                loss=sup+h.get('distillation_weight',t['distillation_weight'])*kd
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite student loss; no automatic changes to hyperparameters.')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(),t['gradient_clip'])
            before=scaler.get_scale();scaler.step(optimizer);scaler.update();steps+=int(scaler.get_scale()>=before)
            total+=np.array([float(loss.detach()),float(sup.detach()),float(kd.detach())]);n_batches+=1
            if index==1 or index%10==0 or index==len(loader):
                progress=dict(task_id=spec['task_id'],stage=spec['stage'],domain=spec['domain'],seed=spec['seed'],fold=spec['fold'],
                    epoch=epoch+1,max_epochs=epochs,batch=index,batches=len(loader),optimizer_steps=steps,
                    loss=total[0]/n_batches,supervised_loss=total[1]/n_batches,distillation_loss=total[2]/n_batches)
                atomic_json(folder/'current.json',progress);event(c['output_dir'],'student_batch',**progress)
        if n_batches==0:raise RuntimeError('Empty training DataLoader.')
        scheduler.step();e=epoch+1;improved=False
        if not selecting_clinical:
            best_epoch=e;best_pair=(float(e),0.,0.);improved=True
        elif e==1 or e%t['validate_every']==0 or e==epochs:
            latest,rows=evaluate(net,validation,data['surfaces'],c,spec['task_id'],honor_pause=False);pair=objective(latest)
            if not math.isfinite(pair[0]):raise RuntimeError('Clinical validation AUROC is undefined.')
            if pair>best_pair:best_pair=pair;best_epoch=e;improved=True
            write_csv(folder/'latest_validation_surfaces.csv',rows)
        finished=(e==epochs or (selecting_clinical and e>=t['minimum_epochs'] and e-best_epoch>=t['patience']))
        weights={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
        if any(v.is_floating_point() and not torch.isfinite(v).all() for v in weights.values()):
            raise FloatingPointError('Nonfinite student parameters; checkpoint rejected.')
        state=dict(network_weights=weights,epoch=e,signature=spec['signature'],architecture_signature=c['architecture_signature'],
                   optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
                   best_pair=best_pair,best_epoch=best_epoch,optimizer_steps=steps,training_finished=finished)
        save_torch(last,state)
        if improved:save_torch(best,{k:state[k] for k in ('network_weights','epoch','signature','architecture_signature')})
        status=dict(task_id=spec['task_id'],stage=spec['stage'],domain=spec['domain'],seed=spec['seed'],fold=spec['fold'],
                    epoch=e,max_epochs=epochs,best_epoch=best_epoch,optimizer_steps=steps,seconds=time.monotonic()-timer,
                    loss=total[0]/n_batches,supervised_loss=total[1]/n_batches,distillation_loss=total[2]/n_batches,
                    metrics=latest,learning_rate=optimizer.param_groups[0]['lr'])
        atomic_json(folder/'current.json',status);atomic_json(folder/'epochs'/f'{e:04d}.json',status)
        event(c['output_dir'],'student_epoch',**status);check_pause(c['output_dir'])
        del state,weights
        if finished:break
    if not best.is_file():raise RuntimeError('No selected student checkpoint produced.')
    ck=torch.load(best,map_location='cpu',weights_only=False);net.load_state_dict(ck['network_weights'],strict=True)
    metrics=None
    if spec['stage']!='V':
        metrics,rows=evaluate(net,validation,data['surfaces'],c,spec['task_id']);write_csv(folder/'validation_surfaces.csv',rows)
    end=torch.load(last,map_location='cpu',weights_only=False)
    result=dict(signature=spec['signature'],checkpoint=str(best),checkpoint_sha256=file_hash(best),
                last_checkpoint=str(last),last_checkpoint_sha256=file_hash(last),best_epoch=ck['epoch'],
                epochs_completed=end['epoch'],optimizer_steps=end['optimizer_steps'],
                metrics=metrics,spec=spec,architecture_signature=c['architecture_signature'],
                selection=spec['selection'],optimizer_reset_at_stage_start=True)
    del net,ck,end
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return result

def predict(c,model,images,folder,task_id):
    import torch
    seed_everything(model['spec']['seed']);net=build_model(c)
    load_weights(net,model,c);net.to('cpu' if c['training']['device']=='cpu' else 'cuda:0')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);paths={}
    for index,(image,p) in enumerate(probabilities(net,images,c),1):
        check_pause(c['output_dir']);path=folder/f'{image["sample_id"]}.npy'
        b=io.BytesIO();np.save(b,p,allow_pickle=False);atomic_bytes(path,b.getvalue())
        paths[image['sample_id']]=str(path)
        event(c['output_dir'],'student_prediction_image',task_id=task_id,completed=index,total=len(images))
    del net
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return paths
