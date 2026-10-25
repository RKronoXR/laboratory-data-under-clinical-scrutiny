"""Historical MONAI hyperparameters, transforms and loss; full-image input.
Validation positive Dice uses the paired proximal surfaces, including misses.
"""
import io, time, random
from pathlib import Path
import numpy as np
from PIL import Image
from .common import *

def build(cfg,device):
    from monai.networks.nets import SegResNet
    m=cfg['monai']
    return SegResNet(spatial_dims=2,in_channels=1,out_channels=1,init_filters=int(m['init_filters']),
                    blocks_down=tuple(m['blocks_down']),blocks_up=tuple(m['blocks_up']),norm='BATCH').to(device)

def transforms(train):
    import torch
    from monai.data import PILReader
    from monai.transforms import (Compose,LoadImaged,EnsureChannelFirstd,ScaleIntensityd,RandFlipd,
        RandAffined,RandScaleIntensityd,RandShiftIntensityd,EnsureTyped)
    items=[LoadImaged(keys=['image','label'],reader=PILReader(reverse_indexing=True),image_only=True),
           EnsureChannelFirstd(keys=['image','label']),ScaleIntensityd(keys=['image'],minv=0.,maxv=1.)]
    if train:
        items += [RandFlipd(keys=['image','label'],prob=.5,spatial_axis=1),
                  RandAffined(keys=['image','label'],prob=.30,rotate_range=(np.deg2rad(5.),),translate_range=(8,8),
                              scale_range=(.05,.05),mode=('bilinear','nearest'),padding_mode='zeros'),
                  RandScaleIntensityd(keys=['image'],factors=.10,prob=.30),RandShiftIntensityd(keys=['image'],offsets=.10,prob=.30)]
    return Compose(items+[EnsureTyped(keys=['image','label'],dtype=torch.float32)])

def rng_nodes(transform):
    # Save all MONAI Randomizable nodes (including nested affine generators).
    seen=set();result=[]
    def visit(obj):
        if id(obj) in seen: return
        seen.add(id(obj))
        if hasattr(obj,'R') and hasattr(obj.R,'get_state'): result.append(obj)
        if hasattr(obj,'__dict__'):
            for val in vars(obj).values():
                if isinstance(val,(list,tuple)):
                    for x in val:
                        if hasattr(x,'set_random_state') or hasattr(x,'transforms'):visit(x)
                elif hasattr(val,'set_random_state') or hasattr(val,'transforms'):visit(val)
    visit(transform);return result

def save_torch(path,payload):
    import torch
    buffer=io.BytesIO();torch.save(payload,buffer);atomic_bytes(path,buffer.getvalue())

def early_stop(epoch,stale,cfg):
    return epoch >= cfg['training']['max_epochs'] or stale >= cfg['training']['patience']

def probabilities(net,images,device):
    import torch
    from dcbr_reference.inference import _monai_input
    from dcbr_compare.evaluation import restore_probability
    net.eval()
    with torch.inference_mode():
        for row in images.to_dict('records'):
            x=_monai_input(row['image_path']).unsqueeze(0).to(device)
            p=net(x).sigmoid()[0,0].detach().cpu().numpy().T.copy()
            yield row,restore_probability(p,row)

def validation(net,images,surfaces,device,project,seed,fold,epoch,save=None):
    from dcbr_compare.baseline import project_full_to_crop,meta
    from dcbr_compare.metrics import surface_record
    groups={str(k):v for k,v in surfaces.groupby('bitewing_id')}
    records=[];whole=[]
    for i,(row,p) in enumerate(probabilities(net,images,device),1):
        bid=str(row['bitewing_id'])
        if save:
            buffer=io.BytesIO();np.save(buffer,p.astype(np.float32),allow_pickle=False)
            atomic_bytes(save/'evaluation/probabilities'/f'{bid}.npy',buffer.getvalue())
        for r in groups.get(bid,surfaces.iloc[:0]).to_dict('records'):
            with Image.open(mask_path(r)) as im: ref=np.asarray(im)>0
            rec=surface_record(ref,project_full_to_crop(p,r),.5)
            rec.update(meta(r,seed,fold,'full'),threshold=.5,threshold_name='fixed');records.append(rec)
        if save:
            with Image.open(row['native_mask_path']) as im: ref=np.asarray(im)>0
            wr=surface_record(ref,p,.5);wr.update(seed=seed,fold=fold,bitewing_id=bid,analysis='whole_image_descriptive');whole.append(wr)
        activity(project,state='validation' if save is None else 'inference_development',seed=seed,fold=fold,epoch=epoch,
                 processed=i,total=len(images),unit='bitewings')
    if len(records)!=len(surfaces): raise RuntimeError('Validación incompleta de superficies')
    vals=[r['dice_positive'] for r in records if r['reference_positive']]
    if not vals: raise RuntimeError('Fold sin superficies positivas para seleccionar checkpoint')
    if save:
        write_csv(save/'test_surfaces.csv',records);write_csv(save/'full_image_results.csv',whole)
    return float(np.mean(vals))

def train(project,seed,fold):
    import torch
    from monai.data import DataLoader,Dataset
    from monai.losses import DiceFocalLoss
    from monai.utils import set_determinism
    project=Path(project);cfg=effective(project);tc=cfg['training'];signature=read_json(out(project)/'protocol.json')['signature']
    directory=folder(project,seed,fold)/'training';directory.mkdir(parents=True,exist_ok=True)
    if (directory/'result.json').exists():
        result=read_json(directory/'result.json')
        if result['signature']!=signature or file_hash(result['checkpoint'])!=result['checkpoint_sha256']:raise RuntimeError('Checkpoint MONAI cambiado')
        return result
    if not torch.cuda.is_available(): raise RuntimeError('CUDA no disponible. No se inicia entrenamiento en CPU.')
    device=torch.device('cuda:0');set_determinism(seed=seed)
    images,surfaces=frames(project);training=images[images.fold.astype(int)!=fold];val=images[images.fold.astype(int)==fold]
    surfaces=surfaces[surfaces.fold.astype(int)==fold]
    tr=transforms(True)
    records=[dict(image=r['image_path'],label=r['mask_path']) for r in training.to_dict('records')]
    loader=DataLoader(Dataset(records,tr),batch_size=int(tc['batch_size']),shuffle=True,num_workers=int(tc['num_workers']),pin_memory=True)
    net=build(cfg,device);optimizer=torch.optim.AdamW(net.parameters(),lr=float(tc['learning_rate']),weight_decay=float(tc['weight_decay']))
    scaler=torch.amp.GradScaler('cuda',enabled=True);loss_fn=DiceFocalLoss(sigmoid=True,lambda_dice=1.,lambda_focal=1.)
    best=-1.;best_epoch=0;stale=0;steps=0;start_epoch=1;elapsed=0.;history=[]
    last=directory/'last.pt';bestfile=directory/'best.pt'
    if last.exists():
        ck=torch.load(last,map_location='cpu',weights_only=False)
        if ck['signature']!=signature or ck['seed']!=seed or ck['fold']!=fold:raise RuntimeError('Reanudación incompatible')
        net.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);scaler.load_state_dict(ck['scaler'])
        best,best_epoch,stale,steps,start_epoch,elapsed=ck['best'],ck['best_epoch'],ck['stale'],ck['steps'],ck['epoch']+1,ck['elapsed']
        history=ck['history'];random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
        nodes=rng_nodes(tr)
        if len(nodes)!=len(ck['transform_rng']):raise RuntimeError('Cambió el grafo de transformaciones MONAI')
        for node,rng in zip(nodes,ck['transform_rng']):node.R.set_state(rng)
        if file_hash(bestfile)!=ck['best_sha256']:raise RuntimeError('Mejor checkpoint cambiado desde última época')
    started=time.monotonic()
    for epoch in range(start_epoch,int(tc['max_epochs'])+1):
        if history and early_stop(epoch-1,stale,cfg):break
        check_stop(project);net.train();losses=[]
        for batch_index,batch in enumerate(loader,1):
            x=batch['image'].to(device);y=(batch['label']>0).float().to(device)
            if x.shape[1]!=1:raise RuntimeError('MONAI requiere un canal gris; entrada no compatible')
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda',enabled=True):loss=loss_fn(net(x),y)
            if not torch.isfinite(loss):raise RuntimeError('Loss no finito; entrenamiento detenido')
            scaler.scale(loss).backward();scaler.step(optimizer);scaler.update();steps+=1;losses.append(float(loss.detach()))
            if batch_index==1 or batch_index%5==0 or batch_index==len(loader):
                activity(project,state='training',seed=seed,fold=fold,epoch=epoch,max_epochs=int(tc['max_epochs']),
                         batch=batch_index,batches=len(loader),optimizer_steps=steps,elapsed_seconds=elapsed+time.monotonic()-started,
                         best_epoch=best_epoch,best_metric=best,stale=stale,patience=int(tc['patience']))
        metric=validation(net,val,surfaces,device,project,seed,fold,epoch)
        improved=metric>best+float(tc['min_delta'])
        if improved:
            best,best_epoch,stale=metric,epoch,0
            save_torch(bestfile,dict(model=net.state_dict(),epoch=epoch,metric=metric,signature=signature,seed=seed,fold=fold))
        else:stale+=1
        row=dict(epoch=epoch,train_loss=float(np.mean(losses)),positive_dice=metric,best=best,best_epoch=best_epoch,
                 stale=stale,optimizer_steps=steps,elapsed_seconds=elapsed+time.monotonic()-started)
        history.append(row)
        payload=dict(model=net.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),signature=signature,
                     seed=seed,fold=fold,epoch=epoch,best=best,best_epoch=best_epoch,stale=stale,steps=steps,elapsed=row['elapsed_seconds'],
                     history=history,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all(),transform_rng=[n.R.get_state() for n in rng_nodes(tr)],best_sha256=file_hash(bestfile))
        save_torch(last,payload);write_csv(directory/'history.csv',history);atomic_json(directory/'epochs'/f'{epoch:04d}.json',row)
        activity(project,state='epoch_saved',seed=seed,fold=fold,max_epochs=int(tc['max_epochs']),**row)
        print(f'MONAI seed={seed} fold={fold} epoch={epoch}/{tc["max_epochs"]} Dice+={metric:.4f} stale={stale}/{tc["patience"]}',flush=True)
        check_stop(project)
        if early_stop(epoch,stale,cfg):break
    if not history:raise RuntimeError('No hay épocas de entrenamiento')
    write_csv(directory/'history.csv',history)
    for saved_row in history:
        path=directory/'epochs'/f"{saved_row['epoch']:04d}.json"
        if not path.exists():atomic_json(path,saved_row)
    result=dict(signature=signature,checkpoint=str(bestfile),checkpoint_sha256=file_hash(bestfile),best_epoch=best_epoch,
                epochs_completed=history[-1]['epoch'],image_size=cfg['project']['image_size'],selection_metric='validation_positive_surface_Dice',
                best_metric=best,early_stopping=True,early_stopping_triggered=history[-1]['epoch']<tc['max_epochs'],optimizer_steps=steps,
                seconds=elapsed+time.monotonic()-started,architecture='monai_segresnet')
    atomic_json(directory/'result.json',result);return result

def evaluate(project,seed,fold,result):
    import torch
    from dcbr_compare.metrics import summarize_records
    project=Path(project);cfg=effective(project);net=build(cfg,torch.device('cuda:0'))
    ck=torch.load(result['checkpoint'],map_location='cpu',weights_only=False);net.load_state_dict(ck['model'])
    images,surfaces=frames(project);f=folder(project,seed,fold)
    validation(net,images[images.fold.astype(int)==fold],surfaces[surfaces.fold.astype(int)==fold],torch.device('cuda:0'),project,seed,fold,result['epochs_completed'],save=f)
    from dcbr_reference.metrics import read_csv
    atomic_json(f/'complete.json',dict(protocol_signature=result['signature'],seed=seed,fold=fold,condition='full',architecture='monai_segresnet',
                new_training=True,hpo=False,trained=result,summary=summarize_records(read_csv(f/'test_surfaces.csv')),
                artifacts={str(p):file_hash(p) for p in [f/'test_surfaces.csv',*sorted((f/'evaluation/probabilities').glob('*.npy'))]}))
