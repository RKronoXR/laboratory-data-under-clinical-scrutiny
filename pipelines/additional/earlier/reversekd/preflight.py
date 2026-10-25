"""Automatic real forward/backward checks on Spark before any study fit."""
from pathlib import Path
import numpy as np
from .io import event,atomic_json,digest

def preflight(c,data,teacher,teacher_maps,folder):
    import torch
    from . import student,professor
    device=torch.device('cpu' if c['training']['device']=='cpu' else 'cuda:0')
    student.seed_everything(17);torch.set_num_threads(c['training']['cpu_threads'])
    net=student.build_model(c).to(device);criterion=student.build_loss(c)
    states={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
    model=professor.build_model(c,teacher,device)
    with torch.inference_mode():
        x=torch.from_numpy(np.repeat(student.load_image(data['acta'][0]['image_path'])[None,None],3,axis=1).astype(np.float32)/255).to(device)
        raw=model(x);raw=raw[0] if isinstance(raw,(list,tuple)) else raw
        if raw.ndim!=4 or raw.shape[1]!=2 or not torch.isfinite(raw).all():raise RuntimeError('Teacher forward check failed.')
    if any(p.requires_grad for p in model.parameters()):raise RuntimeError('Teacher is not frozen.')
    del model
    tested=[]
    sid=next(iter(teacher_maps));clinical=next(i for i in data['images'] if i['sample_id']==sid)
    for domain,images,maps in [('dcbr',[clinical],teacher_maps),('acta',data['acta'][:1],{})]:
        net.load_state_dict(states,strict=True);net.train()
        ds=student.ImageDataset(images,maps);batch=[a[None].to(device) for a in ds[0]]
        x,y,v,p,roi=student.augment(*batch,c)
        optimizer=torch.optim.SGD(net.parameters(),lr=.001,momentum=.99)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type,enabled=c['training']['amp'] and device.type=='cuda'):
            raw=net(x);sup=student.supervised_loss(raw,y,criterion)
            logits=raw[0] if isinstance(raw,(list,tuple)) else raw
            kd=student.kl_loss(logits,p,roi,c['training']['temperature']) if domain=='dcbr' else sup.new_zeros(())
            loss=sup+c['training']['distillation_weight']*kd
        if not torch.isfinite(loss):raise FloatingPointError('Preflight loss is nonfinite.')
        loss.backward()
        if not any(p.grad is not None and torch.any(p.grad!=0) for p in net.parameters()):
            raise RuntimeError('Preflight produced no nonzero student gradient.')
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):
            raise RuntimeError('Preflight produced a nonfinite gradient.')
        optimizer.step();tested.append(dict(domain=domain,input_shape=list(x.shape),loss=float(loss.detach()),
            supervised_loss=float(sup.detach()),distillation_loss=float(kd.detach()),backward_passed=True))
        event(c['output_dir'],'preflight_domain_passed',domain=domain,input_size=x.shape[-1])
        # Exercise the same checkpoint writer/loader used between study stages.
        checkpoint=Path(folder)/f'{domain}_roundtrip.pth'
        student.save_torch(checkpoint,dict(network_weights={k:v.detach().cpu().clone() for k,v in net.state_dict().items()},
            architecture_signature=c['architecture_signature'],optimizer=optimizer.state_dict()))
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        for name,value in net.state_dict().items():
            if not torch.equal(value.detach().cpu(),saved['network_weights'][name]):
                raise RuntimeError('Checkpoint tensor roundtrip failed.')
        net.load_state_dict(saved['network_weights'],strict=True)
        optimizer.load_state_dict(saved['optimizer']);del saved
    # Prove ROI KL does not transmit teacher targets outside the ROI.
    logits=torch.randn((1,2,16,16),device=device,requires_grad=True)
    q=torch.full((1,16,16),.7,device=device);roi=torch.zeros_like(q);roi[:,4:12,4:12]=1
    kd=student.kl_loss(logits,q,roi,1.);kd.backward()
    outside=(1-roi)[:,None].expand_as(logits)
    if torch.any(logits.grad[outside.bool()]!=0):raise RuntimeError('KD gradient escaped proximal ROI support.')
    if not torch.any(logits.grad[(1-outside).bool()]!=0):raise RuntimeError('KD has no gradient inside the proximal ROI.')
    del net,states,optimizer
    if torch.cuda.is_available():torch.cuda.empty_cache()
    result=dict(real_framework_preflight=True,teacher_frozen=True,student_domains=tested,
                kl_gradient_inside_roi_only=True,checkpoint_roundtrip_passed=True,study_weights_modified=False)
    atomic_json(Path(folder)/'PREFLIGHT.json',result);return result
