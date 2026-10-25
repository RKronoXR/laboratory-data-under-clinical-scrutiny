#!/usr/bin/env python3
"""Run ONE selected model from a portable MODELS.zip bundle."""
from __future__ import annotations
import argparse,io,json,os
from pathlib import Path
import numpy as np
from PIL import Image
from actakd.io import read_json,file_hash,atomic_json,atomic_bytes
from actakd.data import letterbox,restore_probability
from actakd._source import _radiograph


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--bundle',type=Path,default=Path('.'));a.add_argument('--model',required=True)
    a.add_argument('--image',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--threshold',choices=['fixed','youden'],default='fixed');a.add_argument('--device',default='0')
    args=a.parse_args()
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        a.error('Prediction output must be a new or empty directory. Existing predictions are never overwritten.')
    folder=args.bundle/'models'/args.model;m=read_json(folder/'model.json')
    if m['role']=='shared_pretraining_module':raise ValueError('P is a pretraining module, not a clinical endpoint model.')
    if file_hash(folder/m['weights'])!=m['weights_sha256']:raise RuntimeError('Weights checksum mismatch.')
    os.environ['CUDA_VISIBLE_DEVICES']='' if args.device=='cpu' else args.device
    os.environ['nnUNet_compile']='false'
    import torch
    device=torch.device('cpu' if args.device=='cpu' else 'cuda:0')
    image,_=_radiograph(args.image);x,geometry=letterbox(image,m['input_size'])
    ck=torch.load(folder/m['weights'],map_location='cpu',weights_only=False)
    if m['architecture']=='yolo26_sem':
        from actakd.student import build_model
        net=build_model({'student_model':m['student_model']});net.load_state_dict(ck['state_dict'],strict=True);net.to(device).eval()
        xx=torch.from_numpy(np.repeat(x[None,None],3,axis=1).astype(np.float32)/255).to(device)
        with torch.inference_mode():
            logits=net(xx)
            if isinstance(logits,(list,tuple)):logits=logits[0]
            p=torch.nn.functional.interpolate(logits.float(),size=x.shape,mode='bilinear',align_corners=False).softmax(1)[0,1].cpu().numpy()
    else:
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        pm=PlansManager(str(folder/m['plans']));cm=pm.get_configuration('2d');dataset=read_json(folder/m['dataset'])
        net=get_network_from_plans(cm.network_arch_class_name,cm.network_arch_init_kwargs,cm.network_arch_init_kwargs_req_import,1,2,allow_init=True,deep_supervision=False)
        net.load_state_dict(ck['state_dict'],strict=True);net.to(device).eval()
        pred=nnUNetPredictor(tile_step_size=.5,use_gaussian=True,use_mirroring=m['mirroring'],device=device,perform_everything_on_device=device.type=='cuda',allow_tqdm=False)
        pred.manual_initialization(net,pm,cm,[net.state_dict()],dataset,'nnUNetTrainerACTAKD',m['inference_allowed_mirroring_axes'])
        with torch.inference_mode():_,pr=pred.predict_single_npy_array(x[None,None].astype(np.float32),{'spacing':(999,1,1)},save_or_return_probabilities=True)
        p=np.asarray(pr)[1,0] if np.asarray(pr).ndim==4 else np.asarray(pr)[1]
    p=restore_probability(p,geometry);threshold=m['thresholds'][args.threshold]
    args.output.mkdir(parents=True,exist_ok=True)
    np.save(args.output/'probabilities.npy',p.astype(np.float32),allow_pickle=False)
    Image.fromarray((p>=threshold).astype(np.uint8)*255).save(args.output/'mask.png')
    atomic_json(args.output/'prediction.json',dict(model_id=m['model_id'],threshold=threshold,threshold_setting=args.threshold,
        threshold_source='original DCBR development validation' if args.threshold=='youden' else 'prespecified',input_image=str(args.image),input_sha256=file_hash(args.image),
        native_shape=list(p.shape),weights_sha256=m['weights_sha256'],teachers_used_for_student_inference=False))
    print(args.output.resolve())

if __name__=='__main__':main()
