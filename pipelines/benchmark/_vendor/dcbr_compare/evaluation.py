"""Native-coordinate probabilities for whole bitewings.

All geometric transforms use input geometry, never reference lesion location.
Reference overlap is used only for the explicitly labeled localization endpoint.
"""
import numpy as np
from PIL import Image

def restore_probability(prob, row):
    """Invert the recorded letterbox, then restore the original pixel grid."""
    left, top = int(row['lb_left']), int(row['lb_top'])
    width, height = int(row['lb_width']), int(row['lb_height'])
    native_w = int(row.get('native_width', row.get('width', 0)))
    native_h = int(row.get('native_height', row.get('height', 0)))
    if min(width,height,native_w,native_h)<1: raise ValueError('Invalid image geometry')
    if left<0 or top<0 or left+width>prob.shape[1] or top+height>prob.shape[0]:
        raise ValueError('Letterbox exceeds prediction dimensions')
    cut=np.ascontiguousarray(prob[top:top+height,left:left+width],dtype=np.float32)
    restored=np.asarray(Image.fromarray(cut).resize((native_w,native_h), Image.Resampling.BILINEAR))
    if not np.isfinite(restored).all() or restored.min()<0 or restored.max()>1:
        raise ValueError('Invalid native probabilities')
    return restored

def probability_maps(net, images, cfg):
    import torch
    device=next(net.parameters()).device
    net.eval()
    records=images.to_dict('records')
    batch=max(1,int(cfg['batch']))
    for start in range(0,len(records),batch):
        chunk=records[start:start+batch]
        arr=[]
        for row in chunk:
            with Image.open(row['image_path']) as im:
                arr.append(np.asarray(im.convert('RGB'),dtype=np.float32).transpose(2,0,1)/255)
        x=torch.from_numpy(np.stack(arr)).to(device)
        with torch.inference_mode():
            raw=net(x)
            if isinstance(raw,(tuple,list)): raw=raw[0]
            if not torch.is_tensor(raw) or raw.ndim!=4 or raw.shape[1]!=2:
                raise RuntimeError('Expected semantic B x 2 x H x W logits; hard masks cannot produce ROC/PR.')
            logits=torch.nn.functional.interpolate(raw.float(),size=x.shape[-2:],mode='bilinear',align_corners=False)
            probabilities=logits.softmax(dim=1)[:,1].cpu().numpy()
        for row,p in zip(chunk,probabilities): yield row, restore_probability(p,row)


def load_network(checkpoint, cfg):
    import torch
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    net=ck.get('ema')
    if net is None: net=ck.get('model')
    if net is None: raise ValueError('Checkpoint has no semantic model')
    device='cpu' if str(cfg['device'])=='cpu' else f"cuda:{cfg['device']}"
    return net.float().to(device).eval()
