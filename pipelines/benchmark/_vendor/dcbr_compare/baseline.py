"""Recorded input and geometry helpers used by the portable benchmark."""
from pathlib import Path
import numpy as np
from PIL import Image
def meta(row,seed,fold,condition):
    return dict(sample_id=row['sample_id'],bitewing_id=str(row['bitewing_id']),group_id=row['group_id'],
        patient_id=row.get('patient_id',''),surface=row.get('surface',''),eligible=1,ambiguity_reason='',
        seed=seed,fold=fold,condition=condition)

def project_full_to_crop(prob,row):
    """Project prediction only into the stored crop grid; never create crop input files."""
    x1,y1,x2,y2=(int(row[x]) for x in ('x1','y1','x2','y2'))
    cut=prob[y1:y2,x1:x2]
    if cut.shape!=(y2-y1,x2-x1):raise ValueError('Full prediction and original ROI coordinates disagree')
    w,h=int(row['lb_resized_width']),int(row['lb_resized_height'])
    left,top=int(row['lb_pad_left']),int(row['lb_pad_top'])
    size=int(row['lb_output_size'])
    q=np.zeros((size,size),dtype=np.float32)
    q[top:top+h,left:left+w]=np.asarray(Image.fromarray(cut.astype(np.float32)).resize((w,h),Image.Resampling.BILINEAR))
    return q
