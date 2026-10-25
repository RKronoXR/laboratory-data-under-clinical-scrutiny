"""Immutable native-grid references and aspect-preserving model inputs."""
from __future__ import annotations
import hashlib, io, os, re, shutil, tempfile, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from . import _source as src
from .io import atomic_json, file_hash, read_json, digest, write_csv, event, check_pause


def letterbox(array,size,is_mask=False):
    h,w=array.shape; r=min(size/w,size/h); rw,rh=max(1,round(w*r)),max(1,round(h*r))
    x,y=(size-rw)//2,(size-rh)//2
    out=np.zeros((size,size),dtype=array.dtype)
    mode=Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
    out[y:y+rh,x:x+rw]=np.asarray(Image.fromarray(array).resize((rw,rh),mode))
    return out,dict(width=w,height=h,resized_width=rw,resized_height=rh,left=x,top=y,size=size)


def restore_probability(p,meta):
    p=np.asarray(p,dtype=np.float32)
    if p.shape!=(int(meta['size']),int(meta['size'])): raise ValueError('Probability/model grid mismatch.')
    x,y,rw,rh=[int(meta[k]) for k in ('left','top','resized_width','resized_height')]
    result=np.asarray(Image.fromarray(p[y:y+rh,x:x+rw]).resize((int(meta['width']),int(meta['height'])),Image.Resampling.BILINEAR),dtype=np.float32)
    return np.clip(result,0,1)


def load_image(p):
    with Image.open(p) as im: return np.asarray(im).copy()


def save_png(build,final,relative,array):
    p=build/relative;p.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(array).save(p,compress_level=3)
    return str(final/relative)


def _extract(archive,domains,raw):
    inventory=[]
    with zipfile.ZipFile(archive) as z:
        selected=src._select_sources(src._safe_members(z),domains)
        for domain,entries in selected.items():
            for info,parts in entries:
                p=raw/domain/Path(*parts);p.parent.mkdir(parents=True,exist_ok=True)
                with z.open(info) as a,p.open('xb') as b: shutil.copyfileobj(a,b)
                inventory.append(dict(domain=domain,path='/'.join(parts),sha256=file_hash(p)))
    return inventory


def _acta(raw,build,final,c):
    images=raw/'3_Radiographs/2_Proximal_Sides'; masks=raw/'4_Radiograph_Annotations/2_Proximal_Sides_Annotations'
    paths=sorted(p for p in images.iterdir() if p.suffix.lower() in src.EXTENSIONS)
    if len(paths)!=c['expected_acta_samples']: raise ValueError(f'ACTA-DIRECT: expected {c["expected_acta_samples"]} images, found {len(paths)}.')
    if {p.name for p in paths}!={p.name for p in masks.iterdir() if p.suffix.lower() in src.EXTENSIONS}: raise ValueError('ACTA image/mask filenames do not match.')
    rows=[];pixels=[]
    for index,p in enumerate(paths,1):
        check_pause(c["output_dir"])
        if index%100==0:event(c["output_dir"],"prepare_acta",completed=index,total=len(paths))
        m=re.fullmatch(r'(\d+)\.(\d+)\.([lr])\.(tif|tiff)',p.name,re.I)
        if not m: raise ValueError(f'Unexpected ACTA filename {p.name}')
        tooth,proj,side=f'{int(m[1]):04}',int(m[2]),m[3].lower()
        sid=f'direct_{tooth}_p{proj}_{side}'
        image,ph=src._radiograph(p); mask=src._mask(masks/p.name)
        adjusted=False
        if image.shape!=mask.shape:
            if p.name.lower()=='0177.5.r.tif' and not mask.any(): mask=np.zeros_like(image);adjusted=True
            else: raise ValueError(f'Nonmatching ACTA image/mask geometry: {p.name}')
        im,meta=letterbox(image,c['crop_image_size']);ma,_=letterbox(mask,c['crop_image_size'],True)
        if mask.any() and not ma.any(): raise ValueError(f'ACTA lesion lost on model grid: {sid}. Increase crop_image_size before training.')
        ip=save_png(build,final,f'acta/images/{sid}.png',im)
        mp=save_png(build,final,f'acta/masks/{sid}.png',ma)
        rows.append(dict(sample_id=sid,source='acta',group_id='tooth_'+tooth,image_path=ip,mask_path=mp,
                         lesion_present=int(ma.any()),geometry=meta,shape_adjusted=adjusted))
        pixels.append(dict(source='acta',group_id='tooth_'+tooth,pixel_sha256=ph))
    if len({r['group_id'] for r in rows})!=c['expected_acta_teeth']: raise ValueError('Unexpected ACTA tooth count.')
    return rows,pixels


def _bitewings(raw,build,final,c,domain):
    ref=domain=='reference25';expected=c['expected_reference_bitewings'] if ref else c['expected_dcbr_bitewings']
    images=src._numbered_files(raw/('1_Bitewings' if ref else 'bitewings'))
    masks=src._numbered_files(raw/('2_Proximal_Caries_Lesion_Delineations' if ref else 'masks'))
    regions=src._regions(raw/'Proximal_Side_Regions.csv');ids=set(range(1,expected+1))
    if set(images)!=ids or set(masks)!=ids or set(regions._bw)!=ids: raise ValueError(f'{domain}: image/mask/region coverage incomplete.')
    expected_n=c['expected_reference_surfaces'] if ref else c['expected_dcbr_surfaces']
    if len(regions)!=expected_n: raise ValueError(f'{domain}: expected {expected_n} surfaces; found {len(regions)}.')
    reviews={} if ref else src._surface_review(raw,regions,c['require_surface_review'])
    rows=[];surfaces=[];pixels=[];audit=[]
    for index,n in enumerate(sorted(images),1):
        check_pause(c["output_dir"]);event(c["output_dir"],"prepare_bitewing",domain=domain,completed=index,total=len(images))
        bid=f'{n:02}' if ref else f'{n:03}';sid=f'{domain}_{bid}'
        image,ph=src._radiograph(images[n]);mask=src._mask(masks[n])
        if image.shape!=mask.shape: raise ValueError(f'{domain} {bid}: image and mask have different geometry.')
        regs=regions.loc[regions._bw==n];patient=str(regs.iloc[0]['_patient'])
        group=f'{domain}_patient_{patient}' if patient else sid
        im,meta=letterbox(image,c['full_image_size']);ma,_=letterbox(mask,c['full_image_size'],True)
        ip=save_png(build,final,f'{domain}/images/{sid}.png',im)
        mp=save_png(build,final,f'{domain}/masks/{sid}.png',ma)
        native_i=save_png(build,final,f'{domain}/native_images/{sid}.png',image)
        native_m=save_png(build,final,f'{domain}/native_masks/{sid}.png',mask)
        teacher_im, teacher_meta = letterbox(image, c['teacher_image_size'])
        teacher_ma, _ = letterbox(mask, c['teacher_image_size'], True)
        teacher_ip = save_png(build, final, f'{domain}/teacher_images/{sid}.png', teacher_im)
        teacher_mp = save_png(build, final, f'{domain}/teacher_masks/{sid}.png', teacher_ma)
        rows.append(dict(sample_id=sid,bitewing_id=bid,source=domain,group_id=group,patient_id=patient,
                         image_path=ip,mask_path=mp,native_image_path=native_i,native_mask_path=native_m,
                         geometry=meta, teacher_image_path=teacher_ip, teacher_mask_path=teacher_mp,
                         teacher_geometry=teacher_meta, lesion_present=int(mask.any())))
        reconstructed=restore_probability(ma.astype(np.float32),meta)>0
        for r in regs.to_dict('records'):
            x1,y1,x2,y2=[int(r[k]) for k in ('x1','y1','x2','y2')]
            if not (0<=x1<x2<=image.shape[1] and 0<=y1<y2<=image.shape[0]): raise ValueError(f'ROI outside image: {sid} {r["Surface"]}')
            crop=mask[y1:y2,x1:x2].copy();review=reviews.get((n,r['_surface_key']),{})
            excluded=bool(review.get('exclude_context_mask',0))
            if excluded: crop[:]=0
            if review and int(crop.any())!=review['reviewed_lesion_present'] and c.get('annotation_version', 'recorded') != 'updated': raise ValueError(f'Review disagrees with mask: {sid}, {r["Surface"]}.')
            tag=digest([bid,x1,y1,x2,y2])[:12];ss=f'{sid}_{tag}'
            refpath=save_png(build,final,f'{domain}/surface_masks/{ss}.png',crop)
            lost=bool(crop.any() and not (reconstructed[y1:y2,x1:x2]&crop.astype(bool)).any())
            surfaces.append(dict(sample_id=ss,image_id=sid,bitewing_id=bid,group_id=group,patient_id=patient,source=domain,
                surface=str(r['Surface']),reference_positive=int(crop.any()),reference_path=refpath,
                x1=x1,y1=y1,x2=x2,y2=y2,exclude_context_mask=int(excluded)))
            audit.append(dict(source=domain,sample_id=ss,image_id=sid,native_foreground_pixels=int(crop.sum()),
                original_width=image.shape[1],original_height=image.shape[0],model_size=c['full_image_size'],
                lost_on_model_grid=lost))
            if c.get('annotation_version', 'recorded') == 'updated':
                audit[-1].update(source_review_lesion_present=review.get('reviewed_lesion_present'),
                    native_reference_positive=int(crop.any()),
                    review_presence_disagreement=bool(review and int(crop.any()) != review['reviewed_lesion_present']),
                    context_exclusion_applied=excluded)
            # Preserve the source teacher's historical 384 grid. Losses are
            # recorded; source_baselines independently verifies exact original
            # image/mask pixels before allowing any new training.
        pixels.append(dict(source=domain,group_id=group,pixel_sha256=ph))
    return rows,surfaces,pixels,audit


def make_splits(images, surfaces, c):
    """Load the exact historical folds; never generate or subdivide a split.

    The validation fold is used for checkpoint/HPO/threshold selection and for
    descriptive development reporting. It is NOT a separate unbiased test set.
    """
    from .reuse import legacy_splits, legacy_split_path
    archived = legacy_splits(c)
    if archived is None:
        raise FileNotFoundError(
            f'Original DCBR split file is required: {legacy_split_path(c)}. '
            'No random or internal replacement split has been generated.')
    all_ids = {r['sample_id'] for r in images}
    if len(all_ids) != len(images):
        raise ValueError('Duplicate DCBR image identifiers.')
    lookup = {r['sample_id']: r for r in images}
    assignment = {}
    result = []
    for part in archived:
        f = part['fold']
        train, val = set(part['train_ids']), set(part['validation_ids'])
        if train & val or train | val != all_ids or train != all_ids - val:
            raise ValueError(f'Original fold {f} does not cover the current DCBR images exactly.')
        tg = {lookup[x]['group_id'] for x in train}
        vg = {lookup[x]['group_id'] for x in val}
        if tg & vg:
            raise ValueError(f'Original fold {f} splits a recorded patient/group across training and validation. '
                             'Stop and review the original protocol; no silent repartitioning or retraining.')
        for x in val:
            if x in assignment:
                raise ValueError('One DCBR image belongs to multiple original validation folds.')
            assignment[x] = f
        # Keep original list order as well as original membership.
        result.append(dict(fold=f, train_ids=list(part['train_ids']),
            validation_ids=list(part['validation_ids']), train_groups=sorted(tg),
            validation_groups=sorted(vg), validation_role='development_selection_and_reporting'))
    if set(assignment) != all_ids:
        raise ValueError('Original validation folds do not cover every DCBR image exactly once.')
    for r in images:
        r['fold'] = assignment[r['sample_id']]
    for r in surfaces:
        r['fold'] = assignment[r['image_id']]
    # Cross-check the frozen full-bitewing manifest used by the earlier models,
    # when present. The authoritative nnU-Net split file is always required.
    from .reuse import legacy_root, canonical_id
    manifest = legacy_root(c).parent/'output/data/bitewings.csv'
    if manifest.is_file():
        old = pd.read_csv(manifest, dtype=str, keep_default_na=False)
        if not {'bitewing_id', 'fold'} <= set(old.columns):
            raise ValueError('Original full-bitewing manifest has no bitewing/fold columns.')
        mapping = {canonical_id(r.bitewing_id): int(r.fold) for r in old.itertuples()}
        if len(old) != len(mapping) or mapping != assignment:
            raise ValueError('Original nnU-Net splits disagree with the shared full-bitewing manifest. '
                             'No new partitions have been generated.')
    return result


def teacher_images(images):
    """View each full bitewing on the original teacher grid, not the student grid."""
    return [{**r, 'image_path': r.get('teacher_image_path', r['image_path']),
             'mask_path': r.get('teacher_mask_path', r['mask_path']),
             'geometry': r.get('teacher_geometry', r['geometry'])} for r in images]


def probability_to_grid(probability, source_geometry, target_geometry):
    """Align soft targets via the native image, using geometry only (no masks).

    Teacher evaluation uses its own grid directly; this conversion is used only
    for teacher-to-student targets when the two model input sizes differ.
    """
    p = np.asarray(probability, dtype=np.float32)
    if source_geometry == target_geometry:
        return p.copy()
    native = restore_probability(p, source_geometry)
    target, geometry = letterbox(native, int(target_geometry['size']))
    if geometry != target_geometry:
        raise ValueError('Teacher/student images have different native geometry.')
    return np.clip(target, 0, 1).astype(np.float32)


def _verify(target,c,mode):
    p=target/'READY.json'
    if not p.exists():raise RuntimeError(f'Incomplete uncommitted data directory: {target}')
    m=read_json(p)
    if m['source_zip_sha256']!=file_hash(c['datasets_zip']):raise RuntimeError('Datasets.zip changed. Choose a new output directory.')
    if any(m[k]!=c[k] for k in ('full_image_size','crop_image_size','teacher_image_size')):raise RuntimeError('Input sizes changed.')
    if not mode:
        from .reuse import legacy_split_path
        if m.get('original_splits_sha256') != file_hash(legacy_split_path(c)):
            raise RuntimeError('Original split file changed after preparation. Existing results are protected.')
    for rel,h in m['artifacts'].items():
        if not (target/rel).is_file() or file_hash(target/rel)!=h:raise RuntimeError(f'Prepared artifact changed: {rel}')
    return read_json(target/'bundle.json')


def prepare(c,reference=False):
    out=Path(c['output_dir']);target=out/('reference_data' if reference else 'data')
    if reference and not (out/'DEVELOPMENT_FROZEN.json').is_file():raise RuntimeError('Reference25 is locked until all development models and thresholds are frozen.')
    if target.exists():return _verify(target,c,reference)
    out.mkdir(parents=True,exist_ok=True)
    from .cache import reuse_data
    reused=reuse_data(c,target,reference)
    if reused is not None:return reused
    with tempfile.TemporaryDirectory(prefix='.prepare_',dir=out) as td:
        temp=Path(td);raw=temp/'raw';build=temp/'build';build.mkdir()
        inventory=_extract(c['datasets_zip'],['bw25'] if reference else ['acta_direct','dcbr'],raw)
        if reference:
            images,surfaces,pixels,audit=_bitewings(raw/'bw25',build,target,c,'reference25')
            previous=read_json(out/'data/pixels.json');_duplicates(previous+pixels)
            bundle=dict(images=images,surfaces=surfaces)
        else:
            acta,ap=_acta(raw/'acta_direct',build,target,c)
            images,surfaces,pixels,audit=_bitewings(raw/'dcbr',build,target,c,'dcbr')
            pixels=ap+pixels;_duplicates(pixels)
            splits=make_splits(images,surfaces,c)
            bundle=dict(acta=acta,images=images,surfaces=surfaces,splits=splits)
            from .reuse import legacy_split_path
            split_source = legacy_split_path(c)
            shutil.copy2(split_source, build/'original_splits_final.json')
            atomic_json(build/'ORIGINAL_PARTITION_AUDIT.json', dict(
                policy=c['partition_policy'], source=str(split_source), source_sha256=file_hash(split_source),
                copied_exactly=True, new_internal_partitions_generated=False,
                reference25_used_for_selection=False,
                validation_role='development_selection_and_reporting',
                folds=[dict(fold=s['fold'],train_ids=s['train_ids'],validation_ids=s['validation_ids']) for s in splits]))
            write_csv(build/'split_counts.csv',[dict(fold=s['fold'],train_bitewings=len(s['train_ids']),
               validation_bitewings=len(s['validation_ids']),new_internal_partitions=0) for s in splits])
        atomic_json(build/'bundle.json',bundle);atomic_json(build/'pixels.json',pixels)
        atomic_json(build/'source_inventory.json',inventory);write_csv(build/'resolution_audit.csv',audit)
        write_csv(build/'surfaces.csv',surfaces)
        artifacts={str(p.relative_to(build)):file_hash(p) for p in build.rglob('*') if p.is_file()}
        atomic_json(build/'READY.json',dict(source_zip_sha256=file_hash(c['datasets_zip']),full_image_size=c['full_image_size'],
            crop_image_size=c['crop_image_size'],teacher_image_size=c['teacher_image_size'],
            original_splits_sha256=None if reference else file_hash(split_source),artifacts=artifacts,mode='reference25' if reference else 'development',
            n_bitewings=len(images),n_surfaces=len(surfaces),reference_labels_used_for_selection=False))
        os.rename(build,target)
    return bundle


def _duplicates(pixels):
    seen={}
    for r in pixels:
        key=r['pixel_sha256'];group=(r['source'],r['group_id'])
        if key in seen and seen[key]!=group: raise ValueError(f'Identical decoded radiographs in different groups/domains: {group}, {seen[key]}.')
        seen[key]=group
