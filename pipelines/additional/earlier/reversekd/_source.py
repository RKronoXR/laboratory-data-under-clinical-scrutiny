"""Source-data readers retained from the supplied working project.
Only parsing/audit functions are reused. No prior training/evaluation code is run.
"""
from __future__ import annotations
import hashlib,csv,os,re,stat,unicodedata
from pathlib import Path,PurePosixPath
import numpy as np
import pandas as pd
from PIL import Image
from .io import file_hash
RAW_NAMES={'acta_direct':'ACTA-DIRECT dataset v2','dcbr':'DCBR proximal caries lesions dataset','bw25':'ACTA-Bw25-RefStd'}
EXTENSIONS={'.png','.tif','.tiff'}
UNKNOWN={'','?','nan','none','null','na','n/a','unk','unknown'}
def _safe_members(zf):
    """Validate archive metadata before any extraction, including unselected names."""
    files, seen = [], {}
    total = 0
    for info in zf.infolist():
        name = info.filename
        parts = PurePosixPath(name).parts
        if ('\\' in name or '\x00' in name or not parts or name.startswith('/')
                or any(p in {'..', '.'} for p in name.split('/'))
                or any(':' in p for p in parts)):
            raise ValueError(f'Unsafe ZIP path: {name!r}')
        canonical = unicodedata.normalize('NFC', name.rstrip('/')).casefold()
        if canonical in seen:
            raise ValueError(f'Duplicate ZIP entry or name collision: {name}')
        seen[canonical] = info.is_dir()
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))):
            raise ValueError(f'Symlink or special file not allowed in ZIP: {name}')
        if info.flag_bits & 1:
            raise ValueError(f'Encrypted ZIP not supported: {name}')
        if not info.is_dir():
            total += info.file_size
            files.append(info)
    for path in seen:
        for parent in PurePosixPath(path).parents:
            if str(parent) != '.' and seen.get(str(parent)) is False:
                raise ValueError(f'ZIP file/directory path collision: {path}')
    if total > 128 * 1024 ** 3:
        raise ValueError('ZIP declares more than 128 GiB. Review the archive before extraction.')
    return files

def _select_sources(infos, domains):
    selected = {}
    for domain in domains:
        candidates = {}
        for info in infos:
            parts = PurePosixPath(info.filename).parts
            if RAW_NAMES[domain] in parts:
                idx = parts.index(RAW_NAMES[domain])
                prefix = tuple(parts[:idx + 1])
                candidates.setdefault(prefix, []).append((info, parts[idx + 1:]))
        if len(candidates) != 1:
            raise ValueError(f'Expected exactly one folder {RAW_NAMES[domain]!r}; found {len(candidates)}.')
        entries = next(iter(candidates.values()))
        selected[domain] = entries
    return selected

def _array(path):
    with Image.open(path) as im:
        if getattr(im, 'n_frames', 1) != 1:
            raise ValueError(f'Multipage image not supported: {path.name}')
        arr = np.array(im)
    if arr.ndim == 3 and arr.shape[2] in {3, 4}:
        if arr.shape[2] == 4 and not np.all(arr[:, :, 3] == np.iinfo(arr.dtype).max):
            raise ValueError(f'Nonopaque alpha channel: {path.name}')
        if not np.all(arr[:, :, :3] == arr[:, :, :1]):
            raise ValueError(f'Color image/mask requires source review: {path.name}')
        arr = arr[:, :, 0]
    if arr.ndim != 2 or not arr.size or not np.isfinite(arr).all():
        raise ValueError(f'Invalid image: {path.name}')
    return arr

def _pixel_hash(arr):
    """Canonical decoded values: catches different encoding, dtype and compression."""
    a = np.ascontiguousarray(arr, dtype='<f8')
    h = hashlib.sha256(str(tuple(a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()

def _radiograph(path):
    raw = _array(path)
    if np.unique(raw).size <= 4:
        raise ValueError(f'Radiograph appears binary or constant: {path.name}')
    if raw.dtype == np.uint8:
        return raw, _pixel_hash(raw)
    if not np.issubdtype(raw.dtype, np.integer) or raw.min() < 0 or raw.max() > 65535:
        raise ValueError(f'Radiograph outside supported uint8/12/16-bit scale: {path.name}')
    denom = 4095 if int(raw.max()) <= 4095 else 65535
    out = np.rint(raw.astype(np.float64) * 255 / denom).clip(0, 255).astype(np.uint8)
    return out, _pixel_hash(raw)

def _mask(path):
    a = _array(path)
    if not np.isin(a, [0, 1, 255]).all():
        raise ValueError(f'Mask must contain binary 0/1 or 0/255 values: {path.name}')
    return (a > 0).astype(np.uint8)

def _id(value):
    value = str(value).strip()
    if not re.fullmatch(r'\d+(?:\.0+)?', value):
        raise ValueError(f'Nonnumeric bitewing ID: {value!r}')
    value = int(value.split('.')[0])
    if value < 1:
        raise ValueError('Bitewing ID must be positive.')
    return value

def _numbered_files(folder):
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    out = {}
    for p in sorted(folder.iterdir()):
        if not p.is_file() or p.suffix.lower() not in EXTENSIONS:
            continue
        n = _id(p.stem)
        if n in out:
            raise ValueError(f'Multiple files for bitewing {n}: {out[n].name}, {p.name}')
        out[n] = p
    return out

def _regions(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        sample = handle.read(8192)
        handle.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=',;\t')
        except csv.Error as exc:
            raise ValueError(f'CSV separator could not be identified: {path.name}') from exc
        header = [str(c).strip() for c in next(csv.reader(handle, dialect))]
        if len(header) != len(set(header)):
            raise ValueError('Duplicate columns in region CSV.')
    frame = pd.read_csv(path, sep=None, engine='python', dtype=str, keep_default_na=False,
                        encoding='utf-8-sig')
    frame.columns = [str(c).strip() for c in frame.columns]
    required = {'Bitewing', 'x1', 'y1', 'x2', 'y2'}
    if not required <= set(frame) or not frame.columns.is_unique:
        raise ValueError(f'Invalid region CSV: missing {required - set(frame)} or duplicate columns.')
    frame['_bw'] = frame.Bitewing.map(_id)
    frame['Surface'] = frame.get('Surface', pd.Series('?', index=frame.index)).map(lambda x: str(x).strip())
    aliases = [c for c in frame if re.sub(r'[^a-z0-9]', '', c.casefold()) in {'patient', 'patientid'}]
    if len(aliases) > 1:
        raise ValueError('Multiple patient ID columns; specify a unique identifier.')
    if aliases:
        frame['_patient'] = frame[aliases[0]].str.strip()
        if frame._patient.str.casefold().isin(UNKNOWN).any():
            raise ValueError('Patient column contains missing identifiers.')
        if frame.groupby('_bw')._patient.nunique().max() != 1:
            raise ValueError('One bitewing is assigned to multiple patients.')
    else:
        frame['_patient'] = ''
    for c in ['x1', 'y1', 'x2', 'y2']:
        numbers = pd.to_numeric(frame[c], errors='raise')
        if not np.isfinite(numbers).all() or not (numbers == np.floor(numbers)).all():
            raise ValueError(f'Noninteger coordinates: {c}')
        frame[c] = numbers.astype(np.int64)
    if frame.duplicated(['_bw', 'x1', 'y1', 'x2', 'y2']).any():
        raise ValueError('Duplicate region: same bitewing and box, even if the Surface label differs.')
    frame['_surface_key'] = frame.Surface.str.upper().str.replace(r'\s+', '', regex=True)
    known = ~frame.Surface.str.casefold().isin(UNKNOWN)
    if frame[known].duplicated(['_bw', '_surface_key']).any():
        raise ValueError('Repeated known surface identifier within a bitewing.')
    # Order and sample IDs do not depend on source CSV row order.
    return frame.sort_values(['_bw', 'x1', 'y1', 'x2', 'y2', '_surface_key']).reset_index(drop=True)

def _surface_review(raw, regions, required=False):
    """Consume the portable, explicitly reviewed DCBR 085 surface annotations.

    Schema and crop-mask semantics come from update_datasets_zip.py (2026-10-06).
    No surface label is inferred from anatomy or overwritten without this file.
    """
    path = raw / 'Surface_Review_085.csv'
    other = [p.name for p in raw.glob('Surface_Review_*.csv') if p.name != path.name]
    if other:
        raise ValueError(f'Unsupported surface review file; it will not be ignored: {other}.')
    if not path.is_file():
        if required or (raw / 'README_085_CORRECTION.txt').is_file():
            raise ValueError('Corrected Datasets.zip lacks Surface_Review_085.csv; reviewed labels will not be inferred.')
        return {}
    columns = ['Bitewing', 'Surface', 'reviewed_lesion_present', 'exclude_context_mask']
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != columns:
            raise ValueError(f'Invalid columns in {path.name}: required {columns}.')
        records = list(reader)
    if len(records) != 14:
        raise ValueError(f'{path.name} must contain the 14 reviewed surfaces of bitewing 085.')
    reviewed = {}
    for record in records:
        if set(record) != set(columns) or any(value is None for value in record.values()):
            raise ValueError(f'Incomplete row or extra fields in {path.name}.')
        image_id = _id(record['Bitewing'])
        surface = record['Surface'].strip()
        key = (image_id, re.sub(r'\s+', '', surface.upper()))
        if image_id != 85 or surface.casefold() in UNKNOWN or key in reviewed:
            raise ValueError(f'Repeated, unknown or non-085 surface in {path.name}: {key}.')
        values = {}
        for field in columns[2:]:
            raw_value = record[field].strip()
            if raw_value not in {'0', '1'}:
                raise ValueError(f'{path.name}: {field} must be 0 or 1.')
            values[field] = int(raw_value)
        if values['exclude_context_mask'] and values['reviewed_lesion_present']:
            raise ValueError(f'{path.name}: a positive surface cannot exclude its entire mask.')
        reviewed[key] = dict(**values, surface_review_source=path.name,
                             surface_review_sha256=file_hash(path))
    expected = {(85, row['_surface_key']) for row in regions[regions._bw == 85].to_dict('records')}
    if set(reviewed) != expected or len(expected) != 14:
        raise ValueError(f'{path.name} must cover exactly the 14 regions of bitewing 085 in the proximal CSV.')
    if sum(row['reviewed_lesion_present'] for row in reviewed.values()) != 4:
        raise ValueError(f'{path.name} must document 4 positive and 10 sound surfaces according to the source review.')
    return reviewed
