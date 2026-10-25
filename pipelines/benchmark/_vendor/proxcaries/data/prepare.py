from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from proxcaries.data.splits import assign_grouped_folds
from proxcaries.data.transforms import crop_xyxy, letterbox_square, load_binary_mask, load_radiograph_uint8, meta_to_dict
from proxcaries.data.validate import inspect_bw25_source
from proxcaries.utils import save_json

_DIRECT_RE = re.compile(r"^(?P<tooth>\d+)\.(?P<projection>\d+)\.(?P<side>[lr])\.(?:tif|tiff)$", re.I)


def extract_zip(zip_path: Path, extracted_dir: Path, force: bool = False) -> Path:
    marker = extracted_dir / ".extracted_ok"
    if marker.exists() and not force:
        return extracted_dir
    if force and extracted_dir.exists():
        shutil.rmtree(extracted_dir)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extracted_dir)
    marker.write_text("ok\n", encoding="utf-8")
    return extracted_dir


def _datasets_root(extracted_dir: Path) -> Path:
    candidate = extracted_dir / "Datasets"
    return candidate if candidate.exists() else extracted_dir


def _save_png(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr.astype(np.uint8)).save(path, compress_level=3)


def prepare_acta_direct(root: Path, out_root: Path, image_size: int, n_folds: int, fold_seed: int, source_directory=None) -> pd.DataFrame:
    src = Path(source_directory) if source_directory is not None else root / "ACTA-DIRECT dataset v2"
    image_dir = src / "3_Radiographs" / "2_Proximal_Sides"
    mask_dir = src / "4_Radiograph_Annotations" / "2_Proximal_Sides_Annotations"
    out_img = out_root / "acta_direct" / "images"
    out_mask = out_root / "acta_direct" / "masks"
    image_files = sorted([*image_dir.glob("*.tif"), *image_dir.glob("*.tiff")])
    rows = []
    for img_path in tqdm(image_files, desc="ACTA-DIRECT"):
        m = _DIRECT_RE.match(img_path.name)
        if not m:
            raise ValueError(f"Unexpected ACTA-DIRECT filename: {img_path.name}")
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        img = load_radiograph_uint8(img_path)
        mask = load_binary_mask(mask_path)
        shape_adjusted = 0
        if img.shape != mask.shape:
            # The supplied ACTA-DIRECT archive contains one known sound-surface
            # mask with a small geometry mismatch (0177.5.r.tif). Resizing a
            # non-empty lesion annotation would risk changing its localization,
            # so only an empty mask may be safely reconciled this way.
            if mask.any():
                raise ValueError(
                    f"Non-empty mask shape mismatch for {img_path.name}: "
                    f"{img.shape} vs {mask.shape}. Manual source-data review required."
                )
            mask = np.zeros(img.shape, dtype=np.uint8)
            shape_adjusted = 1
        proc_img, meta = letterbox_square(img, image_size, is_mask=False)
        proc_mask, _ = letterbox_square(mask, image_size, is_mask=True)
        tooth = m.group("tooth")
        proj = int(m.group("projection"))
        side = m.group("side").lower()
        sample_id = f"direct_{tooth}_p{proj}_{side}"
        out_i = out_img / f"{sample_id}.png"
        out_m = out_mask / f"{sample_id}.png"
        _save_png(proc_img, out_i)
        _save_png(proc_mask, out_m)
        rows.append({
            "sample_id": sample_id,
            "source": "acta_direct",
            "image_path": str(out_i.resolve()),
            "mask_path": str(out_m.resolve()),
            "group_id": f"direct_tooth_{tooth}",
            "bitewing_id": "",
            "surface": side,
            "tooth_id": tooth,
            "projection": proj,
            "lesion_present": int(proc_mask.any()),
            "shape_adjusted": shape_adjusted,
            "x1": 0, "y1": 0, "x2": img.shape[1], "y2": img.shape[0],
            **{f"lb_{k}": v for k, v in meta_to_dict(meta).items()},
        })
    df = assign_grouped_folds(pd.DataFrame(rows), n_splits=n_folds, seed=fold_seed)
    manifest = out_root / "acta_direct" / "manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest, index=False)
    return df


def _prepare_bitewing_dataset(
    source_name: str,
    src: Path,
    image_subdir: str,
    mask_subdir: str,
    out_root: Path,
    image_size: int,
    n_folds: int | None,
    fold_seed: int,
    annotation_version: str = "recorded",
) -> pd.DataFrame:
    regions = pd.read_csv(src / "Proximal_Side_Regions.csv", dtype={"Bitewing": str})
    image_dir = src / image_subdir
    mask_dir = src / mask_subdir
    # DCBR coverage guard: never silently omit a source image.
    if source_name == "dcbr":
        def _index_dcbr_files(directory):
            indexed = {}
            for p in directory.iterdir():
                if not p.is_file() or p.suffix.lower() not in {".png", ".tif", ".tiff"}:
                    continue
                if not p.stem.isdigit():
                    raise ValueError(f"Unrecognized DCBR filename: {p}")
                key = str(int(p.stem)).zfill(3)
                if key in indexed:
                    raise ValueError(f"Duplicate DCBR ID {key}: {indexed[key]} and {p}")
                indexed[key] = p
            return indexed
        approved_085 = {"101_right", "102_left", "102_right", "103_left", "103_right", "104_left", "104_right", "201_left", "201_right", "202_left", "202_right", "203_left", "203_right", "204_left"}
        rows_085 = regions[regions["Bitewing"].map(lambda v: str(int(str(v).strip())).zfill(3)) == "085"]
        if len(rows_085) != 14 or set(rows_085["Surface"].astype(str)) != approved_085:
            raise ValueError("085 must have exactly the 14 reviewed surface identifiers")
        image_ids = set(_index_dcbr_files(image_dir))
        mask_ids = set(_index_dcbr_files(mask_dir))
        csv_ids = {str(int(str(v).strip())).zfill(3) for v in regions["Bitewing"]}
        expected_ids = {str(i).zfill(3) for i in range(1, 101)}
        if image_ids != expected_ids or mask_ids != expected_ids or csv_ids != expected_ids:
            raise ValueError(
                f"DCBR coverage mismatch. Missing images={sorted(expected_ids-image_ids)}, "
                f"missing masks={sorted(expected_ids-mask_ids)}, missing CSV IDs={sorted(expected_ids-csv_ids)}; "
                f"extra IDs={sorted((image_ids|mask_ids|csv_ids)-expected_ids)}"
            )
        # DCBR_DUPLICATE_GUARD_V2: '?' denotes an unknown surface, not a unique ID.
        _dcbr_rows = regions.assign(
            _id=regions["Bitewing"].map(lambda v: str(int(str(v).strip())).zfill(3)),
            _surface=regions["Surface"].fillna("").astype(str).str.strip(),
        )
        _box_keys = ["_id", "x1", "y1", "x2", "y2"]
        if _dcbr_rows.duplicated(_box_keys).any():
            raise ValueError("Duplicate DCBR bounding boxes within a bitewing")
        _known = ~_dcbr_rows["_surface"].isin(["", "?"])
        if _dcbr_rows.loc[_known].duplicated(["_id", "_surface"]).any():
            raise ValueError("Duplicate DCBR known surface IDs within a bitewing")
    out_img = out_root / source_name / "images"
    out_mask = out_root / source_name / "masks"
    rows = []
    # A bitewing contributes many proximal regions. Cache each full radiograph
    # and mask once so preprocessing scales with the number of bitewings rather
    # than repeatedly decoding the same large image for every surface.
    source_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for row_idx, r in tqdm(regions.iterrows(), total=len(regions), desc=source_name):
        raw_id = str(r["Bitewing"]).strip()
        if raw_id not in source_cache:
            candidates = [raw_id, raw_id.zfill(3), raw_id.zfill(2)]
            source_image = None
            source_mask = None
            for stem in candidates:
                for ext in (".png", ".tif", ".tiff"):
                    if source_image is None and (image_dir / f"{stem}{ext}").exists():
                        source_image = image_dir / f"{stem}{ext}"
                    if source_mask is None and (mask_dir / f"{stem}{ext}").exists():
                        source_mask = mask_dir / f"{stem}{ext}"
            if source_image is None or source_mask is None:
                raise FileNotFoundError(f"Could not pair bitewing {raw_id} in {src}")
            img_cached = load_radiograph_uint8(source_image)
            mask_cached = load_binary_mask(source_mask)
            if img_cached.shape != mask_cached.shape:
                raise ValueError(
                    f"Shape mismatch for bitewing {raw_id}: "
                    f"{img_cached.shape} vs {mask_cached.shape}"
                )
            source_cache[raw_id] = (img_cached, mask_cached)
        img, mask = source_cache[raw_id]
        x1, y1, x2, y2 = (int(r[c]) for c in ["x1", "y1", "x2", "y2"])
        crop_img = crop_xyxy(img, x1, y1, x2, y2)
        crop_mask = crop_xyxy(mask, x1, y1, x2, y2)
        # Author-confirmed 2026-10-06: these two 085 surfaces are sound.
        # Foreground in their rectangular crops belongs to neighboring surfaces.
        # Replace only the per-surface target; never mutate the parent lesion mask.
        if source_name == "dcbr" and raw_id.zfill(3) == "085" and str(r.get("Surface", "")).strip() in {"201_right", "203_left"}:
            crop_mask = np.zeros_like(crop_mask)
        proc_img, meta = letterbox_square(crop_img, image_size, is_mask=False)
        proc_mask, _ = letterbox_square(crop_mask, image_size, is_mask=True)
        if source_name == "dcbr" and raw_id.zfill(3) == "085" and annotation_version == "recorded":
            expected_positive = str(r.get("Surface", "")).strip() in {"102_right", "103_left", "202_left", "202_right"}
            if bool(proc_mask.any()) != expected_positive:
                raise ValueError(f"085 surface annotation mismatch: {r.get('Surface')}")
        normalized_bw = raw_id.zfill(3) if source_name == "dcbr" else raw_id.zfill(2)
        surface = str(r.get("Surface", "?")).strip()
        safe_surface = re.sub(r"[^A-Za-z0-9]+", "UNK", surface) or "UNK"
        sample_id = f"{source_name}_{normalized_bw}_r{row_idx:04d}_{safe_surface}"
        out_i = out_img / f"{sample_id}.png"
        out_m = out_mask / f"{sample_id}.png"
        _save_png(proc_img, out_i)
        _save_png(proc_mask, out_m)
        rows.append({
            "sample_id": sample_id,
            "source": source_name,
            "image_path": str(out_i.resolve()),
            "mask_path": str(out_m.resolve()),
            "group_id": f"{source_name}_bw_{normalized_bw}",
            "bitewing_id": normalized_bw,
            "surface": surface,
            "tooth_id": "",
            "projection": "",
            "lesion_present": int(proc_mask.any()),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "parent_width": img.shape[1], "parent_height": img.shape[0],
            **{f"lb_{k}": v for k, v in meta_to_dict(meta).items()},
        })
    df = pd.DataFrame(rows)
    if n_folds is not None:
        df = assign_grouped_folds(df, n_splits=n_folds, seed=fold_seed)
    manifest = out_root / source_name / "manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest, index=False)
    return df


def prepare_all(zip_path: Path, extracted_dir: Path, processed_dir: Path, image_size: int, n_folds: int, fold_seed: int, force_extract: bool = False) -> dict:
    extracted_dir = extract_zip(zip_path, extracted_dir, force=force_extract)
    root = _datasets_root(extracted_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    direct = prepare_acta_direct(root, processed_dir, image_size, n_folds, fold_seed)
    dcbr_src = root / "DCBR proximal caries lesions dataset"
    dcbr = _prepare_bitewing_dataset("dcbr", dcbr_src, "bitewings", "masks", processed_dir, image_size, n_folds, fold_seed)
    dcbr_source_stems = {
        p.stem.zfill(3)
        for p in (dcbr_src / "bitewings").iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".tif", ".tiff"}
    }
    dcbr_used_stems = set(dcbr["bitewing_id"].astype(str).str.zfill(3))
    dcbr_without_regions = sorted(dcbr_source_stems - dcbr_used_stems)

    bw_src = root / "ACTA-Bw25-RefStd"
    bw_validation = inspect_bw25_source(
        bw_src / "1_Bitewings",
        bw_src / "2_Proximal_Caries_Lesion_Delineations",
        bw_src / "Proximal_Side_Regions.csv",
    )
    save_json(bw_validation, processed_dir / "bw25_source_validation.json")
    bw25_count = 0
    if bw_validation["test_ready"]:
        bw25 = _prepare_bitewing_dataset(
            "bw25", bw_src, "1_Bitewings", "2_Proximal_Caries_Lesion_Delineations",
            processed_dir, image_size, None, fold_seed,
        )
        bw25_count = len(bw25)
    else:
        issue = processed_dir / "bw25" / "BLOCKED.txt"
        issue.parent.mkdir(parents=True, exist_ok=True)
        issue.write_text(
            "Independent-test preprocessing was blocked because source validation failed.\n"
            + "\n".join(f"- {x}" for x in bw_validation["issues"]) + "\n",
            encoding="utf-8",
        )

    pooled = pd.concat([direct, dcbr], ignore_index=True)
    pooled.to_csv(processed_dir / "pooled_manifest.csv", index=False)
    report = {
        "acta_direct_samples": len(direct),
        "acta_direct_groups": int(direct.group_id.nunique()),
        "acta_direct_empty_mask_geometry_adjustments": int(direct.get("shape_adjusted", pd.Series(dtype=int)).sum()),
        "dcbr_samples": len(dcbr),
        "dcbr_groups": int(dcbr.group_id.nunique()),
        "dcbr_source_bitewings": len(dcbr_source_stems),
        "dcbr_bitewings_without_regions": dcbr_without_regions,
        "bw25_samples": bw25_count,
        "bw25_test_ready": bool(bw_validation["test_ready"]),
        "image_size": image_size,
        "n_folds": n_folds,
    }
    save_json(report, processed_dir / "preparation_report.json")
    return report
