from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from proxcaries.utils import sha256_file


def inspect_bw25_source(bitewing_dir: Path, mask_dir: Path, regions_csv: Path) -> dict:
    images = sorted(bitewing_dir.glob("*.tif")) + sorted(bitewing_dir.glob("*.tiff"))
    masks = sorted(mask_dir.glob("*.tif")) + sorted(mask_dir.glob("*.tiff"))
    image_by_name = {p.name: p for p in images}
    mask_by_name = {p.name: p for p in masks}
    common = sorted(set(image_by_name) & set(mask_by_name))
    identical = [name for name in common if sha256_file(image_by_name[name]) == sha256_file(mask_by_name[name])]
    binary_like_images = []
    for name, p in image_by_name.items():
        with Image.open(p) as im:
            arr = np.asarray(im)
        if np.unique(arr).size <= 4:
            binary_like_images.append(name)
    region_count = len(pd.read_csv(regions_csv)) if regions_csv.exists() else None
    csv_expected = bitewing_dir.parent / "ACTA-Bw25-RefStandard.csv"
    issues = []
    if identical:
        issues.append(f"{len(identical)} test bitewing file(s) are byte-identical to lesion masks")
    if binary_like_images:
        issues.append(f"{len(binary_like_images)} test bitewing file(s) look binary rather than radiographic")
    if not csv_expected.exists():
        issues.append("ACTA-Bw25-RefStandard.csv described by the dataset README is absent")
    return {
        "n_bitewing_files": len(images),
        "n_mask_files": len(masks),
        "n_regions": region_count,
        "identical_image_mask_pairs": identical,
        "binary_like_bitewing_files": binary_like_images,
        "reference_csv_present": csv_expected.exists(),
        "issues": issues,
        "test_ready": not identical and not binary_like_images and len(images) == 25 and len(masks) == 25,
    }


def validate_processed_manifest(manifest_path: Path) -> list[str]:
    df = pd.read_csv(manifest_path)
    problems = []
    for col in ["sample_id", "image_path", "mask_path", "group_id", "lesion_present"]:
        if col not in df.columns:
            problems.append(f"Missing column: {col}")
    if "sample_id" in df.columns and not df["sample_id"].is_unique:
        problems.append("sample_id is not unique")
    for col in ["image_path", "mask_path"]:
        if col in df.columns:
            missing = [str(p) for p in df[col] if not Path(p).exists()]
            if missing:
                problems.append(f"{len(missing)} missing path(s) in {col}")
    return problems
