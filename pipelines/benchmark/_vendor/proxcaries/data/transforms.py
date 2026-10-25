from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class LetterboxMeta:
    original_width: int
    original_height: int
    output_size: int
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int
    scale: float


def _to_gray_array(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.asarray(im)
    if arr.ndim == 3:
        original_dtype = arr.dtype
        rgb = arr[..., :3].astype(np.float32)
        gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
        # Preserve the original integer intensity scale. In particular, RGB/RGBA
        # uint8 bitewings must remain on the fixed 0..255 scale instead of being
        # normalized by each image's own maximum.
        if np.issubdtype(original_dtype, np.integer):
            info = np.iinfo(original_dtype)
            arr = np.clip(np.round(gray), info.min, info.max).astype(original_dtype)
        else:
            arr = gray.astype(original_dtype, copy=False)
    return np.asarray(arr)


def load_radiograph_uint8(path: str | Path) -> np.ndarray:
    """Load L/RGB/RGBA/16-bit radiographs as deterministic 8-bit grayscale.

    12-bit images stored in uint16 are scaled with a 4095 denominator. True
    16-bit images use 65535. This avoids per-image min-max normalization.
    """
    arr = _to_gray_array(path)
    if arr.dtype == np.uint8:
        return arr
    arrf = arr.astype(np.float32)
    maxv = float(np.nanmax(arrf)) if arrf.size else 0.0
    minv = float(np.nanmin(arrf)) if arrf.size else 0.0
    if minv < 0:
        arrf = arrf - minv
        maxv = float(np.nanmax(arrf))
    if np.issubdtype(arr.dtype, np.integer):
        denom = 4095.0 if maxv <= 4095.0 else 65535.0
    else:
        denom = max(maxv, 1.0)
    arrf = np.clip(arrf / denom, 0.0, 1.0)
    return np.round(arrf * 255.0).astype(np.uint8)


def load_binary_mask(path: str | Path) -> np.ndarray:
    arr = _to_gray_array(path)
    return (arr > 0).astype(np.uint8)


def crop_xyxy(arr: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    h, w = arr.shape[:2]
    x1 = max(0, min(int(x1), w))
    x2 = max(0, min(int(x2), w))
    y1 = max(0, min(int(y1), h))
    y2 = max(0, min(int(y2), h))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid crop ({x1}, {y1}, {x2}, {y2}) for image {w}x{h}")
    return arr[y1:y2, x1:x2]


def letterbox_square(arr: np.ndarray, size: int, is_mask: bool = False) -> tuple[np.ndarray, LetterboxMeta]:
    h, w = arr.shape[:2]
    scale = min(size / w, size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    mode = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
    resized = np.asarray(Image.fromarray(arr).resize((new_w, new_h), mode))
    out = np.zeros((size, size), dtype=np.uint8)
    left = (size - new_w) // 2
    top = (size - new_h) // 2
    out[top : top + new_h, left : left + new_w] = resized
    if is_mask:
        out = (out > 0).astype(np.uint8)
    meta = LetterboxMeta(
        original_width=w,
        original_height=h,
        output_size=size,
        resized_width=new_w,
        resized_height=new_h,
        pad_left=left,
        pad_top=top,
        scale=scale,
    )
    return out, meta


def undo_letterbox(mask_or_prob: np.ndarray, meta: LetterboxMeta, is_probability: bool = False) -> np.ndarray:
    y0, x0 = meta.pad_top, meta.pad_left
    core = mask_or_prob[y0 : y0 + meta.resized_height, x0 : x0 + meta.resized_width]
    pil_mode = Image.Resampling.BILINEAR if is_probability else Image.Resampling.NEAREST
    if is_probability:
        im = Image.fromarray(core.astype(np.float32), mode="F")
        return np.asarray(im.resize((meta.original_width, meta.original_height), pil_mode), dtype=np.float32)
    im = Image.fromarray(core.astype(np.uint8))
    return (np.asarray(im.resize((meta.original_width, meta.original_height), pil_mode)) > 0).astype(np.uint8)


def meta_to_dict(meta: LetterboxMeta) -> dict[str, float | int]:
    return asdict(meta)
