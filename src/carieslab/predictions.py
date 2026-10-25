"""Portable, anonymous regional predictions shared by assessment and analysis."""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import math
import os
import tempfile
from pathlib import Path

COLUMNS = ("architecture", "strategy", "seed", "fold", "region", "bitewing", "threshold_mode", "threshold",
           "reference_positive", "reference_pixels", "predicted_pixels", "intersection_pixels", "pixel_tp", "pixel_fp",
           "pixel_tn", "pixel_fn", "score", "overlap_score", "dice_positive", "detected", "correct_overlap")
STRATEGIES = {
    "nnunet_clinical": ("nnU-Net", "clinical_only"),
    "nnunet_laboratory_pretraining": ("nnU-Net", "laboratory_pretraining"),
    "nnunet_laboratory_distillation": ("nnU-Net", "knowledge_distillation"),
    "nnunet_pretraining_and_distillation": ("nnU-Net", "pretraining_and_distillation"),
    "yolo_clinical": ("YOLO", "clinical_only"),
    "yolo_laboratory_pretraining": ("YOLO", "laboratory_pretraining"),
    "yolo_clinical_distillation": ("YOLO", "knowledge_distillation"),
    "yolo_pretraining_and_distillation": ("YOLO", "pretraining_and_distillation"),
}
CANONICAL_STRATEGIES = {s for _, s in STRATEGIES.values()}


def _architecture(value):
    key = str(value).casefold().replace("-", "").replace("_", "")
    if key in {"nnunet", "nnunetv2"}:
        return "nnU-Net"
    if key in {"yolo", "yolo26", "yolo26n"}:
        return "YOLO"
    raise ValueError(f"Unrecognised assessment architecture: {value!r}")


def _integer(value, field):
    number = float(value)
    if not math.isfinite(number) or number != int(number) or number < 0:
        raise ValueError(f"Expected a nonnegative integer in {field}")
    return int(number)


def _probability(value, field, allow_empty=False):
    if value in (None, ""):
        if allow_empty:
            return None
        raise ValueError(f"Missing {field}")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"Invalid probability in {field}")
    return number


def canonical_predictions(rows) -> list[dict]:
    """Convert the main assessment rows to the analysis schema.

    Both the main workflow's eight strategy names and the canonical four keys
    are accepted. Anonymous keys are computed once across the whole collection,
    preserving region/image pairing and the sorted order of source group keys.
    No original filename, patient field, machine path or internal experiment code
    is included in the returned rows. Lesion-overlap decisions are recomputed
    from pixel counts; any supplied decision must agree.
    """
    converted = []
    reference = {}
    seen = set()
    for raw in rows:
        name = raw.get("strategy")
        if name in STRATEGIES:
            arch, strategy = STRATEGIES[name]
            if raw.get("architecture") and _architecture(raw["architecture"]) != arch:
                raise ValueError("Strategy and architecture disagree")
        elif name in CANONICAL_STRATEGIES:
            arch, strategy = _architecture(raw.get("architecture")), name
        else:
            raise ValueError(f"Unrecognised main assessment strategy: {name!r}")
        region = str(raw.get("region") or raw.get("sample_id") or "").strip()
        bitewing = str(raw.get("bitewing") or raw.get("bitewing_id") or raw.get("image_id") or "").strip()
        if not region or not bitewing:
            raise ValueError("Each prediction needs stable region and bitewing identifiers")
        if bitewing.isdigit():
            bitewing = str(int(bitewing))
        mode = str(raw.get("threshold_mode") or raw.get("threshold_name") or "").lower()
        if mode not in {"fixed", "youden"}:
            raise ValueError(f"Unrecognised threshold mode: {mode!r}")
        row = dict(architecture=arch, strategy=strategy, seed=_integer(raw["seed"], "seed"), fold=_integer(raw["fold"], "fold"),
                   region=region, bitewing=bitewing, threshold_mode=mode, threshold=float(raw["threshold"]))
        if not math.isfinite(row["threshold"]) or (mode == "fixed" and row["threshold"] != 0.5):
            raise ValueError("Invalid fixed or saved model-specific threshold")
        for field in ("reference_positive", "reference_pixels", "predicted_pixels", "intersection_pixels", "pixel_tp", "pixel_fp", "pixel_tn", "pixel_fn"):
            row[field] = _integer(raw[field], field)
        positive = row["reference_positive"]
        if positive not in {0, 1} or positive != int(row["reference_pixels"] > 0):
            raise ValueError("Reference flag and reference pixel count disagree")
        intersection, predicted, target = row["intersection_pixels"], row["predicted_pixels"], row["reference_pixels"]
        if (intersection > min(predicted, target) or row["pixel_tp"] != intersection
                or row["pixel_fp"] != predicted - intersection or row["pixel_fn"] != target - intersection):
            raise ValueError("Inconsistent regional pixel counts")
        row["score"] = _probability(raw.get("score"), "score")
        row["overlap_score"] = _probability(raw.get("overlap_score"), "overlap_score", allow_empty=not positive)
        row["dice_positive"] = 2 * intersection / (target + predicted) if positive else None
        row["detected"] = int(intersection > 0)
        row["correct_overlap"] = int(intersection > 0 if positive else predicted == 0)
        for field in ("detected", "correct_overlap"):
            if raw.get(field) not in (None, "") and _integer(raw[field], field) != row[field]:
                raise ValueError(f"Stored {field} disagrees with lesion-overlap counts")
        if positive and raw.get("dice_positive") not in (None, "") and abs(float(raw["dice_positive"]) - row["dice_positive"]) > 1e-12:
            raise ValueError("Stored positive-region Dice disagrees with pixel counts")
        identity = (bitewing, positive, target)
        if region in reference and reference[region] != identity:
            raise ValueError("Reference region identity changed between fitted models")
        reference[region] = identity
        key = (arch, strategy, row["seed"], row["fold"], mode, region)
        if key in seen:
            raise ValueError("Duplicate fitted-model/threshold/region record")
        seen.add(key)
        converted.append(row)
    if not converted:
        raise ValueError("No main regional predictions were supplied")
    regions = {key: f"R{i:03d}" for i, key in enumerate(sorted(reference), 1)}
    bitewings = {key: f"B{i:03d}" for i, key in enumerate(sorted({r["bitewing"] for r in converted}), 1)}
    for row in converted:
        row["region"] = regions[row["region"]]
        row["bitewing"] = bitewings[row["bitewing"]]
    return converted


def write_regional_predictions(rows, output) -> dict:
    """Write the canonical CSV or CSV.gz atomically and return coverage metadata."""
    rows = canonical_predictions(rows)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False) as handle:
            temporary = Path(handle.name)
        with temporary.open("wb") as raw:
            binary = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) if output.suffix == ".gz" else raw
            with io.TextIOWrapper(binary, encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        os.replace(temporary, output)
        temporary = None
        return {"path": str(output), "records": len(rows), "sha256": digest,
                "regions": len({r["region"] for r in rows}), "bitewings": len({r["bitewing"] for r in rows}),
                "fitted_models": len({(r["architecture"], r["strategy"], r["seed"], r["fold"]) for r in rows}),
                "threshold_modes": sorted({r["threshold_mode"] for r in rows})}
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
