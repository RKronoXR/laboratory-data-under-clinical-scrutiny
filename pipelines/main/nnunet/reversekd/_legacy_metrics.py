"""Surface endpoints. Overlap AP is evaluation-only, not a deployable score.

All decisions use p >= threshold. A diseased surface is detected only when a
prediction overlaps its reference mask; an alarm on a sound surface is FP.
Conventional surface ROC/AP instead uses max(p) for EVERY surface.
"""
from __future__ import annotations

import math
import numpy as np


def as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "1.0", "true", "yes"}
    return bool(value)


def _number(value):
    if value is None or value == "":
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def select_records(rows, threshold_name="fixed", eligible_only=True):
    if hasattr(rows, "to_dict"):
        rows = rows.to_dict("records")
    return [r for r in rows
            if r.get("threshold_name", "fixed") == threshold_name
            and (not eligible_only or as_bool(r.get("eligible", True)))]


def surface_record(reference, probability, threshold=0.5):
    reference = np.asarray(reference) > 0
    probability = np.asarray(probability)
    if reference.shape != probability.shape or reference.ndim != 2 or not reference.size:
        raise ValueError("Reference and probability must be nonempty matching 2D arrays.")
    if not np.issubdtype(probability.dtype, np.floating):
        raise ValueError("Floating point probabilities are required.")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Probabilities must be finite and in [0, 1].")
    if not math.isfinite(float(threshold)) or threshold < 0:
        raise ValueError("Threshold must be finite and nonnegative.")
    predicted = probability >= threshold
    rp, pp = int(reference.sum()), int(predicted.sum())
    intersection = int((reference & predicted).sum())
    return {"reference_positive": int(rp > 0), "reference_pixels": rp,
            "predicted_pixels": pp, "intersection_pixels": intersection,
            "score": float(probability.max()),
            "overlap_score": float(probability[reference].max()) if rp else None,
            "dice_positive": 2.0 * intersection / (rp + pp) if rp else None,
            "detected": int(intersection > 0), "threshold": float(threshold)}


def count_metrics(tp, fp, tn, fn):
    def div(a, b):
        return float(a / b) if b else None
    sensitivity, specificity = div(tp, tp + fn), div(tn, tn + fp)
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            "n_surfaces": int(tp + fp + tn + fn), "n_positive": int(tp + fn),
            "n_negative": int(tn + fp), "sensitivity": sensitivity,
            "specificity": specificity, "ppv": div(tp, tp + fp), "npv": div(tn, tn + fn),
            "f1": div(2 * tp, 2 * tp + fp + fn), "accuracy": div(tp + tn, tp + fp + tn + fn),
            "balanced_accuracy": (sensitivity + specificity) / 2 if sensitivity is not None and specificity is not None else None,
            "mcc": float((tp * tn - fp * fn) / math.sqrt(denominator)) if denominator else None,
            "youden_j": sensitivity + specificity - 1 if sensitivity is not None and specificity is not None else None}


def curve_records(rows, mode="overlap"):
    """Return threshold curve and AP/ROC areas, or None for empty/one-class data.

    AP is the step integral (average precision), not trapezoidal PR area.
    The overlap curve uses reference-mask maxima on diseased surfaces only.
    """
    if mode not in {"overlap", "surface"}:
        raise ValueError("mode must be overlap or surface")
    positive = [r for r in rows if as_bool(r["reference_positive"])]
    negative = [r for r in rows if not as_bool(r["reference_positive"])]
    if not positive or not negative:
        return None
    p = np.asarray([float(r["overlap_score" if mode == "overlap" else "score"]) for r in positive])
    n = np.asarray([float(r["score"]) for r in negative])
    if not np.isfinite(p).all() or not np.isfinite(n).all():
        raise ValueError("Nonfinite surface scores.")
    thresholds = np.unique(np.r_[p, n, 0.0, 0.5, 1.0, float(np.nextafter(np.float32(1), np.float32(np.inf)))])[::-1]
    tp = len(p) - np.searchsorted(np.sort(p), thresholds, side="left")
    fp = len(n) - np.searchsorted(np.sort(n), thresholds, side="left")
    recall, fpr = tp / len(p), fp / len(n)
    precision = np.divide(tp, tp + fp, out=np.ones(len(tp), dtype=float), where=(tp + fp) != 0)
    return {"threshold": thresholds, "recall": recall, "precision": precision, "fpr": fpr,
            "youden_j": recall - fpr, "integer_j": tp * len(n) - fp * len(p),
            "auprc_ap": float(np.sum(np.diff(recall) * precision[1:])),
            "auprc_trapezoid": float(np.sum(np.diff(recall) * (precision[1:] + precision[:-1]) / 2)),
            "auroc": float(np.sum(np.diff(fpr) * (recall[1:] + recall[:-1]) / 2))}


def summarize_records(rows, threshold_name="fixed", eligible_only=True):
    """Summarize saved decisions, preserving each fold's own tuned threshold."""
    selected = select_records(rows, threshold_name, eligible_only)
    tp = fp = tn = fn = misplaced = 0
    dices, hit_dices, thresholds = [], [], []
    for row in selected:
        positive = as_bool(row["reference_positive"])
        hit = int(float(row["intersection_pixels"])) > 0
        alarm = int(float(row["predicted_pixels"])) > 0
        threshold = _number(row.get("threshold"))
        if threshold is not None:
            thresholds.append(threshold)
        if positive:
            tp += int(hit)
            fn += int(not hit)
            misplaced += int(not hit and alarm)
            dice = _number(row.get("dice_positive"))
            if dice is None:
                rp, pp, inter = [int(float(row[k])) for k in ("reference_pixels", "predicted_pixels", "intersection_pixels")]
                dice = 2.0 * inter / (rp + pp)
            dices.append(dice)
            if hit:
                hit_dices.append(dice)
        else:
            fp += int(alarm)
            tn += int(not alarm)
    result = count_metrics(tp, fp, tn, fn)
    result.update(mislocalized_positive_surfaces=misplaced, n_dice_positive=len(dices), n_dice_tp=len(hit_dices),
                  dice_positive_mean=float(np.mean(dices)) if dices else None,
                  dice_positive_median=float(np.median(dices)) if dices else None,
                  dice_positive_std=float(np.std(dices, ddof=1)) if len(dices) > 1 else None,
                  dice_tp_mean=float(np.mean(hit_dices)) if hit_dices else None,
                  dice_tp_median=float(np.median(hit_dices)) if hit_dices else None,
                  dice_tp_std=float(np.std(hit_dices, ddof=1)) if len(hit_dices) > 1 else None,
                  threshold_min=min(thresholds) if thresholds else None,
                  threshold_max=max(thresholds) if thresholds else None)
    for mode in ("overlap", "surface"):
        curve = curve_records(selected, mode)
        for key in ("auprc_ap", "auprc_trapezoid", "auroc"):
            result[f"{mode}_{key}"] = curve[key] if curve else None
    return result


def choose_youden(rows, eligible_only=True):
    """Tuning-set overlap Youden; ties closest to .5 then smaller threshold.

    No class contrast means an undefined optimum: use the prespecified .5.
    """
    selected = select_records(rows, "fixed", eligible_only)
    curve = curve_records(selected, "overlap")
    if curve is None:
        return 0.5
    best = np.flatnonzero(curve["integer_j"] == curve["integer_j"].max())
    i = min(best, key=lambda j: (abs(curve["threshold"][j] - .5), curve["threshold"][j]))
    return float(curve["threshold"][i])
