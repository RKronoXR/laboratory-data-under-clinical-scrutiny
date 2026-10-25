from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def dice_binary(reference: np.ndarray, prediction: np.ndarray) -> float:
    ref = reference.astype(bool)
    pred = prediction.astype(bool)
    denom = int(ref.sum()) + int(pred.sum())
    if denom == 0:
        return float("nan")
    return 2.0 * float(np.logical_and(ref, pred).sum()) / denom


def evaluate_surface(reference: np.ndarray, probability: np.ndarray, threshold: float = 0.5) -> dict:
    ref = reference.astype(bool)
    prob = np.asarray(probability, dtype=np.float32)
    pred = prob >= threshold
    ref_positive = bool(ref.any())
    pred_any = bool(pred.any())
    overlap = bool(np.logical_and(ref, pred).any())
    if ref_positive:
        tp = int(overlap)
        fn = int(not overlap)
        fp = 0
        tn = 0
        dsc = dice_binary(ref, pred) if overlap else 0.0
    else:
        tp = fn = 0
        fp = int(pred_any)
        tn = int(not pred_any)
        dsc = float("nan")
    return {
        "reference_positive": int(ref_positive),
        "score": float(prob.max()) if prob.size else 0.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "correct": int(tp + tn > 0),
        "dice": dsc,
        "predicted_pixels": int(pred.sum()),
        "reference_pixels": int(ref.sum()),
    }


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def summarize_surface_results(df: pd.DataFrame) -> dict:
    tp, fp, tn, fn = (int(df[c].sum()) for c in ["tp", "fp", "tn", "fn"])
    accuracy = _safe_div(tp + tn, tp + fp + tn + fn)
    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    ppv = _safe_div(tp, tp + fp)
    npv = _safe_div(tn, tn + fn)
    balanced = np.nanmean([sensitivity, specificity])
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    y = df["reference_positive"].to_numpy(dtype=int)
    s = df["score"].to_numpy(dtype=float)
    auroc = float(roc_auc_score(y, s)) if np.unique(y).size == 2 else float("nan")
    auprc = float(average_precision_score(y, s)) if np.unique(y).size == 2 else float("nan")
    positive_dice = df.loc[df["reference_positive"] == 1, "dice"].astype(float)
    dice = float(positive_dice.mean()) if len(positive_dice) else float("nan")
    return {
        "n_surfaces": len(df),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "balanced_accuracy": float(balanced),
        "f1": f1,
        "auroc": auroc,
        "auprc": auprc,
        "dice": dice,
    }


def cluster_bootstrap(
    df: pd.DataFrame,
    cluster_col: str = "bitewing_id",
    iterations: int = 10_000,
    seed: int = 2026,
) -> pd.DataFrame:
    """Paired cluster bootstrap by bitewing.

    If a table contains multiple strategies, all rows for a sampled bitewing are
    carried together, preserving the paired strategy predictions.
    """
    rng = np.random.default_rng(seed)
    clusters = np.asarray(sorted(df[cluster_col].astype(str).unique()))
    if len(clusters) == 0:
        raise ValueError("No clusters available for bootstrap")
    out = []
    strategy_col = "strategy" if "strategy" in df.columns else None
    for i in range(iterations):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        parts = []
        for j, cluster in enumerate(sampled):
            x = df[df[cluster_col].astype(str) == cluster].copy()
            x["_bootstrap_cluster"] = f"{j}:{cluster}"
            parts.append(x)
        boot = pd.concat(parts, ignore_index=True)
        groups = boot.groupby(strategy_col) if strategy_col else [("all", boot)]
        for strategy, sub in groups:
            metrics = summarize_surface_results(sub)
            metrics.update({"iteration": i, "strategy": strategy})
            out.append(metrics)
    return pd.DataFrame(out)


def bootstrap_ci(bootstrap_df: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    rows = []
    for strategy, sub in bootstrap_df.groupby("strategy"):
        for metric in metrics:
            vals = sub[metric].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            rows.append({
                "strategy": strategy,
                "metric": metric,
                "lower_95": float(np.quantile(vals, 0.025)) if len(vals) else math.nan,
                "upper_95": float(np.quantile(vals, 0.975)) if len(vals) else math.nan,
            })
    return pd.DataFrame(rows)
