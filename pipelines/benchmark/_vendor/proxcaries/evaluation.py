from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from proxcaries.metrics import evaluate_surface, summarize_surface_results
from proxcaries.utils import save_json


def load_mask(path: str | Path) -> np.ndarray:
    with Image.open(path) as im:
        return (np.asarray(im) > 0).astype(np.uint8)


def evaluate_prediction_arrays(
    manifest: pd.DataFrame,
    probability_dir: str | Path,
    threshold: float,
    output_csv: str | Path,
    metrics_json: str | Path,
) -> dict:
    probability_dir = Path(probability_dir)
    rows = []
    for _, r in manifest.iterrows():
        sample_id = str(r["sample_id"])
        prob_path = probability_dir / f"{sample_id}.npy"
        if not prob_path.exists():
            raise FileNotFoundError(prob_path)
        prob = np.load(prob_path)
        ref = load_mask(r["mask_path"])
        if prob.shape != ref.shape:
            raise ValueError(f"Shape mismatch {sample_id}: prob={prob.shape}, ref={ref.shape}")
        result = evaluate_surface(ref, prob, threshold)
        result.update({
            "sample_id": sample_id,
            "source": r.get("source", ""),
            "group_id": r.get("group_id", ""),
            "bitewing_id": r.get("bitewing_id", ""),
            "surface": r.get("surface", ""),
        })
        rows.append(result)
    out = pd.DataFrame(rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    metrics = summarize_surface_results(out)
    save_json(metrics, metrics_json)
    return metrics
