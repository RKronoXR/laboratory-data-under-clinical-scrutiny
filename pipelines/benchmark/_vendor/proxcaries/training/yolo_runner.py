from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
import yaml

from proxcaries.evaluation import evaluate_prediction_arrays
from proxcaries.training.common import RunSpec, finish_run, split_manifest, start_run


def _link_or_copy(src: str | Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = Path(src)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _prepare_yolo_dataset(train_df: pd.DataFrame, val_df: pd.DataFrame, dataset_root: Path) -> Path:
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    for split, df in [("train", train_df), ("val", val_df)]:
        for r in df.itertuples():
            _link_or_copy(r.image_path, dataset_root / "images" / split / f"{r.sample_id}.png")
            _link_or_copy(r.mask_path, dataset_root / "masks" / split / f"{r.sample_id}.png")
    data = {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "masks_dir": "masks",
        "names": {0: "background", 1: "caries"},
    }
    yaml_path = dataset_root / "dataset.yaml"
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return yaml_path


def _raw_probability(model, image_path: str | Path, device: torch.device) -> np.ndarray | None:
    """Try to obtain class-1 probabilities directly from YOLO26-sem logits.

    Falls back to hard semantic masks when a future Ultralytics return signature
    differs. This keeps the pipeline executable while recording whether true
    probabilities were available.
    """
    try:
        arr = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            raw = model.model(x)
        if isinstance(raw, (tuple, list)):
            raw = raw[0]
        if not torch.is_tensor(raw) or raw.ndim != 4:
            return None
        raw = torch.nn.functional.interpolate(raw, size=arr.shape[:2], mode="bilinear", align_corners=False)
        if raw.shape[1] == 1:
            p = torch.sigmoid(raw[:, 0])
        else:
            p = torch.softmax(raw, dim=1)[:, 1]
        return p[0].detach().cpu().numpy().astype(np.float32)
    except Exception:
        return None


def run_yolo(df: pd.DataFrame, spec: RunSpec, cfg: dict, initialization_checkpoint: Path | None = None) -> dict:
    from ultralytics import YOLO

    tc = cfg["training"]
    max_epochs = int(tc["max_epochs"])
    start = start_run(spec, max_epochs)
    train_df, val_df = split_manifest(df, spec.fold)
    yaml_path = _prepare_yolo_dataset(train_df, val_df, spec.run_dir / "dataset")
    if initialization_checkpoint:
        model = YOLO(str(initialization_checkpoint))
    else:
        model = YOLO(str(cfg["yolo"]["model"]))
    results = model.train(
        data=str(yaml_path),
        epochs=max_epochs,
        patience=int(tc["patience"]),
        imgsz=int(cfg["project"]["image_size"]),
        batch=int(tc["batch_size"]),
        seed=spec.seed,
        deterministic=True,
        device=cfg["yolo"]["device"],
        project=str(spec.run_dir),
        name="ultralytics",
        exist_ok=True,
        verbose=True,
    )
    save_dir = Path(results.save_dir)
    best_path = save_dir / "weights" / "best.pt"
    trained = YOLO(str(best_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        trained.model.to(device).eval()
    except Exception:
        pass
    probability_dir = spec.run_dir / "probabilities"
    probability_dir.mkdir(exist_ok=True)
    probability_count = 0
    for r in val_df.itertuples():
        prob = _raw_probability(trained, r.image_path, device)
        if prob is not None:
            probability_count += 1
        else:
            result = trained.predict(source=r.image_path, imgsz=int(cfg["project"]["image_size"]), verbose=False)[0]
            prob = result.semantic_mask.data.detach().cpu().numpy().astype(np.float32)
            prob = (prob == 1).astype(np.float32)
        np.save(probability_dir / f"{r.sample_id}.npy", prob)
    metrics = evaluate_prediction_arrays(
        val_df, probability_dir, float(cfg["project"]["threshold"]),
        spec.run_dir / "surface_results.csv", spec.run_dir / "metrics.json",
    )
    finish_run(spec, start, metrics, {
        "best_checkpoint": str(best_path),
        "true_probability_maps": probability_count,
        "validation_samples": len(val_df),
    })
    return metrics


def train_yolo_stage(df: pd.DataFrame, run_dir: Path, cfg: dict, seed: int, fold: int, source: str, initialization_checkpoint: Path | None = None) -> tuple[dict, Path]:
    spec = RunSpec("yolo26_sem", source, seed, fold, run_dir)
    metrics = run_yolo(df, spec, cfg, initialization_checkpoint)
    return metrics, run_dir / "ultralytics" / "weights" / "best.pt"
