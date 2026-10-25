from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from monai.data import DataLoader, Dataset
from monai.losses import DiceFocalLoss
from monai.networks.nets import SegResNet
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    RandAffined,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ScaleIntensityd,
)
from monai.utils import set_determinism

from proxcaries.evaluation import evaluate_prediction_arrays
from proxcaries.status import write_status
from proxcaries.training.common import RunSpec, finish_run, split_manifest, start_run


def _records(df: pd.DataFrame) -> list[dict]:
    return [{"image": r.image_path, "label": r.mask_path, "sample_id": r.sample_id} for r in df.itertuples()]


def _transforms(train: bool):
    items = [
        LoadImaged(keys=["image", "label"], image_only=True),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
    ]
    if train:
        items += [
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandAffined(
                keys=["image", "label"], prob=0.30,
                rotate_range=(np.deg2rad(5.0),),
                translate_range=(8, 8),
                scale_range=(0.05, 0.05),
                mode=("bilinear", "nearest"), padding_mode="zeros",
            ),
            RandScaleIntensityd(keys=["image"], factors=0.10, prob=0.30),
            RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.30),
        ]
    items.append(EnsureTyped(keys=["image", "label"], dtype=torch.float32))
    return Compose(items)


def _build_model(cfg: dict, device: torch.device) -> torch.nn.Module:
    mc = cfg["monai"]
    return SegResNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        init_filters=int(mc["init_filters"]),
        blocks_down=tuple(mc["blocks_down"]),
        blocks_up=tuple(mc["blocks_up"]),
        norm="BATCH",
    ).to(device)


def _positive_dice_values(logits: torch.Tensor, target: torch.Tensor) -> list[float]:
    """Per-surface Dice values for reference-positive surfaces only."""
    pred = (torch.sigmoid(logits) >= 0.5)
    target = target > 0.5
    vals: list[float] = []
    for p, t in zip(pred, target):
        if t.any():
            inter = (p & t).sum().item()
            denom = p.sum().item() + t.sum().item()
            vals.append(0.0 if denom == 0 else 2.0 * inter / denom)
    return vals


def run_monai(df: pd.DataFrame, spec: RunSpec, cfg: dict, initialization_checkpoint: Path | None = None) -> dict:
    tc = cfg["training"]
    max_epochs = int(tc["max_epochs"])
    start = start_run(spec, max_epochs)
    set_determinism(seed=spec.seed)
    train_df, val_df = split_manifest(df, spec.fold)
    batch_size = int(tc["batch_size"])
    workers = int(tc["num_workers"])
    train_loader = DataLoader(Dataset(_records(train_df), _transforms(True)), batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(Dataset(_records(val_df), _transforms(False)), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(cfg, device)
    if initialization_checkpoint:
        ckpt = torch.load(initialization_checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=True)
    loss_fn = DiceFocalLoss(sigmoid=True, lambda_dice=1.0, lambda_focal=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tc["learning_rate"]), weight_decay=float(tc["weight_decay"]))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best = -1.0
    stale = 0
    best_path = spec.run_dir / "best.pt"
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            x = batch["image"].to(device)
            y = (batch["label"] > 0).float().to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
        model.eval()
        val_dices = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device)
                y = (batch["label"] > 0).float().to(device)
                logits = model(x)
                val_dices.extend(_positive_dice_values(logits, y))
        metric = float(np.mean(val_dices)) if val_dices else 0.0
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "positive_dice": metric})
        improved = metric > best + float(tc["min_delta"])
        if improved:
            best = metric
            stale = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "metric": metric}, best_path)
        else:
            stale += 1
        write_status(spec.status_path, current_epoch=epoch, best_metric=best, seconds_elapsed=time.time() - start)
        pd.DataFrame(history).to_csv(spec.run_dir / "history.csv", index=False)
        if stale >= int(tc["patience"]):
            break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    probability_dir = spec.run_dir / "probabilities"
    probability_dir.mkdir(exist_ok=True)
    eval_loader = DataLoader(Dataset(_records(val_df), _transforms(False)), batch_size=batch_size, shuffle=False, num_workers=workers)
    with torch.no_grad():
        for batch in eval_loader:
            prob = torch.sigmoid(model(batch["image"].to(device))).cpu().numpy()[:, 0]
            ids = batch["sample_id"]
            for sample_id, p in zip(ids, prob):
                np.save(probability_dir / f"{sample_id}.npy", p.astype(np.float32))
    metrics = evaluate_prediction_arrays(
        val_df, probability_dir, float(cfg["project"]["threshold"]),
        spec.run_dir / "surface_results.csv", spec.run_dir / "metrics.json",
    )
    finish_run(spec, start, metrics, {"best_checkpoint": str(best_path), "best_epoch": int(checkpoint["epoch"])})
    return metrics


def train_monai_stage(df: pd.DataFrame, run_dir: Path, cfg: dict, seed: int, fold: int, source: str, initialization_checkpoint: Path | None = None) -> tuple[dict, Path]:
    spec = RunSpec("monai_segresnet", source, seed, fold, run_dir)
    metrics = run_monai(df, spec, cfg, initialization_checkpoint)
    return metrics, run_dir / "best.pt"
