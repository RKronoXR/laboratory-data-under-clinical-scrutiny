from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from proxcaries.evaluation import evaluate_prediction_arrays
from proxcaries.training.common import RunSpec, finish_run, split_manifest, start_run

DATASET_NAMES = {
    "acta_direct": "ACTADIRECT",
    "dcbr": "DCBR",
    "pooled": "POOLED",
}


def nnunet_env(project_root: Path, cfg: dict) -> dict[str, str]:
    work = project_root / cfg["data"]["work_dir"] / "nnunet"
    env = os.environ.copy()
    env["nnUNet_raw"] = str((work / "raw").resolve())
    env["nnUNet_preprocessed"] = str((work / "preprocessed").resolve())
    env["nnUNet_results"] = str((work / "results").resolve())
    for k in ["nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"]:
        Path(env[k]).mkdir(parents=True, exist_ok=True)
    return env


def dataset_folder_name(source: str, cfg: dict) -> str:
    did = int(cfg["nnunet"]["dataset_ids"][source])
    return f"Dataset{did:03d}_{DATASET_NAMES[source]}"


def export_raw_dataset(source: str, manifest: pd.DataFrame, project_root: Path, cfg: dict) -> Path:
    env = nnunet_env(project_root, cfg)
    folder = Path(env["nnUNet_raw"]) / dataset_folder_name(source, cfg)
    images = folder / "imagesTr"
    labels = folder / "labelsTr"
    if folder.exists():
        shutil.rmtree(folder)
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for r in manifest.itertuples():
        image_dst = images / f"{r.sample_id}_0000.png"
        label_dst = labels / f"{r.sample_id}.png"
        try:
            os.link(r.image_path, image_dst)
            os.link(r.mask_path, label_dst)
        except OSError:
            shutil.copy2(r.image_path, image_dst)
            shutil.copy2(r.mask_path, label_dst)
    dataset_json = {
        "channel_names": {"0": "grayscale"},
        "labels": {"background": 0, "caries": 1},
        "numTraining": int(len(manifest)),
        "file_ending": ".png",
    }
    (folder / "dataset.json").write_text(json.dumps(dataset_json, indent=2), encoding="utf-8")
    return folder


def write_splits(source: str, manifest: pd.DataFrame, project_root: Path, cfg: dict) -> Path:
    env = nnunet_env(project_root, cfg)
    pre = Path(env["nnUNet_preprocessed"]) / dataset_folder_name(source, cfg)
    pre.mkdir(parents=True, exist_ok=True)
    splits = []
    for fold in sorted(manifest["fold"].astype(int).unique()):
        train = manifest.loc[manifest.fold.astype(int) != fold, "sample_id"].astype(str).tolist()
        val = manifest.loc[manifest.fold.astype(int) == fold, "sample_id"].astype(str).tolist()
        splits.append({"train": train, "val": val})
    path = pre / "splits_final.json"
    path.write_text(json.dumps(splits, indent=2), encoding="utf-8")
    return path


def run_cmd(cmd: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, check=True)


def plan_and_preprocess(source: str, manifest: pd.DataFrame, project_root: Path, cfg: dict) -> None:
    env = nnunet_env(project_root, cfg)
    did = str(int(cfg["nnunet"]["dataset_ids"][source]))
    run_cmd(["nnUNetv2_plan_and_preprocess", "-d", did, "-c", str(cfg["nnunet"]["configuration"]), "--verify_dataset_integrity"], env)
    write_splits(source, manifest, project_root, cfg)


def prepare_transfer_plans(pretraining_source: str, finetuning_source: str, pretraining_manifest: pd.DataFrame, project_root: Path, cfg: dict) -> str:
    """Align pretraining topology to the finetuning dataset as required by nnU-Net."""
    env = nnunet_env(project_root, cfg)
    pre_id = str(int(cfg["nnunet"]["dataset_ids"][pretraining_source]))
    fine_id = str(int(cfg["nnunet"]["dataset_ids"][finetuning_source]))
    transferred = f"ProxCariesFrom{finetuning_source.replace('_', '').upper()}"
    run_cmd(["nnUNetv2_plan_and_preprocess", "-d", fine_id, "-c", str(cfg["nnunet"]["configuration"]), "--verify_dataset_integrity"], env)
    run_cmd(["nnUNetv2_extract_fingerprint", "-d", pre_id], env)
    run_cmd([
        "nnUNetv2_move_plans_between_datasets",
        "-s", fine_id, "-t", pre_id,
        "-sp", "nnUNetPlans", "-tp", transferred,
    ], env)
    # Some nnU-Net releases do not copy dataset.json into the preprocessed source folder.
    pre_folder = Path(env["nnUNet_preprocessed"]) / dataset_folder_name(pretraining_source, cfg)
    pre_folder.mkdir(parents=True, exist_ok=True)
    raw_json = Path(env["nnUNet_raw"]) / dataset_folder_name(pretraining_source, cfg) / "dataset.json"
    if not (pre_folder / "dataset.json").exists():
        shutil.copy2(raw_json, pre_folder / "dataset.json")
    run_cmd(["nnUNetv2_preprocess", "-d", pre_id, "-plans_name", transferred, "-c", str(cfg["nnunet"]["configuration"])], env)
    write_splits(pretraining_source, pretraining_manifest, project_root, cfg)
    return transferred


def _result_fold(project_root: Path, source: str, fold: int, plans: str, cfg: dict) -> Path:
    env = nnunet_env(project_root, cfg)
    return Path(env["nnUNet_results"]) / dataset_folder_name(source, cfg) / f"nnUNetTrainerCariesEarlyStop__{plans}__{cfg['nnunet']['configuration']}" / f"fold_{fold}"


def _convert_validation_probabilities(validation_dir: Path, val_df: pd.DataFrame, out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for r in val_df.itertuples():
        src = validation_dir / f"{r.sample_id}.npz"
        if not src.exists():
            raise FileNotFoundError(f"Expected nnU-Net probability archive not found: {src}")
        data = np.load(src)
        if "probabilities" in data:
            p = data["probabilities"]
        elif len(data.files) == 1:
            p = data[data.files[0]]
        else:
            raise KeyError(f"Cannot identify probabilities in {src}: {data.files}")
        # nnU-Net NaturalImage2DIO may save 2D probabilities as
        # (classes, 1, height, width). Remove only the synthetic singleton
        # depth axis, preserving the class axis and the 2D probability map.
        if p.ndim == 4 and p.shape[1] == 1:
            p = p[:, 0, :, :]
        elif p.ndim == 4 and p.shape[0] == 1:
            # Defensive support for arrays exported as (1, classes, H, W).
            p = p[0, :, :, :]

        if p.ndim == 3 and p.shape[0] >= 2:
            caries = p[1]
        elif p.ndim == 3 and p.shape[0] == 1:
            caries = p[0]
        else:
            raise ValueError(f"Unexpected nnU-Net probability shape {p.shape} in {src}")
        np.save(out_dir / f"{r.sample_id}.npy", caries.astype(np.float32))
        count += 1
    return count


def run_nnunet(
    df: pd.DataFrame,
    spec: RunSpec,
    cfg: dict,
    project_root: Path,
    plans: str = "nnUNetPlans",
    initialization_checkpoint: Path | None = None,
) -> dict:
    max_epochs = int(cfg["training"]["max_epochs"])
    start = start_run(spec, max_epochs)
    _, val_df = split_manifest(df, spec.fold)
    env = nnunet_env(project_root, cfg)

    # Keep native nnU-Net outputs independent for each random seed.
    seed_results = Path(env["nnUNet_results"]) / f"seed_{spec.seed}"
    seed_results.mkdir(parents=True, exist_ok=True)
    env["nnUNet_results"] = str(seed_results.resolve())

    env["PROXCARIES_NNUNET_MAX_EPOCHS"] = str(max_epochs)
    env["PROXCARIES_NNUNET_PATIENCE"] = str(int(cfg["training"]["patience"]))
    env["PROXCARIES_NNUNET_MIN_DELTA"] = str(float(cfg["training"]["min_delta"]))
    env["PROXCARIES_SEED"] = str(spec.seed)
    did = str(int(cfg["nnunet"]["dataset_ids"][spec.source]))
    cmd = [
        "nnUNetv2_train", did, str(cfg["nnunet"]["configuration"]), str(spec.fold),
        "-tr", "nnUNetTrainerCariesEarlyStop", "-p", plans, "--npz", "--val_best",
    ]
    if initialization_checkpoint:
        cmd += ["-pretrained_weights", str(initialization_checkpoint)]
    result_fold = (
        Path(env["nnUNet_results"])
        / dataset_folder_name(spec.source, cfg)
        / f"nnUNetTrainerCariesEarlyStop__{plans}__{cfg['nnunet']['configuration']}"
        / f"fold_{spec.fold}"
    )
    validation_dir = result_fold / "validation"
    best_ckpt = result_fold / "checkpoint_best.pth"
    expected_validation = [validation_dir / f"{sid}.npz" for sid in val_df["sample_id"].astype(str)]
    validation_complete = best_ckpt.exists() and all(path.exists() for path in expected_validation)
    if validation_complete:
        print(
            f"Reusing completed nnU-Net fold {spec.fold}: "
            f"{len(expected_validation)} validation probability archives already exist.",
            flush=True,
        )
    else:
        run_cmd(cmd, env)
    probability_dir = spec.run_dir / "probabilities"
    n = _convert_validation_probabilities(result_fold / "validation", val_df, probability_dir)
    metrics = evaluate_prediction_arrays(
        val_df, probability_dir, float(cfg["project"]["threshold"]),
        spec.run_dir / "surface_results.csv", spec.run_dir / "metrics.json",
    )
    finish_run(spec, start, metrics, {"best_checkpoint": str(best_ckpt), "validation_samples": n, "plans": plans})
    return metrics
