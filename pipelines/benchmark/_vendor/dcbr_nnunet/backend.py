"""Isolated native nnU-Net 2d supplement, with no planning search or crop training."""
from __future__ import annotations
import contextlib
import copy
import importlib.metadata
import io
import os
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image
from dcbr_compare.io import atomic_bytes, atomic_json, digest, event, file_hash, read_json


@contextlib.contextmanager
def _environment(values):
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update({key: str(value) for key, value in values.items()})
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _preprocess(bundle, cfg, run_root, stop_file):
    """Use the copied native plan once, without calling a planner or optimizing."""
    from nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor import DatasetFingerprintExtractor
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    directory = Path(bundle["preprocessed_directory"])
    raw = Path(bundle["raw_directory"])
    plan = PlansManager(str(bundle["plans_file"]))
    config = plan.get_configuration(bundle.get("configuration", "2d"))
    prepared = directory / config.data_identifier
    prepared.mkdir(parents=True, exist_ok=True)
    dataset = read_json(bundle["dataset_json_file"])
    source_signature = digest({"plans": file_hash(bundle["plans_file"]),
                               "dataset": file_hash(bundle["dataset_json_file"]),
                               "full_frame": file_hash(bundle["full_frame_path"])})
    receipt = directory / "full_preprocessing.json"
    if receipt.exists():
        old = read_json(receipt)
        if old.get("signature") != source_signature:
            raise RuntimeError("The frozen full-bitewing preprocessing inputs changed.")
        if not all(Path(p).is_file() and file_hash(p) == h for p, h in old["files"].items()):
            raise RuntimeError("A preprocessed nnU-Net file changed or disappeared.")
        return old
    if not (directory / "dataset_fingerprint.json").is_file():
        event(run_root, "nnunet_full_fingerprint", state="preprocessing", hpo=False)
        DatasetFingerprintExtractor(bundle["dataset_name"],
                                    num_processes=int(cfg.get("nnunet_preprocess_workers", 2)),
                                    verbose=False).run(overwrite_existing=False)
    preprocessor = config.preprocessor_class(verbose=False)
    identifiers = sorted(p.name[:-9] for p in (raw / "imagesTr").glob("*_0000.png"))
    if len(identifiers) != int(dataset["numTraining"]):
        raise ValueError("Raw full-bitewing case count disagrees with dataset.json.")
    gt = directory / "gt_segmentations"
    gt.mkdir(exist_ok=True)
    files = {}
    for index, case_id in enumerate(identifiers, 1):
        if Path(stop_file).exists():
            raise KeyboardInterrupt("STOP requested during full-bitewing preprocessing.")
        prefix = prepared / case_id
        marker = directory / "full_case_receipts" / (case_id + ".json")
        image = raw / "imagesTr" / (case_id + "_0000.png")
        mask = raw / "labelsTr" / (case_id + ".png")
        signature = digest({"source": source_signature, "image": file_hash(image), "mask": file_hash(mask)})
        expected = [Path(str(prefix) + suffix) for suffix in (".b2nd", "_seg.b2nd", ".pkl")]
        if not marker.exists():
            temporary = prepared / (case_id + "_temporary")
            preprocessor.run_case_save(str(temporary), [str(image)], str(mask), plan, config, dataset)
            for suffix, destination in zip((".b2nd", "_seg.b2nd", ".pkl"), expected):
                os.replace(str(temporary) + suffix, destination)
            atomic_json(marker, {"signature": signature, "files": {str(p): file_hash(p) for p in expected}})
        old = read_json(marker)
        if old.get("signature") != signature or not all(p.is_file() and file_hash(p) == old["files"].get(str(p)) for p in expected):
            raise RuntimeError(f"nnU-Net preprocessed case integrity failure: {case_id}")
        files.update(old["files"])
        target = gt / mask.name
        if target.exists() and file_hash(target) != file_hash(mask):
            raise RuntimeError("Full-bitewing validation reference changed.")
        if not target.exists():
            shutil.copy2(mask, target)
        event(run_root, "nnunet_full_preprocessing", state="preprocessing", completed=index,
              total=len(identifiers), case_id=case_id, hpo=False)
    outcome = {"signature": source_signature, "files": files, "cases": identifiers,
               "plans_file": str(bundle["plans_file"]), "planning_search": False}
    atomic_json(receipt, outcome)
    return outcome


def _convert_probabilities(paths, raw_directory, target):
    """Validate original-grid native NPZs and expose one float32 NPY per bitewing."""
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    results, archives = {}, {}
    for case_id, filename in paths.items():
        filename = Path(filename)
        with np.load(filename, allow_pickle=False) as archive:
            if "probabilities" not in archive:
                raise ValueError(f"Missing native probabilities in {filename}")
            probabilities = archive["probabilities"]
        if probabilities.ndim == 4 and probabilities.shape[1] == 1:
            probabilities = probabilities[:, 0]
        if probabilities.ndim != 3 or probabilities.shape[0] != 2:
            raise ValueError(f"Expected two-class native probabilities, got {probabilities.shape}")
        with Image.open(Path(raw_directory) / "imagesTr" / f"{case_id}_0000.png") as image:
            width, height = image.size
        if probabilities.shape[1:] != (height, width):
            raise ValueError("Native nnU-Net probabilities were not restored to the prepared full image grid.")
        if not np.isfinite(probabilities).all() or probabilities.min() < 0 or probabilities.max() > 1:
            raise ValueError("Invalid native nnU-Net probabilities.")
        if not np.allclose(probabilities.astype(np.float32).sum(axis=0), 1, atol=.002):
            raise ValueError("Native nnU-Net class probabilities do not sum to one.")
        bitewing_id = case_id[2:] if case_id.startswith("bw") else case_id
        output = target / f"{bitewing_id}.npy"
        buffer = io.BytesIO()
        np.save(buffer, probabilities[1].astype(np.float32), allow_pickle=False)
        atomic_bytes(output, buffer.getvalue())
        results[bitewing_id] = str(output)
        archives[bitewing_id] = str(filename)
    return results, archives


def _predict_native(trainer, expected_ids, stop_file, run_root):
    """Native inference and native export, sequential and atomic per radiograph."""
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.inference.export_prediction import export_prediction_from_logits
    from nnunetv2.inference.sliding_window_prediction import compute_gaussian

    _, identifiers = trainer.do_split()
    if set(identifiers) != set(expected_ids):
        raise RuntimeError("Native validation identifiers disagree with the frozen radiograph folds.")
    dataset = trainer.dataset_class(trainer.preprocessed_dataset_folder, identifiers,
                                    folder_with_segs_from_previous_stage=None)
    validation = Path(trainer.output_folder) / "validation"
    staging = Path(trainer.output_folder) / "prediction_staging"
    validation.mkdir(exist_ok=True)
    staging.mkdir(exist_ok=True)
    trainer.set_deep_supervision_enabled(False)
    trainer.network.eval()
    predictor = nnUNetPredictor(tile_step_size=.5, use_gaussian=True, use_mirroring=True,
                                perform_everything_on_device=True, device=trainer.device,
                                verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    predictor.manual_initialization(trainer.network, trainer.plans_manager, trainer.configuration_manager,
                                    None, trainer.dataset_json, trainer.__class__.__name__,
                                    trainer.inference_allowed_mirroring_axes)
    try:
        for index, case_id in enumerate(dataset.identifiers, 1):
            if Path(stop_file).exists():
                raise KeyboardInterrupt("STOP requested between nnU-Net validation predictions.")
            if not (validation / f"{case_id}.npz").is_file():
                data, _, _, properties = dataset.load_case(case_id)
                tensor = torch.from_numpy(np.array(data[:], copy=True))
                prediction = predictor.predict_sliding_window_return_logits(tensor).cpu()
                prefix = staging / case_id
                export_prediction_from_logits(prediction, properties, trainer.configuration_manager,
                                               trainer.plans_manager, trainer.dataset_json, str(prefix), True)
                generated = list(staging.glob(case_id + ".*"))
                # Publish the probabilities last so their presence marks a
                # complete native export rather than a half-written archive.
                for source in sorted(generated, key=lambda path: path.suffix == ".npz"):
                    os.replace(source, validation / source.name)
            event(run_root, "nnunet_validation_case", state="predicting", case_id=case_id,
                  completed=index, total=len(identifiers), seed=trainer._pc_seed, fold=trainer.fold)
    finally:
        trainer.set_deep_supervision_enabled(True)
        compute_gaussian.cache_clear()


def train_and_predict(spec, data_bundle, cfg):
    """Train one isolated seed/fold and export held-out full-bitewing probabilities.

    `probability_paths` maps bitewing ID (e.g. "001") to class-1 NPY arrays in
    the prepared full-image grid. Original two-class NPZ archives are retained.
    """
    if importlib.metadata.version("nnunetv2") != str(cfg.get("nnunet_version", "2.8.1")):
        raise RuntimeError("The nnU-Net supplement requires the historical nnunetv2 version 2.8.1.")
    if data_bundle.get("configuration", "2d") != "2d":
        raise ValueError("Only the previous native 2d configuration is supported.")
    directory = Path(spec["directory"]).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    run_root = Path(spec["run_root"]).resolve()
    stop_file = Path(spec.get("stop_file", run_root.parent / "STOP"))
    result_file = directory / "result.json"
    if result_file.exists():
        result = read_json(result_file)
        if result.get("signature") != spec["signature"]:
            raise RuntimeError("Completed nnU-Net run signature mismatch.")
        for filename, expected in result["artifact_hashes"].items():
            if not Path(filename).is_file() or file_hash(filename) != expected:
                raise RuntimeError("A completed nnU-Net run artifact changed.")
        return result
    if stop_file.exists():
        raise KeyboardInterrupt("STOP requested before nnU-Net training.")
    environment = {"nnUNet_raw": data_bundle["raw_root"],
                   "nnUNet_preprocessed": data_bundle["preprocessed_root"],
                   # Both seed and fold are isolated, preventing historical seed reuse.
                   "nnUNet_results": str(directory / "native_results")}
    if "nnunet_n_proc_da" in cfg:
        environment["nnUNet_n_proc_DA"] = cfg["nnunet_n_proc_da"]
    if "nnunet_compile" in cfg:
        environment["nnUNet_compile"] = "true" if cfg["nnunet_compile"] else "false"
    with _environment(environment):
        _preprocess(data_bundle, cfg, run_root, stop_file)
        import torch
        from . import native_trainer
        native_trainer.configure({"spec": spec, "cfg": cfg, "stop_file": str(stop_file)})
        plans = copy.deepcopy(read_json(data_bundle["plans_file"]))
        model_dir = (Path(environment["nnUNet_results"]) / data_bundle["dataset_name"] /
                     f"nnUNetTrainerDCBRFullBW__{data_bundle.get('plans_name', 'nnUNetPlans')}__2d" /
                     f"fold_{int(spec['fold'])}")
        latest, best = model_dir / "checkpoint_latest.pth", model_dir / "checkpoint_best.pth"
        done_file = directory / "training_done.json"
        plans["continue_training"] = latest.exists() or done_file.exists()
        device_string = str(cfg.get("device", "0"))
        if device_string != "cpu" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use CPU only for explicit smoke tests.")
        device = torch.device("cpu" if device_string == "cpu" else "cuda")
        trainer = native_trainer.nnUNetTrainerDCBRFullBW(plans, "2d", int(spec["fold"]),
                                                       read_json(data_bundle["dataset_json_file"]), device)
        if Path(trainer.output_folder).resolve() != model_dir.resolve():
            raise RuntimeError("Native nnU-Net output escaped its isolated seed/fold directory.")
        if not done_file.exists():
            if latest.is_file():
                trainer.load_checkpoint(str(latest))
            started = time.monotonic()
            trainer.run_training()
            if not best.is_file():
                raise RuntimeError("Native nnU-Net training produced no best checkpoint.")
            atomic_json(done_file, {"signature": spec["signature"], "epochs_completed": trainer.current_epoch,
                                   "best_epoch": trainer._pc_best_epoch, "best_ema_foreground_dice": trainer._best_ema,
                                   "seconds": trainer._pc_seconds + time.monotonic() - trainer._pc_started,
                                   "seconds_this_attempt": time.monotonic() - started})
        done = read_json(done_file)
        if done.get("signature") != spec["signature"]:
            raise RuntimeError("nnU-Net training completion signature mismatch.")
        validation_ids = data_bundle["validation_ids_by_fold"][str(int(spec["fold"]))]
        native_paths = {case_id: model_dir / "validation" / f"{case_id}.npz" for case_id in validation_ids}
        if not all(path.is_file() for path in native_paths.values()):
            if stop_file.exists():
                raise KeyboardInterrupt("STOP requested before nnU-Net validation predictions.")
            trainer.load_checkpoint(str(best))
            event(run_root, "nnunet_predicting", state="predicting", architecture="nnunetv2", condition="full",
                  seed=int(spec["seed"]), fold=int(spec["fold"]), cases=len(native_paths))
            _predict_native(trainer, validation_ids, stop_file, run_root)
        probability_paths, npz_paths = _convert_probabilities(native_paths, data_bundle["raw_directory"], directory / "probabilities")
        artifacts = [best, *map(Path, probability_paths.values()), *map(Path, npz_paths.values())]
        result = {"signature": spec["signature"], "architecture": "nnunetv2", "condition": "full",
                  "seed": int(spec["seed"]), "fold": int(spec["fold"]), "hpo": False,
                  "checkpoint": str(best), "checkpoint_sha256": file_hash(best),
                  "probability_paths": probability_paths, "native_npz_paths": npz_paths,
                  "probability_grid": "prepared_full_image", "selection_metric": "EMA_foreground_Dice",
                  "native_validation_mirroring": True, "native_validation_step_size": .5,
                  "artifact_hashes": {str(path): file_hash(path) for path in artifacts}, **done}
        atomic_json(result_file, result)
        event(run_root, "nnunet_completed", state="completed", architecture="nnunetv2", condition="full",
              seed=int(spec["seed"]), fold=int(spec["fold"]), best_epoch=done["best_epoch"],
              epochs_completed=done["epochs_completed"])
        del trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result
