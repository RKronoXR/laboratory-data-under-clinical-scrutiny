"""Native YOLO26 semantic training using the prior architecture-selection protocol.

Native validation mIoU selects checkpoints, without HPO. Checkpoints preserve the
live model separately from EMA, with the optimizer, scheduler, scaler and RNG.
"""
from __future__ import annotations

import copy
import io
import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .io import atomic_bytes, atomic_json, digest, event, file_hash, read_json


def _dataset(folder, train_frame, tune_frame, condition):
    """Link prepared binary PNGs without changing original images or masks."""
    import yaml

    manifest = []
    image_size = None
    for split, frame in (("train", train_frame), ("tune", tune_frame)):
        if frame.empty:
            raise ValueError(f"No {split} images for {condition}.")
        if frame["image_path"].astype(str).duplicated().any():
            raise ValueError(f"Duplicate {split} training image paths.")
        for index, row in enumerate(frame.to_dict("records")):
            sources = [Path(row[key]).resolve() for key in ("image_path", "mask_path")]
            if not all(p.is_file() for p in sources):
                raise FileNotFoundError(f"Missing prepared image/mask pair: {sources}")
            # Prepared images are already letterboxed. A second size adjustment
            # would silently invalidate the comparison's recorded resolution.
            with Image.open(sources[0]) as im:
                width, height = im.size
            with Image.open(sources[1]) as mask:
                if mask.size != (width, height):
                    raise ValueError("Prepared training image and mask have different sizes.")
            if width != height or width % 32:
                raise ValueError("Prepared images must be square with dimensions divisible by 32.")
            if image_size is not None and width != image_size:
                raise ValueError("Prepared images within one condition have inconsistent sizes.")
            image_size = width
            key = f"{index:06d}_{digest(str(sources[0]))[:12]}.png"
            for kind, source in zip(("images", "masks"), sources):
                target = folder / "dataset" / kind / split / key
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.is_symlink():
                    if target.resolve() != source:
                        raise RuntimeError(f"Dataset link changed: {target}")
                elif target.exists():
                    raise RuntimeError(f"Refusing to replace a non-link dataset file: {target}")
                else:
                    target.symlink_to(source)
            manifest.append({"split": split, "key": key,
                             "image_path": str(sources[0]), "mask_path": str(sources[1])})
    manifest_path = folder / "dataset" / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise RuntimeError("Training dataset changed within an existing candidate directory.")
    atomic_json(manifest_path, manifest)
    data = {"path": str((folder / "dataset").resolve()), "train": "images/train",
            "val": "images/tune", "masks_dir": "masks",
            "names": {0: "background", 1: "proximal_caries"},
            "label_mapping": {0: 0, 1: 1, 255: 1}}
    path = folder / "dataset" / "data.yaml"
    atomic_bytes(path, yaml.safe_dump(data, sort_keys=False).encode())
    return path, image_size


def train_candidate(spec, train_frame, tune_images, tune_surfaces, cfg):
    """Train or resume one proximal-crop seed/fold, then evaluate its selected EMA.

    A STOP file is honored after a complete epoch and its atomic checkpoint.
    Interrupting during an epoch replays it from the last completed checkpoint.
    Native gradient accumulation and multiworker augmentation mean a resumed
    trajectory is not guaranteed bit-identical to uninterrupted training.
    """
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.models.yolo.semantic.train import SemanticSegmentationTrainer

    if ultralytics.__version__ != cfg["ultralytics_version"]:
        raise RuntimeError(f"Required ultralytics {cfg['ultralytics_version']}; installed {ultralytics.__version__}.")
    condition = spec["condition"]
    if condition != "crops":
        raise ValueError("This queue trains only the corrected DCBR proximal crops.")
    folder = Path(spec["directory"]).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    run_root = Path(spec["run_root"]).resolve()
    stop_file = Path(spec["stop_file"])
    imported_args = dict(cfg.get("training_args", {}))
    epochs = int(spec.get("epochs", imported_args.get("epochs", cfg.get("epochs", 250))))
    batch_size = int(imported_args.get("batch", cfg.get("batch", 16)))
    lr_label = imported_args.get("lr0", 0.01)
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch must be positive integers.")
    if "group_id" in train_frame and "group_id" in tune_surfaces:
        overlap = set(train_frame.group_id.astype(str)) & set(tune_surfaces.group_id.astype(str))
        if overlap:
            raise ValueError(f"Training/tuning group leakage: {sorted(overlap)}")
    result_path = folder / "result.json"
    if result_path.exists():
        result = read_json(result_path)
        if result.get("signature") != spec["signature"]:
            raise RuntimeError("Candidate result belongs to a different configuration.")
        if file_hash(result["checkpoint"]) != result["checkpoint_sha256"]:
            raise RuntimeError("Completed candidate checkpoint hash changed.")
        return result
    if stop_file.exists():
        raise KeyboardInterrupt("STOP requested. Remove output/STOP before resuming.")
    data_yaml, image_size = _dataset(folder, train_frame, tune_images, condition)
    complete = folder / "training_done.json"
    native = folder / "native"
    last, best = native / "weights" / "last.pt", native / "weights" / "best.pt"
    t0 = time.monotonic()
    previous_seconds = 0.0
    latest_metrics = {}

    def status(state, **extra):
        payload = {"state": state, "architecture": "yolo26_sem", "condition": condition, "seed": int(spec["seed"]),
                   "fold": int(spec["fold"]), "lr0_configured": float(lr_label), "hpo": False,
                   "optimizer": imported_args.get("optimizer", "auto"),
                   "selection_metric": "native_validation_mIoU",
                   "directory": str(folder), "max_epochs": epochs,
                   "seconds": previous_seconds + time.monotonic() - t0, **extra}
        atomic_json(folder / "current.json", payload)
        atomic_json(run_root / "current.json", payload)
        event(run_root, "candidate_" + state, **payload)
        return payload

    class ProtocolTrainer(SemanticSegmentationTrainer):
        def _build_train_pipeline(self):
            # Ultralytics can halve batch automatically after CUDA OOM. Fail
            # before rebuilding so no changed-batch checkpoint can be reused.
            if self.batch_size != batch_size:
                raise RuntimeError("Native OOM recovery changed batch. Choose an explicit smaller batch for both conditions.")
            return super()._build_train_pipeline()

        def validate(self):
            nonlocal latest_metrics
            metrics, fitness = super().validate()
            if fitness is None or not math.isfinite(float(fitness)):
                raise ValueError("Native validation mIoU is undefined.")
            latest_metrics = dict(metrics)
            # Match native best.pt semantics, including saving the latest tie.
            self._protocol_best = float(fitness) == float(self.best_fitness)
            if self._protocol_best:
                self.protocol_best_epoch = self.epoch + 1
                self.protocol_best_optimizer_steps = self.protocol_optimizer_steps
            atomic_json(folder / "latest_validation.json", {"epoch": self.epoch + 1,
                        "selection_metric": "native_validation_mIoU", "fitness": fitness, **metrics})
            return metrics, fitness

        def save_model(self):
            if self.batch_size != batch_size:
                raise RuntimeError("Refusing a checkpoint trained with an unplanned batch size.")
            for state_dict in (self.model.state_dict(), self.ema.ema.state_dict()):
                if any(value.is_floating_point() and not torch.isfinite(value).all() for value in state_dict.values()):
                    raise FloatingPointError("Nonfinite model weights; checkpoint not written.")
            ema = copy.deepcopy(self.ema.ema).cpu().float()
            if hasattr(ema, "criterion"):
                ema.criterion = None
            payload = {"epoch": self.epoch, "model": None, "ema": ema,
                       "updates": self.ema.updates,
                       "training_model_state": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
                       "best_fitness": self.best_fitness,
                       "optimizer": copy.deepcopy(self.optimizer.state_dict()),
                       "scaler": self.scaler.state_dict(), "scheduler": self.scheduler.state_dict(),
                       "train_args": vars(self.args),
                       "train_metrics": {**self.metrics, "fitness": self.fitness},
                       "train_results": self.read_results_csv(), "version": ultralytics.__version__,
                       "protocol_best_epoch": self.protocol_best_epoch,
                       "protocol_optimizer_steps": self.protocol_optimizer_steps,
                       "protocol_best_optimizer_steps": self.protocol_best_optimizer_steps,
                       "stopper": vars(self.stopper), "stop_reached": bool(self.stop),
                       "rng_python": random.getstate(), "rng_numpy": np.random.get_state(),
                       "rng_torch": torch.get_rng_state(),
                       "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                       "protocol_signature": spec["signature"],
                       "seconds": previous_seconds + time.monotonic() - t0}
            buf = io.BytesIO()
            torch.save(payload, buf)
            serialized = buf.getvalue()
            atomic_bytes(self.last, serialized)
            if self._protocol_best:
                atomic_bytes(self.best, serialized)
            if self.save_period > 0 and self.epoch % self.save_period == 0:
                atomic_bytes(self.wdir / f"epoch{self.epoch}.pt", serialized)
            return True

        def _load_checkpoint_state(self, checkpoint):
            super()._load_checkpoint_state(checkpoint)
            self.model.load_state_dict(checkpoint["training_model_state"], strict=True)
            self.protocol_best_epoch = checkpoint["protocol_best_epoch"]
            self.protocol_optimizer_steps = checkpoint["protocol_optimizer_steps"]
            self.protocol_best_optimizer_steps = checkpoint["protocol_best_optimizer_steps"]
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            self._protocol_rng = checkpoint

        def final_eval(self):
            # Native final_eval strips optimizers and performs mIoU evaluation.
            # Native mIoU has already selected EMA; retain the complete resume state.
            atomic_json(complete, {"signature": spec["signature"],
                                  "best_epoch": self.protocol_best_epoch,
                                  "epochs_completed": self.epoch + 1,
                                  "optimizer_steps": self.protocol_optimizer_steps,
                                  "best_optimizer_steps": self.protocol_best_optimizer_steps,
                                  "seconds": previous_seconds + time.monotonic() - t0})

        def _handle_nan_recovery(self, epoch):
            if self.loss is not None and not torch.isfinite(self.loss).all():
                raise FloatingPointError("Nonfinite loss; stopping without silently changing the trajectory.")
            if self.fitness is not None and not math.isfinite(float(self.fitness)):
                raise FloatingPointError("Nonfinite native validation objective.")
            return False

    def restore_rng(trainer):
        checkpoint = getattr(trainer, "_protocol_rng", None)
        if checkpoint is not None:
            trainer.stopper.__dict__.update(checkpoint["stopper"])
            random.setstate(checkpoint["rng_python"])
            np.random.set_state(checkpoint["rng_numpy"])
            torch.set_rng_state(checkpoint["rng_torch"])
            if checkpoint["rng_cuda"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(checkpoint["rng_cuda"])
        else:
            trainer.protocol_best_epoch = 0
            trainer.protocol_optimizer_steps = 0
            trainer.protocol_best_optimizer_steps = 0
        if trainer.batch_size != batch_size:
            raise RuntimeError("Native trainer changed the prespecified batch size.")
        def count_step(optimizer, args, kwargs):
            trainer.protocol_optimizer_steps += 1
        trainer._protocol_step_hook = trainer.optimizer.register_step_post_hook(count_step)
        atomic_json(folder / "effective_training_args.json", vars(trainer.args))
        atomic_json(folder / "optimizer.json", [{k: v for k, v in g.items() if k != "params"}
                                                for g in trainer.optimizer.param_groups])

    def report_batch(trainer):
        trainer.protocol_batch = getattr(trainer, "protocol_batch", 0) + 1
        if trainer.protocol_batch == 1 or trainer.protocol_batch % 5 == 0 or trainer.protocol_batch == len(trainer.train_loader):
            status("training", epoch=trainer.epoch + 1, batch=trainer.protocol_batch, batches=len(trainer.train_loader),
                   optimizer_steps=trainer.protocol_optimizer_steps, best_epoch=trainer.protocol_best_epoch)

    def report_epoch_start(trainer):
        trainer.protocol_batch = 0
        # Keep the native data-loader and augmentation streams unchanged.
        status("training", epoch=trainer.epoch + 1, best_epoch=trainer.protocol_best_epoch,
               optimizer_steps=trainer.protocol_optimizer_steps,
               n_train=len(trainer.train_loader.dataset), image_size=image_size)

    def report_epoch(trainer):
        if trainer.batch_size != batch_size:
            raise RuntimeError("OOM changed the batch size. Set an explicit smaller batch for both conditions and restart.")
        losses = {k: float(v) for k, v in trainer.tloss.items()} if isinstance(trainer.tloss, dict) else {"loss": float(trainer.tloss.mean())}
        payload = status("epoch_completed", epoch=trainer.epoch + 1,
                         best_epoch=trainer.protocol_best_epoch,
                         best=float(trainer.best_fitness), metrics=latest_metrics, losses=losses,
                         optimizer_steps=trainer.protocol_optimizer_steps,
                         best_optimizer_steps=trainer.protocol_best_optimizer_steps,
                         learning_rates=[float(g["lr"]) for g in trainer.optimizer.param_groups],
                         image_size=image_size)
        atomic_json(folder / "epochs" / f"{trainer.epoch + 1:04d}.json", payload)
        if stop_file.exists():
            raise KeyboardInterrupt("STOP requested; complete epoch checkpoint saved.")

    try:
        if complete.exists() and read_json(complete).get("signature") != spec["signature"]:
            raise RuntimeError("Completed training belongs to another configuration.")
        if not complete.exists():
            resume = False
            if last.exists():
                checkpoint = torch.load(last, map_location="cpu", weights_only=False)
                if checkpoint.get("protocol_signature") != spec["signature"]:
                    raise RuntimeError("Checkpoint belongs to another configuration.")
                previous_seconds = float(checkpoint.get("seconds", 0))
                if checkpoint["protocol_best_epoch"] == checkpoint["epoch"] + 1:
                    # Recover a crash between the atomic last and best writes.
                    atomic_bytes(best, last.read_bytes())
                if checkpoint["epoch"] + 1 >= epochs or checkpoint.get("stop_reached"):
                    atomic_json(complete, {"signature": spec["signature"],
                                          "best_epoch": checkpoint["protocol_best_epoch"],
                                          "epochs_completed": checkpoint["epoch"] + 1,
                                          "optimizer_steps": checkpoint["protocol_optimizer_steps"],
                                          "best_optimizer_steps": checkpoint["protocol_best_optimizer_steps"],
                                          "seconds": previous_seconds})
                else:
                    resume = True
                del checkpoint
            if not complete.exists():
                random.seed(int(spec["seed"]))
                np.random.seed(int(spec["seed"]))
                torch.manual_seed(int(spec["seed"]))
                model = YOLO(str(last if resume else cfg["model"]))
                model.add_callback("on_train_start", restore_rng)
                model.add_callback("on_train_epoch_start", report_epoch_start)
                model.add_callback("on_fit_epoch_end", report_epoch)
                model.add_callback("on_train_batch_end", report_batch)
                from ultralytics.cfg import DEFAULT_CFG_DICT
                unknown = set(imported_args) - set(DEFAULT_CFG_DICT) - {"save_dir"}
                if unknown:
                    raise ValueError(f"Unrecognized prior native training arguments: {sorted(unknown)}")
                # Reuse imported native settings. The legacy runner did not
                # override optimizer, scheduler, augmentation or accumulation.
                args = {k: v for k, v in imported_args.items() if k in DEFAULT_CFG_DICT}
                for key in ("model", "data", "project", "name", "save_dir", "resume", "mode"):
                    args.pop(key, None)
                args.update(data=str(data_yaml), epochs=epochs,
                            patience=int(imported_args.get("patience", cfg.get("patience", 30))),
                            imgsz=image_size, batch=batch_size,
                            seed=int(spec["seed"]), deterministic=True,
                            device=cfg["device"],
                            project=str(folder), name="native", exist_ok=True, verbose=True,
                            val=True, plots=False, save=True)
                for key in ("workers", "amp"):
                    if key in cfg and key not in imported_args:
                        args[key] = cfg[key]
                if resume:
                    args["resume"] = str(last)
                model.train(trainer=ProtocolTrainer, **args)
                del model
        if not best.is_file():
            raise RuntimeError("Best native-validation checkpoint missing.")
        checkpoint = torch.load(best, map_location="cpu", weights_only=False)
        if checkpoint.get("protocol_signature") != spec["signature"]:
            raise RuntimeError("Selected checkpoint signature mismatch.")
        done = read_json(complete)
        summary = dict(checkpoint.get("train_metrics", {}))
        result = {"signature": spec["signature"], "condition": condition,
                  "seed": int(spec["seed"]), "fold": int(spec["fold"]), "hpo": False,
                  "checkpoint": str(best), "checkpoint_sha256": file_hash(best),
                  "best_epoch": checkpoint["protocol_best_epoch"],
                  "epochs_completed": done["epochs_completed"],
                  "optimizer_steps": done["optimizer_steps"],
                  "best_optimizer_steps": checkpoint["protocol_best_optimizer_steps"],
                  "seconds": done["seconds"], "seconds_this_attempt": time.monotonic() - t0,
                  "image_size": image_size, "summary": summary, **summary,
                  "objective": float(checkpoint["best_fitness"]),
                  "selection_metric": "native_validation_mIoU"}
        atomic_json(result_path, result)
        status("completed", epoch=done["epochs_completed"], best_epoch=result["best_epoch"], metrics=summary)
        del checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result
    except KeyboardInterrupt:
        status("paused", checkpoint=str(last) if last.exists() else None)
        raise
    except Exception as error:
        status("failed", error=str(error), checkpoint=str(last) if last.exists() else None)
        raise
