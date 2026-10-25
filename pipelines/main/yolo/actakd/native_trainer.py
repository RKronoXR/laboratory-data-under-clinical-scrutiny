"""Local nnU-Net 2.8.1 extension; never installed over the original trainer."""
from __future__ import annotations
import copy
import io
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from .io import atomic_bytes, atomic_json, event

_CONTEXT = None


def configure(context):
    global _CONTEXT
    _CONTEXT = copy.deepcopy(context)


class nnUNetTrainerACTAKD(nnUNetTrainer):
    """The previous EMA foreground Dice stopping rule plus atomic resume state."""
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        if _CONTEXT is None:
            raise RuntimeError("Call native_trainer.configure before constructing the local trainer.")
        self.protocol = copy.deepcopy(_CONTEXT)
        super().__init__(plans, configuration, fold, dataset_json, device)
        cfg = self.protocol["cfg"]
        self.num_epochs = int(cfg["nnunet_max_epochs"])
        self._pc_patience = int(cfg["nnunet_patience"])
        self._pc_min_delta = float(cfg["nnunet_min_delta"])
        self._pc_seed = int(self.protocol["spec"]["seed"])
        # Explicit study/HPO overrides. The remaining native trainer settings are inherited.
        for config_key, attribute, cast in (
            ("nnunet_initial_lr", "initial_lr", float),
            ("nnunet_weight_decay", "weight_decay", float),
            ("nnunet_train_iterations", "num_iterations_per_epoch", int),
            ("nnunet_validation_iterations", "num_val_iterations_per_epoch", int),
            ("nnunet_oversample_foreground", "oversample_foreground_percent", float),
        ):
            if config_key in cfg:
                setattr(self, attribute, cast(cfg[config_key]))
        self._pc_early_best = -float("inf")
        self._pc_stale = 0
        self._pc_best_epoch = 0
        self._pc_resume_rng = None
        self._pc_seconds = 0.0
        self._pc_started = time.monotonic()
        self._checkpoint_next_epoch = None

    def do_split(self):
        # Exact ORIGINAL training/validation IDs from the audited task. No internal split.
        spec=self.protocol['spec']
        train=list(spec['train_ids']);val=list(spec['validation_ids'])
        if not train or not val or set(train)&set(val):
            raise RuntimeError('Invalid native teacher train/development-validation split.')
        return train,val

    def plot_network_architecture(self):
        # Architecture is recorded in plans.json; no optional graphviz/PDF dependency.
        return None

    def _status(self, state, **fields):
        spec = self.protocol["spec"]
        payload = {"state": state, "architecture": "nnunetv2", "condition": "full",
                   "seed": int(spec["seed"]), "fold": int(spec["fold"]),
                   "epoch": self.current_epoch, "max_epochs": self.num_epochs,
                   "best_epoch": self._pc_best_epoch,
                   "seconds": self._pc_seconds + time.monotonic() - self._pc_started,
                   "selection_metric": "EMA_foreground_Dice", "hpo": "teacher" in self.protocol["spec"]["config"].get("hpo_enabled", []), **fields}
        atomic_json(Path(spec["directory"]) / "current.json", payload)
        event(Path(spec["run_root"]), "nnunet_" + state, **payload)
        return payload

    def on_train_start(self):
        random.seed(self._pc_seed)
        np.random.seed(self._pc_seed)
        torch.manual_seed(self._pc_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._pc_seed)
        super().on_train_start()
        # Restores parent RNG streams. Native augmentation workers have their
        # own streams, so cross-process continuation is not promised bit-exact.
        if self._pc_resume_rng:
            rng = self._pc_resume_rng
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            if rng.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng["cuda"])

    def save_checkpoint(self, filename):
        if self.disable_checkpointing or self.local_rank != 0:
            return
        model = self.network.module if self.is_ddp else self.network
        model = getattr(model, "_orig_mod", model)
        states = model.state_dict()
        if any(v.is_floating_point() and not torch.isfinite(v).all() for v in states.values()):
            raise FloatingPointError("Nonfinite nnU-Net weights; checkpoint rejected.")
        next_epoch = self.current_epoch + 1 if self._checkpoint_next_epoch is None else self._checkpoint_next_epoch
        if Path(filename).name == "checkpoint_best.pth":
            self._pc_best_epoch = int(next_epoch)
        payload = {
            "network_weights": {k: v.detach().cpu().clone() for k, v in states.items()},
            "optimizer_state": self.optimizer.state_dict(),
            "grad_scaler_state": self.grad_scaler.state_dict() if self.grad_scaler else None,
            "scheduler_state": self.lr_scheduler.state_dict(),
            "logging": self.logger.get_checkpoint(), "_best_ema": self._best_ema,
            "current_epoch": int(next_epoch), "init_args": self.my_init_kwargs,
            "trainer_name": self.__class__.__name__,
            "inference_allowed_mirroring_axes": self.inference_allowed_mirroring_axes,
            "protocol_signature": self.protocol["spec"]["signature"],
            "protocol_seed": self._pc_seed,
            "protocol_early_best": self._pc_early_best, "protocol_stale": self._pc_stale,
            "protocol_best_epoch": self._pc_best_epoch,
            "protocol_seconds": self._pc_seconds + time.monotonic() - self._pc_started,
            "protocol_rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                             "torch": torch.get_rng_state(),
                             "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
        }
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        atomic_bytes(filename, buffer.getvalue())

    def load_checkpoint(self, filename_or_checkpoint):
        if not isinstance(filename_or_checkpoint, (str, Path)):
            raise TypeError("Use a checkpoint path for nnU-Net resume.")
        checkpoint = torch.load(str(filename_or_checkpoint), map_location="cpu", weights_only=False)
        if checkpoint.get("protocol_signature") != self.protocol["spec"]["signature"]:
            raise RuntimeError("nnU-Net checkpoint signature mismatch.")
        if checkpoint.get("protocol_seed") != self._pc_seed:
            raise RuntimeError("nnU-Net checkpoint belongs to another seed.")
        # Native initialization on resume occurs before on_train_start.
        random.seed(self._pc_seed)
        np.random.seed(self._pc_seed)
        torch.manual_seed(self._pc_seed)
        super().load_checkpoint(str(filename_or_checkpoint))
        self.lr_scheduler.load_state_dict(checkpoint["scheduler_state"])
        self._pc_early_best = checkpoint["protocol_early_best"]
        self._pc_stale = checkpoint["protocol_stale"]
        self._pc_best_epoch = checkpoint["protocol_best_epoch"]
        self._pc_resume_rng = checkpoint["protocol_rng"]
        self._pc_seconds = checkpoint.get("protocol_seconds", 0.0)
        self._pc_started = time.monotonic()

    def _close_loaders(self):
        for loader in (self.dataloader_train, self.dataloader_val):
            if hasattr(loader, "_finish"):
                loader._finish()

    def run_training(self):
        stop_path = Path(self.protocol["stop_file"])
        if stop_path.exists():
            raise KeyboardInterrupt("STOP requested.")
        self.on_train_start()
        try:
            if self.current_epoch >= self.num_epochs or self._pc_stale >= self._pc_patience:
                self._status("training_complete")
                return
            for _ in range(self.current_epoch, self.num_epochs):
                self._status("training", epoch=self.current_epoch + 1)
                self.on_epoch_start()
                self.on_train_epoch_start()
                train_outputs = [self.train_step(next(self.dataloader_train))
                                 for _ in range(self.num_iterations_per_epoch)]
                if any(not np.isfinite(value["loss"]).all() for value in train_outputs):
                    raise FloatingPointError("Nonfinite nnU-Net training loss.")
                self.on_train_epoch_end(train_outputs)
                with torch.no_grad():
                    self.on_validation_epoch_start()
                    val_outputs = [self.validation_step(next(self.dataloader_val))
                                   for _ in range(self.num_val_iterations_per_epoch)]
                    self.on_validation_epoch_end(val_outputs)
                self.on_epoch_end()
                current = float(self.logger.get_value("ema_fg_dice", step=-1))
                if not math.isfinite(current):
                    raise FloatingPointError("Nonfinite nnU-Net EMA foreground Dice.")
                if current > self._pc_early_best + self._pc_min_delta:
                    self._pc_early_best, self._pc_stale = current, 0
                else:
                    self._pc_stale += 1
                self._checkpoint_next_epoch = self.current_epoch
                self.save_checkpoint(str(Path(self.output_folder) / "checkpoint_latest.pth"))
                if self._pc_best_epoch == self.current_epoch:
                    self.save_checkpoint(str(Path(self.output_folder) / "checkpoint_best.pth"))
                self._checkpoint_next_epoch = None
                payload = self._status("epoch_completed", metrics={
                    "ema_foreground_dice": current,
                    "foreground_dice": float(self.logger.get_value("mean_fg_dice", step=-1)),
                    "train_loss": float(self.logger.get_value("train_losses", step=-1)),
                    "validation_loss": float(self.logger.get_value("val_losses", step=-1)),
                    "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                }, stale_epochs=self._pc_stale)
                atomic_json(Path(self.protocol["spec"]["directory"]) / "epochs" / f"{self.current_epoch:04d}.json", payload)
                if stop_path.exists():
                    raise KeyboardInterrupt("STOP requested; nnU-Net epoch checkpoint saved.")
                if self._pc_stale >= self._pc_patience:
                    break
            self.on_train_end()
            # Native on_train_end deletes latest. Retain a full resumable copy.
            final = Path(self.output_folder) / "checkpoint_final.pth"
            if final.exists():
                atomic_bytes(Path(self.output_folder) / "checkpoint_latest.pth", final.read_bytes())
        except KeyboardInterrupt:
            self._status("paused")
            raise
        finally:
            self._close_loaders()
