"""nnU-Net v2 trainer with the study's finite budget and early stopping.

The values are read from environment variables so the same file can be reused by
architecture selection and quick smoke tests:
  PROXCARIES_NNUNET_MAX_EPOCHS (default 250)
  PROXCARIES_NNUNET_PATIENCE   (default 30)
  PROXCARIES_NNUNET_MIN_DELTA  (default 0.001)
  PROXCARIES_SEED              (default 2026)
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerCariesEarlyStop(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = int(os.getenv("PROXCARIES_NNUNET_MAX_EPOCHS", "250"))
        self._pc_patience = int(os.getenv("PROXCARIES_NNUNET_PATIENCE", "30"))
        self._pc_min_delta = float(os.getenv("PROXCARIES_NNUNET_MIN_DELTA", "0.001"))
        self._pc_seed = int(os.getenv("PROXCARIES_SEED", "2026"))

    def on_train_start(self):
        random.seed(self._pc_seed)
        np.random.seed(self._pc_seed)
        torch.manual_seed(self._pc_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self._pc_seed)
        super().on_train_start()

    def run_training(self):
        # This mirrors nnU-Net v2's public run_training loop, adding only the
        # pre-specified patience/min_delta criterion on EMA foreground Dice.
        self.on_train_start()
        best = -float("inf")
        stale = 0
        for _epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()
            self.on_train_epoch_start()
            train_outputs = []
            for _batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)
            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for _batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)
            self.on_epoch_end()

            current = float(self.logger.get_value("ema_fg_dice", step=-1))
            if current > best + self._pc_min_delta:
                best = current
                stale = 0
            else:
                stale += 1
            if stale >= self._pc_patience:
                self.print_to_log_file(
                    f"Early stopping after {self.current_epoch} epochs; "
                    f"no EMA foreground Dice improvement > {self._pc_min_delta} "
                    f"for {self._pc_patience} epochs."
                )
                break
        self.on_train_end()
