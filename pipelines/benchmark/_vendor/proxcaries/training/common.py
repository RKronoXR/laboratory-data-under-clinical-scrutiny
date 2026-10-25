from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from proxcaries.status import write_status


@dataclass(frozen=True)
class RunSpec:
    architecture: str
    source: str
    seed: int
    fold: int
    run_dir: Path

    @property
    def status_path(self) -> Path:
        return self.run_dir / "status.json"


def split_manifest(df: pd.DataFrame, fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["fold"] != fold].reset_index(drop=True)
    val = df[df["fold"] == fold].reset_index(drop=True)
    if train.empty or val.empty:
        raise ValueError(f"Empty train/validation split for fold {fold}")
    overlap = set(train.group_id.astype(str)) & set(val.group_id.astype(str))
    if overlap:
        raise RuntimeError(f"Group leakage in fold {fold}: {sorted(overlap)[:5]}")
    return train, val


def start_run(spec: RunSpec, max_epochs: int) -> float:
    spec.run_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    write_status(
        spec.status_path,
        state="running",
        architecture=spec.architecture,
        source=spec.source,
        seed=spec.seed,
        fold=spec.fold,
        current_epoch=0,
        max_epochs=max_epochs,
        best_metric=None,
        seconds_elapsed=0.0,
    )
    return start


def finish_run(spec: RunSpec, start: float, metrics: dict, extra: dict | None = None) -> None:
    elapsed = time.time() - start
    payload = dict(metrics)
    if extra:
        payload.update(extra)
    (spec.run_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_status(spec.status_path, state="completed", seconds_elapsed=elapsed, best_metric=metrics.get("dice"))


def fail_run(spec: RunSpec, start: float, exc: BaseException) -> None:
    write_status(spec.status_path, state="failed", seconds_elapsed=time.time() - start, error=repr(exc))
