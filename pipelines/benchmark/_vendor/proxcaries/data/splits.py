from __future__ import annotations

import pandas as pd



def assign_grouped_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 2026) -> pd.DataFrame:
    from sklearn.model_selection import StratifiedGroupKFold
    required = {"group_id", "lesion_present"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for grouped split: {sorted(missing)}")
    if df["group_id"].nunique() < n_splits:
        raise ValueError("Number of unique groups is smaller than n_splits")
    out = df.copy()
    out["fold"] = -1
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    x = out[["sample_id"]]
    y = out["lesion_present"].astype(int)
    groups = out["group_id"].astype(str)
    for fold, (_, val_idx) in enumerate(cv.split(x, y, groups)):
        out.loc[out.index[val_idx], "fold"] = fold
    if (out["fold"] < 0).any():
        raise RuntimeError("Some samples were not assigned to a fold")
    # Leakage check: every group must occur in exactly one validation fold.
    if out.groupby("group_id")["fold"].nunique().max() != 1:
        raise RuntimeError("Group leakage detected during fold creation")
    return out
