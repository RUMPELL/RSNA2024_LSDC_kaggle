from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def add_group_kfold(df: pd.DataFrame, n_splits: int = 5, group_col: str = "study_id", seed: int = 42) -> pd.DataFrame:
    """Add a `fold` column using GroupKFold over `group_col`."""
    df = df.copy()
    gkf = GroupKFold(n_splits=n_splits)
    df["fold"] = -1
    X = np.zeros(len(df))
    for fold, (_, val_idx) in enumerate(gkf.split(X, groups=df[group_col].values)):
        df.loc[val_idx, "fold"] = fold
    return df
