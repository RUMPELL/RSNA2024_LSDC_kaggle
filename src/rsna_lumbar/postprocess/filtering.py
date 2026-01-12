from __future__ import annotations

from typing import Iterable, List, Sequence

import pandas as pd


def filter_top_confidence_per_class(df: pd.DataFrame, by: Sequence[str] = ("class_id",), n: int = 1) -> pd.DataFrame:
    """Keep top-N detections per group (e.g., per class)."""
    if df.empty:
        return df
    keep = []
    for _, g in df.groupby(list(by)):
        keep.append(g.nlargest(n, "confidence"))
    return pd.concat(keep, ignore_index=True) if keep else df.iloc[:0]


def fill_missing_classes(
    df: pd.DataFrame,
    required_classes: Iterable[int],
    fallback_conf: float = 0.0,
) -> pd.DataFrame:
    """Ensure all required class_ids exist; insert empty rows if missing.

    This helps downstream code avoid crashing when a disc level is not detected.
    """
    df = df.copy()
    present = set(df["class_id"].tolist()) if not df.empty and "class_id" in df.columns else set()
    rows = []
    for c in required_classes:
        if c not in present:
            rows.append(dict(class_id=c, confidence=fallback_conf))
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    return df
