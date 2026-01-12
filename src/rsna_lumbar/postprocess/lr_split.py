from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd


def infer_lr_direction(image_position_x: Optional[float]) -> Optional[str]:
    """Infer Left/Right direction hint from ImagePositionPatient[0].

    This is a heuristic; some datasets may have inconsistent conventions.
    Returns:
        "Right" / "Left" / None
    """
    if image_position_x is None:
        return None
    # heuristic: sign determines direction (tuned per dataset; you may flip if needed)
    return "Right" if image_position_x > 0 else "Left"


def add_lr_direction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["LR_Direction"] = df["image_position_x"].apply(infer_lr_direction)
    return df
