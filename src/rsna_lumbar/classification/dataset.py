# src/rsna_lumbar/classification/dataset.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DatasetConfig:
    """
    Dataset configuration aligned with the training notebook.

    Expected dataframe columns:
      - study_id: str/int
      - disease: one of {'spinal_canal_stenosis','neural_foraminal_narrowing','subarticular_stenosis'}
      - side: 'left'/'right' for non-spinal_canal_stenosis
      - level: one of {'l1_l2','l2_l3','l3_l4','l4_l5','l5_s1'}
      - value: int in {0,1,2} for 3-class severity
    """
    png_root: Path
    image_size: int = 224


class MedicalImageDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, cfg: DatasetConfig, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.cfg = cfg
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, idx: int):
        row = self.dataframe.iloc[idx]

        # Notebook-compatible filename pattern:
        # image_name = f"{study_id}_disc{level[1]}.png" where level is 'l1_l2' -> level[1] == '1'
        study_id = row["study_id"]
        level = str(row["level"])
        disc_idx = level[1]  # 'l1_l2' -> '1'
        image_name = f"{study_id}_disc{disc_idx}.png"

        disease = str(row["disease"])
        if disease == "spinal_canal_stenosis":
            image_path = self.cfg.png_root / disease / image_name
        else:
            side = str(row["side"])
            image_path = self.cfg.png_root / f"{side}_{disease}" / image_name

        image = Image.open(image_path)
        image = np.array(image)  # (H,W,3) uint8
        image = cv2.resize(image, (self.cfg.image_size, self.cfg.image_size), interpolation=cv2.INTER_CUBIC)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        label = int(row["value"])
        one_hot = np.zeros(3, dtype=np.float32)
        one_hot[label] = 1.0

        return image, torch.tensor(one_hot, dtype=torch.float32)
