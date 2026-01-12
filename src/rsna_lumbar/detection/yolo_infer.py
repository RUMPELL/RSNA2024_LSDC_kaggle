from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from rsna_lumbar.dicom.io import read_dicom, get_image_position_x
from rsna_lumbar.dicom.preprocess import resize_and_pad

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore


@dataclass
class YoloInferConfig:
    img_size: int = 640
    conf: float = 0.25
    iou: float = 0.7


def run_yolo_on_series(
    model_path: str | Path,
    dicom_paths: List[str],
    cfg: YoloInferConfig,
    series_description: str,
) -> pd.DataFrame:
    """Run YOLO on a list of DICOM files and return detections DataFrame.

    Output columns:
      - dicom_path
      - class (int)
      - confidence (float)
      - x1,y1,x2,y2 (in original image pixel coords)
      - image_position_x (for LR heuristics)
      - series_description
    """
    if YOLO is None:
        raise ImportError("ultralytics is required for YOLO inference.")

    model = YOLO(str(model_path))
    rows = []
    for p in dicom_paths:
        img = read_dicom(p)
        padded, scale, pad_left, pad_top = resize_and_pad(img, (cfg.img_size, cfg.img_size))
        # ultralytics expects 3-channel; convert grayscale to 3-channel
        inp = np.stack([padded, padded, padded], axis=-1)

        results = model.predict(inp, conf=cfg.conf, iou=cfg.iou, verbose=False)
        if not results:
            continue
        r0 = results[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None:
            continue

        for b in boxes:
            cls_id = int(b.cls.item())
            conf = float(b.conf.item())
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            # map back to original image coords
            x1 = (x1 - pad_left) / scale
            x2 = (x2 - pad_left) / scale
            y1 = (y1 - pad_top) / scale
            y2 = (y2 - pad_top) / scale

            rows.append(
                dict(
                    dicom_path=p,
                    class_id=cls_id,
                    confidence=conf,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    image_position_x=get_image_position_x(p),
                    series_description=series_description,
                )
            )

    return pd.DataFrame(rows)
