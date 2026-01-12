from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from rsna_lumbar.utils.config import load_yaml
from rsna_lumbar.classification.model import EfficientNetSeverity
from rsna_lumbar.classification.transforms import build_valid_transforms
from rsna_lumbar.detection.yolo_infer import YoloInferConfig, run_yolo_on_series
from rsna_lumbar.postprocess.lr_split import add_lr_direction
from rsna_lumbar.postprocess.filtering import filter_top_confidence_per_class, fill_missing_classes
from rsna_lumbar.preprocess.png_export import PngExportConfig, export_3ch_png


# Competition constants (5 disc levels)
LEVELS = [1, 2, 3, 4, 5]
SEVERITY_CLASSES = ["normal_mild", "moderate", "severe"]


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


@torch.no_grad()
def ensemble_predict(models: List[torch.nn.Module], x: torch.Tensor) -> np.ndarray:
    logits = None
    for m in models:
        out = m(x)
        logits = out if logits is None else logits + out
    logits = logits / len(models)
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
    return probs


def load_ensemble(weights: List[str | Path], cfg: Dict, device: torch.device) -> List[torch.nn.Module]:
    models = []
    for w in weights:
        m = EfficientNetSeverity(
            backbone=str(cfg.get("backbone", "tf_efficientnet_b4")),
            pretrained=False,
            dropout=float(cfg.get("dropout", 0.2)),
            num_classes=3,
        )
        m.load_state_dict(torch.load(str(w), map_location="cpu"))
        m.to(device).eval()
        models.append(m)
    return models


def build_submission_template(study_id: int | str) -> pd.DataFrame:
    rows = []
    # Conditions x levels x (side handling) are competition-specific; we generate row_id format used by Kaggle.
    # row_id format example: '{study_id}_{condition}_{level}' with optional side prefix depending on condition.
    # Adjust if your competition row_id differs.
    for level in LEVELS:
        rows.append((f"{study_id}_spinal_canal_stenosis_l{level}",))
        rows.append((f"{study_id}_left_neural_foraminal_narrowing_l{level}",))
        rows.append((f"{study_id}_right_neural_foraminal_narrowing_l{level}",))
        rows.append((f"{study_id}_left_subarticular_stenosis_l{level}",))
        rows.append((f"{study_id}_right_subarticular_stenosis_l{level}",))
    df = pd.DataFrame(rows, columns=["row_id"])
    for c in SEVERITY_CLASSES:
        df[c] = 0.0
    return df


def fallback_probs() -> np.ndarray:
    # configurable fallback; default mirrors common competition heuristic
    return np.array([0.4, 0.4, 0.2], dtype=np.float32)


def make_submission(
    test_series_df: pd.DataFrame,
    paths_cfg: Dict,
    cls_cfg: Dict,
    yolo_models: Dict[str, str],
    cls_weights: Dict[str, List[str]],
    out_csv: Path,
) -> None:
    """End-to-end submission generation (skeleton).

    This function is intentionally explicit and conservative:
    - It logs missing sequences
    - It fills missing disc detections with fallback probabilities
    - It writes a valid `submission.csv`

    You should customize row_id formatting to match your exact Kaggle submission.
    """
    device = torch.device(cls_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    image_size = int(cls_cfg["image_size"])
    tfm = build_valid_transforms(image_size)

    # load ensembles per condition
    ensembles = {
        k: load_ensemble(v, cls_cfg, device) for k, v in cls_weights.items()
    }

    png_cfg = PngExportConfig(image_size=image_size)

    submissions = []
    for study_id, sdf in test_series_df.groupby("study_id"):
        sub_df = build_submission_template(study_id)

        # TODO: implement per-series selection and YOLO inference using your competition's metadata
        # This skeleton expects you to build dicom_paths for each required series description and run YOLO.

        # For now, we fallback entirely:
        probs = fallback_probs()
        for i in range(len(sub_df)):
            sub_df.loc[i, SEVERITY_CLASSES] = probs

        submissions.append(sub_df)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(submissions, ignore_index=True).to_csv(out_csv, index=False)
