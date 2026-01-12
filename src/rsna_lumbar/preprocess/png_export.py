# src/rsna_lumbar/preprocess/png_export.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut


Array = np.ndarray
BBoxXYXY = Tuple[int, int, int, int]


@dataclass(frozen=True)
class DicomToImageConfig:
    """Configuration for converting DICOM slices to uint8 images."""
    try_lut: bool = True
    fix_monochrome: bool = True
    clip_percentiles: Tuple[float, float] = (1.0, 99.0)


def dicom_array_to_image(
    dicom: pydicom.dataset.FileDataset,
    arr: Array,
    cfg: DicomToImageConfig,
) -> Array:
    """
    Convert a DICOM pixel array to a normalized uint8 image.

    This matches the notebook behavior:
    - apply_modality_lut and apply_voi_lut when available (cfg.try_lut)
    - handle MONOCHROME1 inversion (cfg.fix_monochrome)
    - clip by percentiles (1,99) then normalize to [0,255]
    """
    if cfg.try_lut:
        arr = apply_modality_lut(arr, dicom)

    if cfg.try_lut:
        # index=0 is used in the notebook
        arr = apply_voi_lut(arr, dicom, index=0)

    if cfg.fix_monochrome and getattr(dicom, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = np.amax(arr) - arr

    lower, upper = np.percentile(arr, cfg.clip_percentiles)
    arr = np.clip(arr, lower, upper)
    arr = arr - np.min(arr)
    arr = arr / (np.max(arr) + 1e-6)
    arr = (arr * 255).astype(np.uint8)
    return arr


def dicom_path_to_image(path: Path, cfg: DicomToImageConfig) -> Array:
    """
    Load a DICOM file from `path` and convert it to uint8 image.
    """
    dicom = pydicom.dcmread(str(path))
    arr = dicom.pixel_array
    return dicom_array_to_image(dicom, arr, cfg)


def crop_image(
    image: Array,
    x: float,
    y: float,
    crop_size: int = 64,
    offset_x: int = 0,
    offset_y: int = 0,
) -> Array:
    """
    Center-crop around (x,y) with optional offsets.
    This matches the notebook behavior (no padding, boundary clamping by slicing).
    """
    half = crop_size // 2
    x = int(x) + int(offset_x)
    y = int(y) + int(offset_y)

    y1 = max(0, y - half)
    y2 = min(y + half, image.shape[0])
    x1 = max(0, x - half)
    x2 = min(x + half, image.shape[1])

    return image[y1:y2, x1:x2]


def level_to_disc_index(level: str | int) -> int:
    """
    Convert level labels into disc index used in file naming.
    Supported:
    - 'l1_l2' -> 1 ... 'l5_s1' -> 5
    - 'disc5' -> 5
    - int / numeric strings -> int
    """
    if isinstance(level, int):
        return level

    s = str(level).strip().lower()
    if s.isdigit():
        return int(s)

    if s.startswith("disc") and s[4:].isdigit():
        return int(s[4:])

    mapping = {
        "l1_l2": 1,
        "l2_l3": 2,
        "l3_l4": 3,
        "l4_l5": 4,
        "l5_s1": 5,
    }
    if s in mapping:
        return mapping[s]

    # Fallback: try to find a digit 1..5 in the string
    m = next((ch for ch in s if ch in "12345"), None)
    if m is not None:
        return int(m)

    raise ValueError(f"Unable to map level to disc index: {level!r}")


def condition_to_dir(condition: str) -> str:
    """
    Map condition string (notebook title-case) into directory names
    expected by the training notebook.
    """
    s = str(condition).strip().lower()
    # normalize common variants
    s = s.replace("  ", " ")
    mapping = {
        "spinal canal stenosis": "spinal_canal_stenosis",
        "left subarticular stenosis": "left_subarticular_stenosis",
        "right subarticular stenosis": "right_subarticular_stenosis",
        "left neural foraminal narrowing": "left_neural_foraminal_narrowing",
        "right neural foraminal narrowing": "right_neural_foraminal_narrowing",
        # already snake_case
        "spinal_canal_stenosis": "spinal_canal_stenosis",
        "left_subarticular_stenosis": "left_subarticular_stenosis",
        "right_subarticular_stenosis": "right_subarticular_stenosis",
        "left_neural_foraminal_narrowing": "left_neural_foraminal_narrowing",
        "right_neural_foraminal_narrowing": "right_neural_foraminal_narrowing",
    }
    if s in mapping:
        return mapping[s]
    raise ValueError(f"Unknown condition for directory mapping: {condition!r}")


@dataclass(frozen=True)
class CropRuleConfig:
    """
    Crop sizing rules that match the notebook.
    """
    axial_scale: float = 0.13
    non_axial_scale: float = 0.20
    axial_min_small: int = 64
    axial_min_large: int = 112
    sagittal_offset_x_frac: float = -0.05
    sagittal_offset_y_frac: float = 0.05


def compute_crop_params(
    img_h: int,
    img_w: int,
    series_description: str,
    disc_index: int,
    rules: CropRuleConfig,
) -> tuple[int, int, int]:
    """
    Compute crop_size and offsets exactly as in the notebook.
    """
    series = str(series_description)

    offset_x = int(img_w * rules.sagittal_offset_x_frac) if series == "Sagittal T2/STIR" else 0
    offset_y = int(img_h * rules.sagittal_offset_y_frac) if (series == "Sagittal T2/STIR" and disc_index == 5) else 0

    if series == "Axial T2":
        crop_size = int(min(img_h, img_w) * rules.axial_scale)
        crop_size = max(crop_size, rules.axial_min_small) if crop_size < rules.axial_min_small else max(crop_size, rules.axial_min_large)
    else:
        crop_size = int(min(img_h, img_w) * rules.non_axial_scale)

    return crop_size, offset_x, offset_y


def export_crop_3ch_png(
    train_images_dir: Path,
    out_root: Path,
    study_id: int | str,
    series_id: int | str,
    instance_number: int,
    level: str | int,
    series_description: str,
    condition: str,
    x: float,
    y: float,
    dicom_cfg: Optional[DicomToImageConfig] = None,
    crop_rules: Optional[CropRuleConfig] = None,
) -> Path:
    """
    Export a 3-channel PNG crop following the notebook logic.

    DICOM paths are resolved as:
        {train_images_dir}/{study_id}/{series_id}/{instance_number}.dcm
    And channels are stacked as:
        [prev, center, next]
    """
    dicom_cfg = dicom_cfg or DicomToImageConfig()
    crop_rules = crop_rules or CropRuleConfig()

    study_id = str(study_id)
    series_id = str(series_id)

    inst = int(instance_number)
    if inst == 1:
        inst = 2

    def p(i: int) -> Path:
        return train_images_dir / study_id / series_id / f"{i}.dcm"

    # Notebook-style fallback: if paths fail, shift center by -1 and retry.
    try:
        center_path = p(inst)
        prev_path = p(inst - 1)
        next_path = p(inst + 1)

        img_center = dicom_path_to_image(center_path, dicom_cfg)
        img_prev = dicom_path_to_image(prev_path, dicom_cfg)
        img_next = dicom_path_to_image(next_path, dicom_cfg)
    except Exception:
        inst2 = max(2, inst - 1)
        center_path = p(inst2 - 1)
        prev_path = p(inst2 - 2)
        next_path = p(inst2)

        img_center = dicom_path_to_image(center_path, dicom_cfg)
        img_prev = dicom_path_to_image(prev_path, dicom_cfg)
        img_next = dicom_path_to_image(next_path, dicom_cfg)

    img_h, img_w = img_center.shape[:2]
    disc_idx = level_to_disc_index(level)

    crop_size, offset_x, offset_y = compute_crop_params(
        img_h=img_h,
        img_w=img_w,
        series_description=series_description,
        disc_index=disc_idx,
        rules=crop_rules,
    )

    crop_center = crop_image(img_center, x, y, crop_size, offset_x, offset_y)
    crop_prev = crop_image(img_prev, x, y, crop_size, offset_x, offset_y)
    crop_next = crop_image(img_next, x, y, crop_size, offset_x, offset_y)

    crop_3ch = np.stack([crop_prev, crop_center, crop_next], axis=-1).astype(np.uint8)

    cond_dir = condition_to_dir(condition)
    out_dir = out_root / cond_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{study_id}_disc{disc_idx}.png"

    # Use Pillow-like PNG saving behavior through imageio if available, else OpenCV.
    # OpenCV will write 3-channel uint8 just fine.
    import cv2
    ok = cv2.imwrite(str(out_path), crop_3ch)
    if not ok:
        raise IOError(f"Failed to write PNG: {out_path}")

    return out_path
