from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pydicom

try:
    from pydicom.pixel_data_handlers.util import apply_voi_lut
except Exception:  # pragma: no cover
    apply_voi_lut = None  # type: ignore


def read_dicom(path: str | Path, try_lut: bool = True, fix_monochrome: bool = True) -> np.ndarray:
    """Read a DICOM and return uint8 image scaled to [0, 255].

    This mirrors common Kaggle RSNA competition preprocessing patterns:
    - apply VOI LUT if available
    - fix MONOCHROME1
    - min-max normalize to 8-bit
    """
    dcm = pydicom.dcmread(str(path))
    img = dcm.pixel_array.astype(np.float32)

    if try_lut and apply_voi_lut is not None:
        try:
            img = apply_voi_lut(img, dcm).astype(np.float32)
        except Exception:
            pass

    if fix_monochrome and getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = np.max(img) - img

    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img


def get_image_position_x(path: str | Path) -> Optional[float]:
    """Return ImagePositionPatient[0] (x) if available."""
    try:
        dcm = pydicom.dcmread(str(path), stop_before_pixels=True)
        ipp = getattr(dcm, "ImagePositionPatient", None)
        if ipp is None:
            return None
        return float(ipp[0])
    except Exception:
        return None
