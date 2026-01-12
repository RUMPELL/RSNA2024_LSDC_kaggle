from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def resize_and_pad(image: np.ndarray, size: Tuple[int, int]) -> tuple[np.ndarray, float, int, int]:
    """Resize image maintaining aspect ratio and pad to target size.

    Returns:
        padded_image, scale, pad_left, pad_top
    """
    h, w = image.shape[:2]
    target_h, target_w = size

    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    pad_h = target_h - new_h
    pad_w = target_w - new_w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return padded, scale, left, top
