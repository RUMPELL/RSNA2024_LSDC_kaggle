from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2


def build_train_transforms(image_size: int):
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.2),
            A.Rotate(limit=5, p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.3),
            A.Normalize(mean=[0.456, 0.456, 0.456], std=[0.224, 0.224, 0.224]),
            ToTensorV2(),
        ]
    )


def build_valid_transforms(image_size: int):
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.456, 0.456, 0.456], std=[0.224, 0.224, 0.224]),
            ToTensorV2(),
        ]
    )
