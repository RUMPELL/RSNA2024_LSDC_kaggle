# Method Overview

This repository implements a two-stage pipeline:

1) **Detection**: YOLOv8 detects disc-level ROIs for each MRI sequence.
2) **Classification**: EfficientNet predicts severity from 2.5D (3-slice) PNG crops.

## 2.5D PNG export

We stack adjacent slices (t−1, t, t+1) as channels to preserve minimal 3D context while using a 2D CNN.
Cropping is centered around the detected ROI (or annotated center points for training).

## Generalization notes

Public/private leaderboard gaps may occur due to distribution shift:
- scanner/vendor differences
- protocol variations across sites
- slice thickness and orientation changes
