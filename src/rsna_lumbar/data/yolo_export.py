from __future__ import annotations

"""YOLO training dataset exporter.

This is intentionally kept as a separate module because YOLO dataset generation
is very competition-specific (coordinate conventions, label mapping, etc.).

You can port the logic from your notebook:
- resize + pad
- bbox scaling + padding adjustments
- YOLO label normalization

Recommended outputs:
- images/{train|val}/*.png
- labels/{train|val}/*.txt  (YOLO format)
"""

# TODO: Implement based on your notebook logic.
