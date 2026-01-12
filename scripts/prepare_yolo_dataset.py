from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rsna_lumbar.utils.config import load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/paths.yaml")
    ap.add_argument("--out", required=True, help="output root for YOLO dataset")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # This script is a thin wrapper. For a full YOLO export pipeline, see:
    # - src/rsna_lumbar/data/yolo_export.py (you can implement using your notebook logic)
    # We keep it minimal here to avoid leaking competition-specific assumptions.
    print("TODO: Implement YOLO export pipeline here using your coordinate CSVs.")
    print(f"train_csv={cfg['train_csv']}")
    print(f"train_label_coordinates_csv={cfg['train_label_coordinates_csv']}")
    print(f"train_images_dir={cfg['train_images_dir']}")
    print(f"out={out_root}")


if __name__ == "__main__":
    main()
