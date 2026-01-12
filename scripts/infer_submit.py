from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rsna_lumbar.utils.config import load_yaml
from rsna_lumbar.pipeline.submit import make_submission


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", required=True, help="configs/paths.yaml")
    ap.add_argument("--cls", required=True, help="configs/cls_effnet.yaml")
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()

    paths_cfg = load_yaml(args.paths)
    cls_cfg = load_yaml(args.cls)

    test_series_df = pd.read_csv(paths_cfg["test_series_descriptions_csv"])

    # Map series type -> YOLO model path (edit in your configs or here)
    yolo_models = {
        "Sagittal T2/STIR": "path/to/yolo_scs.pt",
        "Sagittal T1": "path/to/yolo_nfs.pt",
        "Axial T2": "path/to/yolo_sc.pt",
    }

    # Map condition -> list of fold weight paths
    cls_weights = {
        "spinal_canal_stenosis": [],
        "neural_foraminal_narrowing": [],
        "subarticular_stenosis": [],
    }

    make_submission(
        test_series_df=test_series_df,
        paths_cfg=paths_cfg,
        cls_cfg=cls_cfg,
        yolo_models=yolo_models,
        cls_weights=cls_weights,
        out_csv=Path(args.out),
    )


if __name__ == "__main__":
    main()
