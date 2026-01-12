# scripts/prepare_png_dataset.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml
from tqdm import tqdm

from rsna_lumbar.preprocess.png_export import (
    CropRuleConfig,
    DicomToImageConfig,
    export_crop_3ch_png,
)


@dataclass(frozen=True)
class ColumnMap:
    study_id: str
    series_id: str
    instance_number: str
    level: str
    series_description: str
    condition: str
    x: str
    y: str


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to configs/png_export.yaml")
    ap.add_argument("--paths", type=str, required=True, help="Path to configs/paths.yaml")
    ap.add_argument("--csv", type=str, required=True, help="Crop instruction CSV (rows describe crops)")
    ap.add_argument("--limit", type=int, default=-1, help="Limit rows for debugging")
    return ap.parse_args()


def build_column_map(cfg: Dict[str, Any]) -> ColumnMap:
    c = cfg["columns"]
    return ColumnMap(
        study_id=c["study_id"],
        series_id=c["series_id"],
        instance_number=c["instance_number"],
        level=c["level"],
        series_description=c["series_description"],
        condition=c["condition"],
        x=c["x"],
        y=c["y"],
    )


def main() -> None:
    args = parse_args()
    cfg = load_yaml(Path(args.config))
    paths = load_yaml(Path(args.paths))

    train_images_dir = Path(paths["train_images_dir"]).expanduser().resolve()
    out_root = Path(paths["png_out_dir"]).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    col = build_column_map(cfg)

    dicom_cfg = DicomToImageConfig(
        try_lut=bool(cfg["dicom"].get("try_lut", True)),
        fix_monochrome=bool(cfg["dicom"].get("fix_monochrome", True)),
        clip_percentiles=tuple(cfg["dicom"].get("clip_percentiles", [1.0, 99.0])),
    )

    crop_rules = CropRuleConfig(
        axial_scale=float(cfg["crop_rules"].get("axial_scale", 0.13)),
        non_axial_scale=float(cfg["crop_rules"].get("non_axial_scale", 0.20)),
        axial_min_small=int(cfg["crop_rules"].get("axial_min_small", 64)),
        axial_min_large=int(cfg["crop_rules"].get("axial_min_large", 112)),
        sagittal_offset_x_frac=float(cfg["crop_rules"].get("sagittal_offset_x_frac", -0.05)),
        sagittal_offset_y_frac=float(cfg["crop_rules"].get("sagittal_offset_y_frac", 0.05)),
    )

    df = pd.read_csv(args.csv)
    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    failures = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="export png crops"):
        try:
            export_crop_3ch_png(
                train_images_dir=train_images_dir,
                out_root=out_root,
                study_id=row[col.study_id],
                series_id=row[col.series_id],
                instance_number=int(row[col.instance_number]),
                level=row[col.level],
                series_description=row[col.series_description],
                condition=row[col.condition],
                x=float(row[col.x]),
                y=float(row[col.y]),
                dicom_cfg=dicom_cfg,
                crop_rules=crop_rules,
            )
        except Exception as e:
            failures.append(
                {
                    "error": repr(e),
                    "study_id": row.get(col.study_id, None),
                    "series_id": row.get(col.series_id, None),
                    "instance_number": row.get(col.instance_number, None),
                    "level": row.get(col.level, None),
                    "condition": row.get(col.condition, None),
                }
            )

    if failures:
        fail_path = out_root / "export_failures.yaml"
        with open(fail_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(failures, f, sort_keys=False, allow_unicode=True)
        print(f"[WARN] Completed with failures: {len(failures)}")
        print(f"[WARN] Failure log saved to: {fail_path}")
    else:
        print("[OK] Export completed with no failures.")


if __name__ == "__main__":
    main()
