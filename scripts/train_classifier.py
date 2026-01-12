from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rsna_lumbar.utils.config import load_yaml
from rsna_lumbar.utils.seed import set_seed
from rsna_lumbar.data.make_splits import add_group_kfold
from rsna_lumbar.classification.run_fold import run_fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/cls_effnet.yaml")
    ap.add_argument("--paths", required=True, help="configs/paths.yaml")
    ap.add_argument("--condition", required=True, choices=[
        "spinal_canal_stenosis",
        "neural_foraminal_narrowing",
        "subarticular_stenosis",
    ])
    ap.add_argument("--fold", type=int, default=None, help="train a single fold (0..n_folds-1); default: all folds")
    ap.add_argument("--train-df", default=None, help="Optional prepared training dataframe CSV with columns: study_id, level, severity, fold(optional), side(optional)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    paths = load_yaml(args.paths)

    set_seed(int(cfg.get("seed", 42)))

    png_root = Path(paths["png_root"])
    models_root = Path(paths["models_root"])
    out_dir = models_root / args.condition
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load a prepared dataframe.
    # For a clean public repo, you should generate this from Kaggle's train.csv + your preprocessing.
    if args.train_df is None:
        raise ValueError("--train-df is required. Provide a CSV with columns: study_id, level, severity, (optional) side, fold")
    df = pd.read_csv(args.train_df)

    # add folds if missing
    if "fold" not in df.columns:
        df = add_group_kfold(df, n_splits=int(cfg.get("n_folds", 5)), group_col="study_id")

    folds = [args.fold] if args.fold is not None else sorted(df["fold"].unique().tolist())
    for f in folds:
        run_fold(df=df, fold=int(f), condition=args.condition, png_root=png_root, out_dir=out_dir, cfg=cfg)


if __name__ == "__main__":
    main()
