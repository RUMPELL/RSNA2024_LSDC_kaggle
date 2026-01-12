from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from rsna_lumbar.classification.dataset import MedicalImageDataset, PngRouting
from rsna_lumbar.classification.model import EfficientNetSeverity
from rsna_lumbar.classification.transforms import build_train_transforms, build_valid_transforms
from rsna_lumbar.classification.train_loop import TrainConfig, train_one_epoch, validate_one_epoch
from rsna_lumbar.classification.metrics import save_confusion_matrix, save_classification_report


@dataclass
class FoldRunArtifacts:
    best_path: Path
    best_val_loss: float


def run_fold(
    df: pd.DataFrame,
    fold: int,
    condition: str,
    png_root: Path,
    out_dir: Path,
    cfg: Dict,
) -> FoldRunArtifacts:
    """Train one fold and save best checkpoint."""
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    image_size = int(cfg["image_size"])
    batch_size = int(cfg["batch_size"])
    num_workers = int(cfg.get("num_workers", 4))
    epochs = int(cfg["epochs"])
    lr = float(cfg["lr"])
    wd = float(cfg.get("weight_decay", 1e-4))
    amp = bool(cfg.get("amp", True))
    patience = int(cfg.get("early_stopping_patience", 5))

    fold_col = cfg.get("fold_col", "fold")
    target_col = cfg.get("target_col", "severity")

    tr = df[df[fold_col] != fold].copy()
    va = df[df[fold_col] == fold].copy()

    routing = PngRouting(png_root=png_root)
    # side is encoded in df for non-SCS conditions; expected columns: side in ['Left','Right'] or 'left'/'right'
    side = None
    if condition != "spinal_canal_stenosis":
        # you can run two trainings (left/right) or include side in df and route per-row;
        # for simplicity, we expect df already filtered to one side when condition != SCS.
        side = str(tr["side"].iloc[0]).lower() if "side" in tr.columns else "left"

    ds_tr = MedicalImageDataset(tr, routing=routing, transform=build_train_transforms(image_size), condition=condition, side=side)
    ds_va = MedicalImageDataset(va, routing=routing, transform=build_valid_transforms(image_size), condition=condition, side=side)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)

    model = EfficientNetSeverity(
        backbone=str(cfg.get("backbone", "tf_efficientnet_b4")),
        pretrained=bool(cfg.get("pretrained", True)),
        dropout=float(cfg.get("dropout", 0.2)),
        num_classes=3,
    ).to(device)

    class_weights = torch.tensor(cfg.get("class_weights", [1.0, 2.0, 4.0]), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / f"{condition}_fold{fold}_best.pth"
    best_val = float("inf")
    bad = 0

    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    for ep in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, dl_tr, optimizer, criterion, device, TrainConfig(amp=amp), scaler=scaler)
        va_loss = validate_one_epoch(model, dl_va, criterion, device)
        scheduler.step()

        # early stop
        if va_loss < best_val:
            best_val = va_loss
            bad = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad += 1

        # simple log
        print(f"[fold {fold}] epoch {ep}/{epochs} train_loss={tr_loss:.4f} val_loss={va_loss:.4f} best={best_val:.4f} bad={bad}/{patience}")

        if bad >= patience:
            break

    return FoldRunArtifacts(best_path=best_path, best_val_loss=best_val)
