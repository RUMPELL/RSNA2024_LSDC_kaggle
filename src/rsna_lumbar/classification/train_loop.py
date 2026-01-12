from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm


@dataclass
class TrainConfig:
    amp: bool = True


def train_one_epoch(model, loader, optimizer, criterion, device: torch.device, cfg: TrainConfig, scaler: GradScaler | None = None) -> float:
    model.train()
    running = 0.0
    n = 0
    pbar = tqdm(loader, desc="train", leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if cfg.amp:
            if scaler is None:
                scaler = GradScaler()
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        running += loss.item() * bs
        n += bs
        pbar.set_postfix(loss=running / max(n, 1))
    return running / max(n, 1)


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device: torch.device) -> float:
    model.eval()
    running = 0.0
    n = 0
    pbar = tqdm(loader, desc="valid", leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        bs = x.size(0)
        running += loss.item() * bs
        n += bs
        pbar.set_postfix(loss=running / max(n, 1))
    return running / max(n, 1)
