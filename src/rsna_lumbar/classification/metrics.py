from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve


def save_confusion_matrix(y_true, y_pred, out_path: Path, labels=(0,1,2)) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    fig = plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.colorbar()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def save_classification_report(y_true, y_pred, out_path: Path) -> None:
    rep = classification_report(y_true, y_pred, digits=4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rep, encoding="utf-8")
