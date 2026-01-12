# RSNA 2024 Lumbar Spine Degenerative Classification (Two-Stage: YOLOv8 → EfficientNet)

This repository packages a **reproducible**, **GitHub-public-ready** pipeline for the Kaggle competition
**RSNA 2024 Lumbar Spine Degenerative Classification**, based on a two-stage approach:

1. **Localization (Stage 1):** YOLOv8 detects disc/lesion-level regions per MRI sequence.
2. **Severity classification (Stage 2):** EfficientNet consumes **2.5D (3-slice) PNG patches** and predicts severity.

> Note: This repo **does not** redistribute the RSNA dataset. You must download data via Kaggle and configure paths.

---

## Method Overview

### Stage 1 — YOLOv8 detection (per sequence)
- **Sagittal T2/STIR** → Spinal Canal Stenosis (SCS)
- **Sagittal T1** → Neural Foraminal Narrowing (NFS)
- **Axial T2** → Subarticular Stenosis (SC)

### Stage 2 — EfficientNet severity classification (2.5D)
- Each detected disc region is converted to a **3-channel PNG** by stacking adjacent slices: *(t−1, t, t+1)*.
- Classifier is trained per condition and can be ensembled across folds.

**Slice ordering:** when exporting 2.5D patches, DICOM slices are ordered by `InstanceNumber` (with filename as a stable tie-breaker) to match the original competition notebook behavior.

---

## Repository Structure

- `src/rsna_lumbar/`
  - `dicom/` : DICOM reading and normalization
  - `detection/` : YOLO inference utilities
  - `preprocess/` : DICOM → 3-channel PNG patch export
  - `classification/` : EfficientNet dataset/model/train/infer
  - `pipeline/` : end-to-end orchestration and submission creation
- `scripts/` : runnable entrypoints
- `configs/` : YAML configs (paths, hyperparameters)

---

## Setup

```bash
# recommended
python -m venv .venv
source .venv/bin/activate

pip install -e .
```

---

## Configure Paths

Copy and edit:

```bash
cp configs/paths.example.yaml configs/paths.yaml
```

---

## Expected PNG Layout (Stage-2 input)

`prepare_png_dataset.py` will create (or expects) a layout like:

```
png_root/
├─ spinal_canal_stenosis/
│  ├─ {study_id}_disc{level}.png
├─ left_neural_foraminal_narrowing/
├─ right_neural_foraminal_narrowing/
├─ left_subarticular_stenosis/
└─ right_subarticular_stenosis/
```

This matches the dataset routing in `src/rsna_lumbar/classification/dataset.py`.

---

## Quickstart

### 1) Export YOLO training dataset (optional, if training YOLO)
```bash
python scripts/prepare_yolo_dataset.py --config configs/paths.yaml --out outputs/yolo_dataset
```

### 2) Export 3-channel PNG patches for EfficientNet (recommended)
```bash
python scripts/prepare_png_dataset.py \
  --config configs/png_export.yaml \
  --paths configs/paths.yaml \
  --csv ./data/crop_instructions.csv
```

### 3) Train classifier (per condition)
```bash
python scripts/train_classifier.py --config configs/cls_effnet.yaml --paths configs/paths.yaml --condition spinal_canal_stenosis
```

### 4) Create submission (requires trained weights and YOLO models)
```bash
python scripts/infer_submit.py --paths configs/paths.yaml --cls configs/cls_effnet.yaml
```

---

## Reproducibility Notes

- All randomness is controlled via `rsna_lumbar/utils/seed.py`.
- Splits should be **study-level** to avoid leakage (see `rsna_lumbar/data/make_splits.py`).

---

## Disclaimer

This is a research/competition codebase repackaged for clarity and reproducibility.
Competition-specific heuristics (e.g., fallback probabilities) are configurable and logged.




## Notebook parity (important)

This repository was refactored from the original Kaggle notebooks. The PNG export pipeline is designed to match
`save_crop-3ch.ipynb` as closely as possible:

- DICOM -> uint8 conversion uses modality/VOI LUT (when available), MONOCHROME1 fix, and percentile clipping (1,99).
- 3-channel crops are created by stacking **[prev, center, next]** slices.
- Crop sizing/offset rules follow the notebook for `Sagittal T2/STIR` and `Axial T2`.

If you want identical inputs to the notebook, make sure your crop-instruction CSV uses the same columns and semantics.

## Prepare 3-channel PNG crops

1) Copy configs:

```bash
cp configs/paths.example.yaml configs/paths.yaml
```

2) Edit `configs/paths.yaml` and set:
- `train_images_dir`: path to Kaggle-style DICOM tree (`train_images/{study_id}/{series_id}/{instance_number}.dcm`)
- `png_out_dir`: output folder

3) Run exporter:

```bash
python scripts/prepare_png_dataset.py \
  --config configs/png_export.yaml \
  --paths configs/paths.yaml \
  --csv /path/to/crop_instructions.csv
```

The exporter expects each row in `crop_instructions.csv` to describe one crop with:
`study_id, series_id, instance_number, level, series_description, condition, x, y`.

Outputs will be written under `png_out_dir` with the same folder layout used in the training notebook.
