# RSNA 2024 Lumbar Spine Degenerative Classification — two-stage MRI pipeline

[![CI](https://github.com/RUMPELL/RSNA2024_LSDC_kaggle/actions/workflows/ci.yml/badge.svg)](https://github.com/RUMPELL/RSNA2024_LSDC_kaggle/actions/workflows/ci.yml)

A two-stage pipeline for the Kaggle competition *RSNA 2024 Lumbar Spine Degenerative
Classification*: **YOLOv8 disc localisation** on each MRI sequence, followed by
**EfficientNet severity classification** on 2.5D (three-slice) crops. The repository is the
competition notebooks refactored into an installable package with YAML-configured scripts.

> The RSNA dataset is not redistributed. Download it from Kaggle and point
> `configs/paths.yaml` at it. No trained weights or competition scores are included.

---

## Problem

For each study, the task is to grade five disc levels (L1/L2 … L5/S1) for three conditions —
spinal canal stenosis, neural foraminal narrowing (left/right), and subarticular stenosis
(left/right) — into *normal/mild*, *moderate*, or *severe*, from three MRI sequences:

| Sequence | Condition it is used for |
|---|---|
| Sagittal T2/STIR | spinal canal stenosis |
| Sagittal T1 | neural foraminal narrowing |
| Axial T2 | subarticular stenosis |

## Pipeline

```
DICOM series ──▶ dicom/io + preprocess ──▶ YOLOv8 (per sequence) ──▶ disc-level boxes
                 VOI/modality LUT,           detection/yolo_infer      + L/R side from
                 MONOCHROME1 fix,                                       ImagePositionPatient
                 1–99 % clip, resize+pad                                (postprocess/)
                                                                              │
                                                                              ▼
                                         preprocess/png_export: crop [t−1, t, t+1] ──▶ 3-channel PNG
                                                                              │
                                                                              ▼
                              classification/: EfficientNet-B4 (timm) per condition, GroupKFold by study
                                                                              │
                                                                              ▼
                                                  pipeline/submit: row_id template + fallback probabilities
```

## What is implemented

| Stage | Module | Status |
|---|---|---|
| DICOM → uint8 image (modality/VOI LUT, MONOCHROME1 inversion, percentile clipping) | `dicom/io.py`, `preprocess/png_export.py` | implemented |
| Resize-and-pad with scale/offset bookkeeping for box back-projection | `dicom/preprocess.py` | implemented |
| YOLOv8 inference over a DICOM series → detections in original pixel coordinates | `detection/yolo_infer.py` | implemented |
| Top-confidence-per-class filtering, missing-class fill, left/right side inference | `postprocess/` | implemented |
| 2.5D three-slice crop export with per-sequence crop-size rules (`configs/png_export.yaml`) | `preprocess/png_export.py`, `scripts/prepare_png_dataset.py` | implemented |
| Study-level `GroupKFold` split | `data/make_splits.py` | implemented |
| EfficientNet classifier, class-weighted cross-entropy, AdamW + cosine schedule, AMP, early stopping, best-checkpoint saving | `classification/` , `scripts/train_classifier.py` | implemented |
| Fold-ensemble logit averaging → softmax, and the `submission.csv` row template | `pipeline/submit.py` | implemented |
| YOLO **training-set** export from coordinate CSVs | `data/yolo_export.py`, `scripts/prepare_yolo_dataset.py` | **not implemented** (stub) |
| End-to-end test-time orchestration (series selection → YOLO → crop → classifier per study) | `pipeline/submit.py` | **not implemented** — writes fallback probabilities for every row |

## Data preprocessing

`scripts/prepare_png_dataset.py` reads one crop instruction per row from a CSV with columns
`study_id, series_id, instance_number, level, series_description, condition, x, y`, and writes:

```
png_root/
├─ spinal_canal_stenosis/{study_id}_disc{level}.png
├─ left_neural_foraminal_narrowing/
├─ right_neural_foraminal_narrowing/
├─ left_subarticular_stenosis/
└─ right_subarticular_stenosis/
```

Each PNG stacks the previous, centre, and next slice as channels. Slices are ordered by
`InstanceNumber` with filename as a stable tie-breaker. Crop size and offset rules for
sagittal vs. axial sequences are in `configs/png_export.yaml` and mirror the original
notebook so exported inputs match what the models were trained on.

## Training and inference

- **Training** (`scripts/train_classifier.py`): one condition at a time; expects a prepared
  dataframe (`study_id, level, severity, [side], [fold]`) and adds a study-level
  `GroupKFold` column if absent. Hyper-parameters — `tf_efficientnet_b4`, 380 px, class
  weights `[1, 2, 4]`, AdamW 3e-4, cosine schedule, AMP, patience 5 — are in
  `configs/cls_effnet.yaml`.
- **Inference** (`scripts/infer_submit.py`): loads fold ensembles and writes a
  `submission.csv`. Because per-study orchestration is not implemented, the current script
  fills every row with the configured fallback probabilities (see *Limitations*).

## Repository structure

```
src/rsna_lumbar/
  dicom/            DICOM reading, LUT/monochrome handling, resize+pad
  detection/        YOLOv8 series inference
  postprocess/      confidence filtering, missing-class fill, L/R side inference
  preprocess/       2.5D crop export
  data/             GroupKFold splits; YOLO dataset export (stub)
  classification/   dataset, EfficientNet model, transforms, train loop, fold runner, metrics
  pipeline/         submission assembly
  utils/            YAML loader, seeding
scripts/            prepare_png_dataset, prepare_yolo_dataset (stub), train_classifier, infer_submit
configs/            paths.example.yaml, png_export.yaml, cls_effnet.yaml
docs/               method_overview.md
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .

cp configs/paths.example.yaml configs/paths.yaml   # then edit every path

# 1) export 2.5D PNG crops
python scripts/prepare_png_dataset.py \
  --config configs/png_export.yaml --paths configs/paths.yaml \
  --csv /path/to/crop_instructions.csv

# 2) train one condition (all folds)
python scripts/train_classifier.py \
  --config configs/cls_effnet.yaml --paths configs/paths.yaml \
  --condition spinal_canal_stenosis --train-df /path/to/train_df.csv

# 3) write a submission template (fallback probabilities only — see Limitations)
python scripts/infer_submit.py --paths configs/paths.yaml --cls configs/cls_effnet.yaml
```

`pip install -e .` pulls in torch, timm, ultralytics, pydicom, and OpenCV; a CUDA GPU is
assumed for training.

## Reproducibility

- Seeds for Python, NumPy, and PyTorch are set in `utils/seed.py`.
- Splits are study-level to avoid leakage between slices of the same patient.
- Crop rules and DICOM conversion are configuration-driven and documented against the
  original notebook (`docs/method_overview.md`).
- Competition results are **not** recorded in this repository; nothing here should be read
  as a leaderboard claim.
- **Tests.** `python -m unittest discover -s tests -t .` runs 50 dependency-light checks on
  synthetic inputs: DICOM → uint8 conversion (LUT pass-through, MONOCHROME1 inversion,
  percentile clipping), crop clamping and crop-rule arithmetic, `[prev, center, next]` slice
  selection at the first/interior/last instance, resize-and-pad back-projection, the
  study-level `GroupKFold` leakage guard, detection post-processing, and config loading. CI
  runs them on Python 3.10/3.11 without torch, weights, or data.

## Limitations

- **YOLO training-set export is a stub.** `data/yolo_export.py` and
  `scripts/prepare_yolo_dataset.py` print a TODO; YOLO models must be trained elsewhere.
- **No end-to-end inference.** `pipeline/submit.py` builds the `row_id` template and loads
  classifier ensembles, but the per-study loop that selects series, runs YOLO, crops, and
  classifies is not written; every row receives the fallback `[0.4, 0.4, 0.2]`.
- `scripts/infer_submit.py` has placeholder YOLO/classifier weight paths that must be edited.
- Tests cover only the utility layers above; the classifier, YOLO inference, and training
  loop are not exercised in CI.
- The 2.5D crop rules were tuned on the competition data; generalisation across scanners,
  protocols, and slice thicknesses is not evaluated.

## License

[MIT](LICENSE)
