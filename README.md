# YOLO Skin Cancer (Benign vs Malignant)

Prepare a shared YOLO dataset from **HAM10000** for comparing **YOLOv3 / YOLOv5 / YOLOv8** on binary lesion detection:

- `0` benign: `nv`, `bkl`, `df`, `vasc`
- `1` malignant: `mel`, `bcc`, `akiec`

Binary skin-lesion detection with a fair comparison of **YOLOv5su / YOLOv8s / YOLOv3-tinyu** on the same balanced HAM10000 split.

## Hardware target

- NVIDIA RTX 3060 (12GB)
- Python 3.12
- Image size: `640`

## Quick start

### 1) Create GPU environment

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
.\.venv\Scripts\Activate.ps1
```

Verify:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
```

### 2) Download raw data

**Recommended (no API key):**

```powershell
# 1) metadata + masks from Harvard Dataverse (~11 MB)
python scripts\download_ham10000.py --source dataverse --skip-images

# 2) images from ISIC Archive API (resumable; skips existing files)
python scripts\download_images_isic.py --workers 4
```

Alternatives for images:

```powershell
# Hugging Face GenSeg mirror (single ~1.7 GB tar)
python scripts\download_ham10000_hf.py

# Full Dataverse image zips (~2.7 GB)
python scripts\download_ham10000.py --source dataverse
```

Optional Kaggle source (if you have `%USERPROFILE%\.kaggle\kaggle.json`):

```powershell
python scripts\download_ham10000.py --source kaggle
```

Accept Kaggle dataset terms first if using that path:
- [skin-cancer-mnist-ham10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- [ham10000-lesion-segmentations](https://www.kaggle.com/datasets/tschandl/ham10000-lesion-segmentations)

Large downloads use local `tools/aria2` when present (multi-connection resume).

### 4) Convert masks → YOLO labels + stratified split

```powershell
python scripts\prepare_yolo_dataset.py
```

Creates `data/processed/` with `train/val/test` (70/15/15, seed=42) and updates `configs/data.yaml`.

To enlarge a partial image set (priority malignant):

```powershell
python scripts\download_missing_by_class.py --target-malignant 800 --target-benign 2000 --workers 4
python scripts\prepare_yolo_dataset.py
```

### 5) Verify

```powershell
python scripts\verify_dataset.py
```

Outputs stats in the terminal and a mosaic at `runs/verify/sample_mosaic.png`.

Or run prepare+verify together after downloads:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\finish_pipeline.ps1
```

## Project layout

```text
configs/data.yaml          # shared YOLO data config
data/raw/                  # images, masks, metadata
data/processed/            # YOLO images/ + labels/
scripts/
  setup_env.ps1
  download_ham10000.py
  prepare_yolo_dataset.py
  verify_dataset.py
models/                    # YOLO repos (later)
runs/                      # training / verification outputs
```

## Training (fair comparison)

Dataset uses a **lesion-level** stratified 70/15/15 split (`lesion_id`) so the same lesion never leaks across train/val/test.

Balanced recipe (recommended):

```powershell
.\.venv\Scripts\Activate.ps1
python scripts\prepare_yolo_dataset.py --balance-train --malignant-ratio 1.2
python scripts\train_yolo.py --model yolov5su --config configs/train_balanced.yaml --name yolov5su_balanced
python scripts\train_yolo.py --model yolov8s --config configs/train_balanced.yaml --name yolov8s_balanced
python scripts\train_yolo.py --model yolov3-tinyu --config configs/train_balanced.yaml --name yolov3-tinyu_balanced
```

Shared balanced hyperparameters: `configs/train_balanced.yaml` (imgsz 640, AdamW, cls=2.5).

Evaluate image-level discrimination:

```powershell
python scripts\test_generalization.py --models yolov5su_balanced yolov8s_balanced yolov3-tinyu_balanced
```

Notes:
- Full `yolov3u` is too heavy for RTX 3060 12GB checkpointing; use `yolov3-tinyu`.
- Comparison metrics: `runs/compare_summary.json` and `runs/reports/model_comparison_report.md`.

## Google Colab

Notebook ready for retrain / inference:

- [`notebooks/YOLO_HAM10000_Colab.ipynb`](notebooks/YOLO_HAM10000_Colab.ipynb)

Open directly:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlrgabeAh2025/yolo-testing-project/blob/main/notebooks/YOLO_HAM10000_Colab.ipynb)

The notebook is split into:
1. setup + dataset download
2. **Training** section (retrain balanced models)
3. **Run / Inference** section (load `best.pt` and predict)

## Disclaimer

Research / educational use only. Not a medical device and not for clinical diagnosis.
