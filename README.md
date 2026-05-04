# A VISION TRANSFORMER (VIT)-BASED MODEL FOR CLASSIFICATION OF SCHIZOPHRENIA SEVERITY CATEGORIES USING MANDALA DRAWINGS

## Project Structure

```
final_model/
├── config/
│   └── config.py                   # All hyperparameters and dataset settings
├── models/
│   └── vit_classifier.py           # ViT backbone + late-fusion classification head
├── training/
│   ├── train.py                    # Main training script
│   └── trainEx1.py                 # Alternative training experiment
├── evaluation/
│   └── metrics.py                  # Metrics, confusion matrix, plot utilities
├── explainability/
│   ├── explain.py                  # CLI for Grad-CAM and attribution methods
│   ├── attribution.py              # Grad-CAM, saliency, integrated gradients, occlusion
│   ├── visualize.py                # Heatmap overlay rendering
│   └── io_utils.py                 # Checkpoint loading and image preprocessing
├── tools/
│   ├── inference.py                # Single-image and batch inference
│   ├── gradcam_inference.py        # Grad-CAM visualization tool
│   ├── demo_app.py                 # Interactive demo application
│   └── regen_confusion_matrix.py   # Regenerate confusion matrix plots
├── utils/
│   ├── dataset.py                  # PyTorch Dataset for images + completion time
│   └── transforms.py               # Train and eval image transform pipelines
├── dataset/                        # Place dataset here (gitignored)
│   ├── healthy/
│   ├── minor/
│   ├── medium/
│   └── severe/
├── checkpoints/                    # Saved model checkpoints (gitignored)
└── explainability_outputs/         # Grad-CAM outputs (gitignored)
```

---

## Installation

**Requirements:** Python 3.8+, NVIDIA GPU with CUDA recommended.

```bash
git clone <repository-url>
cd final_model

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Dataset Setup

### Directory Structure

Place mandala images into class subdirectories under `dataset/`:

```
dataset/
├── healthy/
│   └── HC_001.png
├── minor/
│   └── SZ_001.png
├── medium/
│   └── SZ_010.png
└── severe/
    └── SZ_020.png
```

### Completion Time CSV

Create `dataset/dataset.csv` with task completion times:

```csv
sample_name,image_name,image_path,severity_class,completion_time
HC_001,HC_001.png,dataset/healthy/HC_001.png,healthy,2340
SZ_001,SZ_001.png,dataset/minor/SZ_001.png,minor,870
SZ_010,SZ_010.png,dataset/medium/SZ_010.png,medium,1080
SZ_020,SZ_020.png,dataset/severe/SZ_020.png,severe,420
```

| Column | Description |
|--------|-------------|
| `sample_name` | Unique sample ID — must match the image filename stem |
| `image_name` | Image filename |
| `image_path` | Path to the image file |
| `severity_class` | One of: `healthy`, `minor`, `medium`, `severe` |
| `completion_time` | Task duration in seconds |

---

## Configuration

All settings are in `config/config.py`. Key options:

```python
# Paths
data_dir              = "dataset"
completion_time_csv   = "dataset.csv"
checkpoint_dir        = "checkpoints"

# Model
model_name            = "vit_base_patch16_224_in21k"
pretrained            = True
use_time_feature      = True
dropout               = 0.30
num_classes           = 4
class_names           = ["healthy", "minor", "medium", "severe"]

# Data split (50% train / 25% val / 25% test)
test_size             = 0.25
val_size              = 0.333
random_state          = 7
image_size            = 224

# Training
batch_size            = 2
max_epochs            = 50
head_only_epochs      = 10        # epochs to train head only (backbone frozen)
head_lr               = 1e-4
backbone_lr           = 1e-7
weight_decay          = 0.05
label_smoothing       = 0.05
early_stopping_patience = 5
unfreeze_last_n_blocks  = 2

# Augmentation
rotation_degrees      = 20.0
horizontal_flip_prob  = 0.5
vertical_flip_prob    = 0.3
```

---

## Training

```bash
python training/train.py
```

Outputs saved to `checkpoints/run_YYYYMMDD_HHMMSS/`:

| File | Description |
|------|-------------|
| `vit_mandala_classifier_best.pth` | Best model checkpoint |
| `training.log` | Full training log |
| `training_curves.png` | Loss, accuracy, macro F1 per epoch |
| `val_confusion_matrix.png` | Confusion matrix on validation set |
| `test_confusion_matrix.png` | Confusion matrix on test set |
| `test_per_class_metrics.csv` | Per-class precision, recall, F1 |
| `test_predictions.csv` | Per-sample predictions and probabilities |
| `model_summary.txt` | Config and final test metrics |

---

## Inference

### Single Image

```bash
python tools/inference.py \
  --checkpoint checkpoints/run_TIMESTAMP/vit_mandala_classifier_best.pth \
  --image path/to/mandala.png
```

### Batch

```bash
python tools/inference.py \
  --checkpoint checkpoints/run_TIMESTAMP/vit_mandala_classifier_best.pth \
  --data-dir dataset/
```

---

## Explainability (Grad-CAM)

### Single method

```bash
python explainability/explain.py \
  --checkpoint checkpoints/run_TIMESTAMP/vit_mandala_classifier_best.pth \
  --image-path path/to/mandala.png \
  --completion-time 1200 \
  --method gradcam \
  --output-dir explainability_outputs/
```

### All methods

```bash
python explainability/explain.py \
  --checkpoint checkpoints/run_TIMESTAMP/vit_mandala_classifier_best.pth \
  --image-path path/to/mandala.png \
  --completion-time 1200 \
  --method all \
  --output-dir explainability_outputs/
```

**Available methods:** `gradcam`, `saliency`, `attention`, `integrated_gradients`, `occlusion`

Outputs saved per method:
- `<method>_overlay_class_<idx>_<name>.png` — heatmap on mandala image
- `<method>_map_class_<idx>_<name>.png` — grayscale attribution map
- `<method>_summary.json` — predicted class, probabilities, file paths
- `run_summary.json` — full run metadata

---

## Regenerate Confusion Matrix

```bash
python tools/regen_confusion_matrix.py \
  --checkpoint checkpoints/run_TIMESTAMP/vit_mandala_classifier_best.pth \
  --data-dir dataset/
```

---

## Demo App

```bash
python tools/demo_app.py
```

---

## Troubleshooting

**CUDA out of memory:**
Reduce `image_size` in `config.py`. `batch_size` is already at minimum (2).

**Images not found:**
Class subdirectory names must exactly match `class_names` in `config.py`: `healthy`, `minor`, `medium`, `severe`.

**CSV sample IDs not matched:**
`sample_name` in the CSV must match the image filename without extension (e.g., `SZ_001` for `SZ_001.png`).

**Slow training without GPU:**
Set `num_workers > 0` in `config.py` and ensure CUDA is available (`torch.cuda.is_available()`).

---

## Dependencies

| Package | Version |
|---------|---------|
| `torch` | >= 2.0.0 |
| `torchvision` | >= 0.15.0 |
| `timm` | >= 0.9.0 |
| `scikit-learn` | >= 1.2.0 |
| `pandas` | >= 2.0.0 |
| `numpy` | >= 1.24.0 |
| `matplotlib` | >= 3.7.0 |
| `Pillow` | >= 9.5.0 |

---

## Author

**D.M.I.P.B Gunasinghe** — Department of Computing, Rajarata University of Sri Lanka
Contact: pradeepisuru31@gmail.com
