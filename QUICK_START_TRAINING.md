# Quick Start: Training & Auto-Labeling

## TL;DR

1. **Export labeled images from Label Studio** (YOLO format: images/ + labels/ + classes.txt)
2. **Train**: `python scripts/train.py export_folder --epochs 50`
3. **Auto-label**: `python scripts/auto_label.py extracted_frames --version 1`
4. **Review**: Import predictions.json into Label Studio → delete false positives
5. **Retrain**: `python scripts/train.py reviewed_folder` (auto-increments to v2)

---

## Setup (One-Time)

```bash
conda activate cartwatch
```

---

## Training a Model

### Basic Training

```bash
python scripts/train.py /path/to/label_studio_export \
  --epochs 50
```

Creates: `models/checkpoints/yolo11n_v1/weights/best.pt`

### With Custom Settings

```bash
python scripts/train.py /path/to/label_studio_export \
  --model yolo11n \
  --epochs 100 \
  --batch-size 32 \
  --imgsz 1280 \
  --val-split 0.15 \
  --device cuda
```

### Check Training Results

Ultralytics saves results in each checkpoint folder:
```
models/checkpoints/yolo11n_v1/
├── weights/
│   ├── best.pt        ← Use this for inference/auto-labeling
│   └── last.pt
├── results.csv        ← Training metrics
├── args.yaml          ← Training config
└── plots/             ← Training curves (detection_val_*.png)
```

Open the PNG plots in your browser to see training progress.

---

## Auto-Labeling Frames

### Generate Predictions

```bash
python scripts/auto_label.py data/extracted_frames \
  --model yolo11n \
  --version 1 \
  --output data/labeled/predictions.json
```

Creates: `data/labeled/predictions.json` (Label Studio import format)

### Adjust Confidence

Lower confidence = catch more items (favor recall):
```bash
python scripts/auto_label.py data/extracted_frames \
  --model yolo11n --version 1 \
  --confidence 0.15  # Very permissive
```

Higher confidence = fewer false positives (favor precision):
```bash
python scripts/auto_label.py data/extracted_frames \
  --model yolo11n --version 1 \
  --confidence 0.5   # More restrictive
```

### Use GPU

```bash
python scripts/auto_label.py data/extracted_frames \
  --model yolo11n --version 1 \
  --device cuda
```

---

## Label Studio Workflow

### Import Predictions

1. Open your Label Studio project
2. Click **Import** → **Upload Files**
3. Select `predictions.json`
4. Review predictions (red boxes = AI predictions)

### Review & Correct

- **Delete false positives** (red boxes you don't want)
- **Add missed boxes** if the model missed something
- **Approve** correct predictions

### Export Cleaned Labels

1. Click **Export** (top right)
2. Select **YOLO** format
3. Save to `reviewed_export/`

---

## Retrain with Human-Reviewed Data

```bash
python scripts/train.py reviewed_export \
  --epochs 50 \
  --device cuda
```

Creates: `models/checkpoints/yolo11n_v2/weights/best.pt` (auto-incremented)

---

## Full Active Learning Cycle

```bash
# Iteration 1: Manual labels → v1
python scripts/train.py manual_labels_export --epochs 50 --device cuda
python scripts/auto_label.py data/extracted_frames --model yolo11n --version 1
# → Import & review in Label Studio
# → Export cleaned labels

# Iteration 2: Reviewed labels → v2
python scripts/train.py reviewed_export --epochs 50 --device cuda
python scripts/auto_label.py data/extracted_frames --model yolo11n --version 2
# → Import & review in Label Studio
# → Export cleaned labels

# Iteration 3: Keep going
python scripts/train.py reviewed_export_v2 --epochs 50 --device cuda
# ... repeat until satisfied with accuracy
```

---

## Troubleshooting

### "Checkpoint not found"
```bash
ls models/checkpoints/
# Make sure yolo11n_v1/ exists
# Check --model and --version match the folder name
```

### "No images found"
```bash
# Verify export structure:
ls /path/to/export/images/
ls /path/to/export/labels/
cat /path/to/export/classes.txt
```

### Out of memory (GPU)
```bash
# Reduce batch size
python scripts/train.py export_folder --batch-size 8 --device cuda
```

### Training is slow
```bash
# Try smaller model
python scripts/train.py export_folder --model yolo11n --device cuda
# Or smaller image size
python scripts/train.py export_folder --imgsz 512 --device cuda
```

### Custom Label Studio tag names
```bash
# If your LS project uses different tag names:
python scripts/auto_label.py frames_dir \
  --model yolo11n --version 1 \
  --from-name bbox \
  --to-name photo
```

---

## What Gets Created

### After Training
```
models/checkpoints/yolo11n_v1/
├── weights/best.pt          ← Trained checkpoint
├── dataset.yaml             ← YOLO dataset config
├── train.txt                ← Train image list
├── val.txt                  ← Val image list
├── results.csv              ← Metrics
└── plots/                   ← Visualizations
```

### After Auto-Labeling
```
data/labeled/predictions.json   ← Import into Label Studio
```

---

## Next Level: Python API

```python
from src.training.trainer import YOLOTrainer
from src.inference.auto_label import AutoLabeler
from src.config import Config

# Train
trainer = YOLOTrainer(Config.default())
result = trainer.train("export_folder", "yolo11n")
print(f"Trained v{result.version}")

# Auto-label
labeler = AutoLabeler(
    checkpoint_path=str(result.best_pt),
    confidence_threshold=0.25
)
labeler.label_and_save("frames_dir", "output.json")
```

---

## Key Defaults

| Setting | Default | Tip |
|---------|---------|-----|
| Model | yolo11n | tiny & fast, good for POC |
| Epochs | 100 | reduce for faster iteration |
| Batch Size | 16 | increase if GPU memory allows |
| Image Size | 640 | smaller = faster, less accurate |
| Val Split | 0.2 (80/20) | good for most datasets |
| Train Device | cpu | use cuda if available |
| Auto-label Confidence | 0.25 | catch more, delete false positives in LS |
| Inference Confidence | 0.5 | stricter than auto-labeling |

---

## Version Control

Version numbers auto-increment:
- v1: first trained model
- v2: retrained with reviewed data from v1
- v3: retrained with reviewed data from v2
- etc.

Each version is isolated in its own folder: `models/checkpoints/yolo11n_v1/`, `v2/`, `v3/`, ...
