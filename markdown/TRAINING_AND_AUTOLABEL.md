# Training and Auto-Labeling Pipeline

## Overview

CartWatch has a complete active learning pipeline for training Faster R-CNN models and generating Label Studio-compatible predictions. This enables the core workflow:

**Manual Labels → Train Faster R-CNN v1 → Auto-Label New Frames → Human Review → Retrain Faster R-CNN v2 → Repeat**

---

## New Components

### 1. Training Pipeline (`src/training/`)

**`trainer.py`** — Model training orchestration
- `YOLOTrainer` class: fine-tunes YOLO on Label Studio YOLO exports
- Automatically splits labeled data into 80% train / 20% val
- Saves checkpoints to versioned folders: `models/checkpoints/yolo11n_v1/`, `v2/`, etc.
- Returns `TrainingRunResult` with paths, metrics, and class names

**`dataset_builder.py`** — Dataset preparation
- `discover_class_names()` — reads class names from `classes.txt` or `notes.json`
- `split_dataset()` — random train/val split with reproducible seeding
- `write_split_files()` — generates `train.txt`/`val.txt` (image lists for ultralytics)
- `generate_dataset_yaml()` — creates ultralytics-compatible YAML configuration

**`checkpoint_manager.py`** — Version control
- `get_next_version()` — discovers max version number for a model
- `allocate_version_dir()` — reserves the next version folder path

### 2. Auto-Labeling (`src/inference/`)

**`auto_label.py`** — Prediction generation for review
- `AutoLabeler` class: runs a trained checkpoint over unlabeled frames
- Produces Label Studio prediction-import JSON (ready to import for human review)
- Lower default confidence (0.25 vs 0.5 for inference) → favors recall

**`label_studio_format.py`** — Format conversion
- `detection_to_ls_bbox()` — converts YOLO bbox format (corner, normalized 0-1) to Label Studio format (top-left + width/height, percentage 0-100)
- `build_ls_task()` — builds one task (image + predictions) for Label Studio
- `build_ls_import_payload()` — builds the full JSON array for batch import

### 3. Configuration (`src/config.py`)

Added `TrainingConfig` dataclass:
```python
@dataclass
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 16
    imgsz: int = 640
    val_split: float = 0.2
    patience: int = 50
    seed: int = 42
    device: str = "cpu"
```

### 4. Core Improvements (`src/core/detector.py`)

Fixed `YOLODetector.__init__()` to accept:
- **Stock model names**: `"yolo11n"` → automatically becomes `"yolo11n.pt"` (ultralytics auto-download)
- **Full paths to custom checkpoints**: `"models/checkpoints/yolo11n_v1/weights/best.pt"` → used as-is

Backward compatible: existing code still works unchanged.

### 5. CLI Scripts

**`scripts/train.py`** — Train a model
```bash
python scripts/train.py /path/to/label_studio_export \
  --model yolo11n \
  --epochs 100 \
  --batch-size 16 \
  --val-split 0.2 \
  --device cpu
```

Creates `models/checkpoints/yolo11n_v1/` (auto-increments version number).

**`scripts/auto_label.py`** — Generate predictions for review
```bash
python scripts/auto_label.py data/extracted_frames \
  --model yolo11n \
  --version 1 \
  --output data/labeled/predictions.json \
  --confidence 0.25 \
  --device cpu
```

Outputs Label Studio import JSON ready for import and human review.

---

## Workflow Example

### Step 1: Prepare a Label Studio YOLO Export

Manually label ~500 images in Label Studio and export as YOLO format:
```
export_folder/
├── images/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── labels/
│   ├── img_001.txt      (YOLO format: class_id cx cy w h)
│   ├── img_002.txt
│   └── ...
└── classes.txt          (one class name per line)
```

### Step 2: Train Initial Model (v1)

```bash
python scripts/train.py export_folder \
  --model yolo11n \
  --epochs 50 \
  --device cuda
```

Result: `models/checkpoints/yolo11n_v1/weights/best.pt`

### Step 3: Auto-Label New Extracted Frames

```bash
python scripts/auto_label.py data/extracted_frames \
  --model yolo11n \
  --version 1 \
  --output data/labeled/predictions.json
```

Result: JSON file with 100s of predicted boxes (ready for Label Studio import)

### Step 4: Review in Label Studio

1. Import the JSON: **Import → Upload Files**
2. Review predictions
3. **Delete false positives** (red boxes)
4. **Add missed detections** if needed
5. Export cleaned labels back to YOLO format

### Step 5: Retrain (v2)

```bash
python scripts/train.py reviewed_export_folder \
  --model yolo11n \
  --epochs 50
```

Result: `models/checkpoints/yolo11n_v2/weights/best.pt` (better, human-verified)

### Repeat

Continue the active learning cycle: v2 → auto-label → review → v3 → ...

---

## Configuration Examples

### Use GPU Training

```python
from src.config import Config
config = Config.default()
config.training.device = "cuda"
```

### Adjust Train/Val Split

```python
config.training.val_split = 0.1  # 90/10 instead of 80/20
```

### Custom Image Size

```python
config.training.imgsz = 1280  # instead of default 640
```

### Lower Confidence for Auto-Labeling (Recall Over Precision)

```python
python scripts/auto_label.py frames_dir \
  --model yolo11n --version 1 \
  --confidence 0.15  # very low threshold, catch everything
```

---

## Important Notes

### Label Studio Integration

- **Import Format**: JSON array of tasks with `data.image` (path) and `predictions` (bboxes)
- **Default Tag Names**: Script uses `from_name="label"` and `to_name="image"` — these **must match your Label Studio labeling config XML**. Use `--from-name` and `--to-name` flags to override if needed.
- **Path Format**: The `data.image` field currently uses absolute filesystem paths. Adjust based on your Label Studio storage configuration (local storage, served files, cloud storage, etc.)

### Checkpoint Storage

- Checkpoints are saved to **`models/checkpoints/<model_name>_v<N>/weights/best.pt`**
- Ultralytics' native directory structure is preserved (includes `results.csv`, `args.yaml`, training plots)
- Version numbers auto-increment per model name (no collisions)

### Training Data Requirements

- Expects a **flat folder** of Label Studio YOLO exports (images/ + labels/ + classes.txt)
- Auto-splits 80/20 train/val — no need to pre-split
- Handles missing label files gracefully (treats as background images)

### Auto-Labeling Confidence

- Default: `0.25` (much lower than inference `0.5`) because false positives are cheap to delete in Label Studio UI
- Missed detections are more expensive (require manual box drawing)
- Adjust `--confidence` based on your specific items and tolerance for false positives

---

## Test Coverage

**41 new tests** cover:
- Label Studio bbox format conversion (5 tests)
- Task/payload building (7 tests)
- Checkpoint versioning (8 tests)
- Dataset discovery & splitting (10 tests)
- YAML generation (3 tests)
- Detector path handling (6 tests)

All tests pass: `pytest tests/ -v` (48 total: 7 existing + 41 new)

---

## API Reference

### Training

```python
from src.training.trainer import YOLOTrainer
from src.config import Config

config = Config.default()
trainer = YOLOTrainer(config)

result = trainer.train(
    export_dir="path/to/label_studio_export",
    model_name="yolo11n"
)

print(f"Trained: {result.version_dir}")
print(f"Classes: {result.class_names}")
print(f"Train/Val: {result.num_train_images}/{result.num_val_images}")
```

### Auto-Labeling

```python
from src.inference.auto_label import AutoLabeler

labeler = AutoLabeler(
    checkpoint_path="models/checkpoints/yolo11n_v1/weights/best.pt",
    confidence_threshold=0.25,
    device="cpu"
)

labeler.label_and_save(
    image_dir="data/extracted_frames",
    output_json="data/labeled/predictions.json"
)
```

---

## Next Steps

1. **Test training**: Point `scripts/train.py` at your first manually-labeled dataset
2. **Generate predictions**: Run `scripts/auto_label.py` on new frames
3. **Review in Label Studio**: Import predictions, delete false positives, add missed boxes
4. **Iterate**: Export cleaned data and retrain for v2, v3, etc.
5. **Monitor performance**: Check training curves and mAP metrics in `models/checkpoints/yolo11n_v<N>/`
