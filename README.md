# CartWatch

An inventory detection system using computer vision to monitor items being removed from a shelf. CartWatch detects inventory objects and estimates the quantity of items picked up during interactions.

## Overview

CartWatch is built with an **active learning pipeline** that:
1. **Extracts frames** from video recordings using OpenCV
2. **Trains YOLO object detection models** on manually labeled data
3. **Auto-labels** new frames with trained models for human review
4. **Iteratively improves** through human feedback and retraining

## Technology Stack

- **Python 3.12+**
- **OpenCV** — Video processing and frame extraction
- **Ultralytics YOLO11** — Object detection and model training
- **Supervision** — Bounding box visualization
- **SSIM** — Duplicate frame detection
- **Label Studio** — Data annotation and review

## Project Structure

```
CartWatch/
├── data/
│   ├── raw_videos/              # Source video files
│   ├── extracted_frames/        # Extracted video frames
│   ├── labeled/                 # Auto-labeled predictions
│   └── deduplicated_frames/     # SSIM-deduplicated frames
├── models/
│   ├── checkpoints/             # Trained YOLO models (versioned)
│   │   ├── yolo11n_v1/weights/best.pt
│   │   ├── yolo11n_v2/weights/best.pt
│   │   └── ...
│   └── exports/
├── src/
│   ├── ingestion/               # Video & frame loading
│   ├── preprocessing/           # Deduplication, normalization
│   ├── training/                # YOLO training pipeline
│   ├── inference/               # Detection & auto-labeling
│   ├── core/                    # Core detector class
│   └── utils/                   # Logging, config
├── scripts/
│   ├── extract_frames.py        # Extract frames from video
│   ├── train.py                 # Train YOLO models
│   └── auto_label.py            # Generate predictions for review
├── tests/                       # Unit tests
├── notebooks/                   # Jupyter notebooks
├── requirements.txt
├── CLAUDE.md                    # Project guidelines
├── QUICK_START_TRAINING.md      # Training quickstart
└── TRAINING_AND_AUTOLABEL.md   # Detailed pipeline docs
```

## Quick Start

### Installation

```bash
conda create -n cartwatch python=3.12
conda activate cartwatch
pip install -r requirements.txt
```

### Extract Frames from Video

```bash
python scripts/extract_frames.py video.mp4 \
  --output data/extracted_frames \
  --skip-frames 30 \
  --deduplicate
```

### Train a YOLO Model

First, manually label ~100-300 frames in Label Studio and export as YOLO format:

```bash
python scripts/train.py manual_labels_export \
  --epochs 50 \
  --device cpu
```

Creates: `models/checkpoints/yolo11n_v1/weights/best.pt`

### Auto-Label New Frames

```bash
python scripts/auto_label.py data/extracted_frames \
  --model yolo11n \
  --version 1 \
  --output data/labeled/predictions.json
```

Import `predictions.json` into Label Studio for review.

## Active Learning Workflow

1. **Manual Labeling** → Label ~100-300 frames in Label Studio
2. **Train v1** → `python scripts/train.py manual_labels_export`
3. **Auto-Label** → `python scripts/auto_label.py data/extracted_frames --version 1`
4. **Review** → Import predictions into Label Studio, correct errors
5. **Train v2** → Retrain on corrected labels
6. **Repeat** → Continue improving with v3, v4, etc.

## Model Versioning

Models are automatically versioned:
- `yolo11n_v1/` — First trained model
- `yolo11n_v2/` — Retrained on reviewed data
- `yolo11n_v3/` — And so on...

Each version is isolated and includes:
- `weights/best.pt` — Best checkpoint
- `weights/last.pt` — Last checkpoint
- `dataset.yaml` — Dataset configuration
- `results.csv` — Training metrics
- `plots/` — Training curves

## Testing

```bash
pytest tests/ -v
```

## Documentation

- **[QUICK_START_TRAINING.md](QUICK_START_TRAINING.md)** — Training commands and troubleshooting
- **[TRAINING_AND_AUTOLABEL.md](TRAINING_AND_AUTOLABEL.md)** — Detailed pipeline API reference
- **[CLAUDE.md](CLAUDE.md)** — Project philosophy and development guidelines

## Requirements

See `requirements.txt` for full dependencies.

Key packages:
- opencv-python >= 4.8.0
- ultralytics >= 8.0.0
- supervision >= 0.16.0
- scikit-image >= 0.21.0
- pyyaml >= 6.0

## License

MIT

## Authors

Built with Claude Code by Anthropic
