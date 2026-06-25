# CLAUDE.md

## Project Overview

This project is an inventory detection system that uses computer vision to monitor items being removed from a shelf. A camera observes the shelf area, detects inventory items, and estimates the quantity of items removed during an interaction.

The initial goal is to build a proof-of-concept capable of detecting inventory objects and counting the number of items picked up by a user. Future iterations may expand into object tracking, multi-camera setups, and real-time inventory reconciliation.

---

## Primary Objectives

### Phase 1: Dataset Collection

Build a high-quality dataset containing images and videos of inventory items under various conditions.

Requirements:

* Record 30–60 minute videos of shelf interactions.
* Capture different lighting conditions.
* Capture different item arrangements.
* Primarily use a top-down camera angle.
* Extract frames every 3 seconds.
* Remove duplicate or near-duplicate images using SSIM.

* Label inventory objects using bounding boxes.

### Phase 2: Model Training

Train an object detection model capable of identifying inventory items on shelves.

Requirements:

* Train an initial YOLO model using approximately 500 manually labeled images.
* Use the trained model for assisted labeling of future datasets.
* Review and correct:

  * False positives
  * False negatives
  * Incorrect bounding boxes
* Continuously retrain and improve model performance through active learning.

### Phase 3: Inventory Counting

Detect when a user removes items from a shelf and estimate quantity changes.

Requirements:

* Detect inventory items before interaction.
* Detect inventory items after interaction.
* Compare counts.
* Calculate inventory delta.

Example:

Before:

* Water Bottle: 10

After:

* Water Bottle: 7

Result:

* 3 items removed

---

## Technology Stack

### Language

* Python 3.12+

### Core Libraries

#### OpenCV

Used for:

* Video processing
* Frame extraction
* Image manipulation
* Camera integration

#### Ultralytics YOLO

Used for:

* Object detection
* Model training
* Model evaluation
* Inference

Expected model family:

* YOLO11

#### Supervision

Used for:

* Visualization
* Bounding box rendering
* Dataset inspection
* Evaluation utilities
* Future Multi-Object Tracking (MOT)

#### SSIM (Structural Similarity Index)

Used for:

* Duplicate image removal
* Near-duplicate frame detection

Initial proof-of-concept will use SSIM before introducing embedding-based approaches.

#### CLIP (Future)

Future dataset pipeline improvements may include:

* Semantic image similarity
* Image clustering
* Dataset deduplication
* Dataset quality analysis

CLIP is not required for the initial version.

---

## Dataset Pipeline

### Data Collection

Video Recording
→ Frame Extraction
→ SSIM Deduplication
→ Manual Review
→ Annotation
→ Training Dataset

### Active Learning Pipeline

Manual Labels (500 Images)
→ Train YOLO v1
→ Auto Label New Images
→ Human Verification
→ Retrain YOLO
→ Repeat

---

## Repository Structure

project/

├── data/
│   ├── raw_videos/
│   ├── extracted_frames/
│   ├── deduplicated_frames/
│   └── labeled/
│
├── models/
│   ├── checkpoints/
│   └── exports/
│
├── src/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── training/
│   ├── inference/
│   └── tracking/
│
├── notebooks/
│
├── tests/
│
├── requirements.txt
│
└── CLAUDE.md

---

## Future Roadmap

### Version 2

* CLIP-based similarity filtering
* Automatic image clustering
* Improved auto-labeling workflows

### Version 3

* Multi-Object Tracking (MOT)
* Persistent item tracking
* Shelf interaction events
* Real-time inventory updates

### Version 4

* Multi-camera support
* Edge deployment
* Inventory analytics dashboard
* Automated inventory reconciliation

---

## Development Philosophy

Prioritize data quality over model complexity.

The primary focus should be:

1. Diverse data collection
2. Clean annotations
3. Continuous active learning
4. Reliable counting accuracy

A smaller, high-quality dataset is preferred over a large, noisy dataset.
