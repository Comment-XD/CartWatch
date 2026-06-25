import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.core.detector import Detection


@pytest.fixture
def sample_detections():
    """Create sample detections for testing."""
    return [
        Detection(
            class_id=0,
            class_name="water_bottle",
            confidence=0.95,
            bbox=(0.1, 0.2, 0.4, 0.6),
        ),
        Detection(
            class_id=0,
            class_name="water_bottle",
            confidence=0.87,
            bbox=(0.5, 0.3, 0.8, 0.7),
        ),
        Detection(
            class_id=1,
            class_name="book",
            confidence=0.92,
            bbox=(0.2, 0.5, 0.5, 0.9),
        ),
    ]


@pytest.fixture
def sample_frame():
    """Create a sample frame (BGR numpy array)."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_frames():
    """Create multiple sample frames."""
    return [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(5)]


@pytest.fixture
def tmp_label_studio_export(tmp_path):
    """Build a minimal Label Studio YOLO export on disk.

    Creates:
    - images/ with 10 dummy .jpg files
    - labels/ with matching .txt files (YOLO format)
    - classes.txt with 2 class names

    Returns:
        Path to the export root directory
    """
    export_root = tmp_path / "label_studio_export"
    images_dir = export_root / "images"
    labels_dir = export_root / "labels"

    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    class_names = ["water_bottle", "book"]
    (export_root / "classes.txt").write_text("\n".join(class_names) + "\n")

    for i in range(10):
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        img_path = images_dir / f"image_{i:02d}.jpg"
        cv2.imwrite(str(img_path), img)

        label_path = labels_dir / f"image_{i:02d}.txt"
        lines = []
        for _ in range(np.random.randint(1, 3)):
            class_id = np.random.randint(0, len(class_names))
            cx = np.random.uniform(0.1, 0.9)
            cy = np.random.uniform(0.1, 0.9)
            w = np.random.uniform(0.1, 0.3)
            h = np.random.uniform(0.1, 0.3)
            lines.append(f"{class_id} {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}\n")
        label_path.write_text("".join(lines))

    return export_root
