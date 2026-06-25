"""Dataset building utilities for Label Studio YOLO exports."""

import json
import random
from pathlib import Path
from typing import List, Tuple

import yaml

from src.ingestion.image_loader import ImageLoader
from src.utils.logging import get_logger

logger = get_logger(__name__)


def discover_class_names(export_root: Path) -> List[str]:
    """Discover class names from a Label Studio YOLO export.

    Reads class names in index order from either:
    1. classes.txt (one name per line, line N = class id N)
    2. notes.json with "categories" list sorted by id

    Args:
        export_root: Root of Label Studio YOLO export (contains images/, labels/, classes.txt|notes.json)

    Returns:
        List of class names ordered by class index (list[i] == name for class_id i)

    Raises:
        FileNotFoundError: If neither classes.txt nor notes.json is found
    """
    classes_txt_path = export_root / "classes.txt"
    notes_json_path = export_root / "notes.json"

    if classes_txt_path.exists():
        class_names = classes_txt_path.read_text().strip().split("\n")
        logger.info(f"Discovered {len(class_names)} classes from classes.txt")
        return class_names

    if notes_json_path.exists():
        try:
            notes = json.loads(notes_json_path.read_text())
            if "categories" in notes:
                categories = sorted(notes["categories"], key=lambda x: x.get("id", 0))
                class_names = [cat["name"] for cat in categories]
                logger.info(f"Discovered {len(class_names)} classes from notes.json")
                return class_names
        except (json.JSONDecodeError, KeyError) as e:
            logger.debug(f"Failed to parse notes.json: {e}")

    raise FileNotFoundError(
        f"No classes.txt or notes.json found in {export_root}. "
        "Is this a valid Label Studio YOLO export?"
    )


def split_dataset(
    export_root: Path,
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Path], List[Path]]:
    """Randomly split images in a Label Studio YOLO export into train/val sets.

    Args:
        export_root: Root containing images/ and labels/ subdirectories
        val_split: Fraction of images assigned to validation (0-1)
        seed: Random seed for reproducible splits

    Returns:
        Tuple of (train_image_paths, val_image_paths) — absolute Path lists

    Raises:
        ValueError: If export_root/images has zero images
        FileNotFoundError: If export_root doesn't contain images/ directory
    """
    images_dir = export_root / "images"
    labels_dir = export_root / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        logger.warning(f"Labels directory not found: {labels_dir}. Proceeding with images only.")

    image_paths = ImageLoader.list_images(str(images_dir))
    if not image_paths:
        raise ValueError(f"No images found in {images_dir}")

    logger.info(f"Found {len(image_paths)} images")

    rng = random.Random(seed)
    shuffled = list(image_paths)
    rng.shuffle(shuffled)

    split_idx = int(len(shuffled) * (1 - val_split))
    train_images = [Path(p) for p in shuffled[:split_idx]]
    val_images = [Path(p) for p in shuffled[split_idx:]]

    logger.info(f"Split: {len(train_images)} train, {len(val_images)} val")
    return train_images, val_images


def write_split_files(
    train_images: List[Path],
    val_images: List[Path],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Write train.txt and val.txt image-list files for ultralytics.

    Ultralytics natively supports .txt-based `data:` entries. Each line
    is an absolute path to an image. Labels are resolved by ultralytics
    via a standard path transformation: images/foo.jpg → labels/foo.txt

    Args:
        train_images: Absolute paths to training images
        val_images: Absolute paths to validation images
        output_dir: Directory to write train.txt/val.txt into

    Returns:
        Tuple of (train_txt_path, val_txt_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    train_txt = output_dir / "train.txt"
    val_txt = output_dir / "val.txt"

    train_txt.write_text("\n".join(str(p) for p in train_images) + "\n")
    val_txt.write_text("\n".join(str(p) for p in val_images) + "\n")

    logger.info(f"Wrote {len(train_images)} train paths to {train_txt}")
    logger.info(f"Wrote {len(val_images)} val paths to {val_txt}")

    return train_txt, val_txt


def generate_dataset_yaml(
    export_root: Path,
    train_txt: Path,
    val_txt: Path,
    class_names: List[str],
    output_path: Path,
) -> Path:
    """Write an ultralytics-compatible dataset.yaml.

    Args:
        export_root: Label Studio export root (informational 'path' key)
        train_txt: Path to train.txt image list
        val_txt: Path to val.txt image list
        class_names: Ordered class names (index = class id)
        output_path: Where to write dataset.yaml

    Returns:
        output_path
    """
    yaml_content = {
        "path": str(export_root),
        "train": str(train_txt),
        "val": str(val_txt),
        "nc": len(class_names),
        "names": {i: name for i, name in enumerate(class_names)},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.dump(yaml_content, sort_keys=False))

    logger.info(f"Wrote dataset.yaml with {len(class_names)} classes to {output_path}")
    return output_path
