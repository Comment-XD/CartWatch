"""YOLO model trainer for active learning pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from ultralytics import YOLO

from src.config import Config
from src.training.checkpoint_manager import allocate_version_dir
from src.training.dataset_builder import (
    discover_class_names,
    split_dataset,
    write_split_files,
    generate_dataset_yaml,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingRunResult:
    """Summary of a completed training run."""

    version: int
    version_dir: Path
    best_pt: Path
    last_pt: Path
    class_names: List[str]
    num_train_images: int
    num_val_images: int


class YOLOTrainer:
    """Train YOLO models on Label Studio YOLO-format exports."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize trainer.

        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config = config if config is not None else Config.default()

    def train(
        self,
        export_dir: str,
        model_name: Optional[str] = None,
    ) -> TrainingRunResult:
        """Train YOLO on a flat Label Studio YOLO export, auto-splitting train/val.

        Args:
            export_dir: Path to Label Studio YOLO export root
                        (contains images/, labels/, classes.txt)
            model_name: Base model to fine-tune from, e.g. "yolo11n"
                        (defaults to self.config.detector.model_name)

        Returns:
            TrainingRunResult with version number, checkpoint paths, metrics

        Raises:
            FileNotFoundError: export_dir missing images/labels/classes.txt
            ValueError: export_dir has zero usable images
        """
        export_path = Path(export_dir)
        model_name_to_use = model_name or self.config.detector.model_name

        logger.info(f"Training {model_name_to_use} on {export_path}")

        checkpoints_dir = self.config.paths.model_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        version_dir, version = allocate_version_dir(checkpoints_dir, model_name_to_use)
        logger.info(f"Allocated checkpoint directory: {version_dir} (v{version})")

        class_names = discover_class_names(export_path)
        train_imgs, val_imgs = split_dataset(
            export_path,
            val_split=self.config.training.val_split,
            seed=self.config.training.seed,
        )

        version_dir.mkdir(parents=True, exist_ok=True)

        train_txt, val_txt = write_split_files(train_imgs, val_imgs, version_dir)
        dataset_yaml = generate_dataset_yaml(export_path, train_txt, val_txt, class_names, version_dir / "dataset.yaml")

        logger.info(
            f"Starting training: {model_name_to_use} for {self.config.training.epochs} epochs, "
            f"batch_size={self.config.training.batch_size}, imgsz={self.config.training.imgsz}"
        )

        model = YOLO(f"{model_name_to_use}.pt")
        model.train(
            data=str(dataset_yaml),
            project=str(checkpoints_dir),
            name=f"{model_name_to_use}_v{version}",
            epochs=self.config.training.epochs,
            imgsz=self.config.training.imgsz,
            batch=self.config.training.batch_size,
            patience=self.config.training.patience,
            device=self.config.training.device,
            exist_ok=False,
        )

        best_pt = version_dir / "weights" / "best.pt"
        last_pt = version_dir / "weights" / "last.pt"

        logger.info(f"Training complete!")
        logger.info(f"  Version: {model_name_to_use}_v{version}")
        logger.info(f"  Train images: {len(train_imgs)}")
        logger.info(f"  Val images: {len(val_imgs)}")
        logger.info(f"  Classes: {class_names}")
        logger.info(f"  Best checkpoint: {best_pt}")
        logger.info(f"  Last checkpoint: {last_pt}")

        return TrainingRunResult(
            version=version,
            version_dir=version_dir,
            best_pt=best_pt,
            last_pt=last_pt,
            class_names=class_names,
            num_train_images=len(train_imgs),
            num_val_images=len(val_imgs),
        )
