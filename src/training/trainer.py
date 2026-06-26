"""Faster R-CNN model trainer for active learning pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import cv2

from src.config import Config
from src.training.checkpoint_manager import allocate_version_dir
from src.training.dataset_builder import (
    discover_class_names,
    split_dataset,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingRunResult:
    """Summary of a completed training run."""

    version: int
    version_dir: Path
    best_pth: Path
    class_names: List[str]
    num_train_images: int
    num_val_images: int


class FasterRCNNDataset(Dataset):
    """PyTorch Dataset for Faster R-CNN training."""

    def __init__(self, image_paths: List[Path], label_paths: List[Path]):
        """Initialize dataset.

        Args:
            image_paths: List of image file paths
            label_paths: List of corresponding label file paths
        """
        self.image_paths = image_paths
        self.label_paths = label_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        # Load image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        h, w = image.shape[1:]

        # Load labels (YOLO format: class_id center_x center_y width height)
        boxes = []
        labels = []

        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])

                        # Convert to corner format
                        x1 = (cx - bw / 2) * w
                        y1 = (cy - bh / 2) * h
                        x2 = (cx + bw / 2) * w
                        y2 = (cy + bh / 2) * h

                        boxes.append([x1, y1, x2, y2])
                        labels.append(class_id + 1)  # +1 for background class

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
        }

        return image, target


class FasterRCNNTrainer:
    """Train Faster R-CNN models on Label Studio YOLO-format exports."""

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
        """Train Faster R-CNN on a Label Studio YOLO export, auto-splitting train/val.

        Args:
            export_dir: Path to Label Studio YOLO export root
                        (contains images/, labels/, classes.txt)
            model_name: Model identifier (defaults to fasterrcnn_resnet50)

        Returns:
            TrainingRunResult with version number, checkpoint paths, metrics
        """
        export_path = Path(export_dir)
        model_name_to_use = model_name or "fasterrcnn_resnet50"

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

        # Get corresponding label paths
        def get_label_path(img_path):
            label_path = Path(str(img_path).replace('/images/', '/labels/'))
            label_path = label_path.with_suffix('.txt')
            return label_path

        train_labels = [get_label_path(p) for p in train_imgs]
        val_labels = [get_label_path(p) for p in val_imgs]

        logger.info(
            f"Starting training: {model_name_to_use} for {self.config.training.epochs} epochs, "
            f"batch_size={self.config.training.batch_size}"
        )

        # Create datasets
        train_dataset = FasterRCNNDataset(train_imgs, train_labels)
        val_dataset = FasterRCNNDataset(val_imgs, val_labels)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=lambda x: x,
        )

        # Create model
        num_classes = len(class_names) + 1
        model = fasterrcnn_resnet50_fpn(num_classes=num_classes)
        model = model.to(self.config.training.device)

        # Optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params,
            lr=0.005,
            momentum=0.9,
            weight_decay=0.0005,
        )

        # Training loop
        best_loss = float('inf')
        for epoch in range(self.config.training.epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = 0

            for images, targets in train_loader:
                images = [img.to(self.config.training.device) for img in images]
                targets = [
                    {k: v.to(self.config.training.device) for k, v in t.items()}
                    for t in targets
                ]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                epoch_loss += losses.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches
            logger.info(f"Epoch {epoch + 1}/{self.config.training.epochs}, Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_pth = version_dir / "best.pth"
                torch.save(model.state_dict(), best_pth)
                logger.info(f"Saved best checkpoint: {best_pth}")

        # Save final checkpoint
        final_pth = version_dir / "final.pth"
        torch.save(model.state_dict(), final_pth)

        logger.info(f"Training complete!")
        logger.info(f"  Version: {model_name_to_use}_v{version}")
        logger.info(f"  Train images: {len(train_imgs)}")
        logger.info(f"  Val images: {len(val_imgs)}")
        logger.info(f"  Classes: {class_names}")
        logger.info(f"  Best checkpoint: {best_pth}")

        return TrainingRunResult(
            version=version,
            version_dir=version_dir,
            best_pth=best_pth,
            class_names=class_names,
            num_train_images=len(train_imgs),
            num_val_images=len(val_imgs),
        )
