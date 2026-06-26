#!/usr/bin/env python3
"""Train a Faster R-CNN model on a Label Studio YOLO-format export."""

import argparse
import sys
from pathlib import Path

from src.config import Config
from src.training.trainer import FasterRCNNTrainer
from src.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on a Label Studio YOLO export")
    parser.add_argument("export_dir", help="Path to Label Studio YOLO export (images/, labels/, classes.txt)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()
    setup_logging(level="DEBUG" if args.verbose else "INFO")

    export_path = Path(args.export_dir)
    if not export_path.exists():
        logger.error(f"Export directory not found: {args.export_dir}")
        sys.exit(1)

    config = Config.default()
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.val_split = args.val_split
    config.training.device = args.device
    config.training.seed = args.seed

    trainer = FasterRCNNTrainer(config)

    try:
        result = trainer.train(str(export_path), model_name="fasterrcnn_resnet50")
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

    logger.info(f"Training complete: fasterrcnn_resnet50_v{result.version}")
    logger.info(f"  Train images: {result.num_train_images}, Val images: {result.num_val_images}")
    logger.info(f"  Classes: {result.class_names}")
    logger.info(f"  Best checkpoint: {result.best_pth}")


if __name__ == "__main__":
    main()
