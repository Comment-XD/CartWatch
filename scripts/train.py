#!/usr/bin/env python3
"""Train a YOLO model on a Label Studio YOLO-format export."""

import argparse
import sys
from pathlib import Path

from src.config import Config
from src.training.trainer import YOLOTrainer
from src.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train YOLO on a Label Studio YOLO export")
    parser.add_argument("export_dir", help="Path to Label Studio YOLO export (images/, labels/, classes.txt)")
    parser.add_argument("--model", default="yolo11n", help="Base YOLO model to fine-tune from")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")
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
    config.training.imgsz = args.imgsz
    config.training.val_split = args.val_split
    config.training.patience = args.patience
    config.training.device = args.device
    config.training.seed = args.seed

    trainer = YOLOTrainer(config)

    try:
        result = trainer.train(str(export_path), model_name=args.model)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)

    logger.info(f"Training complete: {args.model}_v{result.version}")
    logger.info(f"  Train images: {result.num_train_images}, Val images: {result.num_val_images}")
    logger.info(f"  Classes: {result.class_names}")
    logger.info(f"  Best checkpoint: {result.best_pt}")
    logger.info(f"  Last checkpoint: {result.last_pt}")


if __name__ == "__main__":
    main()
