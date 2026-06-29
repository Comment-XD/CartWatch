#!/usr/bin/env python3
"""Train Faster R-CNN model on Label Studio YOLO export with MLflow + Azure ML tracking."""

import argparse
import sys
import os
from pathlib import Path
import mlflow

from src.config import Config
from src.training.trainer import FasterRCNNTrainer
from src.training.azure_ml_config import setup_azure_ml_mlflow
from src.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on a Label Studio YOLO export with MLflow + Azure ML")
    parser.add_argument("export_dir", help="Path to Label Studio YOLO export (images/, labels/, classes.txt)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split")
    parser.add_argument("--experiment-name", default="cartwatch-training", help="MLflow experiment name")
    parser.add_argument("--azure-workspace", default=None, help="Azure ML workspace name (optional)")
    parser.add_argument("--azure-resource-group", default=None, help="Azure resource group name (optional)")
    parser.add_argument("--azure-subscription", default=None, help="Azure subscription ID (optional)")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow tracking")
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

    # Setup MLflow with Azure ML if workspace credentials provided
    enable_mlflow = not args.no_mlflow
    if enable_mlflow and args.azure_workspace:
        try:
            setup_azure_ml_mlflow(
                workspace_name=args.azure_workspace,
                resource_group=args.azure_resource_group,
                subscription_id=args.azure_subscription,
                experiment_name=args.experiment_name,
            )
            logger.info(f"Connected to Azure ML workspace: {args.azure_workspace}")
        except Exception as e:
            logger.warning(f"Failed to connect to Azure ML: {e}. Proceeding without MLflow.")
            enable_mlflow = False
    elif enable_mlflow:
        # Setup local MLflow
        mlflow.set_experiment(args.experiment_name)
        logger.info(f"Using local MLflow with experiment: {args.experiment_name}")

    # Start MLflow run
    if enable_mlflow:
        mlflow.start_run()

    trainer = FasterRCNNTrainer(config)

    try:
        result = trainer.train(
            str(export_path),
            model_name="fasterrcnn_resnet50",
            enable_mlflow=enable_mlflow,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Training failed: {e}")
        if enable_mlflow:
            mlflow.end_run()
        sys.exit(1)

    if enable_mlflow:
        mlflow.end_run()

    logger.info(f"Training complete: fasterrcnn_resnet50_v{result.version}")
    logger.info(f"  Train images: {result.num_train_images}, Val images: {result.num_val_images}")
    logger.info(f"  Classes: {result.class_names}")
    logger.info(f"  Best checkpoint: {result.best_pth}")
    if enable_mlflow:
        logger.info(f"  MLflow experiment: {args.experiment_name}")


if __name__ == "__main__":
    main()
