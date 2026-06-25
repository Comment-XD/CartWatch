#!/usr/bin/env python3
"""Run a trained YOLO checkpoint over unlabeled frames and produce Label Studio import JSON."""

import argparse
import sys
from pathlib import Path

from src.config import Config
from src.inference.auto_label import AutoLabeler
from src.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def resolve_checkpoint(model_name: str, version: int, model_dir: Path) -> Path:
    """Resolve a (model_name, version) pair to a best.pt path.

    Args:
        model_name: e.g. "yolo11n"
        version: Checkpoint version number
        model_dir: models/ directory (config.paths.model_dir)

    Returns:
        Path to models/checkpoints/<model_name>_v<version>/weights/best.pt

    Raises:
        FileNotFoundError: if the resolved best.pt doesn't exist
    """
    checkpoint_path = model_dir / "checkpoints" / f"{model_name}_v{version}" / "weights" / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(description="Auto-label frames for Label Studio review")
    parser.add_argument("input", help="Directory of unlabeled image frames")
    parser.add_argument("--output", default="data/labeled/predictions.json", help="Output JSON path")
    parser.add_argument("--model", default="yolo11n", help="Model name (matches checkpoint folder prefix)")
    parser.add_argument("--version", type=int, default=None, help="Checkpoint version to use, e.g. 2 for yolo11n_v2")
    parser.add_argument("--checkpoint", default=None, help="Explicit .pt path (overrides --model/--version)")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device to use")
    parser.add_argument("--from-name", default="label", help="LS RectangleLabels tag name")
    parser.add_argument("--to-name", default="image", help="LS Image tag name")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()
    setup_logging(level="DEBUG" if args.verbose else "INFO")

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input directory not found: {args.input}")
        sys.exit(1)

    config = Config.default()

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
    else:
        if args.version is None:
            logger.error("--version is required when --checkpoint is not specified")
            sys.exit(1)
        try:
            checkpoint_path = resolve_checkpoint(args.model, args.version, config.paths.model_dir)
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)

    logger.info(f"Using checkpoint: {checkpoint_path}")

    labeler = AutoLabeler(
        checkpoint_path=str(checkpoint_path),
        confidence_threshold=args.confidence,
        device=args.device,
        from_name=args.from_name,
        to_name=args.to_name,
    )

    output_path = labeler.label_and_save(str(input_path), args.output)
    logger.info(f"Predictions written to {output_path}")
    logger.info("Import this file in Label Studio via Import > Upload Files")


if __name__ == "__main__":
    main()
