#!/usr/bin/env python3
"""Simple script to run detection on images or video."""

import argparse
import sys
from pathlib import Path

from src.config import Config
from src.inference.detector_runner import DetectionPipeline
from src.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run detection on images or video")
    parser.add_argument("input", help="Path to video file or image directory")
    parser.add_argument("--type", choices=["video", "images"], required=True, help="Input type")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--model", default="yolo11n", help="YOLO model variant")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device to use")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    setup_logging(level="DEBUG" if args.verbose else "INFO")

    # Verify input exists
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path not found: {args.input}")
        sys.exit(1)

    # Create config
    config = Config.default()
    config.detector.confidence_threshold = args.confidence
    config.detector.model_name = args.model
    config.detector.device = args.device

    # Create pipeline
    pipeline = DetectionPipeline(config)

    # Run detection
    if args.type == "video":
        results = pipeline.process_video(str(input_path))
    else:
        results = pipeline.process_images(str(input_path))

    # Print summary
    logger.info(f"Processed {len(results)} frames/images")
    if results:
        first_result = results[0]
        logger.info(f"First result: {len(first_result.detections)} detections")
        for detection in first_result.detections[:5]:
            logger.info(
                f"  - {detection.class_name} "
                f"(confidence: {detection.confidence:.2f})"
            )


if __name__ == "__main__":
    main()
