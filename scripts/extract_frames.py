#!/usr/bin/env python3
"""Extract frames from video with optional deduplication."""

import argparse
import sys
from pathlib import Path

from src.ingestion.frame_iterator import VideoFrameIterator
from src.preprocessing.deduplicator import SSIMDeduplicator
from src.core.frame_processor import FrameProcessor
from src.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--output", default="data/extracted_frames", help="Output directory")
    parser.add_argument("--skip-frames", type=int, default=1, help="Extract every Nth frame")
    parser.add_argument(
        "--deduplicate", action="store_true", help="Remove duplicate frames (SSIM-based)"
    )
    parser.add_argument("--ssim-threshold", type=float, default=0.95, help="SSIM threshold")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    setup_logging(level="DEBUG" if args.verbose else "INFO")

    # Verify input
    video_path = Path(args.video)
    if not video_path.exists():
        logger.error(f"Video not found: {args.video}")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create iterator
    iterator = VideoFrameIterator(str(video_path), skip_frames=args.skip_frames)

    # Optional deduplication
    if args.deduplicate:
        logger.info(f"Deduplicating with SSIM threshold: {args.ssim_threshold}")
        deduplicator = SSIMDeduplicator(threshold=args.ssim_threshold)
        iterator = deduplicator.filter_similar_sequence(iterator, args.ssim_threshold)

    # Extract frames
    frame_count = 0
    for frame, metadata in iterator:
        frame_id = metadata["frame_id"]
        output_path = output_dir / f"frame_{frame_id:06d}.jpg"

        try:
            FrameProcessor.save_frame(frame, str(output_path))
            frame_count += 1

            if frame_count % 50 == 0:
                logger.info(f"Extracted {frame_count} frames...")
        except Exception as e:
            logger.error(f"Failed to save frame {frame_id}: {e}")

    logger.info(f"Extraction complete: {frame_count} frames saved to {output_dir}")


if __name__ == "__main__":
    main()
