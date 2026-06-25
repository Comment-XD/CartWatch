import numpy as np
from skimage.metrics import structural_similarity as ssim
from typing import Generator, Tuple, Dict, Any
from src.ingestion.frame_iterator import FrameIterator
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SSIMDeduplicator:
    """Remove duplicate frames using Structural Similarity Index."""

    def __init__(self, threshold: float = 0.95):
        """Initialize deduplicator.

        Args:
            threshold: SSIM threshold for considering frames identical (0-1)
                      Higher = stricter (more frames kept)
        """
        self.threshold = threshold

    def compute_ssim(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Compute SSIM between two frames.

        Args:
            frame1: First frame (BGR)
            frame2: Second frame (BGR)

        Returns:
            SSIM score (0-1)
        """
        # Convert to grayscale for SSIM
        gray1 = np.mean(frame1, axis=2) if len(frame1.shape) == 3 else frame1
        gray2 = np.mean(frame2, axis=2) if len(frame2.shape) == 3 else frame2

        # Compute SSIM
        try:
            score = ssim(gray1, gray2, data_range=255)
        except Exception as e:
            logger.warning(f"SSIM computation failed: {e}")
            return 0.0

        return float(score)

    def is_duplicate(self, frame1: np.ndarray, frame2: np.ndarray) -> bool:
        """Check if two frames are duplicates.

        Args:
            frame1: First frame
            frame2: Second frame

        Returns:
            True if SSIM > threshold
        """
        score = self.compute_ssim(frame1, frame2)
        return score > self.threshold

    def filter_similar_sequence(
        self,
        frame_iterator: FrameIterator,
        threshold: float = None,
    ) -> Generator[Tuple[np.ndarray, Dict[str, Any]], None, None]:
        """Filter consecutive duplicate frames.

        Args:
            frame_iterator: FrameIterator instance
            threshold: Optional override of default threshold

        Yields:
            (frame, metadata) tuples for unique frames
        """
        if threshold is None:
            threshold = self.threshold

        prev_frame = None
        unique_count = 0
        total_count = 0

        for frame, metadata in frame_iterator:
            total_count += 1
            frame_id = metadata.get("frame_id", total_count)

            if prev_frame is None:
                # Always keep first frame
                prev_frame = frame
                unique_count += 1
                logger.info(f"Frame {frame_id}: KEPT (first frame)")
                yield frame, metadata
            else:
                # Check similarity with previous unique frame
                score = self.compute_ssim(frame, prev_frame)
                if score > threshold:
                    logger.info(f"Frame {frame_id}: DUPLICATE (SSIM={score:.4f} > {threshold})")
                else:
                    logger.info(f"Frame {frame_id}: KEPT (SSIM={score:.4f} <= {threshold})")
                    prev_frame = frame
                    unique_count += 1
                    yield frame, metadata

        logger.info(f"Deduplication complete: Kept {unique_count}/{total_count} frames (threshold={threshold})")
