import cv2
import numpy as np
from typing import Tuple
from src.utils.logging import get_logger

logger = get_logger(__name__)


class FrameNormalizer:
    """Normalize frames for consistent processing."""

    @staticmethod
    def normalize_dimensions(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize frame to target dimensions.

        Args:
            frame: Input frame (H x W x C)
            target_size: Target size (width, height)

        Returns:
            Resized frame
        """
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def normalize_color(frame: np.ndarray) -> np.ndarray:
        """Ensure frame is in BGR format.

        Args:
            frame: Input frame

        Returns:
            Frame in BGR format
        """
        if len(frame.shape) != 3 or frame.shape[2] not in (3, 4):
            logger.warning("Frame color normalization: unexpected shape")
        return frame

    @staticmethod
    def apply_preprocessing(frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Apply full preprocessing pipeline.

        Args:
            frame: Input frame
            target_size: Target dimensions

        Returns:
            Preprocessed frame
        """
        # Ensure color format
        frame = FrameNormalizer.normalize_color(frame)

        # Normalize dimensions
        frame = FrameNormalizer.normalize_dimensions(frame, target_size)

        return frame
