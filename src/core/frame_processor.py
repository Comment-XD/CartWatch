import cv2
import numpy as np
from pathlib import Path


class FrameProcessor:
    """Handle frame I/O and format conversions."""

    @staticmethod
    def load_frame(path: str) -> np.ndarray:
        """Load a frame from disk (returns BGR)."""
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Failed to load frame from {path}")
        return frame

    @staticmethod
    def save_frame(frame: np.ndarray, path: str) -> None:
        """Save a frame to disk (expects BGR)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), frame)

    @staticmethod
    def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
        """Convert BGR to RGB."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    @staticmethod
    def rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
        """Convert RGB to BGR."""
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    @staticmethod
    def normalize(frame: np.ndarray) -> np.ndarray:
        """Normalize frame to [0, 1] range."""
        return frame.astype(np.float32) / 255.0

    @staticmethod
    def denormalize(frame: np.ndarray) -> np.ndarray:
        """Denormalize frame from [0, 1] to [0, 255]."""
        return np.clip(frame * 255, 0, 255).astype(np.uint8)

    @staticmethod
    def resize(frame: np.ndarray, target_size: tuple) -> np.ndarray:
        """Resize frame to target size (width, height)."""
        return cv2.resize(frame, target_size)

    @staticmethod
    def get_frame_info(frame: np.ndarray) -> dict:
        """Get frame metadata."""
        height, width = frame.shape[:2]
        channels = frame.shape[2] if len(frame.shape) == 3 else 1
        return {
            "height": height,
            "width": width,
            "channels": channels,
            "dtype": str(frame.dtype),
        }
