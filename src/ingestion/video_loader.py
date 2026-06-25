import cv2
from pathlib import Path
from src.utils.logging import get_logger

logger = get_logger(__name__)


class VideoLoader:
    """Load and process video files."""

    def __init__(self, video_path: str):
        """Initialize video loader.

        Args:
            video_path: Path to video file
        """
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        logger.info(f"Loaded video: {self.video_path}")

    def get_frame_count(self) -> int:
        """Get total number of frames."""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_fps(self) -> float:
        """Get frames per second."""
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_frame_size(self) -> tuple:
        """Get frame dimensions (width, height)."""
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    def get_metadata(self) -> dict:
        """Get video metadata."""
        return {
            "path": str(self.video_path),
            "frame_count": self.get_frame_count(),
            "fps": self.get_fps(),
            "frame_size": self.get_frame_size(),
            "duration_seconds": self.get_frame_count() / self.get_fps(),
        }

    def read_frame(self) -> tuple:
        """Read next frame.

        Returns:
            (success, frame) tuple
        """
        return self.cap.read()

    def seek(self, frame_number: int) -> None:
        """Seek to specific frame."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def release(self) -> None:
        """Release video resource."""
        self.cap.release()

    def __del__(self):
        """Cleanup on deletion."""
        self.release()
