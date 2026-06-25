from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Generator
import cv2
from pathlib import Path
from src.ingestion.video_loader import VideoLoader
from src.ingestion.image_loader import ImageLoader
from src.utils.logging import get_logger

logger = get_logger(__name__)


class FrameIterator(ABC):
    """Abstract frame iterator interface."""

    @abstractmethod
    def __iter__(self):
        """Return iterator."""
        pass

    @abstractmethod
    def __next__(self) -> Tuple[Any, Dict[str, Any]]:
        """Return (frame, metadata) tuple."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset iterator to beginning."""
        pass

    @abstractmethod
    def get_metadata(self) -> dict:
        """Get source metadata."""
        pass


class VideoFrameIterator(FrameIterator):
    """Iterate over video frames."""

    def __init__(self, video_path: str, skip_frames: int = 1):
        """Initialize video frame iterator.

        Args:
            video_path: Path to video file
            skip_frames: Extract every Nth frame (1 = every frame, 3 = every 3rd frame)
        """
        self.video_path = video_path
        self.skip_frames = skip_frames
        self.loader = VideoLoader(video_path)
        self.frame_index = 0
        self.metadata = self.loader.get_metadata()

    def __iter__(self):
        """Return iterator."""
        self.reset()
        return self

    def __next__(self) -> Tuple[Any, Dict[str, Any]]:
        """Get next frame."""
        while True:
            success, frame = self.loader.read_frame()

            if not success:
                raise StopIteration

            # Skip frames if needed
            if self.frame_index % self.skip_frames == 0:
                metadata = {
                    "frame_id": self.frame_index,
                    "timestamp": self.frame_index / self.loader.get_fps(),
                    "source": str(self.video_path),
                }
                self.frame_index += 1
                return frame, metadata

            self.frame_index += 1

    def reset(self) -> None:
        """Reset to beginning of video."""
        self.loader.seek(0)
        self.frame_index = 0

    def get_metadata(self) -> dict:
        """Get video metadata."""
        return self.metadata


class ImageDirectoryIterator(FrameIterator):
    """Iterate over images in a directory."""

    def __init__(self, directory: str):
        """Initialize image directory iterator.

        Args:
            directory: Path to directory containing images
        """
        self.directory = directory
        self.image_paths = ImageLoader.list_images(directory)
        self.index = 0

    def __iter__(self):
        """Return iterator."""
        self.reset()
        return self

    def __next__(self) -> Tuple[Any, Dict[str, Any]]:
        """Get next image."""
        if self.index >= len(self.image_paths):
            raise StopIteration

        image_path = self.image_paths[self.index]
        frame = ImageLoader.load(image_path)

        metadata = {
            "frame_id": self.index,
            "path": image_path,
            "source": self.directory,
        }

        self.index += 1
        return frame, metadata

    def reset(self) -> None:
        """Reset to beginning."""
        self.index = 0

    def get_metadata(self) -> dict:
        """Get directory metadata."""
        return {
            "directory": self.directory,
            "image_count": len(self.image_paths),
        }
