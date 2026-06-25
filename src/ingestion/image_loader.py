import cv2
from pathlib import Path
from typing import List
from src.utils.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


class ImageLoader:
    """Load and process image files."""

    @staticmethod
    def load(image_path: str):
        """Load single image.

        Args:
            image_path: Path to image file

        Returns:
            Image as numpy array (BGR)
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Failed to load image: {image_path}")

        return frame

    @staticmethod
    def load_batch(image_paths: List[str]) -> List:
        """Load multiple images.

        Args:
            image_paths: List of image file paths

        Returns:
            List of images
        """
        return [ImageLoader.load(path) for path in image_paths]

    @staticmethod
    def list_images(directory: str) -> List[str]:
        """List all images in directory.

        Args:
            directory: Directory path

        Returns:
            List of image file paths
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise ValueError(f"Not a directory: {directory}")

        images = []
        for ext in SUPPORTED_FORMATS:
            images.extend(sorted(dir_path.glob(f"*{ext}")) + sorted(dir_path.glob(f"*{ext.upper()}")))

        # Remove duplicates and sort
        images = sorted(set(images))
        logger.info(f"Found {len(images)} images in {directory}")
        return [str(img) for img in images]
