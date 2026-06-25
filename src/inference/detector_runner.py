import time
from typing import List, Optional
import numpy as np
from src.config import Config
from src.core.detector import YOLODetector
from src.ingestion.frame_iterator import VideoFrameIterator, ImageDirectoryIterator
from src.preprocessing.normalizer import FrameNormalizer
from src.inference.result_formatter import DetectionResult, DetectionResultFormatter
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DetectionPipeline:
    """Orchestrate end-to-end detection pipeline."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize detection pipeline.

        Args:
            config: Configuration object (uses defaults if None)
        """
        self.config: Config = config if config is not None else Config.default()
        self.detector = YOLODetector(
            model_name=self.config.detector.model_name,
            confidence_threshold=self.config.detector.confidence_threshold,
            device=self.config.detector.device,
        )
        self.normalizer = FrameNormalizer()
        logger.info("DetectionPipeline initialized")

    def process_frame(self, frame: np.ndarray, frame_id: str = "0") -> DetectionResult:
        """Process single frame.

        Args:
            frame: Input frame (BGR)
            frame_id: Frame identifier

        Returns:
            DetectionResult object
        """
        # Normalize frame dimensions
        normalized_frame = self.normalizer.apply_preprocessing(
            frame, self.config.preprocessing.target_frame_size
        )

        # Run detection
        start_time = time.time()
        detections = self.detector.detect(normalized_frame)
        inference_time_ms = (time.time() - start_time) * 1000

        # Format result
        result = DetectionResultFormatter.format(
            frame_id=frame_id,
            detections=detections,
            frame_shape=frame.shape,
            model_name=self.config.detector.model_name,
            inference_time_ms=inference_time_ms,
        )

        return result

    def process_video(self, video_path: str) -> List[DetectionResult]:
        """Process entire video.

        Args:
            video_path: Path to video file

        Returns:
            List of DetectionResult objects
        """
        logger.info(f"Processing video: {video_path}")
        iterator = VideoFrameIterator(video_path)
        results = []

        for frame, metadata in iterator:
            result = self.process_frame(frame, frame_id=str(metadata["frame_id"]))
            result.timestamp = metadata.get("timestamp")
            results.append(result)

            if len(results) % 10 == 0:
                logger.info(f"Processed {len(results)} frames")

        logger.info(f"Completed video processing: {len(results)} frames")
        return results

    def process_images(self, image_dir: str) -> List[DetectionResult]:
        """Process all images in directory.

        Args:
            image_dir: Path to image directory

        Returns:
            List of DetectionResult objects
        """
        logger.info(f"Processing images from: {image_dir}")
        iterator = ImageDirectoryIterator(image_dir)
        results = []

        for frame, metadata in iterator:
            result = self.process_frame(frame, frame_id=str(metadata["frame_id"]))
            results.append(result)

            if len(results) % 10 == 0:
                logger.info(f"Processed {len(results)} images")

        logger.info(f"Completed image processing: {len(results)} images")
        return results
