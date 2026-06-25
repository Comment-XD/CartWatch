from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from ultralytics import YOLO

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Detection:
    """Represents a single detection."""

    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 normalized


class YOLODetector:
    """YOLO11 object detector wrapper."""

    def __init__(self, model_name: str = "yolo11n", confidence_threshold: float = 0.5, device: str = "cpu"):
        """Initialize YOLO detector.

        Args:
            model_name: YOLO model variant (yolo11n, yolo11s, yolo11m, etc.) or full path to .pt file
            confidence_threshold: Detection confidence threshold
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device

        weights_path = model_name if model_name.endswith(".pt") else f"{model_name}.pt"
        logger.info(f"Loading {weights_path} on {device}...")
        self.model = YOLO(weights_path)
        self.model.to(device)
        logger.info("Model loaded successfully")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a single frame.

        Args:
            frame: Input frame (BGR or RGB, H x W x 3)

        Returns:
            List of Detection objects
        """
        results = self.model.predict(frame, conf=self.confidence_threshold, verbose=False)
        detections = []

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf[0])

                    # Normalize bbox to [0, 1]
                    h, w = frame.shape[:2]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    bbox = (x1 / w, y1 / h, x2 / w, y2 / h)

                    detections.append(
                        Detection(
                            class_id=class_id,
                            class_name=class_name,
                            confidence=confidence,
                            bbox=bbox,
                        )
                    )

        return detections

    def batch_detect(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Run detection on multiple frames.

        Args:
            frames: List of input frames

        Returns:
            List of detection lists (one per frame)
        """
        return [self.detect(frame) for frame in frames]

    def get_model_info(self) -> dict:
        """Get model metadata."""
        return {
            "model_name": self.model_name,
            "num_classes": len(self.model.names),
            "class_names": self.model.names,
            "confidence_threshold": self.confidence_threshold,
            "device": self.device,
        }
