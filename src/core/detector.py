from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Detection:
    """Represents a single detection."""

    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 normalized


class FasterRCNNDetector:
    """Faster R-CNN with ResNet-50 backbone detector wrapper."""

    def __init__(self, model_path: str = None, class_names: List[str] = None,
                 confidence_threshold: float = 0.5, device: str = "cpu"):
        """Initialize Faster R-CNN detector.

        Args:
            model_path: Path to trained model checkpoint (.pth file)
            class_names: List of class names (required if using trained model)
            confidence_threshold: Detection confidence threshold
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model_path = model_path
        self.class_names = class_names or ["object"]
        self.confidence_threshold = confidence_threshold
        self.device = device

        logger.info(f"Loading Faster R-CNN (ResNet-50) on {device}...")

        num_classes = len(self.class_names) + 1  # +1 for background
        self.model = fasterrcnn_resnet50_fpn(num_classes=num_classes)

        if model_path:
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint)
            logger.info(f"Loaded checkpoint from {model_path}")

        self.model = self.model.to(device)
        self.model.eval()
        logger.info("Model loaded successfully")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a single frame.

        Args:
            frame: Input frame (BGR, H x W x 3)

        Returns:
            List of Detection objects
        """
        h, w = frame.shape[:2]

        # Convert BGR to RGB and normalize
        rgb_frame = frame[..., ::-1]  # BGR to RGB
        tensor = torch.from_numpy(rgb_frame).permute(2, 0, 1).float() / 255.0
        tensor = tensor.to(self.device)

        with torch.no_grad():
            predictions = self.model([tensor])

        detections = []
        if predictions and len(predictions) > 0:
            pred = predictions[0]
            boxes = pred['boxes'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                if score >= self.confidence_threshold:
                    class_id = int(label) - 1  # Subtract 1 for background class
                    if 0 <= class_id < len(self.class_names):
                        x1, y1, x2, y2 = box
                        bbox = (x1 / w, y1 / h, x2 / w, y2 / h)

                        detections.append(
                            Detection(
                                class_id=class_id,
                                class_name=self.class_names[class_id],
                                confidence=float(score),
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
            "model_name": "fasterrcnn_resnet50",
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "confidence_threshold": self.confidence_threshold,
            "device": self.device,
            "model_path": self.model_path,
        }
