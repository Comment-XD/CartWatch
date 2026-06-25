from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json
from src.core.detector import Detection


@dataclass
class DetectionResult:
    """Formatted detection result."""

    frame_id: str
    detections: List[Detection]
    frame_shape: tuple
    model_name: str
    inference_time_ms: float
    timestamp: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "frame_id": self.frame_id,
            "detections": [
                {
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                    "confidence": d.confidence,
                    "bbox": d.bbox,
                }
                for d in self.detections
            ],
            "frame_shape": self.frame_shape,
            "model_name": self.model_name,
            "inference_time_ms": self.inference_time_ms,
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class DetectionResultFormatter:
    """Format raw detections into standardized results."""

    @staticmethod
    def format(
        frame_id: str,
        detections: List[Detection],
        frame_shape: tuple,
        model_name: str,
        inference_time_ms: float,
        timestamp: Optional[float] = None,
    ) -> DetectionResult:
        """Format detections into result object.

        Args:
            frame_id: Frame identifier
            detections: List of Detection objects
            frame_shape: Frame shape (height, width, channels)
            model_name: Model name used
            inference_time_ms: Inference time in milliseconds
            timestamp: Optional timestamp

        Returns:
            DetectionResult object
        """
        return DetectionResult(
            frame_id=frame_id,
            detections=detections,
            frame_shape=frame_shape,
            model_name=model_name,
            inference_time_ms=inference_time_ms,
            timestamp=timestamp,
        )
