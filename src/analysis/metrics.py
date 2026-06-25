from typing import List, Dict
from src.core.detector import Detection
from src.core.counter import InventoryCounter


class DetectionMetrics:
    """Compute detection statistics and metrics."""

    @staticmethod
    def calculate_confidence_mean(detections: List[Detection]) -> float:
        """Calculate mean confidence across all detections.

        Args:
            detections: List of Detection objects

        Returns:
            Mean confidence (0-1)
        """
        if not detections:
            return 0.0
        return sum(d.confidence for d in detections) / len(detections)

    @staticmethod
    def get_per_class_counts(detections: List[Detection]) -> Dict[str, int]:
        """Get object counts per class.

        Args:
            detections: List of Detection objects

        Returns:
            Dictionary mapping class names to counts
        """
        counter = InventoryCounter()
        return counter.count_objects(detections)

    @staticmethod
    def get_detection_distribution(detections: List[Detection]) -> Dict[str, float]:
        """Get percentage distribution of detections by class.

        Args:
            detections: List of Detection objects

        Returns:
            Dictionary mapping class names to percentages
        """
        counter = InventoryCounter()
        return counter.get_class_distribution(detections)

    @staticmethod
    def get_confidence_by_class(detections: List[Detection]) -> Dict[str, Dict[str, float]]:
        """Get confidence statistics by class.

        Args:
            detections: List of Detection objects

        Returns:
            Dictionary with stats per class
        """
        counter = InventoryCounter()
        return counter.get_confidence_stats(detections)

    @staticmethod
    def get_summary(detections: List[Detection]) -> Dict:
        """Get comprehensive metrics summary.

        Args:
            detections: List of Detection objects

        Returns:
            Summary dictionary
        """
        return {
            "total_detections": len(detections),
            "mean_confidence": DetectionMetrics.calculate_confidence_mean(detections),
            "per_class_counts": DetectionMetrics.get_per_class_counts(detections),
            "class_distribution": DetectionMetrics.get_detection_distribution(detections),
            "confidence_by_class": DetectionMetrics.get_confidence_by_class(detections),
        }
