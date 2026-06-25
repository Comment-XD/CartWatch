from typing import List, Dict
from src.core.detector import Detection


class InventoryCounter:
    """Count detected objects by class."""

    @staticmethod
    def count_objects(detections: List[Detection]) -> Dict[str, int]:
        """Count detections by class name.

        Args:
            detections: List of Detection objects

        Returns:
            Dictionary mapping class names to counts
        """
        counts = {}
        for detection in detections:
            counts[detection.class_name] = counts.get(detection.class_name, 0) + 1
        return counts

    @staticmethod
    def get_class_distribution(detections: List[Detection]) -> Dict[str, float]:
        """Get percentage distribution of classes.

        Args:
            detections: List of Detection objects

        Returns:
            Dictionary mapping class names to percentages
        """
        counts = InventoryCounter.count_objects(detections)
        total = sum(counts.values())
        if total == 0:
            return {}
        return {name: (count / total) * 100 for name, count in counts.items()}

    @staticmethod
    def get_confidence_stats(detections: List[Detection]) -> Dict[str, Dict[str, float]]:
        """Get confidence statistics by class.

        Args:
            detections: List of Detection objects

        Returns:
            Dictionary with mean, min, max confidence per class
        """
        class_confidences = {}
        for detection in detections:
            if detection.class_name not in class_confidences:
                class_confidences[detection.class_name] = []
            class_confidences[detection.class_name].append(detection.confidence)

        stats = {}
        for class_name, confidences in class_confidences.items():
            stats[class_name] = {
                "mean": sum(confidences) / len(confidences),
                "min": min(confidences),
                "max": max(confidences),
                "count": len(confidences),
            }
        return stats
