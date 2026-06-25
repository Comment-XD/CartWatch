from dataclasses import dataclass
from typing import List, Dict
from src.core.detector import Detection
from src.core.counter import InventoryCounter


@dataclass
class InventoryDelta:
    """Represents inventory change for a single class."""

    class_name: str
    count_before: int
    count_after: int
    delta: int
    confidence_avg_before: float
    confidence_avg_after: float


class InventoryDeltaCalculator:
    """Calculate inventory changes between two detection sets."""

    @staticmethod
    def calculate_delta(
        before_detections: List[Detection],
        after_detections: List[Detection],
    ) -> List[InventoryDelta]:
        """Calculate inventory delta between before and after.

        Args:
            before_detections: Detections from before interaction
            after_detections: Detections from after interaction

        Returns:
            List of InventoryDelta objects
        """
        counter = InventoryCounter()

        # Count objects
        before_counts = counter.count_objects(before_detections)
        after_counts = counter.count_objects(after_detections)

        # Get confidence stats
        before_stats = counter.get_confidence_stats(before_detections)
        after_stats = counter.get_confidence_stats(after_detections)

        # Calculate deltas
        all_classes = set(before_counts.keys()) | set(after_counts.keys())
        deltas = []

        for class_name in sorted(all_classes):
            count_before = before_counts.get(class_name, 0)
            count_after = after_counts.get(class_name, 0)
            delta = count_after - count_before

            confidence_avg_before = before_stats.get(class_name, {}).get("mean", 0.0)
            confidence_avg_after = after_stats.get(class_name, {}).get("mean", 0.0)

            deltas.append(
                InventoryDelta(
                    class_name=class_name,
                    count_before=count_before,
                    count_after=count_after,
                    delta=delta,
                    confidence_avg_before=confidence_avg_before,
                    confidence_avg_after=confidence_avg_after,
                )
            )

        return deltas

    @staticmethod
    def get_delta_by_class(
        before_detections: List[Detection],
        after_detections: List[Detection],
    ) -> Dict[str, int]:
        """Get delta values indexed by class name.

        Args:
            before_detections: Detections from before
            after_detections: Detections from after

        Returns:
            Dictionary mapping class names to delta (positive = added, negative = removed)
        """
        deltas = InventoryDeltaCalculator.calculate_delta(before_detections, after_detections)
        return {delta.class_name: delta.delta for delta in deltas}

    @staticmethod
    def generate_report(deltas: List[InventoryDelta]) -> str:
        """Generate human-readable report.

        Args:
            deltas: List of InventoryDelta objects

        Returns:
            Formatted report string
        """
        lines = ["=== Inventory Delta Report ===\n"]

        for delta in deltas:
            action = "removed" if delta.delta < 0 else "added" if delta.delta > 0 else "unchanged"
            lines.append(f"{delta.class_name}:")
            lines.append(f"  Before: {delta.count_before} items (confidence: {delta.confidence_avg_before:.2f})")
            lines.append(f"  After:  {delta.count_after} items (confidence: {delta.confidence_avg_after:.2f})")
            lines.append(f"  Delta:  {abs(delta.delta)} items {action}")
            lines.append("")

        return "\n".join(lines)
