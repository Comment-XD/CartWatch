import pytest
from src.core.counter import InventoryCounter


class TestInventoryCounter:
    """Test InventoryCounter functionality."""

    def test_count_objects(self, sample_detections):
        """Test object counting by class."""
        counts = InventoryCounter.count_objects(sample_detections)
        assert counts["water_bottle"] == 2
        assert counts["book"] == 1

    def test_count_empty(self):
        """Test counting with no detections."""
        counts = InventoryCounter.count_objects([])
        assert counts == {}

    def test_class_distribution(self, sample_detections):
        """Test percentage distribution."""
        dist = InventoryCounter.get_class_distribution(sample_detections)
        assert "water_bottle" in dist
        assert "book" in dist
        # water_bottle: 2/3 = 66.67%, book: 1/3 = 33.33%
        assert 66 < dist["water_bottle"] < 67
        assert 33 < dist["book"] < 34

    def test_confidence_stats(self, sample_detections):
        """Test confidence statistics."""
        stats = InventoryCounter.get_confidence_stats(sample_detections)

        # Water bottle: [0.95, 0.87]
        assert stats["water_bottle"]["count"] == 2
        assert abs(stats["water_bottle"]["mean"] - 0.91) < 0.01
        assert stats["water_bottle"]["min"] == 0.87
        assert stats["water_bottle"]["max"] == 0.95

        # Book: [0.92]
        assert stats["book"]["count"] == 1
        assert stats["book"]["mean"] == 0.92
