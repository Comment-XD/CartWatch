import pytest
import numpy as np
from src.core.frame_processor import FrameProcessor


class TestFrameProcessor:
    """Test FrameProcessor functionality."""

    def test_get_frame_info(self, sample_frame):
        """Test frame metadata extraction."""
        info = FrameProcessor.get_frame_info(sample_frame)
        assert info["height"] == 480
        assert info["width"] == 640
        assert info["channels"] == 3

    def test_resize(self, sample_frame):
        """Test frame resizing."""
        resized = FrameProcessor.resize(sample_frame, (320, 240))
        assert resized.shape == (240, 320, 3)

    def test_normalize_denormalize(self, sample_frame):
        """Test normalization and denormalization."""
        normalized = FrameProcessor.normalize(sample_frame.astype(np.float32))
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

        denormalized = FrameProcessor.denormalize(normalized)
        assert denormalized.min() >= 0
        assert denormalized.max() <= 255
