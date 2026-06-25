"""Tests for the YOLO detector, including custom checkpoint loading."""

from unittest.mock import MagicMock, patch

import pytest

from src.core.detector import YOLODetector


class TestYOLODetectorInit:
    """Test YOLODetector initialization with various model paths."""

    @patch("src.core.detector.YOLO")
    def test_model_name_stock_appends_pt_extension(self, mock_yolo):
        """Test that stock model names get .pt appended."""
        mock_model_instance = MagicMock()
        mock_yolo.return_value = mock_model_instance

        detector = YOLODetector(model_name="yolo11n")

        mock_yolo.assert_called_once_with("yolo11n.pt")

    @patch("src.core.detector.YOLO")
    def test_model_name_path_passthrough(self, mock_yolo):
        """Test that full paths ending in .pt pass through unchanged."""
        mock_model_instance = MagicMock()
        mock_yolo.return_value = mock_model_instance

        detector = YOLODetector(model_name="models/checkpoints/yolo11n_v1/weights/best.pt")

        mock_yolo.assert_called_once_with("models/checkpoints/yolo11n_v1/weights/best.pt")

    @patch("src.core.detector.YOLO")
    def test_model_name_path_without_pt(self, mock_yolo):
        """Test that paths without .pt get .pt appended."""
        mock_model_instance = MagicMock()
        mock_yolo.return_value = mock_model_instance

        detector = YOLODetector(model_name="models/checkpoints/yolo11n_v1")

        mock_yolo.assert_called_once_with("models/checkpoints/yolo11n_v1.pt")

    @patch("src.core.detector.YOLO")
    def test_model_stores_original_name(self, mock_yolo):
        """Test that the original model_name is stored."""
        mock_model_instance = MagicMock()
        mock_yolo.return_value = mock_model_instance

        detector = YOLODetector(model_name="yolo11n")

        assert detector.model_name == "yolo11n"

    @patch("src.core.detector.YOLO")
    def test_model_device_setting(self, mock_yolo):
        """Test that device is set on the model."""
        mock_model_instance = MagicMock()
        mock_yolo.return_value = mock_model_instance

        detector = YOLODetector(model_name="yolo11n", device="cuda")

        mock_model_instance.to.assert_called_once_with("cuda")

    @patch("src.core.detector.YOLO")
    def test_model_confidence_threshold_stored(self, mock_yolo):
        """Test that confidence threshold is stored."""
        mock_model_instance = MagicMock()
        mock_yolo.return_value = mock_model_instance

        detector = YOLODetector(model_name="yolo11n", confidence_threshold=0.6)

        assert detector.confidence_threshold == 0.6
