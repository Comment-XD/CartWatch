"""Tests for Label Studio format conversion."""

import pytest

from src.core.detector import Detection
from src.inference.label_studio_format import (
    detection_to_ls_bbox,
    build_ls_task,
    build_ls_import_payload,
)


class TestDetectionToLsBbox:
    """Test bbox conversion from corner format to Label Studio percentage format."""

    def test_corner_bbox_to_ls_percentage(self):
        """Test exact conversion of a typical bbox."""
        bbox = (0.1, 0.2, 0.4, 0.6)
        result = detection_to_ls_bbox(bbox)

        assert abs(result["x"] - 10.0) < 1e-6
        assert abs(result["y"] - 20.0) < 1e-6
        assert abs(result["width"] - 30.0) < 1e-6
        assert abs(result["height"] - 40.0) < 1e-6
        assert result["rotation"] == 0

    def test_full_frame_bbox(self):
        """Test bbox covering the entire frame."""
        bbox = (0.0, 0.0, 1.0, 1.0)
        result = detection_to_ls_bbox(bbox)

        assert result["x"] == 0.0
        assert result["y"] == 0.0
        assert result["width"] == 100.0
        assert result["height"] == 100.0

    def test_zero_area_bbox(self):
        """Test degenerate bbox with zero area."""
        bbox = (0.5, 0.5, 0.5, 0.5)
        result = detection_to_ls_bbox(bbox)

        assert result["width"] == 0.0
        assert result["height"] == 0.0

    def test_clamping_out_of_range(self):
        """Test clamping of out-of-range values due to float imprecision."""
        bbox = (-0.001, 1.0002, 0.5, 1.5)
        result = detection_to_ls_bbox(bbox)

        assert 0.0 <= result["x"] <= 100.0
        assert 0.0 <= result["y"] <= 100.0
        assert 0.0 <= result["width"] <= 100.0
        assert 0.0 <= result["height"] <= 100.0

    def test_small_bbox(self):
        """Test a small bbox."""
        bbox = (0.3, 0.4, 0.35, 0.45)
        result = detection_to_ls_bbox(bbox)

        assert abs(result["x"] - 30.0) < 1e-6
        assert abs(result["y"] - 40.0) < 1e-6
        assert abs(result["width"] - 5.0) < 1e-6
        assert abs(result["height"] - 5.0) < 1e-6


class TestBuildLsTask:
    """Test Label Studio task building."""

    def test_build_ls_task_structure(self, sample_detections):
        """Test that task has correct structure."""
        image_path = "/path/to/image.jpg"
        detections = sample_detections[:2]
        frame_shape = (480, 640, 3)

        task = build_ls_task(image_path, detections, frame_shape)

        assert "data" in task
        assert task["data"]["image"] == image_path
        assert "predictions" in task
        assert len(task["predictions"]) == 1
        assert "result" in task["predictions"][0]
        assert len(task["predictions"][0]["result"]) == 2

    def test_task_with_no_detections(self):
        """Test task building with no detections."""
        image_path = "/path/to/image.jpg"
        detections = []
        frame_shape = (480, 640, 3)

        task = build_ls_task(image_path, detections, frame_shape)

        assert task["data"]["image"] == image_path
        assert task["predictions"][0]["result"] == []

    def test_result_entry_structure(self, sample_detections):
        """Test structure of individual result entries."""
        image_path = "/path/to/image.jpg"
        detections = sample_detections[:1]
        frame_shape = (480, 640, 3)

        task = build_ls_task(image_path, detections, frame_shape)
        result = task["predictions"][0]["result"][0]

        assert result["from_name"] == "label"
        assert result["to_name"] == "image"
        assert result["type"] == "rectanglelabels"
        assert "value" in result
        assert "x" in result["value"]
        assert "y" in result["value"]
        assert "width" in result["value"]
        assert "height" in result["value"]
        assert result["value"]["rectanglelabels"] == ["water_bottle"]
        assert result["score"] == 0.95

    def test_task_model_version(self):
        """Test model version tag in task."""
        image_path = "/path/to/image.jpg"
        detections = []
        frame_shape = (480, 640, 3)

        task = build_ls_task(image_path, detections, frame_shape, model_version="yolo11n_v2")

        assert task["predictions"][0]["model_version"] == "yolo11n_v2"


class TestBuildLsImportPayload:
    """Test full import payload building."""

    def test_payload_length(self, sample_detections):
        """Test that payload has correct number of tasks."""
        image_paths = ["/path1.jpg", "/path2.jpg", "/path3.jpg"]
        detections_per_image = [sample_detections[:1], sample_detections[1:2], sample_detections[2:]]
        frame_shapes = [(480, 640, 3)] * 3

        payload = build_ls_import_payload(image_paths, detections_per_image, frame_shapes)

        assert len(payload) == 3
        assert payload[0]["data"]["image"] == "/path1.jpg"
        assert payload[1]["data"]["image"] == "/path2.jpg"
        assert payload[2]["data"]["image"] == "/path3.jpg"

    def test_payload_with_no_detections(self):
        """Test payload when some images have no detections."""
        image_paths = ["/path1.jpg", "/path2.jpg"]
        detections_per_image = [[], []]
        frame_shapes = [(480, 640, 3)] * 2

        payload = build_ls_import_payload(image_paths, detections_per_image, frame_shapes)

        assert len(payload) == 2
        assert payload[0]["predictions"][0]["result"] == []
        assert payload[1]["predictions"][0]["result"] == []

    def test_payload_custom_tags(self, sample_detections):
        """Test payload with custom tag names."""
        image_paths = ["/path.jpg"]
        detections_per_image = [sample_detections[:1]]
        frame_shapes = [(480, 640, 3)]

        payload = build_ls_import_payload(
            image_paths,
            detections_per_image,
            frame_shapes,
            from_name="bbox",
            to_name="photo",
        )

        result = payload[0]["predictions"][0]["result"][0]
        assert result["from_name"] == "bbox"
        assert result["to_name"] == "photo"
