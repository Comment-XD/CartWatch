"""Tests for checkpoint versioning and directory management."""

import pytest

from src.training.checkpoint_manager import get_next_version, allocate_version_dir


class TestGetNextVersion:
    """Test version number discovery and allocation."""

    def test_get_next_version_empty_dir(self, tmp_path):
        """Test version numbering on an empty directory."""
        version = get_next_version(tmp_path, "yolo11n")
        assert version == 1

    def test_get_next_version_existing(self, tmp_path):
        """Test version numbering with existing versions."""
        (tmp_path / "yolo11n_v1").mkdir()
        (tmp_path / "yolo11n_v2").mkdir()

        version = get_next_version(tmp_path, "yolo11n")
        assert version == 3

    def test_get_next_version_ignores_other_models(self, tmp_path):
        """Test that different model names don't interfere."""
        (tmp_path / "yolo11s_v1").mkdir()
        (tmp_path / "yolo11s_v5").mkdir()

        version = get_next_version(tmp_path, "yolo11n")
        assert version == 1

    def test_get_next_version_ignores_malformed_dirs(self, tmp_path):
        """Test that malformed directory names are ignored."""
        (tmp_path / "yolo11n_v1").mkdir()
        (tmp_path / "yolo11n_vfoo").mkdir()
        (tmp_path / "yolo11n_v").mkdir()
        (tmp_path / "other_dir").mkdir()

        version = get_next_version(tmp_path, "yolo11n")
        assert version == 2

    def test_get_next_version_nonexistent_dir(self, tmp_path):
        """Test with a directory that doesn't exist yet."""
        nonexistent = tmp_path / "nonexistent"
        version = get_next_version(nonexistent, "yolo11n")
        assert version == 1


class TestAllocateVersionDir:
    """Test version directory allocation."""

    def test_allocate_version_dir_does_not_create(self, tmp_path):
        """Test that allocate_version_dir doesn't create the directory."""
        version_dir, version = allocate_version_dir(tmp_path, "yolo11n")

        assert not version_dir.exists()
        assert version == 1
        assert version_dir.name == "yolo11n_v1"

    def test_allocate_version_dir_increments(self, tmp_path):
        """Test that version increments correctly."""
        (tmp_path / "yolo11n_v1").mkdir()

        version_dir, version = allocate_version_dir(tmp_path, "yolo11n")

        assert version == 2
        assert version_dir.name == "yolo11n_v2"
        assert not version_dir.exists()

    def test_allocate_version_dir_multiple_models(self, tmp_path):
        """Test allocation for multiple different models."""
        (tmp_path / "yolo11n_v1").mkdir()
        (tmp_path / "yolo11s_v3").mkdir()

        yolo11n_dir, yolo11n_version = allocate_version_dir(tmp_path, "yolo11n")
        yolo11s_dir, yolo11s_version = allocate_version_dir(tmp_path, "yolo11s")

        assert yolo11n_version == 2
        assert yolo11s_version == 4
        assert yolo11n_dir.name == "yolo11n_v2"
        assert yolo11s_dir.name == "yolo11s_v4"
