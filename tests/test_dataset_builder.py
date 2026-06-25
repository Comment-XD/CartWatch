"""Tests for dataset building from Label Studio exports."""

import json
from pathlib import Path

import pytest
import yaml

from src.training.dataset_builder import (
    discover_class_names,
    split_dataset,
    write_split_files,
    generate_dataset_yaml,
)


class TestDiscoverClassNames:
    """Test class name discovery from Label Studio exports."""

    def test_discover_class_names_from_classes_txt(self, tmp_path):
        """Test reading classes from classes.txt."""
        export_root = tmp_path / "export"
        export_root.mkdir()
        (export_root / "images").mkdir()

        classes = ["water_bottle", "book", "pen"]
        (export_root / "classes.txt").write_text("\n".join(classes) + "\n")

        result = discover_class_names(export_root)
        assert result == classes

    def test_discover_class_names_from_notes_json(self, tmp_path):
        """Test fallback to notes.json for class discovery."""
        export_root = tmp_path / "export"
        export_root.mkdir()
        (export_root / "images").mkdir()

        notes = {
            "categories": [
                {"id": 1, "name": "book"},
                {"id": 0, "name": "water_bottle"},
                {"id": 2, "name": "pen"},
            ]
        }
        (export_root / "notes.json").write_text(json.dumps(notes))

        result = discover_class_names(export_root)
        assert result == ["water_bottle", "book", "pen"]

    def test_discover_class_names_notes_json_out_of_order(self, tmp_path):
        """Test that notes.json categories are sorted by id."""
        export_root = tmp_path / "export"
        export_root.mkdir()
        (export_root / "images").mkdir()

        notes = {
            "categories": [
                {"id": 2, "name": "pen"},
                {"id": 0, "name": "water_bottle"},
                {"id": 1, "name": "book"},
            ]
        }
        (export_root / "notes.json").write_text(json.dumps(notes))

        result = discover_class_names(export_root)
        assert result == ["water_bottle", "book", "pen"]

    def test_discover_class_names_prefers_classes_txt(self, tmp_path):
        """Test that classes.txt is preferred over notes.json."""
        export_root = tmp_path / "export"
        export_root.mkdir()
        (export_root / "images").mkdir()

        (export_root / "classes.txt").write_text("class_a\nclass_b\n")
        notes = {"categories": [{"id": 0, "name": "class_x"}]}
        (export_root / "notes.json").write_text(json.dumps(notes))

        result = discover_class_names(export_root)
        assert result == ["class_a", "class_b"]

    def test_discover_class_names_missing_raises(self, tmp_path):
        """Test that missing both files raises error."""
        export_root = tmp_path / "export"
        export_root.mkdir()
        (export_root / "images").mkdir()

        with pytest.raises(FileNotFoundError):
            discover_class_names(export_root)


class TestSplitDataset:
    """Test train/val splitting."""

    def test_split_dataset_proportions(self, tmp_label_studio_export):
        """Test that split respects the requested proportion."""
        train_imgs, val_imgs = split_dataset(tmp_label_studio_export, val_split=0.2)

        total = len(train_imgs) + len(val_imgs)
        assert total == 10
        assert len(val_imgs) == 2
        assert len(train_imgs) == 8

    def test_split_dataset_reproducible_with_seed(self, tmp_label_studio_export):
        """Test that split is reproducible with the same seed."""
        train1, val1 = split_dataset(tmp_label_studio_export, val_split=0.2, seed=42)
        train2, val2 = split_dataset(tmp_label_studio_export, val_split=0.2, seed=42)

        assert train1 == train2
        assert val1 == val2

    def test_split_dataset_different_splits(self, tmp_label_studio_export):
        """Test different split ratios."""
        train50, val50 = split_dataset(tmp_label_studio_export, val_split=0.5)
        assert len(train50) == 5
        assert len(val50) == 5

        train90, val90 = split_dataset(tmp_label_studio_export, val_split=0.1)
        assert len(train90) == 9
        assert len(val90) == 1

    def test_split_dataset_empty_raises(self, tmp_path):
        """Test that empty images directory raises error."""
        export_root = tmp_path / "export"
        images_dir = export_root / "images"
        images_dir.mkdir(parents=True)

        with pytest.raises(ValueError, match="No images found"):
            split_dataset(export_root)

    def test_split_dataset_missing_images_dir_raises(self, tmp_path):
        """Test that missing images directory raises error."""
        export_root = tmp_path / "export"
        export_root.mkdir()

        with pytest.raises(FileNotFoundError, match="Images directory not found"):
            split_dataset(export_root)


class TestWriteSplitFiles:
    """Test split file writing."""

    def test_write_split_files_content(self, tmp_path, tmp_label_studio_export):
        """Test that split files have correct content."""
        train_imgs, val_imgs = split_dataset(tmp_label_studio_export, val_split=0.2)
        output_dir = tmp_path / "splits"

        train_txt, val_txt = write_split_files(train_imgs, val_imgs, output_dir)

        assert train_txt.exists()
        assert val_txt.exists()

        train_lines = train_txt.read_text().strip().split("\n")
        val_lines = val_txt.read_text().strip().split("\n")

        assert len(train_lines) == 8
        assert len(val_lines) == 2

        for line in train_lines:
            assert Path(line).exists()
        for line in val_lines:
            assert Path(line).exists()

    def test_write_split_files_creates_output_dir(self, tmp_path):
        """Test that output directory is created."""
        output_dir = tmp_path / "deep" / "nested" / "dir"
        assert not output_dir.exists()

        train_txt, val_txt = write_split_files([Path("/fake/img1.jpg")], [], output_dir)

        assert output_dir.exists()
        assert train_txt.parent == output_dir


class TestGenerateDatasetYaml:
    """Test dataset.yaml generation."""

    def test_generate_dataset_yaml_structure(self, tmp_path, tmp_label_studio_export):
        """Test that generated YAML has correct structure."""
        train_imgs, val_imgs = split_dataset(tmp_label_studio_export, val_split=0.2)
        train_txt, val_txt = write_split_files(train_imgs, val_imgs, tmp_path)
        class_names = ["water_bottle", "book"]

        yaml_path = tmp_path / "dataset.yaml"
        generate_dataset_yaml(
            tmp_label_studio_export,
            train_txt,
            val_txt,
            class_names,
            yaml_path,
        )

        assert yaml_path.exists()

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        assert "path" in data
        assert "train" in data
        assert "val" in data
        assert "nc" in data
        assert "names" in data
        assert data["nc"] == 2
        assert data["names"] == {0: "water_bottle", 1: "book"}

    def test_generate_dataset_yaml_paths_are_correct(self, tmp_path, tmp_label_studio_export):
        """Test that YAML contains the correct file paths."""
        train_imgs, val_imgs = split_dataset(tmp_label_studio_export, val_split=0.2)
        train_txt, val_txt = write_split_files(train_imgs, val_imgs, tmp_path)

        yaml_path = tmp_path / "dataset.yaml"
        generate_dataset_yaml(
            tmp_label_studio_export,
            train_txt,
            val_txt,
            ["water_bottle", "book"],
            yaml_path,
        )

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        assert data["train"] == str(train_txt)
        assert data["val"] == str(val_txt)
        assert data["path"] == str(tmp_label_studio_export)

    def test_generate_dataset_yaml_creates_output_dir(self, tmp_path):
        """Test that output directory is created."""
        output_dir = tmp_path / "deep" / "nested" / "dir"
        yaml_path = output_dir / "dataset.yaml"

        generate_dataset_yaml(
            Path("/tmp"),
            Path("/tmp/train.txt"),
            Path("/tmp/val.txt"),
            ["class1"],
            yaml_path,
        )

        assert yaml_path.exists()
