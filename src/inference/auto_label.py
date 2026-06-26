"""Auto-labeling with trained Faster R-CNN checkpoints for Label Studio review."""

import json
from pathlib import Path
from typing import Any, Dict, List

from src.core.detector import FasterRCNNDetector
from src.inference.label_studio_format import build_ls_import_payload
from src.ingestion.frame_iterator import ImageDirectoryIterator
from src.training.dataset_builder import discover_class_names
from src.utils.logging import get_logger

logger = get_logger(__name__)


class AutoLabeler:
    """Run a trained Faster R-CNN checkpoint over unlabeled frames and produce
    Label Studio prediction-import JSON for human review."""

    def __init__(
        self,
        checkpoint_path: str,
        class_names: List[str] = None,
        confidence_threshold: float = 0.25,
        device: str = "cpu",
        from_name: str = "label",
        to_name: str = "image",
    ):
        """Initialize auto-labeler with a custom-trained checkpoint.

        Args:
            checkpoint_path: Full path to a .pth file, e.g.
                             models/checkpoints/fasterrcnn_resnet50_v2/best.pth
            class_names: List of class names (if not provided, attempt to load from checkpoint dir)
            confidence_threshold: Detection confidence threshold
                                  (lower than inference default 0.5 favors recall)
            device: 'cpu' or 'cuda'
            from_name: RectangleLabels tag name in LS labeling config
            to_name: Image tag name in LS labeling config
        """
        # Try to load class names from checkpoint directory if not provided
        if class_names is None:
            checkpoint_dir = Path(checkpoint_path).parent.parent
            if (checkpoint_dir / "classes.txt").exists():
                with open(checkpoint_dir / "classes.txt") as f:
                    class_names = [line.strip() for line in f]
                logger.info(f"Loaded {len(class_names)} classes from {checkpoint_dir / 'classes.txt'}")
            else:
                logger.warning(f"Could not find classes.txt in {checkpoint_dir}, using default")
                class_names = ["object"]

        self.detector = FasterRCNNDetector(
            model_path=checkpoint_path,
            class_names=class_names,
            confidence_threshold=confidence_threshold,
            device=device,
        )
        self.from_name = from_name
        self.to_name = to_name
        self.checkpoint_path = checkpoint_path

    def label_directory(self, image_dir: str) -> List[Dict[str, Any]]:
        """Run detection over every image in a directory and build LS tasks.

        Walks ImageDirectoryIterator directly (not DetectionPipeline) so that
        the original file path survives into the output. Runs FasterRCNNDetector.detect()
        on raw loaded frames (no FrameNormalizer resize) so predicted boxes map
        directly to original image dimensions for Label Studio.

        Args:
            image_dir: Directory of unlabeled frames (flat, no subdirs)

        Returns:
            List of Label Studio task dicts (ready for import JSON)
        """
        iterator = ImageDirectoryIterator(image_dir)
        image_paths, detections_per_image, frame_shapes = [], [], []

        count = 0
        for frame, metadata in iterator:
            try:
                detections = self.detector.detect(frame)
            except Exception as e:
                logger.error(f"Detection failed on {metadata['path']}: {e}")
                continue

            image_paths.append(metadata["path"])
            detections_per_image.append(detections)
            frame_shapes.append(frame.shape)

            count += 1
            if count % 50 == 0:
                logger.info(f"Auto-labeled {count} images...")

        logger.info(f"Auto-labeling complete: {count} images processed")

        return build_ls_import_payload(
            image_paths,
            detections_per_image,
            frame_shapes,
            from_name=self.from_name,
            to_name=self.to_name,
            model_version=Path(self.checkpoint_path).parent.parent.name,
        )

    def label_and_save(self, image_dir: str, output_json: str) -> Path:
        """Run label_directory and write the result to a JSON file.

        Args:
            image_dir: Directory of unlabeled frames
            output_json: Output path for the Label Studio import JSON

        Returns:
            Path to the written JSON file
        """
        payload = self.label_directory(image_dir)
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))
        logger.info(f"Wrote {len(payload)} tasks to {output_path}")
        return output_path
