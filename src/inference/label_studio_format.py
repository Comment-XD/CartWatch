"""Label Studio prediction format conversion for YOLO detections."""

from typing import Any, Dict, List, Tuple

from src.core.detector import Detection
from src.utils.logging import get_logger

logger = get_logger(__name__)


def detection_to_ls_bbox(bbox: Tuple[float, float, float, float]) -> Dict[str, float]:
    """Convert a normalized corner-format bbox to Label Studio percentage format.

    Args:
        bbox: (x1, y1, x2, y2), each normalized to [0, 1] relative to
              image width (x) / height (y), top-left/bottom-right corners.

    Returns:
        Dict with keys x, y, width, height, rotation — all in Label
        Studio's percentage convention (0-100), top-left + width/height.
    """
    x1, y1, x2, y2 = bbox

    def clamp(v: float) -> float:
        return max(0.0, min(100.0, v))

    return {
        "x": clamp(x1 * 100),
        "y": clamp(y1 * 100),
        "width": clamp((x2 - x1) * 100),
        "height": clamp((y2 - y1) * 100),
        "rotation": 0,
    }


def build_ls_task(
    image_path: str,
    detections: List[Detection],
    frame_shape: Tuple[int, int, int],
    from_name: str = "label",
    to_name: str = "image",
    model_version: str = "auto_label",
) -> Dict[str, Any]:
    """Build one Label Studio prediction-import task for a single image.

    Args:
        image_path: Path to the image, as Label Studio should reference it
                    (typically absolute path or URL depending on storage config)
        detections: Detections for this image
        frame_shape: (height, width, channels) of the source frame
        from_name: Name of the RectangleLabels control tag in LS labeling config
        to_name: Name of the Image object tag in LS labeling config
        model_version: String tag identifying the checkpoint that produced predictions

    Returns:
        Dict matching Label Studio's task-with-predictions import schema
    """
    results = []
    for det in detections:
        ls_bbox = detection_to_ls_bbox(det.bbox)
        results.append({
            "from_name": from_name,
            "to_name": to_name,
            "type": "rectanglelabels",
            "value": {
                **ls_bbox,
                "rectanglelabels": [det.class_name],
            },
            "score": det.confidence,
        })

    task = {
        "data": {"image": image_path},
        "predictions": [{
            "model_version": model_version,
            "result": results,
        }],
    }
    return task


def build_ls_import_payload(
    image_paths: List[str],
    detections_per_image: List[List[Detection]],
    frame_shapes: List[Tuple[int, int, int]],
    from_name: str = "label",
    to_name: str = "image",
    model_version: str = "auto_label",
) -> List[Dict[str, Any]]:
    """Build the full Label Studio prediction-import JSON (list of tasks).

    Args:
        image_paths: One path per image, same order as detections_per_image
        detections_per_image: Detections for each image, same order
        frame_shapes: (h, w, c) per image, same order
        from_name: RectangleLabels tag name
        to_name: Image tag name
        model_version: Model version identifier

    Returns:
        List of task dicts — this list IS the JSON structure to write to disk
        (Label Studio's "Import" expects a JSON array of tasks at the top level)
    """
    return [
        build_ls_task(path, dets, shape, from_name=from_name, to_name=to_name, model_version=model_version)
        for path, dets, shape in zip(image_paths, detections_per_image, frame_shapes)
    ]
