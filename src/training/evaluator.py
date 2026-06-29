"""Evaluation module for Faster R-CNN model validation."""

from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader
import cv2

from src.training.metrics import (
    compute_map_at_iou,
    compute_precision_recall_f1,
    compute_confidence_precision_at_thresholds,
    match_detections,
)
from src.training.visualizations import (
    plot_precision_confidence_curve,
    plot_per_class_metrics,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class FasterRCNNEvaluator:
    """Evaluate Faster R-CNN model on validation dataset."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: str = "cpu",
        confidence_threshold: float = 0.5,
    ):
        """Initialize evaluator.

        Args:
            model: Faster R-CNN model
            device: Device to run evaluation on
            confidence_threshold: Confidence threshold for detections
        """
        self.model = model
        self.device = device
        self.confidence_threshold = confidence_threshold

    def evaluate(
        self,
        data_loader: DataLoader,
        class_names: List[str],
        save_plots: bool = True,
        plots_dir: Optional[Path] = None,
    ) -> Dict[str, float]:
        """Evaluate model on validation dataset.

        Args:
            data_loader: DataLoader for validation data
            class_names: List of class names
            save_plots: Whether to save visualization plots
            plots_dir: Directory to save plots

        Returns:
            Dict of evaluation metrics
        """
        logger.info("Starting evaluation...")

        self.model.eval()
        torch.set_grad_enabled(False)

        all_gt_boxes = []
        all_gt_labels = []
        all_pred_boxes = []
        all_pred_labels = []
        all_pred_scores = []
        all_losses = []

        num_classes = len(class_names)

        with torch.no_grad():
            for images, targets in data_loader:
                images = [img.to(self.device) for img in images]
                targets = [
                    {k: v.to(self.device) for k, v in t.items()}
                    for t in targets
                ]

                # Forward pass
                outputs = self.model(images)

                for i, (output, target) in enumerate(zip(outputs, targets)):
                    gt_boxes = target["boxes"].cpu().numpy()
                    gt_labels = target["labels"].cpu().numpy() - 1  # Remove background offset

                    # Filter predictions by confidence
                    keep = output["scores"] >= self.confidence_threshold
                    pred_boxes = output["boxes"][keep].cpu().numpy()
                    pred_labels = output["labels"][keep].cpu().numpy() - 1
                    pred_scores = output["scores"][keep].cpu().numpy()

                    all_gt_boxes.append(gt_boxes)
                    all_gt_labels.append(gt_labels)
                    all_pred_boxes.append(pred_boxes)
                    all_pred_labels.append(pred_labels)
                    all_pred_scores.append(pred_scores)

        # Compute metrics
        map50, per_class_map50 = compute_map_at_iou(
            all_gt_boxes,
            all_gt_labels,
            all_pred_boxes,
            all_pred_labels,
            all_pred_scores,
            iou_threshold=0.5,
            num_classes=num_classes,
        )

        map75, per_class_map75 = compute_map_at_iou(
            all_gt_boxes,
            all_gt_labels,
            all_pred_boxes,
            all_pred_labels,
            all_pred_scores,
            iou_threshold=0.75,
            num_classes=num_classes,
        )

        # Compute average mAP across IoU thresholds
        iou_thresholds = np.arange(0.5, 1.0, 0.05)
        maps = []
        for iou_threshold in iou_thresholds:
            map_iou, _ = compute_map_at_iou(
                all_gt_boxes,
                all_gt_labels,
                all_pred_boxes,
                all_pred_labels,
                all_pred_scores,
                iou_threshold=iou_threshold,
                num_classes=num_classes,
            )
            maps.append(map_iou)

        map_avg = np.mean(maps) if maps else 0.0

        # Compute precision/recall/F1
        all_gt_boxes_flat = np.concatenate(all_gt_boxes) if all_gt_boxes else np.array([])
        num_gt = len(all_gt_boxes_flat)

        # Count TP/FP
        tp_total = 0
        fp_total = 0

        for gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores in zip(
            all_gt_boxes, all_gt_labels, all_pred_boxes, all_pred_labels, all_pred_scores
        ):
            matched_gt, matched_pred, _ = match_detections(
                gt_boxes,
                gt_labels,
                pred_boxes,
                pred_labels,
                pred_scores,
                iou_threshold=0.5,
            )

            tp = len(matched_gt)
            fp = len(pred_boxes) - tp

            tp_total += tp
            fp_total += fp

        precision, recall, f1 = compute_precision_recall_f1(
            np.ones(tp_total, dtype=np.int8),
            np.ones(fp_total, dtype=np.int8),
            num_gt,
        )

        # Compute confidence-precision curve
        confidence_thresholds = np.linspace(0.1, 0.9, 9).tolist()
        conf_precision = compute_confidence_precision_at_thresholds(
            all_pred_scores,
            [np.ones(len(scores), dtype=np.int8) for scores in all_pred_scores],
            confidence_thresholds,
        )

        metrics = {
            "map50": map50,
            "map75": map75,
            "map_avg": map_avg,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "num_gt": num_gt,
            "num_detections": sum(len(b) for b in all_pred_boxes),
        }

        # Per-class metrics
        for class_id, class_name in enumerate(class_names):
            if class_id in per_class_map50:
                metrics[f"map50_{class_name}"] = per_class_map50[class_id]

        logger.info(f"mAP@50: {map50:.4f}")
        logger.info(f"mAP@75: {map75:.4f}")
        logger.info(f"mAP (avg): {map_avg:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")

        # Save plots if requested
        if save_plots and plots_dir is not None:
            plots_dir.mkdir(parents=True, exist_ok=True)

            # Precision-confidence curve
            plot_precision_confidence_curve(
                confidence_thresholds,
                [conf_precision.get(t, 0.0) for t in confidence_thresholds],
                output_path=plots_dir / "precision_confidence.png",
            )

            # Per-class metrics
            per_class_maps = {
                class_name: [per_class_map50.get(idx, 0.0)]
                for idx, class_name in enumerate(class_names)
            }
            plot_per_class_metrics(
                class_names,
                {"mAP@50": [per_class_map50.get(i, 0.0) for i in range(len(class_names))]},
                output_path=plots_dir / "per_class_map50.png",
            )

            logger.info(f"Saved evaluation plots to {plots_dir}")

        return metrics
