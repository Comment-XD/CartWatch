"""Object detection and classification metrics computation."""

from typing import List, Tuple, Dict
import numpy as np
import torch
from pathlib import Path


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute Intersection over Union (IoU) between two boxes.

    Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format

    Returns:
        IoU value (0.0 to 1.0)
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def match_detections(
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    pred_boxes: np.ndarray,
    pred_labels: np.ndarray,
    pred_scores: np.ndarray,
    iou_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match predictions to ground truth using greedy IoU matching.

    Args:
        gt_boxes: Ground truth boxes [N, 4]
        gt_labels: Ground truth labels [N]
        pred_boxes: Predicted boxes [M, 4]
        pred_labels: Predicted labels [M]
        pred_scores: Prediction confidence scores [M]
        iou_threshold: IoU threshold for match

    Returns:
        Tuple of (matched_gt_idx, matched_pred_idx, ious)
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    # Sort predictions by score (descending)
    sort_idx = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[sort_idx]
    pred_labels = pred_labels[sort_idx]
    pred_scores = pred_scores[sort_idx]

    matched_gt = []
    matched_pred = []
    ious = []

    used_gt = set()

    for pred_idx, (pred_box, pred_label, pred_score) in enumerate(
        zip(pred_boxes, pred_labels, pred_scores)
    ):
        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            if gt_idx in used_gt:
                continue
            if pred_label != gt_label:
                continue

            iou = compute_iou(pred_box, gt_box)

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            matched_gt.append(best_gt_idx)
            matched_pred.append(pred_idx)
            ious.append(best_iou)
            used_gt.add(best_gt_idx)

    return np.array(matched_gt), np.array(matched_pred), np.array(ious)


def compute_ap(tp: np.ndarray, fp: np.ndarray, num_gt: int) -> float:
    """Compute Average Precision.

    Args:
        tp: True positive array (binary, per detection)
        fp: False positive array (binary, per detection)
        num_gt: Total number of ground truth objects

    Returns:
        AP value (0.0 to 1.0)
    """
    if num_gt == 0:
        return 0.0

    # Compute precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recalls = tp_cumsum / num_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-9)

    # Append sentinel values
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([1.0], precisions, [0.0]))

    # Compute AP using trapezoidal rule
    ap = 0.0
    for i in range(len(precisions) - 1):
        ap += (recalls[i + 1] - recalls[i]) * precisions[i + 1]

    return ap


def compute_map_at_iou(
    all_gt_boxes: List[np.ndarray],
    all_gt_labels: List[np.ndarray],
    all_pred_boxes: List[np.ndarray],
    all_pred_labels: List[np.ndarray],
    all_pred_scores: List[np.ndarray],
    iou_threshold: float = 0.5,
    num_classes: int = None,
) -> Tuple[float, Dict[int, float]]:
    """Compute mAP at a specific IoU threshold across all images and classes.

    Args:
        all_gt_boxes: List of ground truth boxes per image
        all_gt_labels: List of ground truth labels per image
        all_pred_boxes: List of predicted boxes per image
        all_pred_labels: List of predicted labels per image
        all_pred_scores: List of prediction scores per image
        iou_threshold: IoU threshold (e.g., 0.5 for mAP@50)
        num_classes: Number of classes (auto-detect if None)

    Returns:
        Tuple of (mAP, per_class_ap_dict)
    """
    if num_classes is None:
        max_label = max(
            [max(labels) for labels in all_gt_labels if len(labels) > 0],
            default=0,
        )
        num_classes = int(max_label) + 1

    # Collect all detections by class
    detections_by_class = {c: [] for c in range(num_classes)}
    gt_counts_by_class = {c: 0 for c in range(num_classes)}

    for gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores in zip(
        all_gt_boxes, all_gt_labels, all_pred_boxes, all_pred_labels, all_pred_scores
    ):
        # Count ground truth by class
        for label in gt_labels:
            gt_counts_by_class[label] += 1

        # Match detections
        matched_gt, matched_pred, ious = match_detections(
            gt_boxes,
            gt_labels,
            pred_boxes,
            pred_labels,
            pred_scores,
            iou_threshold,
        )

        # Create TP/FP arrays per prediction
        tp = np.zeros(len(pred_boxes), dtype=np.int8)
        fp = np.ones(len(pred_boxes), dtype=np.int8)

        for gt_idx, pred_idx in zip(matched_gt, matched_pred):
            tp[pred_idx] = 1
            fp[pred_idx] = 0

        # Add to class detections
        for pred_idx, (pred_label, pred_score) in enumerate(
            zip(pred_labels, pred_scores)
        ):
            detections_by_class[pred_label].append(
                {"score": pred_score, "tp": tp[pred_idx], "fp": fp[pred_idx]}
            )

    # Compute AP per class
    per_class_ap = {}
    aps = []

    for class_id in range(num_classes):
        detections = detections_by_class[class_id]
        num_gt = gt_counts_by_class[class_id]

        if num_gt == 0 and len(detections) == 0:
            continue

        if len(detections) == 0:
            per_class_ap[class_id] = 0.0
            aps.append(0.0)
            continue

        # Sort by score
        detections.sort(key=lambda x: x["score"], reverse=True)

        tp = np.array([d["tp"] for d in detections], dtype=np.int8)
        fp = np.array([d["fp"] for d in detections], dtype=np.int8)

        ap = compute_ap(tp, fp, num_gt)
        per_class_ap[class_id] = ap
        aps.append(ap)

    mAP = np.mean(aps) if aps else 0.0

    return mAP, per_class_ap


def compute_precision_recall_f1(
    tp: np.ndarray,
    fp: np.ndarray,
    num_gt: int,
) -> Tuple[float, float, float]:
    """Compute precision, recall, and F1 score.

    Args:
        tp: True positive array
        fp: False positive array
        num_gt: Total ground truth count

    Returns:
        Tuple of (precision, recall, f1)
    """
    tp_sum = np.sum(tp)
    fp_sum = np.sum(fp)

    precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
    recall = tp_sum / num_gt if num_gt > 0 else 0.0

    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return precision, recall, f1


def compute_confidence_precision_at_thresholds(
    all_pred_scores: List[np.ndarray],
    all_tp: List[np.ndarray],
    thresholds: List[float] = None,
) -> Dict[float, float]:
    """Compute precision at different confidence thresholds.

    Args:
        all_pred_scores: List of confidence scores per image
        all_tp: List of TP arrays per image
        thresholds: Confidence thresholds (default: 0.1 to 0.9)

    Returns:
        Dict mapping threshold to precision
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1).tolist()

    # Flatten all scores and TP values
    all_scores = np.concatenate(all_pred_scores) if all_pred_scores else np.array([])
    all_tp_flat = np.concatenate(all_tp) if all_tp else np.array([])

    if len(all_scores) == 0:
        return {t: 0.0 for t in thresholds}

    precision_at_threshold = {}

    for threshold in thresholds:
        mask = all_scores >= threshold
        if np.sum(mask) == 0:
            precision_at_threshold[threshold] = 0.0
            continue

        tp_at_threshold = np.sum(all_tp_flat[mask])
        total_at_threshold = np.sum(mask)

        precision = tp_at_threshold / total_at_threshold
        precision_at_threshold[threshold] = precision

    return precision_at_threshold
