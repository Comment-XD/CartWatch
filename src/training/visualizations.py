"""Visualization utilities for training metrics and model evaluation."""

from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime


def plot_loss_curves(
    epochs: List[int],
    train_loss: List[float],
    val_loss: Optional[List[float]] = None,
    cls_loss: Optional[List[float]] = None,
    box_loss: Optional[List[float]] = None,
    output_path: Optional[Path] = None,
) -> Path:
    """Plot training and validation loss curves.

    Args:
        epochs: List of epoch numbers
        train_loss: Training loss per epoch
        val_loss: Validation loss per epoch (optional)
        cls_loss: Classification loss per epoch (optional)
        box_loss: Box regression loss per epoch (optional)
        output_path: Path to save plot (default: /tmp/loss_curves.png)

    Returns:
        Path to saved plot
    """
    if output_path is None:
        output_path = Path("/tmp/loss_curves.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Total loss
    ax = axes[0]
    ax.plot(epochs, train_loss, label="Train Loss", marker="o", linewidth=2)
    if val_loss is not None:
        ax.plot(epochs, val_loss, label="Val Loss", marker="s", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Total Loss Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Component losses
    ax = axes[1]
    if cls_loss is not None:
        ax.plot(epochs, cls_loss, label="Classification Loss", marker="o", linewidth=2)
    if box_loss is not None:
        ax.plot(epochs, box_loss, label="Box Regression Loss", marker="s", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Components")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def plot_metrics_curves(
    epochs: List[int],
    metrics_dict: Dict[str, List[float]],
    output_path: Optional[Path] = None,
) -> Path:
    """Plot multiple metric curves (precision, recall, F1, mAP).

    Args:
        epochs: List of epoch numbers
        metrics_dict: Dict mapping metric name to list of values
        output_path: Path to save plot

    Returns:
        Path to saved plot
    """
    if output_path is None:
        output_path = Path("/tmp/metrics_curves.png")

    num_metrics = len(metrics_dict)
    fig, axes = plt.subplots(
        (num_metrics + 1) // 2,
        2,
        figsize=(14, 5 * ((num_metrics + 1) // 2)),
    )

    if num_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for ax_idx, (metric_name, values) in enumerate(metrics_dict.items()):
        ax = axes[ax_idx]
        ax.plot(epochs, values, marker="o", linewidth=2, color="steelblue")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric_name.replace("_", " ").title())
        ax.set_title(f"{metric_name.replace('_', ' ').title()} Curve")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.0])

    # Hide unused subplots
    for ax_idx in range(len(metrics_dict), len(axes)):
        axes[ax_idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def plot_precision_confidence_curve(
    confidence_thresholds: List[float],
    precisions: List[float],
    output_path: Optional[Path] = None,
) -> Path:
    """Plot precision vs confidence threshold curve.

    Args:
        confidence_thresholds: Confidence thresholds (0.0 to 1.0)
        precisions: Precision at each threshold
        output_path: Path to save plot

    Returns:
        Path to saved plot
    """
    if output_path is None:
        output_path = Path("/tmp/precision_confidence.png")

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        confidence_thresholds,
        precisions,
        marker="o",
        linewidth=2.5,
        markersize=8,
        color="seagreen",
    )
    ax.axhline(y=np.mean(precisions), color="r", linestyle="--", label="Mean Precision")
    ax.set_xlabel("Confidence Threshold")
    ax.set_ylabel("Precision")
    ax.set_title("Precision vs Confidence Threshold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim([0, 1.0])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def plot_per_class_metrics(
    class_names: List[str],
    class_metrics: Dict[str, List[float]],
    output_path: Optional[Path] = None,
) -> Path:
    """Plot per-class metrics as grouped bar chart.

    Args:
        class_names: List of class names
        class_metrics: Dict mapping metric name to list of values per class
        output_path: Path to save plot

    Returns:
        Path to saved plot
    """
    if output_path is None:
        output_path = Path("/tmp/per_class_metrics.png")

    num_classes = len(class_names)
    x = np.arange(num_classes)
    width = 0.25
    num_metrics = len(class_metrics)

    fig, ax = plt.subplots(figsize=(max(12, num_classes * 1.5), 6))

    for metric_idx, (metric_name, values) in enumerate(class_metrics.items()):
        offset = (metric_idx - num_metrics / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=metric_name.replace("_", " ").title())

    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: List[str],
    output_path: Optional[Path] = None,
) -> Path:
    """Plot confusion matrix as heatmap.

    Args:
        confusion_matrix: Confusion matrix [num_classes, num_classes]
        class_names: List of class names
        output_path: Path to save plot

    Returns:
        Path to saved plot
    """
    if output_path is None:
        output_path = Path("/tmp/confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(10, 10))

    im = ax.imshow(confusion_matrix, cmap="Blues")

    # Set ticks and labels
    num_classes = len(class_names)
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    for i in range(num_classes):
        for j in range(num_classes):
            text = ax.text(
                j,
                i,
                f"{confusion_matrix[i, j]:.0f}",
                ha="center",
                va="center",
                color="white" if confusion_matrix[i, j] > confusion_matrix.max() / 2 else "black",
            )

    ax.set_title("Confusion Matrix")
    ax.set_ylabel("Ground Truth")
    ax.set_xlabel("Prediction")

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path


def plot_detection_examples(
    image: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: List[str],
    pred_boxes: np.ndarray,
    pred_labels: List[str],
    pred_scores: np.ndarray,
    output_path: Optional[Path] = None,
) -> Path:
    """Visualize detections on an image.

    Args:
        image: Image array (H x W x 3)
        gt_boxes: Ground truth boxes [N, 4] in corner format
        gt_labels: Ground truth label strings
        pred_boxes: Predicted boxes [M, 4] in corner format
        pred_labels: Predicted label strings
        pred_scores: Prediction confidence scores
        output_path: Path to save plot

    Returns:
        Path to saved plot
    """
    if output_path is None:
        output_path = Path("/tmp/detection_example.png")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Display image
    if image.max() <= 1.0:
        ax.imshow((image * 255).astype(np.uint8))
    else:
        ax.imshow(image.astype(np.uint8))

    # Draw ground truth boxes (green)
    for box, label in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="green",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, label, color="green", fontsize=10, weight="bold")

    # Draw predicted boxes (red)
    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
            linestyle="--",
        )
        ax.add_patch(rect)
        ax.text(
            x1,
            y2 + 15,
            f"{label} {score:.2f}",
            color="red",
            fontsize=10,
            weight="bold",
        )

    ax.set_title("Detection Example\n(Green=GT, Red=Pred)")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return output_path
