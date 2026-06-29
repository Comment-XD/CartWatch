"""Azure ML + MLflow configuration for experiment tracking."""

from pathlib import Path
from typing import Optional
import mlflow
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

from src.utils.logging import get_logger

logger = get_logger(__name__)


def setup_azure_ml_mlflow(
    workspace_name: str,
    resource_group: str,
    subscription_id: str,
    experiment_name: str = "cartwatch-training",
) -> MLClient:
    """Setup Azure ML workspace with MLflow tracking.

    Args:
        workspace_name: Azure ML workspace name
        resource_group: Azure resource group name
        subscription_id: Azure subscription ID
        experiment_name: MLflow experiment name (created if not exists)

    Returns:
        Configured MLClient instance
    """
    logger.info(f"Connecting to Azure ML workspace: {workspace_name}")

    # Authenticate with Azure
    credential = DefaultAzureCredential()

    # Create MLClient
    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )

    # Configure MLflow to use Azure ML as backend
    mlflow.set_tracking_uri(ml_client.workspace_connection)
    mlflow.set_experiment(experiment_name)

    logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    logger.info(f"MLflow experiment: {experiment_name}")

    return ml_client


def log_training_hyperparams(
    epochs: int,
    batch_size: int,
    learning_rate: float,
    val_split: float,
    device: str,
    model_name: str,
    seed: int,
) -> None:
    """Log training hyperparameters to MLflow.

    Args:
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        val_split: Validation split ratio
        device: Device (cpu/cuda)
        model_name: Model name
        seed: Random seed
    """
    mlflow.log_params({
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "val_split": val_split,
        "device": device,
        "model_name": model_name,
        "seed": seed,
    })
    logger.info("Logged training hyperparameters to MLflow")


def log_training_metrics(
    epoch: int,
    total_loss: float,
    cls_loss: Optional[float] = None,
    box_reg_loss: Optional[float] = None,
    objectness_loss: Optional[float] = None,
    rpn_box_loss: Optional[float] = None,
) -> None:
    """Log training metrics for current epoch to MLflow.

    Args:
        epoch: Epoch number (0-indexed)
        total_loss: Total loss
        cls_loss: Classification loss
        box_reg_loss: Box regression loss
        objectness_loss: Objectness loss
        rpn_box_loss: RPN box loss
    """
    mlflow.log_metric("train/total_loss", total_loss, step=epoch)

    if cls_loss is not None:
        mlflow.log_metric("train/cls_loss", cls_loss, step=epoch)
    if box_reg_loss is not None:
        mlflow.log_metric("train/box_reg_loss", box_reg_loss, step=epoch)
    if objectness_loss is not None:
        mlflow.log_metric("train/objectness_loss", objectness_loss, step=epoch)
    if rpn_box_loss is not None:
        mlflow.log_metric("train/rpn_box_loss", rpn_box_loss, step=epoch)


def log_validation_metrics(
    epoch: int,
    val_loss: float,
    map50: float,
    map75: float,
    map_avg: float,
    precision: float,
    recall: float,
    f1_score: float,
) -> None:
    """Log validation metrics to MLflow.

    Args:
        epoch: Epoch number
        val_loss: Validation loss
        map50: mAP@50
        map75: mAP@75
        map_avg: mAP (averaged across IoU thresholds)
        precision: Overall precision
        recall: Overall recall
        f1_score: Overall F1 score
    """
    mlflow.log_metric("val/loss", val_loss, step=epoch)
    mlflow.log_metric("val/map50", map50, step=epoch)
    mlflow.log_metric("val/map75", map75, step=epoch)
    mlflow.log_metric("val/map_avg", map_avg, step=epoch)
    mlflow.log_metric("val/precision", precision, step=epoch)
    mlflow.log_metric("val/recall", recall, step=epoch)
    mlflow.log_metric("val/f1_score", f1_score, step=epoch)


def log_artifact_to_mlflow(local_path: Path, artifact_path: str = "models") -> None:
    """Log a file artifact to MLflow.

    Args:
        local_path: Local file path
        artifact_path: Destination path in MLflow artifact store
    """
    mlflow.log_artifact(str(local_path), artifact_path)
    logger.info(f"Logged artifact: {local_path.name} → {artifact_path}")


def log_model_to_registry(
    model_path: Path,
    model_name: str,
    version: str,
    metrics: dict,
    hyperparams: dict,
) -> None:
    """Register trained model in Azure ML model registry.

    Args:
        model_path: Path to saved model (.pth file)
        model_name: Model name for registry
        version: Model version
        metrics: Dict of evaluation metrics
        hyperparams: Dict of training hyperparameters
    """
    mlflow.log_artifact(str(model_path), "models")
    mlflow.set_tag("model_version", version)

    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(f"final/{metric_name}", metric_value)

    logger.info(f"Registered model: {model_name} v{version}")
