# MLflow + Azure ML Training Pipeline

CartWatch now supports comprehensive experiment tracking with MLflow and Azure ML Studio for training visualization and model management.

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Key packages:
- `mlflow>=2.8.0` — experiment tracking
- `azure-ai-ml>=1.10.0` — Azure ML workspace management
- `azure-identity>=1.14.0` — Azure authentication

### 2. Azure ML Authentication

The system uses `DefaultAzureCredential`, which tries multiple authentication methods in order:

1. **Environment variables** (if set):
   ```bash
   export AZURE_CLIENT_ID="your-service-principal-client-id"
   export AZURE_CLIENT_SECRET="your-service-principal-secret"
   export AZURE_TENANT_ID="your-tenant-id"
   ```

2. **Azure CLI** (if logged in):
   ```bash
   az login
   ```

3. **Managed Identity** (if running on Azure VM/App Service)

4. **Interactive browser login** (if all above fail)

Pick one method based on your environment. For local development, `az login` is easiest.

---

## Training with MLflow

### Local MLflow (Default)

```bash
python scripts/train.py /path/to/export \
  --epochs 50 \
  --batch-size 8 \
  --device cuda \
  --experiment-name "cartwatch-v1"
```

This logs metrics to a local MLflow tracking server (`./mlruns/`).

**View results:**
```bash
mlflow ui
# Open http://localhost:5000
```

### Azure ML MLflow Integration

```bash
python scripts/train.py /path/to/export \
  --epochs 50 \
  --batch-size 8 \
  --device cuda \
  --experiment-name "cartwatch-training" \
  --azure-workspace "your-workspace-name" \
  --azure-resource-group "your-resource-group" \
  --azure-subscription "your-subscription-id"
```

**View results:** Open your Azure ML Studio workspace → Experiments tab → "cartwatch-training"

---

## What Gets Logged

### Hyperparameters

```
epochs: 50
batch_size: 8
learning_rate: 0.005
val_split: 0.2
device: cuda
model_name: fasterrcnn_resnet50
seed: 42
```

### Training Metrics (Per Epoch)

- `train/total_loss` — total training loss
- `train/cls_loss` — classification loss
- `train/box_reg_loss` — box regression loss
- `train/objectness_loss` — RPN objectness loss
- `train/rpn_box_loss` — RPN box loss

### Validation Metrics (Every 10% of training)

- `val/map50` — mAP at IoU threshold 0.5
- `val/map75` — mAP at IoU threshold 0.75
- `val/map_avg` — mean mAP across IoU thresholds
- `val/precision` — overall precision
- `val/recall` — overall recall
- `val/f1_score` — overall F1 score
- `val/num_gt` — number of ground truth objects
- `val/num_detections` — number of predictions
- `val/map50_<classname>` — per-class mAP@50

### Final Metrics

- `final/map50` — final mAP@50
- `final/map75` — final mAP@75
- `final/map_avg` — final mean mAP
- `final/precision` — final precision
- `final/recall` — final recall
- `final/f1_score` — final F1 score
- `final/num_classes` — number of classes
- `final/train_images` — training set size
- `final/val_images` — validation set size

### Visualizations

Saved to `models/checkpoints/<model>_v<N>/plots/`:

1. **loss_curves.png**
   - Total training loss curve
   - Classification + box regression loss components

2. **metrics_curves.png**
   - mAP@50, mAP@75, precision, recall, F1 over epochs

3. **precision_confidence.png**
   - Precision vs confidence threshold
   - Helps identify optimal confidence cutoff

4. **per_class_map50.png**
   - Per-class mAP@50 breakdown
   - Identify weak/strong class performance

5. **confusion_matrix.png** (if available)
   - Class-level prediction distribution

### Model Artifacts

- `models/checkpoints/<model>_v<N>/best.pth` — best checkpoint by loss
- `models/checkpoints/<model>_v<N>/final.pth` — final checkpoint
- All artifacts logged to MLflow for easy retrieval

---

## Object Detection Metrics Explained

### mAP (Mean Average Precision)

- **mAP@50**: Average precision with IoU threshold 0.5 (loose matching)
- **mAP@75**: Average precision with IoU threshold 0.75 (strict matching)
- **mAP@50-95**: Average across thresholds 0.5, 0.55, 0.60, ..., 0.95 (COCO standard)

Higher is better. Typical ranges:
- **0.0–0.3**: Poor detection
- **0.3–0.5**: Fair detection
- **0.5–0.7**: Good detection
- **0.7+**: Excellent detection

### Precision & Recall

- **Precision**: Of predicted boxes, how many are correct? (avoids false positives)
- **Recall**: Of ground truth boxes, how many are found? (avoids false negatives)
- **F1 Score**: Harmonic mean of precision and recall (0–1, higher is better)

Trade-off: higher precision = lower recall (and vice versa).

### Confidence Threshold

Lower confidence = more detections (higher recall, lower precision).
The precision-confidence curve helps identify the sweet spot for your application.

---

## Comparing Models

### In Azure ML Studio

1. Navigate to **Experiments** → **cartwatch-training**
2. Click **Compare** at the top
3. Select multiple runs to compare side-by-side
4. View metrics, parameters, artifacts in the comparison table

### In MLflow UI

```bash
mlflow ui
```

1. Select an experiment
2. View all runs with their metrics and parameters
3. Click a run to see detailed metrics, plots, and artifacts

### Programmatic Comparison

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get all runs in an experiment
experiment = client.get_experiment_by_name("cartwatch-training")
runs = client.search_runs(experiment_ids=[experiment.experiment_id])

# Compare metrics
for run in runs:
    print(f"Run {run.info.run_id}:")
    print(f"  mAP@50: {run.data.metrics.get('final/map50', 'N/A')}")
    print(f"  F1: {run.data.metrics.get('final/f1_score', 'N/A')}")
```

---

## Hyperparameter Tuning with MLflow

### Multiple Runs with Different Hyperparameters

```bash
# Run 1: Small batch, high learning rate
python scripts/train.py export1 --batch-size 4 --epochs 100 --experiment-name "tuning"

# Run 2: Large batch, low learning rate
python scripts/train.py export2 --batch-size 16 --epochs 100 --experiment-name "tuning"

# Run 3: Medium batch, medium learning rate
python scripts/train.py export3 --batch-size 8 --epochs 100 --experiment-name "tuning"
```

Then compare all three runs in MLflow UI to find the best hyperparameters.

---

## Production Deployment

### Register Best Model

In Azure ML Studio:

1. Navigate to the best run
2. Click **Register Model** button
3. Enter model name and version
4. Model is now in **Models** registry for deployment

### Or Programmatically

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

credential = DefaultAzureCredential()
ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

# Register model
model = ml_client.models.create_or_update(
    Model(
        path="models/checkpoints/fasterrcnn_resnet50_v1/best.pth",
        name="cartwatch-detector",
        version="1",
        description="Faster R-CNN trained on shelf items"
    )
)
```

---

## Troubleshooting

### "Failed to connect to Azure ML"

**Problem**: MLflow can't authenticate to Azure.

**Solutions**:
1. Check you're logged in: `az login`
2. Verify workspace name, resource group, subscription ID
3. Use `--no-mlflow` to skip Azure and use local MLflow
4. Check Azure credential chain: `az account show`

### "No metrics appear in Azure ML Studio"

**Problem**: Training completes but no metrics visible in Studio.

**Solutions**:
1. Check experiment name is correct
2. Verify `mlflow.start_run()` / `mlflow.end_run()` are called
3. Check training logs for errors
4. Refresh Studio UI (Ctrl+R / Cmd+R)

### "OOM (Out of Memory) during training"

**Solution**: Reduce batch size:
```bash
python scripts/train.py export --batch-size 4 --device cuda
```

### "Training very slow on CPU"

**Solution**: Use GPU if available:
```bash
python scripts/train.py export --device cuda
```

---

## Architecture

```
Training Script (train.py)
    ↓
FasterRCNNTrainer (trainer.py)
    ├→ Create datasets & dataloaders
    ├→ Initialize model
    ├→ Training loop:
    │   ├→ Forward pass, loss computation
    │   ├→ Backward pass, optimizer step
    │   └→ Log metrics to MLflow (log_training_metrics)
    ├→ Validation:
    │   ├→ Run FasterRCNNEvaluator
    │   ├→ Compute mAP, precision, recall, F1
    │   ├→ Generate visualizations
    │   └→ Log metrics to MLflow
    └→ Save best checkpoint
    ↓
MLflow / Azure ML
    ├→ Store metrics (time series)
    ├→ Store artifacts (plots, models)
    ├→ Store hyperparameters
    └→ Provide UI for comparison
    ↓
Azure ML Studio UI
    ├→ Experiment browser
    ├→ Metrics charts
    ├→ Run comparison
    ├→ Artifact explorer
    └→ Model registry
```

---

## Next Steps

1. **Initial training**: Run with small dataset, local MLflow
2. **Azure setup**: Configure Azure workspace, run with `--azure-workspace`
3. **Hyperparameter tuning**: Run multiple training runs, compare in MLflow
4. **Model selection**: Choose best model based on mAP + F1 trade-off
5. **Deployment**: Register model in Azure ML, deploy to production

---

## References

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Azure ML Documentation](https://learn.microsoft.com/en-us/azure/machine-learning/)
- [COCO Metrics](https://cocodataset.org/#detection-eval)
- [Object Detection Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics)
