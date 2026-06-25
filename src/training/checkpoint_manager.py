"""Checkpoint versioning and directory management for trained models."""

import re
from pathlib import Path
from typing import Tuple

from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_next_version(checkpoints_dir: Path, model_name: str) -> int:
    """Find the next available version number for a model.

    Scans checkpoints_dir for folders matching '<model_name>_v<N>' and
    returns max(N) + 1, or 1 if none exist.

    Args:
        checkpoints_dir: Path to checkpoints directory (e.g., models/checkpoints/)
        model_name: Model name to match (e.g., "yolo11n")

    Returns:
        Next version integer (>= 1)
    """
    pattern = re.compile(rf"^{re.escape(model_name)}_v(\d+)$")
    existing = []

    if checkpoints_dir.is_dir():
        for entry in checkpoints_dir.iterdir():
            if entry.is_dir():
                match = pattern.match(entry.name)
                if match:
                    existing.append(int(match.group(1)))

    return max(existing, default=0) + 1


def allocate_version_dir(checkpoints_dir: Path, model_name: str) -> Tuple[Path, int]:
    """Compute (but do not create) the next versioned checkpoint directory.

    Args:
        checkpoints_dir: Path to checkpoints directory (e.g., models/checkpoints/)
        model_name: Model name (e.g., "yolo11n")

    Returns:
        Tuple of (version_dir_path, version_number)
        Example: (models/checkpoints/yolo11n_v3, 3)
    """
    version = get_next_version(checkpoints_dir, model_name)
    version_dir = checkpoints_dir / f"{model_name}_v{version}"
    return version_dir, version
