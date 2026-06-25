from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

from src.utils.paths import get_project_root


@dataclass
class DetectorConfig:
    """Configuration for YOLO detector."""

    model_name: str = "yolo11n"
    confidence_threshold: float = 0.5
    device: str = "cpu"


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing."""

    target_frame_size: Tuple[int, int] = (640, 640)
    ssim_threshold: float = 0.95


@dataclass
class TrainingConfig:
    """Configuration for YOLO training."""

    epochs: int = 100
    batch_size: int = 16
    imgsz: int = 640
    val_split: float = 0.2
    patience: int = 50
    seed: int = 42
    device: str = "cpu"


@dataclass
class PathConfig:
    """Configuration for data paths."""

    project_root: Optional[Path] = None
    data_dir: Optional[Path] = None
    model_dir: Optional[Path] = None
    output_dir: Optional[Path] = None

    def __post_init__(self):
        if self.project_root is None:
            self.project_root = get_project_root()
        if self.data_dir is None:
            self.data_dir = self.project_root / "data"
        if self.model_dir is None:
            self.model_dir = self.project_root / "models"
        if self.output_dir is None:
            self.output_dir = self.project_root / "outputs"


@dataclass
class Config:
    """Main configuration class."""

    detector: Optional[DetectorConfig] = None
    preprocessing: Optional[PreprocessingConfig] = None
    paths: Optional[PathConfig] = None
    training: Optional[TrainingConfig] = None

    def __post_init__(self):
        if self.detector is None:
            self.detector = DetectorConfig()
        if self.preprocessing is None:
            self.preprocessing = PreprocessingConfig()
        if self.paths is None:
            self.paths = PathConfig()
        if self.training is None:
            self.training = TrainingConfig()

    @classmethod
    def default(cls) -> "Config":
        """Create a config with default values."""
        return cls()
