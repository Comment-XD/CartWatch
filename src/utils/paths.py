from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def get_data_dir() -> Path:
    """Get the data directory."""
    data_dir = get_project_root() / "data"
    return data_dir


def get_model_dir() -> Path:
    """Get the models directory."""
    model_dir = get_project_root() / "models"
    return model_dir


def ensure_dirs(*dirs: Path) -> None:
    """Ensure directories exist."""
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)


def resolve_path(path_str: str) -> Path:
    """Resolve a path string to a Path object."""
    return Path(path_str).resolve()
