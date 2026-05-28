from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_like: str | Path, base: Path | None = None) -> Path:
    """Resolve a path string while allowing project-relative values."""
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    return (base or PROJECT_ROOT).joinpath(path).resolve()


def load_project_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the project YAML configuration."""
    path = resolve_path(config_path or PROJECT_ROOT / "configs" / "project.yaml")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config["_config_path"] = str(path)
    config["_project_root"] = str(PROJECT_ROOT)
    return config


def class_names(config: dict[str, Any]) -> dict[int, str]:
    """Return class names keyed by integer class id."""
    return {int(k): str(v) for k, v in config["classes"].items()}


def ensure_dir(path_like: str | Path) -> Path:
    """Create a directory if it does not already exist."""
    path = resolve_path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


def configured_path(config: dict[str, Any], key: str) -> Path:
    """Resolve a path from the config paths section."""
    return resolve_path(config["paths"][key])
