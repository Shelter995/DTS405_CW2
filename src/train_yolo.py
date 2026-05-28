from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import configured_path


def augmentation_off_kwargs() -> dict[str, float]:
    """Return Ultralytics training parameters that minimise augmentation."""
    return {
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
    }


def train_yolo(config: dict[str, Any], data_yaml: str | Path | None = None) -> dict[str, Any]:
    """Train the configured YOLO model and return important output paths."""
    from ultralytics import YOLO

    processed_root = configured_path(config, "processed_yolo_root")
    data_yaml = Path(data_yaml or processed_root / "data.yaml")
    runs_root = configured_path(config, "runs_root")
    detect_project = runs_root / "detect"

    model_config = config["model"]
    model = YOLO(model_config["base_model"])

    train_kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": int(model_config["epochs"]),
        "imgsz": int(model_config["imgsz"]),
        "batch": int(model_config["batch"]),
        "patience": int(model_config["patience"]),
        "seed": int(config["project"]["seed"]),
        "project": str(detect_project),
        "name": model_config["run_name"],
        "exist_ok": True,
        "verbose": True,
    }
    if bool(model_config.get("disable_augmentation", False)):
        train_kwargs.update(augmentation_off_kwargs())

    model.train(**train_kwargs)

    run_dir = detect_project / model_config["run_name"]
    best_weights = run_dir / "weights" / "best.pt"
    last_weights = run_dir / "weights" / "last.pt"
    return {
        "run_dir": str(run_dir),
        "best_weights": str(best_weights),
        "last_weights": str(last_weights),
        "data_yaml": str(data_yaml),
    }


def best_weights_path(config: dict[str, Any]) -> Path:
    """Return the expected best.pt path for the configured training run."""
    return (
        configured_path(config, "runs_root")
        / "detect"
        / config["model"]["run_name"]
        / "weights"
        / "best.pt"
    )
