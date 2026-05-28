from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .config import class_names, configured_path


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)
    except (TypeError, ValueError):
        return None


def _array_to_list(value: Any) -> list[float]:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "cpu"):
        value = value.cpu().numpy()
    elif hasattr(value, "tolist"):
        return [float(item) for item in value.tolist()]
    try:
        return [float(item) for item in value]
    except TypeError:
        return []


def f1_score(precision: float | None, recall: float | None) -> float | None:
    """Compute F1 from precision and recall."""
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def extract_metrics(metrics: Any, names: dict[int, str]) -> dict[str, Any]:
    """Extract a compact metrics dictionary from Ultralytics validation output."""
    box = getattr(metrics, "box", None)
    precision = _safe_float(getattr(box, "mp", None))
    recall = _safe_float(getattr(box, "mr", None))

    payload: dict[str, Any] = {
        "precision": precision,
        "recall": recall,
        "f1": f1_score(precision, recall),
        "map50": _safe_float(getattr(box, "map50", None)),
        "map50_95": _safe_float(getattr(box, "map", None)),
    }

    per_class_p = _array_to_list(getattr(box, "p", None))
    per_class_r = _array_to_list(getattr(box, "r", None))
    per_class_map = _array_to_list(getattr(box, "maps", None))
    per_class: list[dict[str, Any]] = []
    for class_id, class_name in names.items():
        p = per_class_p[class_id] if class_id < len(per_class_p) else None
        r = per_class_r[class_id] if class_id < len(per_class_r) else None
        class_map = per_class_map[class_id] if class_id < len(per_class_map) else None
        per_class.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "precision": p,
                "recall": r,
                "f1": f1_score(p, r),
                "map50_95": class_map,
            }
        )
    payload["per_class"] = per_class
    return payload


def validate_yolo(
    config: dict[str, Any],
    model_path: str | Path,
    data_yaml: str | Path | None = None,
    split: str = "val",
    run_name: str | None = None,
) -> dict[str, Any]:
    """Run Ultralytics validation on a configured split."""
    from ultralytics import YOLO

    processed_root = configured_path(config, "processed_yolo_root")
    data_yaml = Path(data_yaml or processed_root / "data.yaml")
    output_root = configured_path(config, "outputs_root") / "metrics"
    output_root.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=int(config["model"]["imgsz"]),
        project=str(output_root),
        name=run_name or f"{split}_metrics",
        exist_ok=True,
        verbose=False,
    )

    payload = extract_metrics(metrics, class_names(config))
    payload["split"] = split
    payload["data_yaml"] = str(data_yaml)
    payload["model_path"] = str(model_path)

    metrics_path = output_root / f"{run_name or split}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    payload["metrics_path"] = str(metrics_path)
    return payload


def write_test_clip_yaml(config: dict[str, Any], clip_name: str) -> Path:
    """Create a temporary YOLO data YAML that evaluates one prefixed test clip."""
    processed_root = configured_path(config, "processed_yolo_root")
    output_root = configured_path(config, "outputs_root") / "eval_lists"
    output_root.mkdir(parents=True, exist_ok=True)

    image_paths = sorted((processed_root / "images" / "test").glob(f"{clip_name}_*.*"))
    if not image_paths:
        raise FileNotFoundError(f"No processed test images found for {clip_name}.")

    list_path = output_root / f"{clip_name}_images.txt"
    with list_path.open("w", encoding="utf-8") as file:
        for image_path in image_paths:
            file.write(str(image_path).replace("\\", "/") + "\n")

    yaml_path = output_root / f"{clip_name}_data.yaml"
    payload = {
        "path": str(processed_root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": str(list_path).replace("\\", "/"),
        "names": class_names(config),
    }
    with yaml_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)
    return yaml_path


def evaluate_standard_and_per_clip(
    config: dict[str, Any],
    model_path: str | Path,
) -> dict[str, Any]:
    """Evaluate validation, combined test, and each configured test clip."""
    processed_root = configured_path(config, "processed_yolo_root")
    data_yaml = processed_root / "data.yaml"

    results: dict[str, Any] = {
        "val": validate_yolo(config, model_path, data_yaml=data_yaml, split="val", run_name="val"),
        "test_overall": validate_yolo(
            config,
            model_path,
            data_yaml=data_yaml,
            split="test",
            run_name="test_overall",
        ),
        "test_by_clip": {},
    }

    for clip_name in config["data"]["test_clips"]:
        clip_yaml = write_test_clip_yaml(config, clip_name)
        results["test_by_clip"][clip_name] = validate_yolo(
            config,
            model_path,
            data_yaml=clip_yaml,
            split="test",
            run_name=f"test_{clip_name}",
        )
    return results
