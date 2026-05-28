from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from .config import class_names, configured_path, resolve_path


def natural_key(path: Path) -> list[Any]:
    """Sort frame names naturally, e.g. 2 before 10."""
    parts = re.split(r"(\d+)", path.stem)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def list_images(directory: str | Path, extensions: list[str]) -> list[Path]:
    """List non-metadata image files in natural order."""
    directory = resolve_path(directory)
    allowed = {ext.lower() for ext in extensions}
    files = [
        path
        for path in directory.iterdir()
        if path.is_file()
        and not path.name.startswith("._")
        and path.suffix.lower() in allowed
    ]
    return sorted(files, key=natural_key)


def label_for_image(image_path: Path, labels_dir: Path) -> Path:
    """Return the YOLO label path matching an image name."""
    return labels_dir / f"{image_path.stem}.txt"


def find_clip_dir(raw_root: Path, split_dir_name: str, clip_name: str) -> Path:
    """Find a clip directory, tolerating the train/trian spelling mismatch."""
    candidates = [raw_root / split_dir_name / clip_name]
    if split_dir_name == "trian":
        candidates.append(raw_root / "train" / clip_name)
    if split_dir_name == "train":
        candidates.append(raw_root / "trian" / clip_name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {clip_name}. Tried: {tried}")


def verify_clip(clip_dir: str | Path, extensions: list[str]) -> dict[str, Any]:
    """Check image/label alignment for one extracted clip."""
    clip_dir = resolve_path(clip_dir)
    images_dir = clip_dir / "images"
    labels_dir = clip_dir / "labels"
    images = list_images(images_dir, extensions)

    missing_labels = [
        str(label_for_image(image_path, labels_dir))
        for image_path in images
        if not label_for_image(image_path, labels_dir).exists()
    ]

    return {
        "clip_dir": str(clip_dir),
        "image_count": len(images),
        "missing_label_count": len(missing_labels),
        "missing_labels": missing_labels,
    }


def _copy_image_and_label(
    image_path: Path,
    source_labels_dir: Path,
    target_images_dir: Path,
    target_labels_dir: Path,
    clip_prefix: str,
) -> dict[str, str]:
    label_path = label_for_image(image_path, source_labels_dir)
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")

    target_stem = f"{clip_prefix}_{image_path.stem}"
    target_image = target_images_dir / f"{target_stem}{image_path.suffix.lower()}"
    target_label = target_labels_dir / f"{target_stem}.txt"

    shutil.copy2(image_path, target_image)
    shutil.copy2(label_path, target_label)

    return {"image": str(target_image), "label": str(target_label)}


def prepare_yolo_dataset(config: dict[str, Any], overwrite: bool = False) -> dict[str, Any]:
    """Create the standard YOLO directory layout from extracted clips."""
    raw_root = configured_path(config, "raw_data_root")
    processed_root = configured_path(config, "processed_yolo_root")
    outputs_root = configured_path(config, "outputs_root")
    extensions = config["data"]["image_extensions"]
    names = class_names(config)

    if overwrite and processed_root.exists():
        try:
            processed_root.resolve().relative_to(outputs_root.resolve())
        except ValueError as exc:
            raise ValueError(
                "Refusing to delete processed_yolo_root because it is not inside "
                f"outputs_root: {processed_root}"
            ) from exc
        shutil.rmtree(processed_root)

    for split in ("train", "val", "test"):
        (processed_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (processed_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    train_clip = config["data"]["train_clip"]
    train_clip_dir = find_clip_dir(raw_root, config["data"]["train_dir_name"], train_clip)
    train_images = list_images(train_clip_dir / "images", extensions)
    train_count = int(config["data"]["train_frames"])
    val_count = int(config["data"]["val_frames"])
    needed = train_count + val_count

    if len(train_images) < needed:
        raise ValueError(
            f"{train_clip} has {len(train_images)} images, but {needed} are required "
            f"for the configured train/val split."
        )

    summary: dict[str, Any] = {
        "processed_root": str(processed_root),
        "splits": {"train": [], "val": [], "test": []},
        "source_clips": {},
    }

    split_images = {
        "train": train_images[:train_count],
        "val": train_images[train_count:needed],
    }
    source_labels_dir = train_clip_dir / "labels"

    for split, images in split_images.items():
        for image_path in images:
            record = _copy_image_and_label(
                image_path,
                source_labels_dir,
                processed_root / "images" / split,
                processed_root / "labels" / split,
                train_clip,
            )
            summary["splits"][split].append(record)

    for clip_name in config["data"]["test_clips"]:
        clip_dir = find_clip_dir(raw_root, config["data"]["test_dir_name"], clip_name)
        summary["source_clips"][clip_name] = str(clip_dir)
        for image_path in list_images(clip_dir / "images", extensions):
            record = _copy_image_and_label(
                image_path,
                clip_dir / "labels",
                processed_root / "images" / "test",
                processed_root / "labels" / "test",
                clip_name,
            )
            summary["splits"]["test"].append(record)

    data_yaml = processed_root / "data.yaml"
    yaml_payload = {
        "path": str(processed_root).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }
    with data_yaml.open("w", encoding="utf-8") as file:
        yaml.safe_dump(yaml_payload, file, sort_keys=False, allow_unicode=True)
    summary["data_yaml"] = str(data_yaml)

    summary_path = processed_root / "split_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    summary["summary_path"] = str(summary_path)
    return summary


def count_labels(labels_dir: str | Path, names: dict[int, str]) -> dict[str, int]:
    """Count YOLO class labels in a labels directory."""
    labels_dir = resolve_path(labels_dir)
    counts: Counter[str] = Counter()
    for label_path in sorted(labels_dir.glob("*.txt")):
        with label_path.open("r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split()
                if not parts:
                    continue
                class_id = int(float(parts[0]))
                counts[names.get(class_id, str(class_id))] += 1
    return dict(counts)


def dataset_statistics(config: dict[str, Any]) -> dict[str, Any]:
    """Return image and label counts for the processed YOLO dataset."""
    processed_root = configured_path(config, "processed_yolo_root")
    names = class_names(config)
    stats: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        image_count = len(list((processed_root / "images" / split).glob("*.*")))
        stats[split] = {
            "images": image_count,
            "labels_by_class": count_labels(processed_root / "labels" / split, names),
        }
    return stats
