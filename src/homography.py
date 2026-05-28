from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import resolve_path


def load_homography_config(path: str | Path) -> dict[str, Any]:
    """Load the saved manual homography point configuration."""
    path = resolve_path(path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_homography_config(config: dict[str, Any], path: str | Path) -> Path:
    """Save manual homography point configuration."""
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    return path


def normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize 2D points for numerically stable DLT."""
    points = np.asarray(points, dtype=np.float64)
    centroid = points.mean(axis=0)
    shifted = points - centroid
    mean_distance = np.mean(np.linalg.norm(shifted, axis=1))
    scale = np.sqrt(2.0) / mean_distance if mean_distance > 0 else 1.0
    transform = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    homogeneous = np.column_stack([points, np.ones(len(points))])
    normalized = (transform @ homogeneous.T).T
    return normalized[:, :2], transform


def compute_homography_dlt(
    source_points: list[list[float]] | np.ndarray,
    target_points: list[list[float]] | np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """Compute H such that target ~ H * source using manual DLT/SVD."""
    src = np.asarray(source_points, dtype=np.float64)
    dst = np.asarray(target_points, dtype=np.float64)

    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2:
        raise ValueError("source_points and target_points must both have shape (N, 2).")
    if len(src) < 4:
        raise ValueError("At least four point correspondences are required.")

    if normalize:
        src_used, src_transform = normalize_points(src)
        dst_used, dst_transform = normalize_points(dst)
    else:
        src_used, dst_used = src, dst
        src_transform = np.eye(3)
        dst_transform = np.eye(3)

    rows = []
    for (u, v), (x, y) in zip(src_used, dst_used):
        rows.append([-u, -v, -1.0, 0.0, 0.0, 0.0, x * u, x * v, x])
        rows.append([0.0, 0.0, 0.0, -u, -v, -1.0, y * u, y * v, y])
    matrix_a = np.asarray(rows, dtype=np.float64)

    _, _, vh = np.linalg.svd(matrix_a)
    h = vh[-1, :]
    h_normalized = h.reshape(3, 3)
    homography = np.linalg.inv(dst_transform) @ h_normalized @ src_transform

    if abs(homography[2, 2]) > 1e-12:
        homography = homography / homography[2, 2]
    return homography


def project_points(points: list[list[float]] | np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Project 2D points with a homography and return Euclidean coordinates."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(1, 2)
    homogeneous = np.column_stack([points, np.ones(len(points))])
    projected = (homography @ homogeneous.T).T
    projected = projected[:, :2] / projected[:, 2:3]
    return projected


def reprojection_errors(
    source_points: list[list[float]] | np.ndarray,
    target_points: list[list[float]] | np.ndarray,
    homography: np.ndarray,
) -> dict[str, Any]:
    """Compute reference-point reprojection errors in target pixel space."""
    target = np.asarray(target_points, dtype=np.float64)
    projected = project_points(source_points, homography)
    errors = np.linalg.norm(projected - target, axis=1)
    return {
        "errors": errors.tolist(),
        "mean": float(errors.mean()),
        "max": float(errors.max()),
        "projected_points": projected.tolist(),
    }


def select_image_points(image_path: str | Path, max_points: int | None = None) -> list[list[int]]:
    """Open an OpenCV window and let the user click calibration points."""
    import cv2

    image = cv2.imread(str(resolve_path(image_path)))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    points: list[list[int]] = []
    preview = image.copy()
    window_name = "Manual calibration: left click points, press Enter when done"

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        del flags, param
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if max_points is not None and len(points) >= max_points:
            return
        points.append([x, y])
        cv2.circle(preview, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(
            preview,
            str(len(points)),
            (x + 7, y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    while True:
        cv2.imshow(window_name, preview)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10):
            break
        if key == 27:
            points.clear()
            break
    cv2.destroyWindow(window_name)
    return points


def draw_reference_points(
    image_path: str | Path,
    points: list[list[float]],
    output_path: str | Path,
) -> Path:
    """Save a calibration frame with numbered reference points."""
    import cv2

    image = cv2.imread(str(resolve_path(image_path)))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    for index, (x, y) in enumerate(points, start=1):
        centre = (int(round(x)), int(round(y)))
        cv2.circle(image, centre, 6, (0, 0, 255), -1)
        cv2.putText(
            image,
            str(index),
            (centre[0] + 8, centre[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return output_path
