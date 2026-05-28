from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import class_names, configured_path, resolve_path
from .dataset import list_images
from .homography import project_points
from .minimap import draw_court_template, draw_minimap_state, point_inside_template
from .visualization import draw_detections, overlay_pip, tuple_color


def _parse_result_boxes(result: Any) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.detach().cpu().numpy()
    cls = boxes.cls.detach().cpu().numpy().astype(int)
    conf = boxes.conf.detach().cpu().numpy()
    track_ids = None
    if getattr(boxes, "id", None) is not None:
        track_ids = boxes.id.detach().cpu().numpy().astype(int)

    detections: list[dict[str, Any]] = []
    for index, box in enumerate(xyxy):
        detections.append(
            {
                "xyxy": box.tolist(),
                "class_id": int(cls[index]),
                "confidence": float(conf[index]),
                "track_id": int(track_ids[index]) if track_ids is not None else None,
            }
        )
    return detections


def _bottom_centre(xyxy: list[float]) -> tuple[float, float]:
    x1, _, x2, y2 = xyxy
    return (x1 + x2) / 2.0, y2


def _box_centre(xyxy: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _resize_minimap(minimap: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(minimap, (width, height), interpolation=cv2.INTER_AREA)


def render_demo_video(
    config: dict[str, Any],
    model_path: str | Path,
    image_dir: str | Path,
    homography: np.ndarray,
    output_video_path: str | Path,
    frame_limit: int | None = None,
    snapshot_indices: set[int] | None = None,
    snapshot_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Render the clip2 demo with detections, tracking IDs, and a BEV mini-map."""
    from ultralytics import YOLO

    image_dir = resolve_path(image_dir)
    output_video_path = resolve_path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_dir_path = resolve_path(snapshot_dir) if snapshot_dir is not None else None
    if snapshot_dir_path is not None:
        snapshot_dir_path.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(image_dir, config["data"]["image_extensions"])
    if frame_limit is not None:
        image_paths = image_paths[: int(frame_limit)]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    first_frame = cv2.imread(str(image_paths[0]))
    if first_frame is None:
        raise FileNotFoundError(f"Could not read first frame: {image_paths[0]}")
    frame_h, frame_w = first_frame.shape[:2]

    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(config["visualization"]["fps"]),
        (frame_w, frame_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_video_path}")

    names = class_names(config)
    colors = {
        key: tuple_color(value)
        for key, value in config["visualization"]["colors_bgr"].items()
    }
    minimap_template = draw_court_template(
        template_size=(
            int(config["visualization"]["minimap_width"] * 2),
            int(config["visualization"]["minimap_height"] * 2),
        ),
        margin=int(config["visualization"]["court_margin"] * 2),
    )

    model = YOLO(str(model_path))
    raw_to_display_id: dict[int, int] = {}
    next_display_id = 1
    trail_length = int(config["visualization"]["trail_length"])
    tracks: dict[int, deque[tuple[float, float]]] = defaultdict(lambda: deque(maxlen=trail_length))
    ball_last: tuple[float, float] | None = None
    ball_missing = int(config["visualization"]["ball_missing_tolerance"]) + 1

    saved_snapshots: list[str] = []
    processed = 0
    try:
        for frame_index, image_path in enumerate(image_paths):
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue

            result = model.track(
                frame,
                persist=True,
                tracker=config["model"]["tracker"],
                conf=float(config["model"]["conf_threshold"]),
                verbose=False,
            )[0]
            detections = _parse_result_boxes(result)

            players_for_map: list[dict[str, Any]] = []
            ball_candidates: list[dict[str, Any]] = []

            for detection in detections:
                class_id = int(detection["class_id"])
                if class_id == 0 and detection.get("track_id") is not None:
                    raw_id = int(detection["track_id"])
                    if raw_id not in raw_to_display_id:
                        raw_to_display_id[raw_id] = next_display_id
                        next_display_id += 1
                    display_id = raw_to_display_id[raw_id]
                    detection["display_id"] = display_id

                    foot_point = _bottom_centre(detection["xyxy"])
                    projected = project_points([foot_point], homography)[0]
                    point = (float(projected[0]), float(projected[1]))
                    if point_inside_template(point, minimap_template):
                        tracks[display_id].append(point)
                        players_for_map.append({"display_id": display_id, "point": point})

                if class_id == 1:
                    ball_candidates.append(detection)

            if ball_candidates:
                best_ball = max(ball_candidates, key=lambda item: item["confidence"])
                ball_point = _box_centre(best_ball["xyxy"])
                projected_ball = project_points([ball_point], homography)[0]
                ball_candidate = (float(projected_ball[0]), float(projected_ball[1]))
                if point_inside_template(ball_candidate, minimap_template):
                    ball_last = ball_candidate
                    ball_missing = 0
                else:
                    ball_missing += 1
            else:
                ball_missing += 1

            ball_to_draw = None
            ball_alpha = 1.0
            tolerance = int(config["visualization"]["ball_missing_tolerance"])
            if ball_last is not None and ball_missing <= tolerance:
                ball_to_draw = ball_last
                ball_alpha = 1.0 - (ball_missing / max(1, tolerance + 1))

            annotated = draw_detections(frame, detections, names, colors)
            minimap = draw_minimap_state(
                minimap_template,
                players_for_map,
                {key: list(value) for key, value in tracks.items()},
                ball=ball_to_draw,
                ball_alpha=ball_alpha,
                colors_bgr=colors,
            )
            minimap = _resize_minimap(
                minimap,
                int(config["visualization"]["minimap_width"]),
                int(config["visualization"]["minimap_height"]),
            )
            combined = overlay_pip(
                annotated,
                minimap,
                position=config["visualization"]["overlay_position"],
                margin=int(config["visualization"]["overlay_margin"]),
            )

            writer.write(combined)
            processed += 1

            if snapshot_indices and frame_index in snapshot_indices and snapshot_dir_path is not None:
                snapshot_path = snapshot_dir_path / f"frame_{frame_index:04d}.jpg"
                cv2.imwrite(str(snapshot_path), combined)
                saved_snapshots.append(str(snapshot_path))
    finally:
        writer.release()

    return {
        "output_video": str(output_video_path),
        "frames_processed": processed,
        "snapshots": saved_snapshots,
    }


def default_clip2_image_dir(config: dict[str, Any]) -> Path:
    """Return the raw image directory for clip2 using configured data paths."""
    raw_root = configured_path(config, "raw_data_root")
    return raw_root / config["data"]["test_dir_name"] / "clip2" / "images"
