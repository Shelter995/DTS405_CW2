from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import class_names, resolve_path
from .homography import load_homography_config, project_points
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


def _box_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _copy_track_ids_to_predict_boxes(
    predict_detections: list[dict[str, Any]],
    tracked_detections: list[dict[str, Any]],
    min_iou: float = 0.3,
) -> None:
    """Attach tracker display IDs to matching prediction boxes for drawing."""
    for tracked in tracked_detections:
        best_detection: dict[str, Any] | None = None
        best_iou = 0.0
        for detection in predict_detections:
            if int(detection["class_id"]) != int(tracked["class_id"]):
                continue
            if detection.get("display_id") is not None:
                continue
            iou = _box_iou(detection["xyxy"], tracked["xyxy"])
            if iou > best_iou:
                best_iou = iou
                best_detection = detection
        if best_detection is not None and best_iou >= min_iou:
            best_detection["display_id"] = tracked["display_id"]


def _resize_minimap(minimap: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(minimap, (width, height), interpolation=cv2.INTER_AREA)


def render_demo_video(
    config: dict[str, Any],
    model_path: str | Path,
    video_path: str | Path,
    homography: np.ndarray,
    output_video_path: str | Path,
    frame_limit: int | None = None,
    snapshot_indices: set[int] | None = None,
    snapshot_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Render the demo video with detections, tracking IDs, and a BEV mini-map."""
    from ultralytics import YOLO

    video_path = resolve_path(video_path)
    output_video_path = resolve_path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_dir_path = resolve_path(snapshot_dir) if snapshot_dir is not None else None
    if snapshot_dir_path is not None:
        snapshot_dir_path.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open demo video: {video_path}")

    ok, first_frame = capture.read()
    if not ok or first_frame is None:
        capture.release()
        raise RuntimeError(f"Could not read first frame from demo video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_h, frame_w = first_frame.shape[:2]
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    fps = source_fps if source_fps and source_fps > 0 else float(config["visualization"]["fps"])

    writer = cv2.VideoWriter(
        str(output_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (frame_w, frame_h),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open video writer: {output_video_path}")

    names = class_names(config)
    colors = {
        key: tuple_color(value)
        for key, value in config["visualization"]["colors_bgr"].items()
    }
    homography_config = load_homography_config(config["paths"]["homography_config"])
    minimap_template = draw_court_template(
        template_size=tuple(int(value) for value in homography_config["template_size"]),
        margin=int(homography_config["court_margin_px"]),
    )

    model = YOLO(str(model_path))
    raw_player_to_display_id: dict[int, int] = {}
    raw_referee_to_display_id: dict[int, int] = {}
    next_player_id = 1
    next_referee_id = 1
    trail_length = int(config["visualization"]["trail_length"])
    player_tracks: dict[int, deque[tuple[float, float]]] = defaultdict(lambda: deque(maxlen=trail_length))
    referee_tracks: dict[int, deque[tuple[float, float]]] = defaultdict(lambda: deque(maxlen=trail_length))
    ball_last: tuple[float, float] | None = None
    ball_missing = int(config["visualization"]["ball_missing_tolerance"]) + 1

    saved_snapshots: list[str] = []
    processed = 0
    try:
        frame_index = 0
        while True:
            if frame_limit is not None and frame_index >= int(frame_limit):
                break
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            predict_result = model.predict(
                frame,
                conf=float(config["model"]["conf_threshold"]),
                verbose=False,
            )[0]
            detections = _parse_result_boxes(predict_result)

            track_result = model.track(
                frame,
                persist=True,
                tracker=config["model"]["tracker"],
                conf=float(config["model"]["conf_threshold"]),
                verbose=False,
            )[0]
            track_detections = _parse_result_boxes(track_result)

            players_for_map: list[dict[str, Any]] = []
            referees_for_map: list[dict[str, Any]] = []
            tracked_for_labels: list[dict[str, Any]] = []
            ball_candidates: list[dict[str, Any]] = []

            for detection in detections:
                if int(detection["class_id"]) == 1:
                    ball_candidates.append(detection)

            for detection in track_detections:
                class_id = int(detection["class_id"])
                if class_id == 0 and detection.get("track_id") is not None:
                    raw_id = int(detection["track_id"])
                    if raw_id not in raw_player_to_display_id:
                        raw_player_to_display_id[raw_id] = next_player_id
                        next_player_id += 1
                    display_id = raw_player_to_display_id[raw_id]
                    detection["display_id"] = display_id

                    foot_point = _bottom_centre(detection["xyxy"])
                    projected = project_points([foot_point], homography)[0]
                    point = (float(projected[0]), float(projected[1]))
                    if point_inside_template(point, minimap_template):
                        player_tracks[display_id].append(point)
                        players_for_map.append({"display_id": display_id, "point": point})
                    tracked_for_labels.append(detection)

                if class_id == 2 and detection.get("track_id") is not None:
                    raw_id = int(detection["track_id"])
                    if raw_id not in raw_referee_to_display_id:
                        raw_referee_to_display_id[raw_id] = next_referee_id
                        next_referee_id += 1
                    display_id = raw_referee_to_display_id[raw_id]
                    detection["display_id"] = display_id

                    foot_point = _bottom_centre(detection["xyxy"])
                    projected = project_points([foot_point], homography)[0]
                    point = (float(projected[0]), float(projected[1]))
                    if point_inside_template(point, minimap_template):
                        referee_tracks[display_id].append(point)
                        referees_for_map.append({"display_id": display_id, "point": point})
                    tracked_for_labels.append(detection)

            _copy_track_ids_to_predict_boxes(detections, tracked_for_labels)

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
                referees_for_map,
                {key: list(value) for key, value in player_tracks.items()},
                {key: list(value) for key, value in referee_tracks.items()},
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
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    return {
        "output_video": str(output_video_path),
        "frames_processed": processed,
        "source_video": str(video_path),
        "fps": float(fps),
        "snapshots": saved_snapshots,
    }
