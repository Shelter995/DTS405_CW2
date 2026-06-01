"""Manually select four image points from the first frame of Demo Video.mp4.

Usage:
1. Make sure data/Demo Video.mp4 exists.
2. Run this script.
3. The script saves the first video frame to outputs/calibration.
4. Left-click four court reference points in the OpenCV window.
5. Press Enter to confirm, or Esc to cancel.

The selected points are printed and, by default, written into
configs/homography_camera_b.json as image_points_px.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.homography import draw_reference_points, select_image_points

DEMO_VIDEO_PATH = PROJECT_ROOT / "data" / "Demo Video.mp4"
HOMOGRAPHY_CONFIG_PATH = PROJECT_ROOT / "configs" / "homography_camera_b.json"
REFERENCE_FRAME_PATH = PROJECT_ROOT / "outputs" / "calibration" / "demo_video_first_frame.jpg"
ANNOTATED_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "calibration" / "demo_video_reference_points.jpg"

MAX_POINTS = 4
SAVE_TO_CONFIG = True
SAVE_ANNOTATED_IMAGE = True


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def extract_first_frame(video_path: Path, output_path: Path) -> Path:
    if not video_path.exists():
        raise FileNotFoundError(f"Demo video does not exist: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open demo video: {video_path}")

    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read the first frame from: {video_path}")

    cv2.imwrite(str(output_path), frame)
    return output_path


def main() -> None:
    reference_frame = extract_first_frame(DEMO_VIDEO_PATH, REFERENCE_FRAME_PATH)
    print(f"Extracted first frame: {reference_frame}")

    points = select_image_points(reference_frame, max_points=MAX_POINTS)
    if len(points) != MAX_POINTS:
        print("Point selection cancelled or incomplete. No config was updated.")
        print(f"Selected points: {points}")
        return

    print("\nSelected image_points_px:")
    print(json.dumps(points, indent=2))

    if SAVE_TO_CONFIG:
        config = load_json(HOMOGRAPHY_CONFIG_PATH)
        config["source_clip"] = "Demo Video.mp4"
        config["reference_frame"] = str(reference_frame)
        config["image_points_px"] = points
        save_json(HOMOGRAPHY_CONFIG_PATH, config)
        print(f"\nUpdated config: {HOMOGRAPHY_CONFIG_PATH}")

    if SAVE_ANNOTATED_IMAGE:
        output_path = draw_reference_points(reference_frame, points, ANNOTATED_OUTPUT_PATH)
        print(f"Saved annotated reference frame: {output_path}")

    print("\nNext step: fill template_points_m in configs/homography_camera_b.json")
    print("Make sure each template point uses the same order as image_points_px.")


if __name__ == "__main__":
    main()
