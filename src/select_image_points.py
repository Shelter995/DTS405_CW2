"""
Manually select four image points for Camera B homography calibration.

Usage:
1. Set REFERENCE_FRAME_PATH below to a clear frame from clip2.
2. Run this script.
3. Left-click four court reference points in the OpenCV window.
4. Press Enter to confirm, or Esc to cancel.

The selected points are printed and, by default, written into
configs/homography_camera_b.json as image_points_px.
"""

from __future__ import annotations

import json
from pathlib import Path

from .homography import draw_reference_points, select_image_points


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Edit this path before running.
REFERENCE_FRAME_PATH = Path(r"D:\path\to\clip2\images\0001.jpg")

HOMOGRAPHY_CONFIG_PATH = PROJECT_ROOT / "configs" / "homography_camera_b.json"
ANNOTATED_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "calibration" / "camera_b_reference_points.jpg"

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


def main() -> None:
    reference_frame = REFERENCE_FRAME_PATH.expanduser()
    if not reference_frame.exists():
        raise FileNotFoundError(
            "REFERENCE_FRAME_PATH does not exist. Edit the path at the top of "
            f"this script first: {reference_frame}"
        )

    points = select_image_points(reference_frame, max_points=MAX_POINTS)
    if len(points) != MAX_POINTS:
        print("Point selection cancelled or incomplete. No config was updated.")
        print(f"Selected points: {points}")
        return

    print("\nSelected image_points_px:")
    print(json.dumps(points, indent=2))

    if SAVE_TO_CONFIG:
        config = load_json(HOMOGRAPHY_CONFIG_PATH)
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
