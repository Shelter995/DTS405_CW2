from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def tuple_color(value: list[int] | tuple[int, int, int]) -> tuple[int, int, int]:
    return int(value[0]), int(value[1]), int(value[2])


def draw_text_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    """Draw readable text with a filled background."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    y = max(y, text_size[1] + 4)
    cv2.rectangle(
        image,
        (x, y - text_size[1] - baseline - 4),
        (x + text_size[0] + 6, y + baseline),
        color,
        -1,
    )
    cv2.putText(image, text, (x + 3, y - 4), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def draw_detections(
    frame: np.ndarray,
    detections: list[dict[str, Any]],
    class_names: dict[int, str],
    colors_bgr: dict[str, tuple[int, int, int]],
) -> np.ndarray:
    """Draw all class boxes on a frame."""
    canvas = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in detection["xyxy"]]
        class_id = int(detection["class_id"])
        class_name = class_names.get(class_id, str(class_id))
        color = colors_bgr.get(class_name, (255, 255, 255))
        confidence = detection.get("confidence")

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = class_name
        if class_name == "player" and detection.get("display_id") is not None:
            label = f"player {detection['display_id']}"
        if confidence is not None:
            label = f"{label} {float(confidence):.2f}"
        draw_text_label(canvas, label, (x1, y1 - 4), color)
    return canvas


def overlay_pip(
    frame: np.ndarray,
    pip: np.ndarray,
    position: str = "top_right",
    margin: int = 20,
    border_color: tuple[int, int, int] = (245, 245, 245),
) -> np.ndarray:
    """Overlay a picture-in-picture image on a frame."""
    canvas = frame.copy()
    frame_h, frame_w = canvas.shape[:2]
    pip_h, pip_w = pip.shape[:2]

    if position == "top_right":
        x = frame_w - pip_w - margin
        y = margin
    elif position == "top_left":
        x = margin
        y = margin
    elif position == "bottom_right":
        x = frame_w - pip_w - margin
        y = frame_h - pip_h - margin
    elif position == "bottom_center":
        x = (frame_w - pip_w) // 2
        y = frame_h - pip_h - margin
    else:
        raise ValueError(f"Unsupported PIP position: {position}")

    x = max(0, min(x, frame_w - pip_w))
    y = max(0, min(y, frame_h - pip_h))

    cv2.rectangle(canvas, (x - 2, y - 2), (x + pip_w + 2, y + pip_h + 2), border_color, 2)
    canvas[y : y + pip_h, x : x + pip_w] = pip
    return canvas
